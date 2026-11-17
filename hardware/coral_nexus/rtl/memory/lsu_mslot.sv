// lsu_mslot.sv — C031 / W4-03: real multi-Context scalar LSU owner over the
// real C029 l1_bank bytes.
//
// Scope (G2_scalar.md #c031, W4 plan §5.3): the scalar LSU itself. It owns a
// bounded per-Context record table (LSU_CREDIT_CAP sealed slots), translates
// trusted per-Context window tables into bank segments, drives one real
// l1_bank instance (one request port per Context) and hands cached-window
// reads to the C031 miss owner (lsu_cache_path) through the independent
// cache_req/cache_rsp channels. The bank's C030 cache data port is bridged
// 1:1 (cp_i/cp_o); its SPM ports are driven only by this module.
//
// Ordering contract (MEM-01..08; MEM-02 no head-of-line serialization):
//   * Launch is ingress-ordered per engine. The bank engine is single-record
//     per Context: a younger bank access never issues before the previous
//     bank record's fin released the holder, so a same-Context load can never
//     overtake a previous store's effects (MEM-06 alias store->load ordering
//     is enforced through the bank fin; SPM effects are never reordered).
//   * The cache engine is multi-record: an older miss waiting external data
//     never blocks younger launches on either engine, and completions are
//     UNORDERED — the response channel presents the oldest READY (terminal)
//     record and never blocks behind an unfinished earlier instruction.
//     Retirement order is preserved by the actual ROB (C026), not here.
//   * MEM-06 fence: request_t.fence with byte_count==0 is the only legal
//     zero-byte request; it performs no bank/cache access, blocks younger
//     launches on both engines while queued, and completes COMPLETED once
//     every older record of its Context reached a terminal state. No other
//     fence semantics are claimed; byte_count!=0 with fence=1 is a fault.
//
// Trusted window table (installed only through the config port while the
// Context has zero outstanding records — generation reuse refused until
// drain). Window mapping (per contract): i = (address-base)/stripe_bytes,
// bank = i % BANK_COUNT, bank_addr = spm_base + (i/BANK_COUNT)*stripe_bytes
// + (address-base) % stripe_bytes. Window bounds are checked before any side
// effect. Cached windows are readable-only, and because the miss engine
// authorizes WHOLE-LINE refills, a cached window whose binding_offset or size
// is not LINE_BYTES-aligned is rejected at config (no silent overread past
// the trusted bound; no sub-line path exists in C031). LSU data stores to
// external HBM are forbidden (ADR-06): stores never take the cached path.
//
// Owner/identity: every accepted request yields exactly one owned completion
// (COMPLETED / ISOLATED / FAULTED). ERR_OWNER covers ctx_slot != port index,
// generation != installed table generation (stale or future), no table
// installed, and a duplicate (ctx_slot, ctx_generation, owner_op_seq,
// subop_id) key still outstanding (a record stays duplicate-relevant until
// its completion is consumed). Faults complete with no memory side effects.
//
// Cancellation (C031-local settle; the cross-module cancel tracker is C032):
// cancel_i marks ONLY records accepted before the cancel edge — the marking
// snapshot is fixed at edge start and the very same edge refuses ingress on
// that Context (one-cycle admission bubble) so a new live operation can never
// leak behind the snapshot. Unissued records (including fences) complete
// ISOLATED immediately; an issued store finishes its effects and completes
// COMPLETED; an issued read drains fully, then completes ISOLATED; a
// cache-waiting read forwards the miss owner's settled final state (the
// shared refill obligation is the miss owner's, never cancelled here). There
// is NO sticky generation disable: once the marked records settle, new
// same-generation requests are accepted without any reconfiguration. A cancel
// whose generation does not match the installed table is a counted no-op
// (stale_o) and does not block ingress.
//
// Credits/conservation: one record slot per accepted request; the slot (and
// the credit) is released exactly at the accepted completion response, so
// accepted_o - terminal_o == outstanding_o holds at every edge (asserted
// live, C031_ASSERT). rsp payloads are held stable under backpressure;
// per-Context response backpressure is independent.
//
// bank_busy_o derives from this module's own active holder channels (the
// segment masks of records between bank accept and fin ack). The cp hold is
// separate by construction (the bank keeps it out of its holder machinery)
// and is observable only through cp_o.busy.
//
// No fake data and no timers: every completion byte comes from the real bank
// beat streams or the real miss-owner response; progress is driven purely by
// handshakes. Assert markers are C031_ASSERT (C029 markers inside l1_bank
// remain that module's).
`timescale 1ns/1ps

module lsu_mslot
  import c029_params_pkg::*;
  import c031_lsu_pkg::*;
#(
  parameter int CONTEXT_COUNT = C2_CONTEXTS,
  parameter int WINDOW_COUNT = SLOT_COUNT
) (
  input logic clk,
  input logic rst_n,

  // Scalar ingress, one channel per Context.
  input  logic [CONTEXT_COUNT-1:0] req_valid_i,
  output logic [CONTEXT_COUNT-1:0] req_ready_o,
  input  request_t req_i [CONTEXT_COUNT],

  // Completions, one independent channel per Context (unordered; oldest
  // ready record presented; payload stable under backpressure).
  output logic [CONTEXT_COUNT-1:0] rsp_valid_o,
  input  logic [CONTEXT_COUNT-1:0] rsp_ready_i,
  output completion_t rsp_o [CONTEXT_COUNT],

  // C031-local cancellation (per accepted records; see header).
  input  logic [CONTEXT_COUNT-1:0] cancel_i,
  input  logic [15:0] cancel_generation_i [CONTEXT_COUNT],

  // Trusted per-Context window table install (quiescent-only).
  input  logic config_valid_i,
  output logic config_ready_o,
  output logic config_reject_o,
  input  logic [31:0] config_context_i,
  input  logic [15:0] config_generation_i,
  input  window_t config_windows_i [WINDOW_COUNT],

  // Global launch gate (data phases always drain).
  input logic issue_enable_i,

  // C030 cache data port bridge into the owned l1_bank (refill/fill traffic
  // of the miss owner; independent of the SPM channels).
  input  cache_port_request_t cp_i,
  output cache_port_response_t cp_o,

  // Miss-owner quiescence input, forwarded to the bank's profile gate.
  input logic cache_quiescent_i,

  // Profile switch passthrough: the bank is the authority (args check,
  // strictly increasing generation, quiescent service); bank_profile_o
  // mirrors the bank's active profile for the miss owner.
  input  logic profile_valid_i,
  input  logic [$clog2(PROFILE_COUNT)-1:0] profile_index_i,
  input  logic [31:0] profile_generation_i,
  output logic profile_ack_o,
  output logic profile_reject_o,
  output logic [$clog2(PROFILE_COUNT)-1:0] bank_profile_o,

  // Cached-window read launches into lsu_cache_path (multi-record per
  // Context, ingress ordered; record_id is the LSU record slot id).
  output logic [CONTEXT_COUNT-1:0] cache_req_valid_o,
  input  logic [CONTEXT_COUNT-1:0] cache_req_ready_i,
  output cache_request_t cache_req_o [CONTEXT_COUNT],
  input  logic [CONTEXT_COUNT-1:0] cache_rsp_valid_i,
  output logic [CONTEXT_COUNT-1:0] cache_rsp_ready_o,
  input  cache_response_t cache_rsp_i [CONTEXT_COUNT],

  // Per-Context attribution/conservation counters.
  output logic [31:0] outstanding_o [CONTEXT_COUNT],
  output logic [63:0] accepted_o [CONTEXT_COUNT],
  output logic [63:0] terminal_o [CONTEXT_COUNT],
  output logic [63:0] wait_bank_o [CONTEXT_COUNT],
  output logic [63:0] wait_mshr_o [CONTEXT_COUNT],
  output logic [63:0] stale_o [CONTEXT_COUNT],

  // This module's active bank holder channels (cp hold separate).
  output logic [BANK_COUNT-1:0] bank_busy_o,

  output logic invariant_error_o
);

  // ----------------------------------------------------------- geometry
  localparam int unsigned CAP = LSU_CREDIT_CAP;
  localparam int unsigned MAX_SEG = BANK_COUNT;
  localparam int unsigned BANK_IDX_W = (BANK_COUNT <= 1) ? 1 : $clog2(BANK_COUNT);
  localparam int unsigned BANK_ADDR_W = (BANK_BYTES <= 1) ? 1 : $clog2(BANK_BYTES);
  localparam int unsigned BANK_LEN_W = $clog2(BANK_BYTES + 1);
  localparam int unsigned SEG_CNT_W = $clog2(MAX_SEG + 1);
  localparam int unsigned MAX_PIECES = 4;  // byte_count <= 4, one run per stripe
  localparam int NONE = CAP;               // selection sentinel
  localparam logic [2:0] S_FREE = 3'd0, S_REG = 3'd1, S_CWAIT = 3'd2,
    S_BWD = 3'd3, S_BRD = 3'd4, S_BFIN = 3'd5, S_TERM = 3'd6;

  initial begin : shape_checks
    assert (CONTEXT_COUNT == C1_CONTEXTS || CONTEXT_COUNT == C2_CONTEXTS)
      else $fatal(1, "C031_ASSERT lsu_mslot unsupported Context count");
    assert (CAP >= 1 && CAP <= SLOT_COUNT)
      else $fatal(1, "C031_ASSERT lsu_mslot credit geometry");
    assert (WINDOW_COUNT == SLOT_COUNT)
      else $fatal(1, "C031_ASSERT lsu_mslot window table geometry");
  end

  // ------------------------------------------------------------ helpers
  // Segment math mirrors the bank's int-domain helpers (geometry-derived,
  // not bank state).
  function automatic int unsigned seg_beats(input logic [BANK_ADDR_W-1:0] adr,
                                            input logic [BANK_LEN_W-1:0] len);
    seg_beats = ((int'(adr) % DATA_BYTES) + int'(len) + DATA_BYTES - 1) / DATA_BYTES;
  endfunction

  function automatic logic [DATA_BYTES-1:0] beat_en(input logic [BANK_ADDR_W-1:0] adr,
                                                    input logic [BANK_LEN_W-1:0] len,
                                                    input int unsigned k);
    int unsigned nb;
    int unsigned last_off;
    beat_en = '1;
    nb = seg_beats(adr, len);
    last_off = ((int'(adr) % DATA_BYTES) + int'(len) - 1) % DATA_BYTES;
    if (k == 0) begin
      for (int b = 0; b < DATA_BYTES; b++) begin
        if (b < (int'(adr) % DATA_BYTES)) beat_en[b] = 1'b0;
      end
    end
    if (k == nb - 1) begin
      for (int b = 0; b < DATA_BYTES; b++) begin
        if (b > last_off) beat_en[b] = 1'b0;
      end
    end
  endfunction

  // Write beat payload: access byte (poff + (bank byte - seg_adr)) lands in
  // the lane of its DATA_BYTES word. Lanes outside the piece may alias other
  // access bytes; the bank masks them with the segment byte enables anyway.
  function automatic logic [DATA_BYTES*8-1:0] wbeat_data(
      input logic [31:0] wdata, input logic [2:0] bc,
      input logic [BANK_ADDR_W-1:0] seg_adr, input logic [1:0] poff,
      input int unsigned k);
    int unsigned word_base;
    int off;
    wbeat_data = '0;
    word_base = ((int'(seg_adr) / DATA_BYTES) + k) * DATA_BYTES;
    for (int b = 0; b < DATA_BYTES; b++) begin
      off = int'(word_base + int'(b)) - int'(seg_adr) + int'(poff);
      if (off >= 0 && off < int'(bc)) begin
        wbeat_data[8*b +: 8] = wdata[8*off +: 8];
      end
    end
  endfunction

  // Requested-byte mask of an access (bc bytes from lane 0).
  function automatic logic [3:0] req_mask(input logic [2:0] bc);
    case (int'(bc))
      1: req_mask = 4'b0001;
      2: req_mask = 4'b0011;
      default: req_mask = 4'b1111;
    endcase
  endfunction

  // Scalar load extension over right-justified bytes (SPM assembly and miss
  // owner responses share this convention).
  function automatic logic [31:0] ext_data(input logic [31:0] rj,
                                           input logic [2:0] bc,
                                           input logic unsigned_load);
    case (int'(bc))
      1: ext_data = unsigned_load ? {24'b0, rj[7:0]} : {{24{rj[7]}}, rj[7:0]};
      2: ext_data = unsigned_load ? {16'b0, rj[15:0]} : {{16{rj[15]}}, rj[15:0]};
      default: ext_data = rj;
    endcase
  endfunction

  function automatic logic [3:0] bank_rej_map(input logic [2:0] code);
    // Bank request faults 1..5 (shape/len0/overflow/reserved/cache window)
    // are address-range class; generation exhaustion is a resource fault.
    bank_rej_map = (int'(code) <= 5) ? ERR_RANGE : ERR_MEMORY;
  endfunction

  // Age order over outstanding records: allocation seq with modular compare;
  // at most CAP records outstanding so the forward distance never reaches 16.
  function automatic logic seq_older(input logic [4:0] a, input logic [4:0] b);
    int unsigned d;
    seq_older = 1'b0;
    if (a != b) begin
      d = (int'(b) - int'(a) + 32) % 32;
      seq_older = (d >= 1) && (d <= 15);
    end
  endfunction

  function automatic logic [63:0] mk_txn(input int c, input int rid,
                                         input owner_t o);
    // Unique among all outstanding bank holders: generation is unique per
    // incarnation (quiescent-gated config), rid unique per Context.
    mk_txn = {o.ctx_generation, o.owner_op_seq, 8'(c), 4'(rid), 4'b0};
  endfunction

  // ------------------------------------------------------- record table
  logic [2:0] s_state [CONTEXT_COUNT][CAP];
  logic [4:0] s_seq [CONTEXT_COUNT][CAP];
  owner_t s_owner [CONTEXT_COUNT][CAP];
  logic [31:0] s_pc [CONTEXT_COUNT][CAP];
  logic [31:0] s_addr [CONTEXT_COUNT][CAP];
  logic [31:0] s_wdata [CONTEXT_COUNT][CAP];
  logic [31:0] s_tdata [CONTEXT_COUNT][CAP];
  logic [4:0] s_rd [CONTEXT_COUNT][CAP];
  logic [2:0] s_bc [CONTEXT_COUNT][CAP];
  logic s_write [CONTEXT_COUNT][CAP];
  logic s_uns [CONTEXT_COUNT][CAP];
  logic s_flt [CONTEXT_COUNT][CAP];
  logic s_fence [CONTEXT_COUNT][CAP];
  logic s_cancelled [CONTEXT_COUNT][CAP];
  logic s_cached [CONTEXT_COUNT][CAP];
  logic [3:0] s_err [CONTEXT_COUNT][CAP];
  logic [1:0] s_final [CONTEXT_COUNT][CAP];
  logic [2:0] s_segcnt [CONTEXT_COUNT][CAP];
  logic [MAX_PIECES-1:0][BANK_IDX_W-1:0] s_seg_bank [CONTEXT_COUNT][CAP];
  logic [MAX_PIECES-1:0][BANK_ADDR_W-1:0] s_seg_addr [CONTEXT_COUNT][CAP];
  logic [MAX_PIECES-1:0][BANK_LEN_W-1:0] s_seg_len [CONTEXT_COUNT][CAP];
  // Offset of each piece's first byte within the access (byte lane 0).
  logic [MAX_PIECES-1:0][1:0] s_seg_off [CONTEXT_COUNT][CAP];
  logic [31:0] s_bid [CONTEXT_COUNT][CAP];
  logic [31:0] s_bgen [CONTEXT_COUNT][CAP];
  logic [31:0] s_coff [CONTEXT_COUNT][CAP];
  logic [31:0] s_arena [CONTEXT_COUNT][CAP];
  logic [31:0] s_scope [CONTEXT_COUNT][CAP];
  logic [63:0] s_txn [CONTEXT_COUNT][CAP];
  logic [63:0] s_bgen_cap [CONTEXT_COUNT][CAP];
  logic [3:0] s_wseg [CONTEXT_COUNT][CAP];
  logic [3:0] s_wbeat [CONTEXT_COUNT][CAP];
  logic [3:0] s_rseg [CONTEXT_COUNT][CAP];
  logic [3:0] s_rbeat [CONTEXT_COUNT][CAP];
  logic [3:0] s_got [CONTEXT_COUNT][CAP];
  logic [31:0] s_rbytes [CONTEXT_COUNT][CAP];

  // ------------------------------------------------------ context state
  window_t c_windows [CONTEXT_COUNT][WINDOW_COUNT];
  logic c_valid [CONTEXT_COUNT];
  logic [15:0] c_gen [CONTEXT_COUNT];
  logic [3:0] c_tail [CONTEXT_COUNT];
  logic [4:0] c_count [CONTEXT_COUNT];
  logic [4:0] c_allocseq [CONTEXT_COUNT];
  logic [63:0] c_acc [CONTEXT_COUNT];
  logic [63:0] c_term [CONTEXT_COUNT];
  logic [63:0] c_wbank [CONTEXT_COUNT];
  logic [63:0] c_wmshr [CONTEXT_COUNT];
  logic [63:0] c_stale [CONTEXT_COUNT];

  // ------------------------------------------------------------ globals
  logic [BANK_COUNT-1:0] bank_held;
  logic cfg_reject_q;
  logic fwd_gen_valid_q;
  logic [31:0] fwd_gen_q;
  logic [31:0] prof_gen32;
  logic inv_q;
  logic rsp_unstable_q;
  completion_t rsp_shadow [CONTEXT_COUNT];
  logic rsp_hold [CONTEXT_COUNT];
  // Parked response presentation: once rsp_valid_o rises, valid+payload are
  // held in registers until the fire edge; the scan reloads only when the
  // channel is idle. A newly-terminal OLDER record can never swap the
  // payload under backpressure (rsp_stable_under_backpressure).
  logic rsp_hold_q [CONTEXT_COUNT];
  logic [RECORD_W-1:0] rsp_slot_q [CONTEXT_COUNT];
  completion_t rsp_reg [CONTEXT_COUNT];

  // -------------------------------------------------- l1_bank wire fans
  logic [CONTEXT_COUNT-1:0] breq_valid, breq_write;
  logic [CONTEXT_COUNT-1:0][63:0] breq_txn;
  logic [CONTEXT_COUNT-1:0][SEG_CNT_W-1:0] breq_segcnt;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_IDX_W-1:0] breq_bank;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_ADDR_W-1:0] breq_addr;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_LEN_W-1:0] breq_len;
  logic [CONTEXT_COUNT-1:0] breq_ready, breq_rej;
  logic [CONTEXT_COUNT-1:0][2:0] breq_rej_code;
  logic [CONTEXT_COUNT-1:0][63:0] breq_generation;
  logic [CONTEXT_COUNT-1:0] bdone;
  logic [CONTEXT_COUNT-1:0][63:0] bdone_txn;
  logic [CONTEXT_COUNT-1:0] bw_valid, bw_ready;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0] bw_data;
  logic [CONTEXT_COUNT-1:0] br_valid, br_ready;
  logic [CONTEXT_COUNT-1:0][63:0] br_txn;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0] br_data;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES-1:0] br_ben;
  logic [CONTEXT_COUNT-1:0] bfin_valid, bfin_ack, bfin_rej;
  logic [CONTEXT_COUNT-1:0][63:0] bfin_txn, bfin_gen;

  l1_bank #(
    .CONTEXT_COUNT (CONTEXT_COUNT)
  ) ubank (
    .clk                  (clk),
    .rstn                 (rst_n),
    .req_valid_i          (breq_valid),
    .req_ready_o          (breq_ready),
    .req_rej_o            (breq_rej),
    .req_rej_code_o       (breq_rej_code),
    .req_txn_i            (breq_txn),
    .req_generation_o     (breq_generation),
    .req_write_i          (breq_write),
    .req_seg_count_i      (breq_segcnt),
    .req_bank_i           (breq_bank),
    .req_addr_i           (breq_addr),
    .req_len_i            (breq_len),
    .req_done_o           (bdone),
    .req_done_txn_o       (bdone_txn),
    .req_done_generation_o(),
    .wdata_valid_i        (bw_valid),
    .wdata_ready_o        (bw_ready),
    .wdata_i              (bw_data),
    .rdata_valid_o        (br_valid),
    .rdata_ready_i        (br_ready),
    .rdata_txn_o          (br_txn),
    .rdata_generation_o   (),
    .rdata_data_o         (br_data),
    .rdata_ben_o          (br_ben),
    .fin_valid_i          (bfin_valid),
    .fin_txn_i            (bfin_txn),
    .fin_generation_i     (bfin_gen),
    .fin_ready_o          (),
    .fin_ack_o            (bfin_ack),
    .fin_rej_o            (bfin_rej),
    .fin_rej_code_o       (),
    .inv_valid_i          ('0),
    .inv_ready_o          (),
    .inv_bank_i           ('0),
    .inv_addr_i           ('0),
    .inv_len_i            ('0),
    .inv_ack_o            (),
    .inv_rej_o            (),
    .inv_rej_code_o       (),
    .profile_set_valid_i  (profile_valid_i),
    .profile_index_i      (profile_index_i),
    .profile_generation_i (64'(profile_generation_i)),
    .profile_set_ready_o  (),
    .profile_ack_o        (profile_ack_o),
    .profile_rej_o        (profile_reject_o),
    .profile_rej_code_o   (),
    .cp_valid_i           (cp_i.valid),
    .cp_write_i           (cp_i.write),
    .cp_bank_i            (cp_i.bank),
    .cp_addr_i            (cp_i.address),
    .cp_beats_i           (cp_i.beats),
    .cp_wmode_i           (cp_i.wmode),
    .cp_wdata_valid_i     (cp_i.wvalid),
    .cp_wdata_i           (cp_i.wdata),
    .cp_wben_i            (cp_i.wben),
    .cp_rdata_ready_i     (cp_i.rready),
    .cache_quiescent_i    (cache_quiescent_i),
    .cp_ready_o           (cp_o.ready),
    .cp_rej_o             (cp_o.reject),
    .cp_rej_code_o        (),
    .cp_wdata_ready_o     (cp_o.wready),
    .cp_rdata_valid_o     (cp_o.rvalid),
    .cp_rdata_o           (cp_o.rdata),
    .cp_rben_o            (cp_o.rben),
    .cp_done_o            (cp_o.done),
    .cp_busy_o            (cp_o.busy),
    .active_profile_o     (bank_profile_o),
    .profile_generation_o ()
  );

  // ------------------------------------------------- per-ctx selections
  int o_term [CONTEXT_COUNT];
  int o_nonterm [CONTEXT_COUNT];
  int b_act [CONTEXT_COUNT];
  int o_reg_spm [CONTEXT_COUNT];
  int o_reg_cac [CONTEXT_COUNT];
  int o_fence [CONTEXT_COUNT];
  int alloc_slot [CONTEXT_COUNT];
  int b_cnt [CONTEXT_COUNT];
  logic cwait_any [CONTEXT_COUNT];
  logic fence_fin [CONTEXT_COUNT];

  always_comb begin : sel_scan
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      o_term[c] = NONE;
      o_nonterm[c] = NONE;
      b_act[c] = NONE;
      o_reg_spm[c] = NONE;
      o_reg_cac[c] = NONE;
      o_fence[c] = NONE;
      cwait_any[c] = 1'b0;
      b_cnt[c] = 0;
      for (int i = 0; i < CAP; i++) begin
        if (s_state[c][i] == S_FREE) begin
          // free: neither terminal nor launchable
        end else if (s_state[c][i] == S_TERM) begin
          if (o_term[c] == NONE) o_term[c] = i;
          else if (seq_older(s_seq[c][i], s_seq[c][o_term[c]])) o_term[c] = i;
        end else begin
          if (o_nonterm[c] == NONE) o_nonterm[c] = i;
          else if (seq_older(s_seq[c][i], s_seq[c][o_nonterm[c]]))
            o_nonterm[c] = i;
          case (s_state[c][i])
            S_REG: begin
              if (s_fence[c][i]) begin
                if (o_fence[c] == NONE) o_fence[c] = i;
                else if (seq_older(s_seq[c][i], s_seq[c][o_fence[c]]))
                  o_fence[c] = i;
              end else if (s_cached[c][i]) begin
                if (o_reg_cac[c] == NONE) o_reg_cac[c] = i;
                else if (seq_older(s_seq[c][i], s_seq[c][o_reg_cac[c]]))
                  o_reg_cac[c] = i;
              end else begin
                if (o_reg_spm[c] == NONE) o_reg_spm[c] = i;
                else if (seq_older(s_seq[c][i], s_seq[c][o_reg_spm[c]]))
                  o_reg_spm[c] = i;
              end
            end
            S_CWAIT: cwait_any[c] = 1'b1;
            default: begin  // S_BWD / S_BRD / S_BFIN
              b_act[c] = i;
              b_cnt[c] = b_cnt[c] + 1;
            end
          endcase
        end
      end
      // A queued fence completes when every older record is terminal.
      fence_fin[c] = 1'b0;
      if (o_nonterm[c] != NONE && s_state[c][o_nonterm[c]] == S_REG)
        fence_fin[c] = s_fence[c][o_nonterm[c]];
      // Allocation slot: RR scan from tail.
      alloc_slot[c] = NONE;
      for (int k = 0; k < CAP; k++) begin
        if (alloc_slot[c] == NONE
            && s_state[c][(int'(c_tail[c]) + k) % CAP] == S_FREE)
          alloc_slot[c] = (int'(c_tail[c]) + k) % CAP;
      end
    end
  end

  // Launch eligibility: a younger record never passes a queued older fence.
  function automatic logic fence_blocks(input int c, input int cand);
    if (o_fence[c] == NONE) fence_blocks = 1'b0;
    else if (cand == NONE) fence_blocks = 1'b1;
    else fence_blocks = seq_older(s_seq[c][o_fence[c]], s_seq[c][cand]);
  endfunction

  // Segment bank mask of a record (its holder channels in the bank).
  function automatic logic [BANK_COUNT-1:0] seg_mask(input int c, input int i);
    logic [BANK_COUNT-1:0] m;
    m = '0;
    if (i != NONE) begin
      for (int p = 0; p < MAX_PIECES; p++) begin
        if (p < int'(s_segcnt[c][i]))
          m |= BANK_COUNT'(1) << s_seg_bank[c][i][p];
      end
    end
    return m;
  endfunction

  // ------------------------------------------------------ cancel decode
  logic [CONTEXT_COUNT-1:0] cxl_eff;
  always_comb begin : cancel_decode
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      cxl_eff[c] = cancel_i[c] && c_valid[c]
                   && (cancel_generation_i[c] == c_gen[c]);
    end
  end

  // ------------------------------------------------------- ingress gate
  logic [CONTEXT_COUNT-1:0] alloc_fire;
  always_comb begin : ingress_gate
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      req_ready_o[c] = (alloc_slot[c] != NONE) && !cxl_eff[c]
                       && !(config_valid_i && (config_context_i == 32'(c)));
      alloc_fire[c] = req_valid_i[c] && req_ready_o[c];
    end
  end

  // ------------------------------------------------------ bank channels
  int bp_idx [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] bp_acc, bp_rej, ba_wf, ba_rf, ba_done, ba_finack;
  logic [CONTEXT_COUNT-1:0] fin_bad, rdata_bad;

  always_comb begin : bank_drive
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      // Launch presentation: bank engine free, oldest eligible SPM record.
      bp_idx[c] = NONE;
      if (b_act[c] == NONE && issue_enable_i && o_reg_spm[c] != NONE
          && !fence_blocks(c, o_reg_spm[c]))
        bp_idx[c] = o_reg_spm[c];
      breq_valid[c] = (bp_idx[c] != NONE);
      breq_txn[c] = (bp_idx[c] != NONE)
        ? mk_txn(c, bp_idx[c], s_owner[c][bp_idx[c]]) : '0;
      breq_write[c] = (bp_idx[c] != NONE) && s_write[c][bp_idx[c]];
      breq_segcnt[c] = (bp_idx[c] != NONE)
        ? SEG_CNT_W'(s_segcnt[c][bp_idx[c]]) : '0;
      for (int s = 0; s < MAX_SEG; s++) begin
        if (bp_idx[c] != NONE && s < int'(s_segcnt[c][bp_idx[c]])) begin
          breq_bank[c][s] = s_seg_bank[c][bp_idx[c]][s];
          breq_addr[c][s] = s_seg_addr[c][bp_idx[c]][s];
          breq_len[c][s] = s_seg_len[c][bp_idx[c]][s];
        end else begin
          breq_bank[c][s] = '0;
          breq_addr[c][s] = '0;
          breq_len[c][s] = '0;
        end
      end

      // Data-phase channels belong to the single bank-active record.
      bw_valid[c] = (b_act[c] != NONE) && (s_state[c][b_act[c]] == S_BWD);
      bw_data[c] = (b_act[c] != NONE)
        ? wbeat_data(s_wdata[c][b_act[c]],
                     s_bc[c][b_act[c]],
                     s_seg_addr[c][b_act[c]][s_wseg[c][b_act[c]]],
                     s_seg_off[c][b_act[c]][s_wseg[c][b_act[c]]],
                     int'(s_wbeat[c][b_act[c]]))
        : '0;
      ba_wf[c] = bw_valid[c] && bw_ready[c];

      br_ready[c] = (b_act[c] != NONE) && (s_state[c][b_act[c]] == S_BRD);
      ba_rf[c] = br_valid[c] && br_ready[c]
                 && (b_act[c] != NONE)
                 && (br_txn[c] == s_txn[c][b_act[c]]);
      rdata_bad[c] = br_valid[c] && br_ready[c] && !ba_rf[c];

      bfin_valid[c] = (b_act[c] != NONE) && (s_state[c][b_act[c]] == S_BFIN);
      bfin_txn[c] = (b_act[c] != NONE) ? s_txn[c][b_act[c]] : '0;
      bfin_gen[c] = (b_act[c] != NONE) ? s_bgen_cap[c][b_act[c]] : '0;
      ba_finack[c] = bfin_valid[c] && bfin_ack[c];
      fin_bad[c] = bfin_valid[c] && bfin_rej[c];

      ba_done[c] = 1'b0;
      if (b_act[c] != NONE) begin
        if (s_state[c][b_act[c]] == S_BWD || s_state[c][b_act[c]] == S_BRD)
          ba_done[c] = bdone[c] && (bdone_txn[c] == s_txn[c][b_act[c]]);
      end
    end
  end

  // Accept/reject fire, in its own block so the bank handshake stays acyclic
  // at Verilator's block granularity (breq_valid is register-derived only).
  always_comb begin : bank_fire
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      bp_acc[c] = breq_valid[c] && breq_ready[c];
      bp_rej[c] = breq_valid[c] && breq_rej[c];
    end
  end

  // ----------------------------------------------------- cache channels
  int cp_idx [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] cp_acc;
  int cw_hit [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] cw_id_bad;

  always_comb begin : cache_drive
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      cp_idx[c] = NONE;
      if (issue_enable_i && o_reg_cac[c] != NONE
          && !fence_blocks(c, o_reg_cac[c]))
        cp_idx[c] = o_reg_cac[c];
      cache_req_valid_o[c] = (cp_idx[c] != NONE);
      cache_req_o[c].owner = (cp_idx[c] != NONE) ? s_owner[c][cp_idx[c]] : '0;
      cache_req_o[c].record_id = (cp_idx[c] != NONE) ? RECORD_W'(cp_idx[c]) : '0;
      cache_req_o[c].binding_id = (cp_idx[c] != NONE) ? s_bid[c][cp_idx[c]] : '0;
      cache_req_o[c].binding_generation =
        (cp_idx[c] != NONE) ? s_bgen[c][cp_idx[c]] : '0;
      cache_req_o[c].offset = (cp_idx[c] != NONE) ? s_coff[c][cp_idx[c]] : '0;
      cache_req_o[c].arena_generation =
        (cp_idx[c] != NONE) ? s_arena[c][cp_idx[c]] : '0;
      cache_req_o[c].scope_epoch =
        (cp_idx[c] != NONE) ? s_scope[c][cp_idx[c]] : '0;
      cache_req_o[c].profile_generation = prof_gen32;
      cache_req_o[c].byte_count = (cp_idx[c] != NONE) ? s_bc[c][cp_idx[c]] : '0;
      cp_acc[c] = cache_req_valid_o[c] && cache_req_ready_i[c];

      cw_hit[c] = NONE;
      if (cache_rsp_valid_i[c]) begin
        for (int i = 0; i < CAP; i++) begin
          if (s_state[c][i] == S_CWAIT
              && (cache_rsp_i[c].record_id == RECORD_W'(i))) begin
            cw_hit[c] = i;
          end
        end
      end
      cache_rsp_ready_o[c] = (cw_hit[c] != NONE);
      cw_id_bad[c] = cache_rsp_valid_i[c] && (cw_hit[c] == NONE);
    end
  end

  // ---------------------------------------------------- response drive
  logic [CONTEXT_COUNT-1:0] rsp_fire;
  always_comb begin : rsp_drive
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      rsp_valid_o[c] = rsp_hold_q[c];
      rsp_o[c] = rsp_reg[c];
      rsp_fire[c] = rsp_valid_o[c] && rsp_ready_i[c];
      outstanding_o[c] = 32'(c_count[c]);
      accepted_o[c] = c_acc[c];
      terminal_o[c] = c_term[c];
      wait_bank_o[c] = c_wbank[c];
      wait_mshr_o[c] = c_wmshr[c];
      stale_o[c] = c_stale[c];
    end
    bank_busy_o = bank_held;
  end

  // Presentation park: load the oldest terminal record only while the
  // channel is idle; hold it (valid+payload+slot id) until the fire edge.
  always_ff @(posedge clk) begin : rsp_park
    if (!rst_n) begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        rsp_hold_q[c] <= 1'b0;
        rsp_slot_q[c] <= '0;
        rsp_reg[c] <= '0;
      end
    end else begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (rsp_fire[c]) begin
          rsp_hold_q[c] <= 1'b0;
        end else if (!rsp_hold_q[c] && (o_term[c] != NONE)) begin
          rsp_hold_q[c] <= 1'b1;
          rsp_slot_q[c] <= RECORD_W'(o_term[c]);
          rsp_reg[c].owner <= s_owner[c][o_term[c]];
          rsp_reg[c].pc <= s_pc[c][o_term[c]];
          rsp_reg[c].data <= s_tdata[c][o_term[c]];
          rsp_reg[c].rd <= s_rd[c][o_term[c]];
          rsp_reg[c].write <= s_write[c][o_term[c]];
          rsp_reg[c].float_reg <= s_flt[c][o_term[c]];
          rsp_reg[c].final_state <= s_final[c][o_term[c]];
          rsp_reg[c].error <= s_err[c][o_term[c]];
        end
      end
    end
  end

  // -------------------------------------------------------- invariants
  logic inv_live;
  logic [CONTEXT_COUNT-1:0] inv_conserve, inv_cap;
  always_comb begin : c031_invariants
    inv_live = 1'b0;
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      inv_conserve[c] = (64'(c_count[c]) != (c_acc[c] - c_term[c]));
      inv_cap[c] = (int'(c_count[c]) > CAP);
      assert (!inv_conserve[c])
        else $error("C031_ASSERT lsu_mslot credit_conservation ctx %0d: out=%0d acc=%0d term=%0d",
                    c, c_count[c], c_acc[c], c_term[c]);
      assert (!inv_cap[c])
        else $error("C031_ASSERT lsu_mslot credit_cap ctx %0d: %0d > %0d",
                    c, c_count[c], CAP);
      assert (!fin_bad[c])
        else $error("C031_ASSERT lsu_mslot exact_bank_finish_generation ctx %0d", c);
      assert (!rdata_bad[c])
        else $error("C031_ASSERT lsu_mslot read_beat_txn_match ctx %0d", c);
      assert (!(bp_acc[c] && bp_rej[c]))
        else $error("C031_ASSERT lsu_mslot bank_accept_never_reject ctx %0d", c);
      // MEM-06 structure: the bank engine is single-record per Context.
      assert (b_cnt[c] <= 1)
        else $error("C031_ASSERT lsu_mslot bank_engine_single_record ctx %0d: %0d",
                    c, b_cnt[c]);
      // A fence never touches bank/cache channels.
      if (bp_idx[c] != NONE)
        assert (!s_fence[c][bp_idx[c]])
          else $error("C031_ASSERT lsu_mslot fence_never_issues_bank ctx %0d", c);
      if (cp_idx[c] != NONE)
        assert (!s_fence[c][cp_idx[c]])
          else $error("C031_ASSERT lsu_mslot fence_never_issues_cache ctx %0d", c);
      inv_live = inv_live || inv_conserve[c] || inv_cap[c] || fin_bad[c]
                 || rdata_bad[c] || (b_cnt[c] > 1);
    end
  end

  always_ff @(posedge clk) begin : rsp_stability
    if (!rst_n) begin
      inv_q <= 1'b0;
      rsp_unstable_q <= 1'b0;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        rsp_hold[c] <= 1'b0;
        rsp_shadow[c] <= '0;
      end
    end else begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (rsp_valid_o[c] && !rsp_ready_i[c]) begin
          if (rsp_hold[c]) begin
            assert (rsp_o[c] == rsp_shadow[c])
              else $error("C031_ASSERT lsu_mslot rsp_stable_under_backpressure ctx %0d", c);
            if (rsp_o[c] != rsp_shadow[c]) rsp_unstable_q <= 1'b1;
          end
          rsp_shadow[c] <= rsp_o[c];
          rsp_hold[c] <= 1'b1;
        end else begin
          rsp_hold[c] <= 1'b0;
        end
      end
      // Sticky, $error-free protocol guard: a cache response naming no
      // waiting record of this Context is an integration fault (injected
      // deliberately only by directed verification at the very end).
      inv_q <= inv_q || (|cw_id_bad) || inv_live || rsp_unstable_q;
    end
  end

  // ---------------------------------------------- sequential core
  // Section order fixes same-edge conflicts: (0) counters, (1) cancel
  // snapshot marking, (2) ingress allocation + decode, (3) launch/data
  // phases, (4) fence completion, (5) response release. Later sections'
  // nonblocking assignments to the same slot win by design (a launch accept
  // overrides the cancel snapshot's ISOLATED for that slot: it is issued and
  // settles through the issued paths with the cancelled flag set).
  always_ff @(posedge clk) begin : main
    // Block locals (single-writer deltas and decode scratch).
    logic [BANK_COUNT-1:0] held_d;
    int cnt_d;
    int unsigned stale_d;
    int ri, bi, pi2, wi, fi;
    logic [3:0] derr;
    logic dup, bad;
    int unsigned dw, cx;
    logic [63:0] dwend, dend;
    int unsigned rel, rem, run, si_, pi_, row_, piece, poff;
    logic [DATA_BYTES-1:0] be;
    int unsigned absb, beats;
    int offb;
    logic [3:0] got_d;

    if (!rst_n) begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        for (int i = 0; i < CAP; i++) begin
          s_state[c][i] <= S_FREE;
          s_seq[c][i] <= '0;
          s_owner[c][i] <= '0;
          s_pc[c][i] <= '0;
          s_addr[c][i] <= '0;
          s_wdata[c][i] <= '0;
          s_tdata[c][i] <= '0;
          s_rd[c][i] <= '0;
          s_bc[c][i] <= '0;
          s_write[c][i] <= 1'b0;
          s_uns[c][i] <= 1'b0;
          s_flt[c][i] <= 1'b0;
          s_fence[c][i] <= 1'b0;
          s_cancelled[c][i] <= 1'b0;
          s_cached[c][i] <= 1'b0;
          s_err[c][i] <= '0;
          s_final[c][i] <= '0;
          s_segcnt[c][i] <= '0;
          s_seg_bank[c][i] <= '0;
          s_seg_addr[c][i] <= '0;
          s_seg_len[c][i] <= '0;
          s_seg_off[c][i] <= '0;
          s_bid[c][i] <= '0;
          s_bgen[c][i] <= '0;
          s_coff[c][i] <= '0;
          s_arena[c][i] <= '0;
          s_scope[c][i] <= '0;
          s_txn[c][i] <= '0;
          s_bgen_cap[c][i] <= '0;
          s_wseg[c][i] <= '0;
          s_wbeat[c][i] <= '0;
          s_rseg[c][i] <= '0;
          s_rbeat[c][i] <= '0;
          s_got[c][i] <= '0;
          s_rbytes[c][i] <= '0;
        end
        c_valid[c] <= 1'b0;
        c_gen[c] <= '0;
        c_tail[c] <= '0;
        c_count[c] <= '0;
        c_allocseq[c] <= '0;
        c_acc[c] <= '0;
        c_term[c] <= '0;
        c_wbank[c] <= '0;
        c_wmshr[c] <= '0;
        c_stale[c] <= '0;
      end
      bank_held <= '0;
      cfg_reject_q <= 1'b0;
      fwd_gen_valid_q <= 1'b0;
      fwd_gen_q <= '0;
      prof_gen32 <= '0;
    end else begin
      held_d = bank_held;

      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        // ---------------- (0) counters / attribution / holder mask
        cnt_d = int'(c_count[c]);
        stale_d = int'(c_stale[c]);
        c_acc[c] <= c_acc[c] + 64'(alloc_fire[c]);
        c_term[c] <= c_term[c] + 64'(rsp_fire[c]);
        if (cancel_i[c] && !cxl_eff[c]) stale_d = stale_d + 1;
        if (cwait_any[c]) c_wmshr[c] <= c_wmshr[c] + 64'd1;
        if (issue_enable_i && o_reg_spm[c] != NONE
            && !fence_blocks(c, o_reg_spm[c])
            && ((b_act[c] != NONE)
                || (bp_idx[c] != NONE && !bp_acc[c] && !bp_rej[c])))
          c_wbank[c] <= c_wbank[c] + 64'd1;

        if (ba_finack[c]) held_d = held_d & ~seg_mask(c, b_act[c]);
        if (bp_acc[c]) held_d = held_d | seg_mask(c, bp_idx[c]);

        // ---------------- (1) cancel snapshot (records accepted before
        // this edge; ingress on this Context is bubbled this cycle).
        if (cxl_eff[c]) begin
          for (int i = 0; i < CAP; i++) begin
            if (s_state[c][i] == S_REG) begin
              if (((bp_idx[c] == i) && bp_acc[c])
                  || ((cp_idx[c] == i) && cp_acc[c])) begin
                s_cancelled[c][i] <= 1'b1;  // launching this edge: issued
              end else begin
                s_state[c][i] <= S_TERM;
                s_final[c][i] <= ISOLATED;
                s_err[c][i] <= ERR_NONE;
                s_tdata[c][i] <= '0;
              end
            end else if (s_state[c][i] == S_CWAIT
                         || s_state[c][i] == S_BWD
                         || s_state[c][i] == S_BRD
                         || s_state[c][i] == S_BFIN) begin
              s_cancelled[c][i] <= 1'b1;
            end
          end
        end

        // ---------------- (2) ingress allocation + decode
        if (alloc_fire[c]) begin
          ri = alloc_slot[c];
          derr = ERR_NONE;
          dup = 1'b0;
          // Duplicate owner key among outstanding records (a record stays
          // duplicate-relevant until its completion is consumed).
          for (int i = 0; i < CAP; i++) begin
            if (s_state[c][i] != S_FREE
                && (s_owner[c][i].ctx_generation == req_i[c].owner.ctx_generation)
                && (s_owner[c][i].owner_op_seq == req_i[c].owner.owner_op_seq)
                && (s_owner[c][i].subop_id == req_i[c].owner.subop_id))
              dup = 1'b1;
          end
          // Identity.
          if (dup || !c_valid[c]
              || (req_i[c].owner.ctx_slot != 32'(c))
              || (req_i[c].owner.ctx_generation != c_gen[c])) begin
            derr = ERR_OWNER;
          end else if (req_i[c].fence) begin
            if (req_i[c].byte_count != 3'd0) derr = ERR_RANGE;
          end else if (!(req_i[c].byte_count inside {3'd1, 3'd2, 3'd4})) begin
            derr = ERR_RANGE;
          end else begin
            // Window lookup (first match) with full containment.
            dw = WINDOW_COUNT;
            for (int w = 0; w < WINDOW_COUNT; w++) begin
              if (dw == WINDOW_COUNT && c_windows[c][w].valid) begin
                dwend = {32'b0, c_windows[c][w].base}
                        + {32'b0, c_windows[c][w].size};
                dend = {32'b0, req_i[c].address}
                       + 64'(32'(req_i[c].byte_count));
                if ((dwend <= 64'd4294967296)
                    && ({32'b0, req_i[c].address}
                        >= {32'b0, c_windows[c][w].base})
                    && (dend <= dwend)) begin
                  dw = unsigned'(w);
                end
              end
            end
            if (dw == WINDOW_COUNT) begin
              derr = ERR_RANGE;
            end else if (req_i[c].write && !c_windows[c][dw].writable) begin
              derr = ERR_PERMISSION;
            end else if (!req_i[c].write && !c_windows[c][dw].readable) begin
              derr = ERR_PERMISSION;
            end else if (c_windows[c][dw].cached && req_i[c].write) begin
              derr = ERR_PERMISSION;  // unreachable: config rejects, guarded
            end
          end

          // Echo fields are always written (fault completions are owned).
          s_owner[c][ri] <= req_i[c].owner;
          s_pc[c][ri] <= req_i[c].pc;
          s_rd[c][ri] <= req_i[c].rd;
          s_write[c][ri] <= req_i[c].write;
          s_flt[c][ri] <= req_i[c].float_reg;
          s_fence[c][ri] <= req_i[c].fence;
          s_cached[c][ri] <= 1'b0;
          s_segcnt[c][ri] <= '0;
          s_final[c][ri] <= FAULTED;
          s_err[c][ri] <= derr;
          s_tdata[c][ri] <= '0;
          s_state[c][ri] <= S_TERM;
          if (derr == ERR_NONE) begin
            s_state[c][ri] <= S_REG;
            s_final[c][ri] <= '0;
            s_addr[c][ri] <= req_i[c].address;
            s_wdata[c][ri] <= req_i[c].data;
            s_bc[c][ri] <= req_i[c].byte_count;
            s_uns[c][ri] <= req_i[c].unsigned_load;
            s_cancelled[c][ri] <= 1'b0;
            if (!req_i[c].fence && c_windows[c][dw].cached) begin
              s_cached[c][ri] <= 1'b1;
              s_bid[c][ri] <= c_windows[c][dw].binding_id;
              s_bgen[c][ri] <= c_windows[c][dw].binding_generation;
              s_coff[c][ri] <= c_windows[c][dw].binding_offset
                               + (req_i[c].address - c_windows[c][dw].base);
              s_arena[c][ri] <= c_windows[c][dw].arena_generation;
              s_scope[c][ri] <= c_windows[c][dw].scope_epoch;
            end else if (!req_i[c].fence) begin
              // SPM segment split at stripe boundaries (address order).
              rel = req_i[c].address - c_windows[c][dw].base;
              rem = int'(req_i[c].byte_count);
              piece = 0;
              s_seg_bank[c][ri] <= '0;
              s_seg_addr[c][ri] <= '0;
              s_seg_len[c][ri] <= '0;
              s_seg_off[c][ri] <= '0;
              poff = 0;
              while (rem > 0 && piece < MAX_PIECES) begin
                si_ = rel / c_windows[c][dw].stripe_bytes;
                pi_ = rel % c_windows[c][dw].stripe_bytes;
                row_ = si_ / int'(BANK_COUNT);
                run = c_windows[c][dw].stripe_bytes - pi_;
                if (run > rem) run = rem;
                if ({32'b0, c_windows[c][dw].spm_base}
                    + 64'(row_) * 64'(c_windows[c][dw].stripe_bytes)
                    + 64'(pi_) + 64'(run) > 64'(BANK_BYTES)) begin
                  derr = ERR_RANGE;
                end else begin
                  s_seg_bank[c][ri][2'(piece)] <= BANK_IDX_W'(si_ % BANK_COUNT);
                  s_seg_addr[c][ri][2'(piece)] <= BANK_ADDR_W'(
                    c_windows[c][dw].spm_base
                    + row_ * c_windows[c][dw].stripe_bytes + pi_);
                  s_seg_len[c][ri][2'(piece)] <= BANK_LEN_W'(run);
                  s_seg_off[c][ri][2'(piece)] <= 2'(poff);
                end
                rel = rel + run;
                rem = rem - run;
                poff = poff + run;
                piece = piece + 1;
              end
              s_segcnt[c][ri] <= 3'(piece);
            end
            if (derr != ERR_NONE) begin
              s_state[c][ri] <= S_TERM;
              s_final[c][ri] <= FAULTED;
              s_err[c][ri] <= derr;
              s_tdata[c][ri] <= '0;
            end
          end
          if (derr == ERR_OWNER) stale_d = stale_d + 1;
          cnt_d = cnt_d + 1;
          c_tail[c] <= 4'((ri + 1) % CAP);
          s_seq[c][ri] <= c_allocseq[c];
          c_allocseq[c] <= c_allocseq[c] + 5'd1;
        end

        // ---------------- (3a) bank launch / data phase
        if (b_act[c] != NONE) begin
          bi = b_act[c];
          case (s_state[c][bi])
            S_BWD: begin
              if (ba_wf[c]) begin
                beats = seg_beats(s_seg_addr[c][bi][s_wseg[c][bi]],
                                  s_seg_len[c][bi][s_wseg[c][bi]]);
                if (s_wbeat[c][bi] == 4'(beats - 1)) begin
                  s_wbeat[c][bi] <= '0;
                  if (s_wseg[c][bi] != 4'(int'(s_segcnt[c][bi]) - 1))
                    s_wseg[c][bi] <= s_wseg[c][bi] + 4'd1;
                end else begin
                  s_wbeat[c][bi] <= s_wbeat[c][bi] + 4'd1;
                end
              end
              if (ba_done[c]) s_state[c][bi] <= S_BFIN;
            end
            S_BRD: begin
              if (ba_rf[c]) begin
                be = beat_en(s_seg_addr[c][bi][s_rseg[c][bi]],
                             s_seg_len[c][bi][s_rseg[c][bi]],
                             int'(s_rbeat[c][bi]));
                got_d = s_got[c][bi];
                for (int b = 0; b < DATA_BYTES; b++) begin
                  if (be[b]) begin
                    absb = ((int'(s_seg_addr[c][bi][s_rseg[c][bi]])
                             / DATA_BYTES) + int'(s_rbeat[c][bi]))
                           * DATA_BYTES + int'(b);
                    offb = int'(absb) - int'(s_seg_addr[c][bi][s_rseg[c][bi]])
                           + int'(s_seg_off[c][bi][s_rseg[c][bi]]);
                    if (offb >= 0 && offb < int'(s_bc[c][bi])) begin
                      // Stored per-byte validity gates the got mask: a
                      // never-written byte is never exposed as valid data.
                      got_d[offb] = br_ben[c][b];
                      if (br_ben[c][b]) begin
                        s_rbytes[c][bi][8*offb +: 8] <= br_data[c][8*b +: 8];
                      end
                    end
                  end
                end
                s_got[c][bi] <= got_d;
                beats = seg_beats(s_seg_addr[c][bi][s_rseg[c][bi]],
                                  s_seg_len[c][bi][s_rseg[c][bi]]);
                if (s_rbeat[c][bi] == 4'(beats - 1)) begin
                  s_rbeat[c][bi] <= '0;
                  if (s_rseg[c][bi] != 4'(int'(s_segcnt[c][bi]) - 1))
                    s_rseg[c][bi] <= s_rseg[c][bi] + 4'd1;
                end else begin
                  s_rbeat[c][bi] <= s_rbeat[c][bi] + 4'd1;
                end
              end
              if (ba_done[c]) s_state[c][bi] <= S_BFIN;
            end
            S_BFIN: begin
              if (ba_finack[c]) begin
                s_state[c][bi] <= S_TERM;
                if (s_write[c][bi]) begin
                  // An issued store always finishes its effects, cancel or
                  // not (C031 contract).
                  s_final[c][bi] <= COMPLETED;
                  s_err[c][bi] <= ERR_NONE;
                  s_tdata[c][bi] <= '0;
                end else if (s_cancelled[c][bi]) begin
                  s_final[c][bi] <= ISOLATED;
                  s_err[c][bi] <= ERR_NONE;
                  s_tdata[c][bi] <= ext_data(s_rbytes[c][bi], s_bc[c][bi],
                                             s_uns[c][bi]);
                end else if (s_got[c][bi] != req_mask(s_bc[c][bi])) begin
                  s_final[c][bi] <= FAULTED;
                  s_err[c][bi] <= ERR_UNINITIALIZED;
                  s_tdata[c][bi] <= '0;
                end else begin
                  s_final[c][bi] <= COMPLETED;
                  s_err[c][bi] <= ERR_NONE;
                  s_tdata[c][bi] <= ext_data(s_rbytes[c][bi], s_bc[c][bi],
                                             s_uns[c][bi]);
                end
              end
            end
            default: ;  // S_FREE/S_REG/S_CWAIT/S_TERM handled elsewhere
          endcase
        end else if (bp_idx[c] != NONE) begin
          pi2 = bp_idx[c];
          if (bp_rej[c]) begin
            s_state[c][pi2] <= S_TERM;
            s_final[c][pi2] <= FAULTED;
            s_err[c][pi2] <= bank_rej_map(breq_rej_code[c]);
            s_tdata[c][pi2] <= '0;
          end else if (bp_acc[c]) begin
            s_state[c][pi2] <= s_write[c][pi2] ? S_BWD : S_BRD;
            s_txn[c][pi2] <= mk_txn(c, pi2, s_owner[c][pi2]);
            s_bgen_cap[c][pi2] <= breq_generation[c];
            s_wseg[c][pi2] <= '0;
            s_wbeat[c][pi2] <= '0;
            s_rseg[c][pi2] <= '0;
            s_rbeat[c][pi2] <= '0;
            s_got[c][pi2] <= '0;
            s_rbytes[c][pi2] <= '0;
          end
        end

        // ---------------- (3b) cache launch / settle
        if (cp_idx[c] != NONE && cp_acc[c]) begin
          s_state[c][cp_idx[c]] <= S_CWAIT;
        end
        if (cw_hit[c] != NONE) begin
          wi = cw_hit[c];
          s_state[c][wi] <= S_TERM;
          s_final[c][wi] <= cache_rsp_i[c].final_state;
          s_err[c][wi] <= cache_rsp_i[c].error;
          s_tdata[c][wi] <= (cache_rsp_i[c].final_state == COMPLETED)
            ? ext_data(cache_rsp_i[c].data, s_bc[c][wi], s_uns[c][wi]) : '0;
        end

        // ---------------- (4) fence completion (all older records terminal;
        // an effective cancel this edge settles the fence ISOLATED instead).
        if (fence_fin[c] && !cxl_eff[c]) begin
          fi = o_nonterm[c];
          s_state[c][fi] <= S_TERM;
          s_final[c][fi] <= COMPLETED;
          s_err[c][fi] <= ERR_NONE;
          s_tdata[c][fi] <= '0;
        end

        // ---------------- (5) response release (credit returns here).
        // Free the PARKED record, not the scan result: a newly-terminal
        // older record may legitimately have moved o_term past the parked
        // slot while its completion was stalled.
        if (rsp_fire[c]) begin
          s_state[c][rsp_slot_q[c]] <= S_FREE;
          cnt_d = cnt_d - 1;
        end

        c_count[c] <= 5'(cnt_d);
        c_stale[c] <= 64'(stale_d);
      end

      bank_held <= held_d;

      // ---------------- config install (quiescent per Context)
      cfg_reject_q <= 1'b0;
      if (config_valid_i) begin
        bad = 1'b0;
        if (config_context_i >= 32'(CONTEXT_COUNT)) begin
          bad = 1'b1;
        end else begin
          cx = unsigned'(config_context_i);
          if (int'(c_count[cx]) != 0) bad = 1'b1;
        end
        for (int w = 0; w < WINDOW_COUNT; w++) begin
          if (!bad && config_windows_i[w].valid) begin
            if (config_windows_i[w].stripe_bytes == '0
                || config_windows_i[w].size == '0
                || (!config_windows_i[w].readable && !config_windows_i[w].writable)
                || (config_windows_i[w].cached && config_windows_i[w].writable)
                || (config_windows_i[w].cached
                    && ((config_windows_i[w].binding_offset % LINE_BYTES) != 0
                        || (config_windows_i[w].size % LINE_BYTES) != 0))
                || (({32'b0, config_windows_i[w].base}
                     + {32'b0, config_windows_i[w].size})
                    > 64'd4294967296)) begin
              bad = 1'b1;
            end
          end
        end
        if (bad) begin
          cfg_reject_q <= 1'b1;
        end else begin
          for (int w = 0; w < WINDOW_COUNT; w++) begin
            c_windows[unsigned'(config_context_i)][w] <= config_windows_i[w];
          end
          c_valid[unsigned'(config_context_i)] <= 1'b1;
          c_gen[unsigned'(config_context_i)] <= config_generation_i;
        end
      end

      // ---------------- profile forwarding latch
      if (profile_valid_i) begin
        fwd_gen_q <= profile_generation_i;
        fwd_gen_valid_q <= 1'b1;
      end
      if (profile_ack_o) begin
        assert (fwd_gen_valid_q)
          else $error("C031_ASSERT lsu_mslot profile_ack_without_forward");
        prof_gen32 <= fwd_gen_q;
      end
    end
  end

  assign config_ready_o = 1'b1;
  assign config_reject_o = cfg_reject_q;
  assign invariant_error_o = inv_q;

endmodule
