// l1_bank.sv — C029 / W4: L1 bank SRAM bytes/validity, per-bank busy and exact
// transaction-holder tracking.
//
// Scope (C029 card, G2_scalar.md#c029): the bank primitive only. Cache/MSHR and
// the program physical port are C030; the LSU/MFE transfer FSM is C031+. This
// module proves real SRAM byte behavior plus atomic multi-bank occupancy, not a
// gather/scatter FSM.
//
// Internal interface contract (boring, standalone; consumed by tb_l1_bank and
// the C029 slot-frame/arena-ledger bench):
//   * One request port per Context. A request carries an opaque transport txn
//     id (TXN_ID_WIDTH, explicitly private transport storage: no silicon
//     identity/epoch width claim, never reinterpreted), a direction (read or
//     write for the whole request) and 1..MAX_SEG byte segments
//     (bank, bank-local byte address, byte length).
//   * A compound request has BANK_COUNT segment positions, sourced from the
//     geometry rather than an independent framing limit. All resource
//     parameters come from the sealed C013/C019 consumer package.
//   * Bank-local layout per bank (profiles.py MemoryProfile + arena.py
//     reconfigure_free_intervals): [0, SYSTEM_RESERVED_BYTES) system reserved,
//     [SYSTEM_RESERVED_BYTES, SPM_BYTES_PER_BANK[profile]) user SPM,
//     [SPM_BYTES_PER_BANK[profile], BANK_BYTES) cache partition (C030; this
//     port never touches it, accesses there are rejected).
//   * All-or-none acquisition (transfer.py try_issue semantics): every
//     requested bank/address/partition is preflighted before any occupancy;
//     any busy bank in the set, or any invalid range, accepts nothing. A
//     conflicting request simply waits (req_ready_o low); an invalid payload
//     is rejected combinationally with a stable fault code and no side
//     effects. Same-cycle multi-port grants require disjoint bank sets
//     (fixed port-index priority).
//   * Busy runs until fin matches port, caller txn and the bank-issued holder
//     generation. The generation advances on every accept and never wraps;
//     reusing a caller txn cannot authorize an old fin against a new holder.
//     fin is accepted only after the data phase completed.
//   * Write stream: per-Context wdata beats (DATA_BYTES wide) accepted in
//     segment order, byte-masks derived from each segment's byte range; only
//     covered bytes are written and marked valid. Read stream: one beat per
//     DATA_BYTES word in segment order; per-byte validity comes from stored
//     validity bits, so never-written bytes are returned with ben=0 and are
//     never exposed as valid zeros. Response payload (data, byte validity,
//     txn echo) is held stable while the consumer backpressures.
//   * Checked range-validity invalidation (agreed C029 lifecycle hook; chosen
//     over per-byte ownership tags): a lifecycle consumer (slot frame / arena
//     ledger retire-reuse path) invalidates ONE exact byte range on one bank.
//     The bank clears the validity bits of exactly the covered bytes — the
//     SRAM bytes themselves are kept, no row/word rounding is applied, and
//     bytes outside the range are untouched. Invalidation exclusively owns
//     the selected bank and clears one DATA_BYTES word per cycle. ACK follows
//     the final clear, never just the request handshake. Existing holders
//     drain first; the lifecycle caller retains the extent until ACK.
//   * Read latency is the sourced synchronous SRAM access
//     (BANK_LATENCY_CYCLES): first rdata beat appears one cycle after the
//     accept edge, then one beat per cycle.
//   * Profile transition mirrors arena.py reconfigure: strictly increasing
//     profile_generation (separate domain from ctx_generation and
//     arena_alloc_gen) and quiescent bank service, including invalidations.
//     Its commit edge accepts no new data or invalidation request.
//
// Fault codes (req_rej_code_o and inv_rej_code_o, priority order):
//   0 none; 1 bad seg_count / bank index (SHAPE); 2 zero length (LEN0);
//   3 addr+len beyond bank bytes (OVERFLOW); 4 reserved bytes (RESERVED);
//   5 cache partition (CACHE); 6 holder generation exhausted (requests only).
//   fin codes: 0 none; 1 stale/duplicate/wrong holder; 2 data incomplete.
//   profile codes: 0 none; 1 bad index or non-increasing generation;
//   2 not quiescent.
`timescale 1ns/1ps

module l1_bank
  import c029_params_pkg::*;
#(
  // Top-level context count; the runner passes -GCONTEXT_COUNT=1/2 (C1/C2).
  parameter int CONTEXT_COUNT = C2_CONTEXTS,
  // Private transport identity storage widths (experimental, no PPA claim).
  parameter int TXN_ID_WIDTH = 64,
  parameter int PROFILE_GEN_WIDTH = 64,
  parameter int HOLDER_GEN_WIDTH = 64,
  localparam int MAX_SEG = BANK_COUNT,
  // Derived widths (localparam entries: resolved from the package above,
  // not overridable by instantiation).
  localparam int unsigned BANK_IDX_W = (BANK_COUNT <= 1) ? 1 : $clog2(BANK_COUNT),
  localparam int unsigned WORDS_PER_BANK = BANK_BYTES / DATA_BYTES,
  localparam int unsigned WORD_ADDR_W = (WORDS_PER_BANK <= 1) ? 1 : $clog2(WORDS_PER_BANK),
  localparam int unsigned BANK_ADDR_W = (BANK_BYTES <= 1) ? 1 : $clog2(BANK_BYTES),
  localparam int unsigned BYTE_OFF_W = (DATA_BYTES <= 1) ? 1 : $clog2(DATA_BYTES),
  localparam int unsigned BANK_LEN_W = $clog2(BANK_BYTES + 1),
  localparam int unsigned SEG_IDX_W = (MAX_SEG <= 1) ? 1 : $clog2(MAX_SEG),
  localparam int unsigned SEG_CNT_W = $clog2(MAX_SEG + 1),
  localparam int unsigned BEAT_W = $clog2(WORDS_PER_BANK + 1),
  localparam int unsigned PORT_IDX_W = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT),
  localparam int unsigned PROFILE_IDX_W = $clog2(PROFILE_COUNT)
) (
  input logic clk,
  input logic rstn,

  // Compound acquire + data request, one channel per Context.
  input  logic [CONTEXT_COUNT-1:0]                    req_valid_i,
  output logic [CONTEXT_COUNT-1:0]                    req_ready_o,
  output logic [CONTEXT_COUNT-1:0]                    req_rej_o,
  output logic [CONTEXT_COUNT-1:0][2:0]               req_rej_code_o,
  input  logic [CONTEXT_COUNT-1:0][TXN_ID_WIDTH-1:0]  req_txn_i,
  output logic [CONTEXT_COUNT-1:0][HOLDER_GEN_WIDTH-1:0] req_generation_o,
  input  logic [CONTEXT_COUNT-1:0]                    req_write_i,
  input  logic [CONTEXT_COUNT-1:0][SEG_CNT_W-1:0]     req_seg_count_i,
  input  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_IDX_W-1:0]  req_bank_i,
  input  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_ADDR_W-1:0] req_addr_i,
  input  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_LEN_W-1:0]  req_len_i,

  // Data-phase completion (level; clear at fin ack).
  output logic [CONTEXT_COUNT-1:0]                    req_done_o,
  output logic [CONTEXT_COUNT-1:0][TXN_ID_WIDTH-1:0]  req_done_txn_o,
  output logic [CONTEXT_COUNT-1:0][HOLDER_GEN_WIDTH-1:0] req_done_generation_o,

  // Write beat stream (segment order, one DATA_BYTES beat per fire).
  input  logic [CONTEXT_COUNT-1:0]                    wdata_valid_i,
  output logic [CONTEXT_COUNT-1:0]                    wdata_ready_o,
  input  logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0]  wdata_i,

  // Read beat stream (segment order; payload stable under backpressure).
  output logic [CONTEXT_COUNT-1:0]                    rdata_valid_o,
  input  logic [CONTEXT_COUNT-1:0]                    rdata_ready_i,
  output logic [CONTEXT_COUNT-1:0][TXN_ID_WIDTH-1:0]  rdata_txn_o,
  output logic [CONTEXT_COUNT-1:0][HOLDER_GEN_WIDTH-1:0] rdata_generation_o,
  output logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0]  rdata_data_o,
  output logic [CONTEXT_COUNT-1:0][DATA_BYTES-1:0]    rdata_ben_o,

  // Explicit holder finish/release, one channel per Context.
  input  logic [CONTEXT_COUNT-1:0]                    fin_valid_i,
  input  logic [CONTEXT_COUNT-1:0][TXN_ID_WIDTH-1:0]  fin_txn_i,
  input  logic [CONTEXT_COUNT-1:0][HOLDER_GEN_WIDTH-1:0] fin_generation_i,
  output logic [CONTEXT_COUNT-1:0]                    fin_ready_o,
  output logic [CONTEXT_COUNT-1:0]                    fin_ack_o,
  output logic [CONTEXT_COUNT-1:0]                    fin_rej_o,
  output logic [CONTEXT_COUNT-1:0][1:0]               fin_rej_code_o,

  // Checked exact-range validity invalidation (lifecycle hook; bytes kept).
  input  logic [CONTEXT_COUNT-1:0]                    inv_valid_i,
  output logic [CONTEXT_COUNT-1:0]                    inv_ready_o,
  input  logic [CONTEXT_COUNT-1:0][BANK_IDX_W-1:0]    inv_bank_i,
  input  logic [CONTEXT_COUNT-1:0][BANK_ADDR_W-1:0]   inv_addr_i,
  input  logic [CONTEXT_COUNT-1:0][BANK_LEN_W-1:0]    inv_len_i,
  output logic [CONTEXT_COUNT-1:0]                    inv_ack_o,
  output logic [CONTEXT_COUNT-1:0]                    inv_rej_o,
  output logic [CONTEXT_COUNT-1:0][2:0]               inv_rej_code_o,

  // Quiescent-only profile transition.
  input  logic                         profile_set_valid_i,
  input  logic [PROFILE_IDX_W-1:0]     profile_index_i,
  input  logic [PROFILE_GEN_WIDTH-1:0] profile_generation_i,
  output logic                         profile_set_ready_o,
  output logic                         profile_ack_o,
  output logic                         profile_rej_o,
  output logic [1:0]                   profile_rej_code_o
);

  // Fault code encodings (documented in the header comment).
  localparam logic [2:0] REJ_NONE = 3'd0;
  localparam logic [2:0] REJ_SHAPE = 3'd1;
  localparam logic [2:0] REJ_LEN0 = 3'd2;
  localparam logic [2:0] REJ_OVERFLOW = 3'd3;
  localparam logic [2:0] REJ_RESERVED = 3'd4;
  localparam logic [2:0] REJ_CACHE = 3'd5;
  localparam logic [2:0] REJ_GENERATION_EXHAUSTED = 3'd6;
  localparam logic [1:0] FIN_REJ_NOT_HOLDER = 2'd1;
  localparam logic [1:0] FIN_REJ_NOT_COMPLETE = 2'd2;
  localparam logic [1:0] PROF_REJ_ARGS = 2'd1;
  localparam logic [1:0] PROF_REJ_QUIESCENT = 2'd2;

  // Fail closed on a package table that violates the physical bank geometry.
  initial begin : pkg_shape_checks
    assert (BANK_LATENCY_CYCLES == 1 && (DATA_BYTES & (DATA_BYTES - 1)) == 0
            && (BANK_BYTES & (BANK_BYTES - 1)) == 0)
      else $fatal(1, "C029_ASSERT l1_bank unsupported synchronous SRAM geometry");
    assert (CONTEXT_COUNT == C1_CONTEXTS || CONTEXT_COUNT == C2_CONTEXTS)
      else $fatal(1, "C029_ASSERT l1_bank unsupported Context count");
    assert (BANK_BYTES % DATA_BYTES == 0)
      else $fatal(1, "C029_ASSERT l1_bank package_geometry: DATA_BYTES must divide BANK_BYTES");
    assert (SYSTEM_RESERVED_BYTES <= SPM_BYTES_PER_BANK[0])
      else $fatal(1, "C029_ASSERT l1_bank package_geometry: reserved exceeds SPM mode 0");
    for (int m = 0; m < PROFILE_COUNT; m++) begin
      assert (SPM_BYTES_PER_BANK[m] >= SYSTEM_RESERVED_BYTES && SPM_BYTES_PER_BANK[m] <= BANK_BYTES)
        else $fatal(1, "C029_ASSERT l1_bank package_geometry: profile %0d violates bank bytes", m);
    end
  end

  // ------------------------------------------------- segment math (int domain)
  // Address arithmetic uses 32-bit ints: every operand fits with room to
  // spare, so overflow is detected by the explicit range checks below, never
  // by silent truncation.
  function automatic int unsigned seg_end(input logic [BANK_ADDR_W-1:0] adr,
                                          input logic [BANK_LEN_W-1:0] len);
    seg_end = int'(adr) + int'(len);  // exclusive end, widened before addition
  endfunction

  function automatic int unsigned seg_beats(input logic [BANK_ADDR_W-1:0] adr,
                                            input logic [BANK_LEN_W-1:0] len);
    seg_beats = ((int'(adr) % DATA_BYTES) + int'(len) + DATA_BYTES - 1) / DATA_BYTES;
  endfunction

  function automatic logic [WORD_ADDR_W-1:0] seg_base_word(input logic [BANK_ADDR_W-1:0] adr);
    seg_base_word = adr[WORD_ADDR_W+BYTE_OFF_W-1:BYTE_OFF_W];
  endfunction

  // Byte enables of beat k of a segment: only bytes inside [adr, adr+len).
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

  // Range/partition fault of one segment (priority: LEN0 > OVERFLOW >
  // RESERVED > CACHE). spm is the active profile's SPM bound.
  function automatic logic [2:0] seg_fault(input logic [BANK_IDX_W-1:0] bank,
                                           input logic [BANK_ADDR_W-1:0] adr,
                                           input logic [BANK_LEN_W-1:0] len,
                                           input int unsigned spm);
    if (int'(bank) >= BANK_COUNT) seg_fault = REJ_SHAPE;
    else if (len == '0) seg_fault = REJ_LEN0;
    else if (seg_end(adr, len) > BANK_BYTES) seg_fault = REJ_OVERFLOW;
    else if (int'(adr) < SYSTEM_RESERVED_BYTES) seg_fault = REJ_RESERVED;
    else if (seg_end(adr, len) > spm) seg_fault = REJ_CACHE;
    else seg_fault = REJ_NONE;
  endfunction

  function automatic int unsigned sel_spm(input int unsigned mode);
    // Variable index into the package parameter array: legal for both packed
    // and unpacked package-side declarations, and evaluates the sealed
    // C019-generated value, not a local copy.
    sel_spm = SPM_BYTES_PER_BANK[mode];
  endfunction

  // ------------------------------------------------------------- SRAM state
  // Real per-bank byte arrays. mem_data is deliberately not reset (SRAM
  // contents are undefined until written); mem_bval is tag-style storage,
  // cleared by reset so never-written bytes are never exposed as valid.
  // Physical bytes and validity are separate: invalidation clears mem_bval
  // only and never rewrites mem_data.
  logic [DATA_BYTES-1:0][7:0] mem_data [BANK_COUNT][WORDS_PER_BANK];
  logic [DATA_BYTES-1:0] mem_bval [BANK_COUNT][WORDS_PER_BANK];

  // ------------------------------------------------------- holder tracking
  logic [BANK_COUNT-1:0] busy;
  logic [BANK_COUNT-1:0][TXN_ID_WIDTH-1:0] holder_txn;
  logic [BANK_COUNT-1:0][PORT_IDX_W-1:0] holder_port;

  // Per-Context accepted request (held verbatim for the txn lifetime).
  logic [CONTEXT_COUNT-1:0] active;
  logic [CONTEXT_COUNT-1:0] pdone;
  logic [CONTEXT_COUNT-1:0][TXN_ID_WIDTH-1:0] ptxn;
  logic [CONTEXT_COUNT-1:0][HOLDER_GEN_WIDTH-1:0] holder_generation;
  logic [CONTEXT_COUNT-1:0] pwrite;
  logic [CONTEXT_COUNT-1:0][SEG_CNT_W-1:0] psegcnt;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_IDX_W-1:0] pbank;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_ADDR_W-1:0] paddr;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_LEN_W-1:0] plen;

  // Profile state (generation strictly increases; separate domain).
  logic [PROFILE_IDX_W-1:0] active_profile;
  logic [PROFILE_GEN_WIDTH-1:0] profile_generation;

  logic [CONTEXT_COUNT-1:0] inv_active, inv_accept, inv_fire;
  logic [CONTEXT_COUNT-1:0][2:0] inv_fault;
  logic [CONTEXT_COUNT-1:0][BANK_IDX_W-1:0] inv_bank;
  logic [CONTEXT_COUNT-1:0][BANK_ADDR_W-1:0] inv_addr;
  logic [CONTEXT_COUNT-1:0][BANK_LEN_W-1:0] inv_len;
  logic [CONTEXT_COUNT-1:0][BEAT_W-1:0] inv_beat;
  logic [BANK_COUNT-1:0] inv_reserved_mask;

  // ------------------------------------------------------- request decode
  logic [CONTEXT_COUNT-1:0][BANK_COUNT-1:0] req_mask;
  logic [CONTEXT_COUNT-1:0][2:0] req_fault;
  logic [CONTEXT_COUNT-1:0] req_accept;
  int unsigned spm_bound;
  assign spm_bound = sel_spm(int'(active_profile));

  always_comb begin : req_decode
    logic [BANK_COUNT-1:0] taken;
    int unsigned cnt;
    taken = inv_reserved_mask;
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      cnt = int'(req_seg_count_i[c]);
      req_fault[c] = REJ_NONE;
      if (cnt < 1 || cnt > MAX_SEG) begin
        req_fault[c] = REJ_SHAPE;
      end else begin
        for (int s = 0; s < MAX_SEG; s++) begin
          if (s < cnt && req_fault[c] == REJ_NONE) begin
            req_fault[c] = seg_fault(req_bank_i[c][s], req_addr_i[c][s],
                                     req_len_i[c][s], spm_bound);
          end
        end
      end
      if (req_fault[c] == REJ_NONE && (&holder_generation[c]))
        req_fault[c] = REJ_GENERATION_EXHAUSTED;
      req_generation_o[c] = active[c] ? holder_generation[c]
        : holder_generation[c] + HOLDER_GEN_WIDTH'(1);
      req_mask[c] = '0;
      for (int s = 0; s < MAX_SEG; s++) begin
        if (s < cnt) req_mask[c] |= (BANK_COUNT'(1) << req_bank_i[c][s]);
      end
      // All-or-none preflight: no partial occupancy, no acceptance while any
      // requested bank is busy (including same-cycle grants of lower ports).
      req_ready_o[c] = !profile_set_valid_i && !active[c] && (req_fault[c] == REJ_NONE)
                       && ((req_mask[c] & taken) == '0);
      req_accept[c] = req_valid_i[c] && req_ready_o[c];
      if (req_accept[c]) taken |= req_mask[c];
      req_rej_o[c] = req_valid_i[c] && (req_fault[c] != REJ_NONE);
      req_rej_code_o[c] = req_fault[c];
    end
  end

  // ------------------------------------------------------ streamer decode
  // rfin_issued: the final response beat of the accepted read has already
  // been issued into out_*; no further issue until the next accept (terminal
  // read handling — the final fire must CLEAR out_valid, not reload).
  logic [CONTEXT_COUNT-1:0] rfin_issued;
  logic [CONTEXT_COUNT-1:0] wfire, rfire, rissue;
  logic [CONTEXT_COUNT-1:0][BANK_IDX_W-1:0] wcur_bank, rcur_bank;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES-1:0] wcur_en, rcur_en;
  logic [CONTEXT_COUNT-1:0][SEG_IDX_W-1:0] wseg, rseg;
  logic [CONTEXT_COUNT-1:0][BEAT_W-1:0] wbeat, rbeat;
  logic [CONTEXT_COUNT-1:0] wlast_seg, wlast_beat;
  logic [CONTEXT_COUNT-1:0] rlast_seg, rlast_beat;
  logic [CONTEXT_COUNT-1:0] out_valid, out_last;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0] out_data;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES-1:0] out_ben;
  logic [CONTEXT_COUNT-1:0][TXN_ID_WIDTH-1:0] out_txn;
  logic [CONTEXT_COUNT-1:0][HOLDER_GEN_WIDTH-1:0] out_generation;

  // Read response outputs.
  always_comb begin : rdata_out
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      rdata_valid_o[c] = out_valid[c];
      rdata_txn_o[c] = out_txn[c];
      rdata_generation_o[c] = out_generation[c];
      rdata_data_o[c] = out_data[c];
      rdata_ben_o[c] = out_ben[c];
    end
  end

  // Completion outputs.
  always_comb begin : done_out
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      req_done_o[c] = active[c] && pdone[c];
      req_done_txn_o[c] = ptxn[c];
      req_done_generation_o[c] = holder_generation[c];
    end
  end

  // Current write/read beat decode (combinational from held request state).
  always_comb begin : stream_decode
    int unsigned beats;
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      wcur_bank[c] = pbank[c][wseg[c]];
      wcur_en[c] = beat_en(paddr[c][wseg[c]], plen[c][wseg[c]], int'(wbeat[c]));
      beats = seg_beats(paddr[c][wseg[c]], plen[c][wseg[c]]);
      wlast_beat[c] = (wbeat[c] == BEAT_W'(beats - 1));
      wlast_seg[c] = (wseg[c] == SEG_IDX_W'(psegcnt[c] - 1));
      wdata_ready_o[c] = active[c] && pwrite[c] && !pdone[c];
      wfire[c] = wdata_ready_o[c] && wdata_valid_i[c];

      rcur_bank[c] = pbank[c][rseg[c]];
      rcur_en[c] = beat_en(paddr[c][rseg[c]], plen[c][rseg[c]], int'(rbeat[c]));
      beats = seg_beats(paddr[c][rseg[c]], plen[c][rseg[c]]);
      rlast_beat[c] = (rbeat[c] == BEAT_W'(beats - 1));
      rlast_seg[c] = (rseg[c] == SEG_IDX_W'(psegcnt[c] - 1));
      rissue[c] = active[c] && !pwrite[c] && !pdone[c] && !rfin_issued[c]
                  && (!out_valid[c] || rdata_ready_i[c]);
      rfire[c] = out_valid[c] && rdata_ready_i[c];
    end
  end

  // Single-writer-per-bank invariant across concurrent ports.
  always_comb begin : single_writer
    for (int c1 = 0; c1 < CONTEXT_COUNT; c1++) begin
      for (int c2 = c1 + 1; c2 < CONTEXT_COUNT; c2++) begin
        assert (!(wfire[c1] && wfire[c2] && (wcur_bank[c1] == wcur_bank[c2])))
          else $error("C029_ASSERT l1_bank single_writer_per_bank: ports %0d/%0d", c1, c2);
      end
    end
  end

  // Invalidation wins over new data on the same bank, never over a holder.
  // Only accepted requests reserve resources in this arbitration pass.
  always_comb begin : inv_decode
    logic [BANK_COUNT-1:0] taken;
    taken = busy;
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      if (inv_active[c]) taken[inv_bank[c]] = 1'b1;
    end
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      inv_fault[c] = seg_fault(inv_bank_i[c], inv_addr_i[c], inv_len_i[c], spm_bound);
      inv_ready_o[c] = !profile_set_valid_i && !inv_active[c]
        && ((inv_fault[c] != REJ_NONE) || !taken[inv_bank_i[c]]);
      inv_accept[c] = inv_valid_i[c] && inv_ready_o[c];
      inv_fire[c] = inv_accept[c] && (inv_fault[c] == REJ_NONE);
      if (inv_fire[c]) taken[inv_bank_i[c]] = 1'b1;
    end
    inv_reserved_mask = taken;
  end

  always_ff @(posedge clk) begin : invalidation
    if (!rstn) begin
      inv_active <= '0;
      inv_bank <= '0;
      inv_addr <= '0;
      inv_len <= '0;
      inv_beat <= '0;
      inv_ack_o <= '0;
      inv_rej_o <= '0;
      inv_rej_code_o <= '0;
    end else begin
      inv_ack_o <= '0;
      inv_rej_o <= '0;
      inv_rej_code_o <= '0;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (inv_accept[c]) begin
          if (inv_fault[c] != REJ_NONE) begin
            inv_rej_o[c] <= 1'b1;
            inv_rej_code_o[c] <= inv_fault[c];
          end else begin
            inv_active[c] <= 1'b1;
            inv_bank[c] <= inv_bank_i[c];
            inv_addr[c] <= inv_addr_i[c];
            inv_len[c] <= inv_len_i[c];
            inv_beat[c] <= '0;
          end
        end else if (inv_active[c]) begin
          if (int'(inv_beat[c]) + 1 == seg_beats(inv_addr[c], inv_len[c])) begin
            inv_active[c] <= 1'b0;
            inv_ack_o[c] <= 1'b1;
          end else begin
            inv_beat[c] <= inv_beat[c] + BEAT_W'(1);
          end
        end
      end
    end
  end

  // ------------------------------------------------------- SRAM write path
  always_ff @(posedge clk) begin : sram_write
    logic [DATA_BYTES-1:0] inv_en;
    if (!rstn) begin
      for (int b = 0; b < BANK_COUNT; b++) begin
        for (int w = 0; w < WORDS_PER_BANK; w++) begin
          mem_bval[b][w] <= '0;
        end
      end
      wseg <= '0;
      wbeat <= '0;
    end else begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (inv_active[c]) begin
          inv_en = beat_en(inv_addr[c], inv_len[c], int'(inv_beat[c]));
          for (int byi = 0; byi < DATA_BYTES; byi++) begin
            if (inv_en[byi]) begin
              mem_bval[inv_bank[c]][WORD_ADDR_W'(seg_base_word(inv_addr[c]) + inv_beat[c])][byi]
                <= 1'b0;
            end
          end
        end
      end
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (req_accept[c]) begin
          wseg[c] <= '0;
          wbeat[c] <= '0;
        end else if (wfire[c]) begin
          assert (active[c] && pwrite[c])
            else $error("C029_ASSERT l1_bank write_only_from_accepted_holder: port %0d", c);
          assert (seg_fault(wcur_bank[c], paddr[c][wseg[c]], plen[c][wseg[c]], spm_bound)
                  == REJ_NONE)
            else $error("C029_ASSERT l1_bank access_within_user_spm: port %0d", c);
          for (int byi = 0; byi < DATA_BYTES; byi++) begin
            if (wcur_en[c][byi]) begin
              mem_data[wcur_bank[c]][WORD_ADDR_W'(seg_base_word(paddr[c][wseg[c]]) + wbeat[c])][byi]
                <= wdata_i[c][8*byi +: 8];
              mem_bval[wcur_bank[c]][WORD_ADDR_W'(seg_base_word(paddr[c][wseg[c]]) + wbeat[c])][byi]
                <= 1'b1;
            end
          end
          if (wlast_beat[c]) begin
            wbeat[c] <= '0;
            if (!wlast_seg[c]) wseg[c] <= wseg[c] + SEG_IDX_W'(1);
          end else begin
            wbeat[c] <= wbeat[c] + BEAT_W'(1);
          end
        end
      end
    end
  end

  // -------------------------------------------------------- SRAM read path
  always_ff @(posedge clk) begin : sram_read
    if (!rstn) begin
      rseg <= '0;
      rbeat <= '0;
      out_valid <= '0;
      out_last <= '0;
      rfin_issued <= '0;
      out_data <= '0;
      out_ben <= '0;
      out_txn <= '0;
      out_generation <= '0;
    end else begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (req_accept[c]) begin
          rseg[c] <= '0;
          rbeat[c] <= '0;
          out_valid[c] <= 1'b0;
          out_last[c] <= 1'b0;
          rfin_issued[c] <= 1'b0;
        end else if (rissue[c]) begin
          assert (seg_fault(rcur_bank[c], paddr[c][rseg[c]], plen[c][rseg[c]], spm_bound)
                  == REJ_NONE)
            else $error("C029_ASSERT l1_bank access_within_user_spm: port %0d", c);
          // Synchronous SRAM access: one sourced BANK_LATENCY_CYCLES stage.
          out_valid[c] <= 1'b1;
          out_data[c] <= mem_data[rcur_bank[c]][WORD_ADDR_W'(seg_base_word(paddr[c][rseg[c]]) + rbeat[c])];
          out_ben[c] <= rcur_en[c]
                        & mem_bval[rcur_bank[c]][WORD_ADDR_W'(seg_base_word(paddr[c][rseg[c]]) + rbeat[c])];
          out_txn[c] <= ptxn[c];
          out_generation[c] <= holder_generation[c];
          out_last[c] <= rlast_seg[c] && rlast_beat[c];
          rfin_issued[c] <= rlast_seg[c] && rlast_beat[c];
          if (rlast_beat[c]) begin
            rbeat[c] <= '0;
            if (!rlast_seg[c]) rseg[c] <= rseg[c] + SEG_IDX_W'(1);
          end else begin
            rbeat[c] <= rbeat[c] + BEAT_W'(1);
          end
        end else if (rfire[c]) begin
          out_valid[c] <= 1'b0;
        end
      end
    end
  end

  // ------------------------------------------------ control (accept/fin)
  logic [CONTEXT_COUNT-1:0] fin_match, fin_accept;

  always_comb begin : fin_decode
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      fin_ready_o[c] = 1'b1;
      fin_match[c] = fin_valid_i[c] && active[c] && (fin_txn_i[c] == ptxn[c])
        && (fin_generation_i[c] == holder_generation[c]);
      fin_accept[c] = fin_match[c] && pdone[c];
    end
  end

  logic profile_quiescent, profile_args_ok;

  always_comb begin : profile_decode
    profile_quiescent = (busy == '0) && (active == '0) && (inv_active == '0);
    profile_args_ok = (int'(profile_index_i) < PROFILE_COUNT)
                      && (profile_generation_i > profile_generation);
    profile_set_ready_o = 1'b1;
  end

  always_ff @(posedge clk) begin : control
    logic [BANK_COUNT-1:0] busy_d;
    int unsigned freed_cnt;
    if (!rstn) begin
      busy <= '0;
      holder_txn <= '0;
      holder_port <= '0;
      active <= '0;
      pdone <= '0;
      ptxn <= '0;
      holder_generation <= '0;
      pwrite <= '0;
      psegcnt <= '0;
      pbank <= '0;
      paddr <= '0;
      plen <= '0;
      fin_ack_o <= '0;
      fin_rej_o <= '0;
      fin_rej_code_o <= '0;
      active_profile <= '0;
      profile_generation <= '0;
      profile_ack_o <= '0;
      profile_rej_o <= '0;
      profile_rej_code_o <= '0;
    end else begin
      busy_d = busy;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (req_accept[c]) begin
          assert (((req_mask[c] & busy) == '0) && (req_fault[c] == REJ_NONE))
            else $error("C029_ASSERT l1_bank atomic_all_or_none: port %0d", c);
          busy_d |= req_mask[c];
          for (int b = 0; b < BANK_COUNT; b++) begin
            if (req_mask[c][b]) begin
              holder_txn[b] <= req_txn_i[c];
              holder_port[b] <= PORT_IDX_W'(c);
            end
          end
          active[c] <= 1'b1;
          pdone[c] <= 1'b0;
          ptxn[c] <= req_txn_i[c];
          holder_generation[c] <= holder_generation[c] + HOLDER_GEN_WIDTH'(1);
          pwrite[c] <= req_write_i[c];
          psegcnt[c] <= req_seg_count_i[c];
          for (int s = 0; s < MAX_SEG; s++) begin
            pbank[c][s] <= req_bank_i[c][s];
            paddr[c][s] <= req_addr_i[c][s];
            plen[c][s] <= req_len_i[c][s];
          end
        end

        if (fin_valid_i[c]) begin
          if (fin_accept[c]) begin
            // Terminal read handling: a released read txn must have fully
            // drained — no residual response beat may remain.
            assert (pwrite[c] || !out_valid[c])
              else $error("C029_ASSERT l1_bank read_drained_at_release: port %0d", c);
            // Free exactly this holder's banks; never another owner's.
            freed_cnt = 0;
            for (int b = 0; b < BANK_COUNT; b++) begin
              if (busy[b] && (holder_port[b] == PORT_IDX_W'(c)) && (holder_txn[b] == ptxn[c])) begin
                busy_d[b] = 1'b0;
                holder_txn[b] <= '0;
                holder_port[b] <= '0;
                freed_cnt++;
              end
            end
            assert (freed_cnt > 0)
              else $error("C029_ASSERT l1_bank fin_releases_exact_holder: port %0d freed none", c);
            active[c] <= 1'b0;
            pdone[c] <= 1'b0;
            fin_ack_o[c] <= 1'b1;
            fin_rej_o[c] <= 1'b0;
            fin_rej_code_o[c] <= '0;
          end else begin
            fin_ack_o[c] <= 1'b0;
            fin_rej_o[c] <= 1'b1;
            fin_rej_code_o[c] <= fin_match[c] ? FIN_REJ_NOT_COMPLETE : FIN_REJ_NOT_HOLDER;
          end
        end else begin
          fin_ack_o[c] <= 1'b0;
          fin_rej_o[c] <= 1'b0;
          fin_rej_code_o[c] <= '0;
        end


        // Data-phase completion (no completion while a fin acks this cycle:
        // fin_accept requires pdone, and streamers stop at pdone).
        if (!req_accept[c] && !fin_accept[c]) begin
          if (pwrite[c]
               ? (wfire[c] && wlast_beat[c] && wlast_seg[c])
               : (rfire[c] && out_last[c])) begin
            pdone[c] <= 1'b1;
          end
        end
      end

      if (profile_set_valid_i) begin
        if (!profile_args_ok) begin
          profile_ack_o <= 1'b0;
          profile_rej_o <= 1'b1;
          profile_rej_code_o <= PROF_REJ_ARGS;
        end else if (!profile_quiescent) begin
          profile_ack_o <= 1'b0;
          profile_rej_o <= 1'b1;
          profile_rej_code_o <= PROF_REJ_QUIESCENT;
        end else begin
          assert (profile_quiescent)
            else $error("C029_ASSERT l1_bank profile_switch_quiescent_only");
          active_profile <= profile_index_i;
          profile_generation <= profile_generation_i;
          profile_ack_o <= 1'b1;
          profile_rej_o <= 1'b0;
          profile_rej_code_o <= '0;
        end
      end else begin
        profile_ack_o <= 1'b0;
        profile_rej_o <= 1'b0;
        profile_rej_code_o <= '0;
      end

      busy <= busy_d;
    end
  end

endmodule
