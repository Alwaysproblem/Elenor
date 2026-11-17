// tb_l1_bank.sv — C029 / W4 unit bench for rtl/memory/l1_bank.sv.
//
// Proves real bank SRAM byte/validity behavior and atomic multi-bank holder
// tracking only. It deliberately does NOT claim any gather/scatter FSM, cache,
// MSHR or program port (C030/C031+).
//
// Fixture provenance: the indexed gather/scatter fixture of
// pipeline_validator/tests/test_tile_indexed.py — DATA rows
// [[10..13],[20..23],[30..33],[40..43]] (little-endian i32 items), gather
// indices (3,0,3), scatter indices (1,1,2). The bench replays that exact
// ordered address/byte sequence through bank reads/writes: gather reads DATA
// rows 3,0,3; scatter writes OUT rows 1,1,2 with the gathered segments in
// ordinal order, so segment ordinal 1 overwrites ordinal 0 on OUT row 1.
// Expected OUT: row 1 == [10,11,12,13], row 2 == [40,41,42,43], rows 0/3
// never written (byte validity 0, never exposed as valid zeros).
//
// Scenarios (all on C1 and C2 unless noted; two-Context overlap cases are
// adapted for CONTEXT_COUNT=1 without weakening the C2 coverage):
//   S0  reset state: never-written user bytes read back validity 0.
//   S1  indexed fixture gather/scatter + ordered overwrite readback.
//   S2  compound multi-bank holder: stalled holder keeps banks; conflicting
//       compound request waits with NO partial occupancy (withdrawn request
//       leaves the free bank immediately usable); junk write beats from a
//       waiting request are never accepted; disjoint request completes while
//       the holder is stalled (C2; C1 adapts to sequential re-issue).
//   S3  (C2) same-cycle overlapping grants arbitrated; same-cycle disjoint
//       grants both accepted; stale/wrong-port/duplicate fin rejected without
//       touching the other owner; fin before data completion rejected.
//   S4  read latency == sourced BANK_LATENCY_CYCLES (1 cycle after accept);
//       backpressured response payload stable; disjoint port progresses while
//       a read is backpressured (C2).
//   S5  fault battery: reserved bytes, reserved-crossing, zero length, bad
//       segment count, bank overflow — all rejected with no side effects;
//       last user byte of the bank writable/readable under profile 0.
//   S6  profile transitions: quiescent-only (busy txn blocks the switch and
//       leaves the boundary unchanged), strictly increasing generation,
//       post-switch SPM/cache boundary faults, validity retained in the
//       still-user region across a switch.
//   S7  completion/release of one port progress while another port's data
//       issue is gated (C2); gated write never completes until resumed.
//   S8  checked exact-range validity invalidation (the agreed C029 lifecycle
//       hook): clears validity only (bytes kept), exactly the covered bytes
//       (no word/row rounding; neighbours untouched), one-cycle ACK before a
//       new binding may be published (ledger-side rule), idempotent acks,
//       fault-coded rejects, and reuse safety — a rewritten sub-range alone
//       becomes valid again while uncleared/neighbouring validity is not
//       fabricated.
//
//   CP1..CP6 (C030 cache data port, identical on C1 and C2 — the port is a
//      single independent channel): cp byte/validity semantics per write
//      mode (CP1), cache-window bounds and fault codes with SPM isolation
//      (CP2), arbitration against SPM holders/invalidations including the
//      fin-identity hazard and same-cycle SPM priority (CP3), CP_MAX_WAIT
//      anti-starvation under sustained back-to-back SPM pressure with the
//      measured bound (CP4), profile-switch gating on cp_busy /
//      cache_quiescent_i and window movement (CP5), and protocol shape:
//      backpressure stability, ready gating, one-cycle done pulse,
//      side-effect-free rejects (CP6).
// Builds: package, then l1_bank.sv, then this bench; Verilator --assert with
// -GCONTEXT_COUNT=1 and =2. Prints C029_PASS tb_l1_bank only after every
// check passed; failures print C029_ASSERT markers and end non-zero.
`timescale 1ns/1ps

module tb_l1_bank
  import c029_params_pkg::*;
#(
  parameter int CONTEXT_COUNT = C2_CONTEXTS,
  // Mirrors the DUT fairness knob so CP4 asserts the bound it configures.
  parameter int CP_MAX_WAIT = 8
);

  localparam int unsigned TXN_W = 64;
  localparam int unsigned MAX_SEG = BANK_COUNT;
  localparam int unsigned BANK_IDX_W = (BANK_COUNT <= 1) ? 1 : $clog2(BANK_COUNT);
  localparam int unsigned BANK_ADDR_W = (BANK_BYTES <= 1) ? 1 : $clog2(BANK_BYTES);
  localparam int unsigned BANK_LEN_W = $clog2(BANK_BYTES + 1);
  localparam int unsigned SEG_CNT_W = $clog2(MAX_SEG + 1);
  localparam int unsigned PROFILE_IDX_W = $clog2(PROFILE_COUNT);
  localparam int unsigned BEAT_W = $clog2(BANK_BYTES / DATA_BYTES + 1);
  localparam int DW = DATA_BYTES * 8;

  // First user-visible byte of every bank (reserved region below it).
  localparam int unsigned USER0 = SYSTEM_RESERVED_BYTES;

  // ------------------------------------------------------------- DUT I/O
  logic clk;
  logic rstn;

  logic [CONTEXT_COUNT-1:0]                    req_valid_i;
  logic [CONTEXT_COUNT-1:0]                    req_ready_o;
  logic [CONTEXT_COUNT-1:0]                    req_rej_o;
  logic [CONTEXT_COUNT-1:0][2:0]               req_rej_code_o;
  logic [CONTEXT_COUNT-1:0][TXN_W-1:0]         req_txn_i;
  logic [CONTEXT_COUNT-1:0][63:0] req_generation_o, accepted_generation;
  logic [CONTEXT_COUNT-1:0]                    req_write_i;
  logic [CONTEXT_COUNT-1:0][SEG_CNT_W-1:0]     req_seg_count_i;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_IDX_W-1:0]  req_bank_i;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_ADDR_W-1:0] req_addr_i;
  logic [CONTEXT_COUNT-1:0][MAX_SEG-1:0][BANK_LEN_W-1:0]  req_len_i;

  logic [CONTEXT_COUNT-1:0]                    req_done_o;
  logic [CONTEXT_COUNT-1:0][TXN_W-1:0]         req_done_txn_o;
  logic [CONTEXT_COUNT-1:0][63:0] req_done_generation_o;

  logic [CONTEXT_COUNT-1:0]                    wdata_valid_i;
  logic [CONTEXT_COUNT-1:0]                    wdata_ready_o;
  logic [CONTEXT_COUNT-1:0][DW-1:0]            wdata_i;

  logic [CONTEXT_COUNT-1:0]                    rdata_valid_o;
  logic [CONTEXT_COUNT-1:0]                    rdata_ready_i;
  logic [CONTEXT_COUNT-1:0][TXN_W-1:0]         rdata_txn_o;
  logic [CONTEXT_COUNT-1:0][63:0] rdata_generation_o;
  logic [CONTEXT_COUNT-1:0][DW-1:0]            rdata_data_o;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES-1:0]    rdata_ben_o;

  logic [CONTEXT_COUNT-1:0]                    fin_valid_i;
  logic [CONTEXT_COUNT-1:0][TXN_W-1:0]         fin_txn_i;
  logic [CONTEXT_COUNT-1:0][63:0] fin_generation_i;
  logic [CONTEXT_COUNT-1:0]                    fin_ready_o;
  logic [CONTEXT_COUNT-1:0]                    fin_ack_o;
  logic [CONTEXT_COUNT-1:0]                    fin_rej_o;
  logic [CONTEXT_COUNT-1:0][1:0]               fin_rej_code_o;

  logic [CONTEXT_COUNT-1:0]                    inv_valid_i;
  logic [CONTEXT_COUNT-1:0]                    inv_ready_o;
  logic [CONTEXT_COUNT-1:0][BANK_IDX_W-1:0]    inv_bank_i;
  logic [CONTEXT_COUNT-1:0][BANK_ADDR_W-1:0]   inv_addr_i;
  logic [CONTEXT_COUNT-1:0][BANK_LEN_W-1:0]    inv_len_i;
  logic [CONTEXT_COUNT-1:0]                    inv_ack_o;
  logic [CONTEXT_COUNT-1:0]                    inv_rej_o;
  logic [CONTEXT_COUNT-1:0][2:0]               inv_rej_code_o;

  logic                         profile_set_valid_i;
  logic [PROFILE_IDX_W-1:0]     profile_index_i;
  logic [63:0]                  profile_generation_i;
  logic                         profile_set_ready_o;
  logic                         profile_ack_o;
  logic                         profile_rej_o;
  logic [1:0]                   profile_rej_code_o;

  // Cache data port (C030).
  logic        cp_valid_i, cp_write_i, cp_wmode_i;
  logic        cp_wdata_valid_i, cp_rdata_ready_i, cache_quiescent_i;
  logic [BANK_IDX_W-1:0]  cp_bank_i;
  logic [BANK_ADDR_W-1:0] cp_addr_i;
  logic [BEAT_W-1:0]      cp_beats_i;
  logic [DW-1:0]          cp_wdata_i;
  logic [DATA_BYTES-1:0]  cp_wben_i;
  logic        cp_ready_o, cp_rej_o, cp_wdata_ready_o;
  logic        cp_rdata_valid_o, cp_done_o, cp_busy_o;
  logic [2:0]  cp_rej_code_o;
  logic [DW-1:0] cp_rdata_o;
  logic [DATA_BYTES-1:0] cp_rben_o;
  logic [PROFILE_IDX_W-1:0] active_profile_o;
  logic [63:0] profile_generation_o;

  l1_bank #(
    .CONTEXT_COUNT(CONTEXT_COUNT),
    .TXN_ID_WIDTH (TXN_W),
    .CP_MAX_WAIT  (CP_MAX_WAIT)
  ) dut (
    .clk                  (clk),
    .rstn                 (rstn),
    .req_valid_i          (req_valid_i),
    .req_ready_o          (req_ready_o),
    .req_rej_o            (req_rej_o),
    .req_rej_code_o       (req_rej_code_o),
    .req_txn_i            (req_txn_i),
    .req_generation_o     (req_generation_o),
    .req_write_i          (req_write_i),
    .req_seg_count_i      (req_seg_count_i),
    .req_bank_i           (req_bank_i),
    .req_addr_i           (req_addr_i),
    .req_len_i            (req_len_i),
    .req_done_o           (req_done_o),
    .req_done_txn_o       (req_done_txn_o),
    .req_done_generation_o(req_done_generation_o),
    .wdata_valid_i        (wdata_valid_i),
    .wdata_ready_o        (wdata_ready_o),
    .wdata_i              (wdata_i),
    .rdata_valid_o        (rdata_valid_o),
    .rdata_ready_i        (rdata_ready_i),
    .rdata_txn_o          (rdata_txn_o),
    .rdata_generation_o   (rdata_generation_o),
    .rdata_data_o         (rdata_data_o),
    .rdata_ben_o          (rdata_ben_o),
    .fin_valid_i          (fin_valid_i),
    .fin_txn_i            (fin_txn_i),
    .fin_generation_i    (fin_generation_i),
    .fin_ready_o          (fin_ready_o),
    .fin_ack_o            (fin_ack_o),
    .fin_rej_o            (fin_rej_o),
    .fin_rej_code_o       (fin_rej_code_o),
    .inv_valid_i          (inv_valid_i),
    .inv_ready_o          (inv_ready_o),
    .inv_bank_i           (inv_bank_i),
    .inv_addr_i           (inv_addr_i),
    .inv_len_i            (inv_len_i),
    .inv_ack_o            (inv_ack_o),
    .inv_rej_o            (inv_rej_o),
    .inv_rej_code_o       (inv_rej_code_o),
    .profile_set_valid_i  (profile_set_valid_i),
    .profile_index_i      (profile_index_i),
    .profile_generation_i (profile_generation_i),
    .profile_set_ready_o  (profile_set_ready_o),
    .profile_ack_o        (profile_ack_o),
    .profile_rej_o        (profile_rej_o),
    .profile_rej_code_o   (profile_rej_code_o),
    .cp_valid_i           (cp_valid_i),
    .cp_write_i           (cp_write_i),
    .cp_bank_i            (cp_bank_i),
    .cp_addr_i            (cp_addr_i),
    .cp_beats_i           (cp_beats_i),
    .cp_wmode_i           (cp_wmode_i),
    .cp_wdata_valid_i     (cp_wdata_valid_i),
    .cp_wdata_i           (cp_wdata_i),
    .cp_wben_i            (cp_wben_i),
    .cp_rdata_ready_i     (cp_rdata_ready_i),
    .cache_quiescent_i    (cache_quiescent_i),
    .cp_ready_o           (cp_ready_o),
    .cp_rej_o             (cp_rej_o),
    .cp_rej_code_o        (cp_rej_code_o),
    .cp_wdata_ready_o     (cp_wdata_ready_o),
    .cp_rdata_valid_o     (cp_rdata_valid_o),
    .cp_rdata_o           (cp_rdata_o),
    .cp_rben_o            (cp_rben_o),
    .cp_done_o            (cp_done_o),
    .cp_busy_o            (cp_busy_o),
    .active_profile_o     (active_profile_o),
    .profile_generation_o (profile_generation_o)
  );

  always #5 clk = ~clk;

  int errors;
  int checks;
  int wait_obs;
  int latency_obs;
  logic [63:0] prior_holder_generation;
  int probe_port;
  int P0 = 0;
  int P1 = 1;

  // Capture the offered incarnation on every real handshake, including
  // simultaneous requests driven outside req_push.
  always @(posedge clk) begin
    if (!rstn) accepted_generation <= '0;
    else begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (req_valid_i[c] && req_ready_o[c]) begin
          check_val(req_generation_o[c], accepted_generation[c] + 64'd1,
                    "holder generation advances on acceptance");
          accepted_generation[c] <= req_generation_o[c];
        end
      end
    end
  end

  // Global cycle counter (posedge count) for arbitration-bound measurement.
  int unsigned cyc;
  always @(posedge clk) begin
    if (!rstn) cyc <= 0;
    else cyc <= cyc + 1;
  end

  // Timeout guard: any hang (e.g. a request that never becomes ready) fails
  // with a C029_ASSERT marker instead of running forever.
  initial begin
    #5_000_000;
    $error("C029_ASSERT tb_l1_bank progress: timeout");
    $display("C029_FAIL tb_l1_bank timeout");
    $fatal(1);
  end

  // -------------------------------------------------------------- helpers
  task automatic check_bit(input logic got, input logic exp, input string what);
    begin
      checks = checks + 1;
      if (got !== exp) begin
        errors = errors + 1;
        $error("C029_ASSERT tb_l1_bank %s: got %0b expected %0b", what, got, exp);
      end
    end
  endtask

  task automatic check_val(input logic [63:0] got, input logic [63:0] exp, input string what);
    begin
      checks = checks + 1;
      if (got !== exp) begin
        errors = errors + 1;
        $error("C029_ASSERT tb_l1_bank %s: got 0x%0h expected 0x%0h", what, got, exp);
      end
    end
  endtask

  task automatic check_ready(input int c, input logic exp, input string what);
    begin
      check_bit(req_ready_o[c], exp, what);
    end
  endtask

  // Compare a read beat: byte validity must match exactly; every enabled byte
  // must carry the expected value. Disabled bytes are never checked (their
  // data is undefined by contract, only validity is guaranteed).
  task automatic check_beat(input logic [DW-1:0] gd, input logic [DATA_BYTES-1:0] gen_,
                            input logic [DW-1:0] ed, input logic [DATA_BYTES-1:0] een,
                            input string what);
    begin
      checks = checks + 1;
      if (gen_ !== een) begin
        errors = errors + 1;
        $error("C029_ASSERT tb_l1_bank %s byte-validity: got 0x%08h expected 0x%08h",
               what, gen_, een);
      end else begin
        for (int byi = 0; byi < DATA_BYTES; byi++) begin
          checks = checks + 1;
          if (een[byi] && (gd[8*byi +: 8] !== ed[8*byi +: 8])) begin
            errors = errors + 1;
            $error("C029_ASSERT tb_l1_bank %s byte %0d: got 0x%02h expected 0x%02h",
                   what, byi, gd[8*byi +: 8], ed[8*byi +: 8]);
          end
        end
      end
    end
  endtask

  // DATA row r packed little-endian into 16 bytes (bytes 0..15).
  function automatic logic [127:0] row128(input int r);
    row128 = {32'((r+1)*10+3), 32'((r+1)*10+2), 32'((r+1)*10+1), 32'((r+1)*10+0)};
  endfunction

  // 32-byte beat: lo at bytes 0..15, hi at bytes 16..31.
  function automatic logic [DW-1:0] mkword(input logic [127:0] lo, input logic [127:0] hi);
    mkword = {hi, lo};
  endfunction

  // ------------------------------------------------------- request staging
  logic [63:0]       s_txn [2];
  logic              s_write [2];
  logic [SEG_CNT_W-1:0] s_cnt [2];
  logic [BANK_IDX_W-1:0]  s_bank [2][MAX_SEG];
  logic [BANK_ADDR_W-1:0] s_addr [2][MAX_SEG];
  logic [BANK_LEN_W-1:0]  s_len [2][MAX_SEG];

  task automatic req_stage(input int pc, input int n, input logic wr, input logic [63:0] t);
    begin
      s_txn[pc] = t;
      s_write[pc] = wr;
      s_cnt[pc] = SEG_CNT_W'(n);
    end
  endtask

  task automatic seg_set(input int pc, input int s, input int bk, input int adr, input int len);
    begin
      s_bank[pc][s] = BANK_IDX_W'(bk);
      s_addr[pc][s] = BANK_ADDR_W'(adr);
      s_len[pc][s] = BANK_LEN_W'(len);
    end
  endtask

  task automatic req_drive(input int c);
    begin
      req_valid_i[c] = 1'b1;
      req_txn_i[c] = s_txn[c];
      req_write_i[c] = s_write[c];
      req_seg_count_i[c] = s_cnt[c];
      for (int s = 0; s < MAX_SEG; s++) begin
        req_bank_i[c][s] = s_bank[c][s];
        req_addr_i[c][s] = s_addr[c][s];
        req_len_i[c][s] = s_len[c][s];
      end
    end
  endtask

  task automatic req_undrive(input int c);
    begin
      req_valid_i[c] = 1'b0;
    end
  endtask

  // Drive a staged request and wait for acceptance. waits counts full cycles
  // spent waiting (conflict visibility).
  task automatic req_push(input int c, output int waits);
    begin
      @(negedge clk);
      req_drive(c);
      #1;
      waits = 0;
      while (!req_ready_o[c]) begin
        @(negedge clk);
        #1;
        waits = waits + 1;
      end
      @(posedge clk);
      #1;
      req_undrive(c);
    end
  endtask

  // Drive a staged request, verify it is rejected with a stable fault code,
  // then withdraw. No bank may change (later scenarios re-verify).
  task automatic req_expect_reject(input int c, input logic [2:0] code, input string what);
    logic [2:0] code_first;
    begin
      @(negedge clk);
      req_drive(c);
      #1;
      check_ready(c, 1'b0, {what, ": rejected request not accepted"});
      check_bit(req_rej_o[c], 1'b1, {what, ": reject flag"});
      check_val({61'd0, req_rej_code_o[c]}, {61'd0, code}, {what, ": fault code"});
      code_first = req_rej_code_o[c];
      // Hold one more full cycle: the fault must stay stable while valid.
      @(negedge clk);
      #1;
      check_bit(req_rej_o[c], 1'b1, {what, ": reject stable"});
      check_val({61'd0, req_rej_code_o[c]}, {61'd0, code_first}, {what, ": fault stable"});
      req_undrive(c);
      #1;
    end
  endtask

  // Push one write beat (writes are always ready for an active write txn).
  task automatic wbeat_push(input int c, input logic [DW-1:0] w);
    begin
      @(negedge clk);
      wdata_valid_i[c] = 1'b1;
      wdata_i[c] = w;
      #1;
      check_bit(wdata_ready_o[c], 1'b1, "wdata ready for active write");
      @(posedge clk);
      #1;
      wdata_valid_i[c] = 1'b0;
    end
  endtask

  // Pop one read beat. stall_cycles holds rdata_ready low for that many full
  // cycles first, checking valid stays high and the payload is byte-stable.
  task automatic rbeat_pop(input int c, input int stall_cycles,
                           output logic [DW-1:0] d, output logic [DATA_BYTES-1:0] en,
                           output logic [63:0] t);
    begin
      @(negedge clk);
      #1;
      while (!rdata_valid_o[c]) begin
        @(negedge clk);
        #1;
      end
      d = rdata_data_o[c];
      en = rdata_ben_o[c];
      t = rdata_txn_o[c];
      check_val(rdata_generation_o[c], accepted_generation[c], "read holder generation");
      for (int i = 0; i < stall_cycles; i++) begin
        @(negedge clk);
        #1;
        check_bit(rdata_valid_o[c], 1'b1, "rdata valid held under backpressure");
        check_bit((rdata_data_o[c] == d) && (rdata_ben_o[c] == en)
                  && (rdata_txn_o[c] == t), 1'b1,
                  "rdata payload stable under backpressure");
        check_val(rdata_generation_o[c], accepted_generation[c], "read generation stable");
      end
      rdata_ready_i[c] = 1'b1;
      #1;
      check_bit(rdata_valid_o[c], 1'b1, "rdata valid at accept");
      @(posedge clk);
      #1;
      rdata_ready_i[c] = 1'b0;
    end
  endtask

  task automatic rpop_check(input int c, input int stall,
                            input logic [DW-1:0] ed, input logic [DATA_BYTES-1:0] een,
                            input logic [63:0] etxn, input string what);
    logic [DW-1:0] d;
    logic [DATA_BYTES-1:0] en;
    logic [63:0] t;
    begin
      rbeat_pop(c, stall, d, en, t);
      check_beat(d, en, ed, een, what);
      check_val(t, etxn, {what, ": txn echo"});
    end
  endtask

  task automatic wait_done(input int c, input int max_wait, input string what);
    int n;
    begin
      n = 0;
      @(negedge clk);
      #1;
      while (!req_done_o[c] && n <= max_wait) begin
        @(negedge clk);
        #1;
        n = n + 1;
      end
      check_bit(req_done_o[c], 1'b1, {what, ": done asserted"});
      check_bit((n <= max_wait), 1'b1, {what, ": done bounded"});
      check_val(req_done_generation_o[c], accepted_generation[c], "done holder generation");
    end
  endtask

  task automatic fin_push(input int c, input logic [63:0] t, input bit exp_ack,
                          input logic [1:0] exp_code, input string what);
    begin
      @(negedge clk);
      fin_valid_i[c] = 1'b1;
      fin_txn_i[c] = t;
      fin_generation_i[c] = accepted_generation[c];
      #1;
      check_bit(fin_ready_o[c], 1'b1, {what, ": fin ready"});
      @(posedge clk);
      #1;
      fin_valid_i[c] = 1'b0;
      if (exp_ack) begin
        check_bit(fin_ack_o[c], 1'b1, {what, ": fin acked"});
        // Terminal handling: after the release edge no response beat may
        // remain (an extra duplicated final beat would show here) and the
        // write stream must be closed.
        check_bit(rdata_valid_o[c], 1'b0, {what, ": no residual response after release"});
        check_bit(wdata_ready_o[c], 1'b0, {what, ": write stream closed after release"});
      end else begin
        check_bit(fin_rej_o[c], 1'b1, {what, ": fin rejected"});
        check_bit(fin_ack_o[c], 1'b0, {what, ": no fin ack"});
        check_val({62'd0, fin_rej_code_o[c]}, {62'd0, exp_code}, {what, ": fin code"});
      end
    end
  endtask

  // Invalidation owns a bank and walks its exact range before acknowledging.
  task automatic inv_push(input int c, input int bk, input int adr, input int len,
                          input bit exp_ack, input logic [2:0] exp_code, input string what);
    int n;
    begin
      @(negedge clk);
      inv_valid_i[c] = 1'b1;
      inv_bank_i[c] = BANK_IDX_W'(bk);
      inv_addr_i[c] = BANK_ADDR_W'(adr);
      inv_len_i[c] = BANK_LEN_W'(len);
      #1;
      n = 0;
      while (!inv_ready_o[c] && n <= BANK_BYTES / DATA_BYTES + 4) begin
        @(negedge clk); #1; n++;
      end
      check_bit(inv_ready_o[c], 1'b1, {what, ": inv accepted"});
      @(posedge clk); #1;
      inv_valid_i[c] = 1'b0;
      n = 0;
      while (!inv_ack_o[c] && !inv_rej_o[c] && n <= BANK_BYTES / DATA_BYTES + 2) begin
        @(posedge clk); #1; n++;
      end
      check_bit(n <= BANK_BYTES / DATA_BYTES + 2, 1'b1, {what, ": bounded clear"});
      if (exp_ack) begin
        check_bit(inv_ack_o[c], 1'b1, {what, ": inv ack after clear"});
        check_bit(inv_rej_o[c], 1'b0, {what, ": no inv reject"});
      end else begin
        check_bit(inv_rej_o[c], 1'b1, {what, ": inv rejected"});
        check_bit(inv_ack_o[c], 1'b0, {what, ": no inv ack"});
        check_val({61'd0, inv_rej_code_o[c]}, {61'd0, exp_code}, {what, ": inv code"});
      end
    end
  endtask

  task automatic profile_push(input logic [PROFILE_IDX_W-1:0] idx, input logic [63:0] gen,
                              input bit exp_ack, input logic [1:0] exp_code, input string what);
    begin
      @(negedge clk);
      profile_set_valid_i = 1'b1;
      profile_index_i = idx;
      profile_generation_i = gen;
      #1;
      check_bit(profile_set_ready_o, 1'b1, {what, ": profile ready"});
      @(posedge clk);
      #1;
      profile_set_valid_i = 1'b0;
      if (exp_ack) begin
        check_bit(profile_ack_o, 1'b1, {what, ": profile acked"});
        check_bit(profile_rej_o, 1'b0, {what, ": no profile reject"});
      end else begin
        check_bit(profile_rej_o, 1'b1, {what, ": profile rejected"});
        check_bit(profile_ack_o, 1'b0, {what, ": no profile ack"});
        check_val({62'd0, profile_rej_code_o}, {62'd0, exp_code}, {what, ": profile code"});
      end
    end
  endtask

  // -------------------------------------------------- cache-port helpers
  // Drive a staged cp request (call at a negedge; hold until cp_push).
  task automatic cp_drive(input bit wr, input int bk, input int adr, input int beats,
                          input bit wmode);
    begin
      cp_valid_i = 1'b1;
      cp_write_i = wr;
      cp_bank_i = BANK_IDX_W'(bk);
      cp_addr_i = BANK_ADDR_W'(adr);
      cp_beats_i = BEAT_W'(beats);
      cp_wmode_i = wmode;
    end
  endtask

  task automatic cp_push(output int waits);
    begin
      waits = 0;
      #1;
      while (!cp_ready_o) begin
        @(negedge clk);
        #1;
        waits = waits + 1;
      end
      @(posedge clk);
      #1;
      cp_valid_i = 1'b0;
    end
  endtask

  // Drive a staged cp request, verify a stable fault code, withdraw; no
  // state may change (busy stays low, no write stream).
  task automatic cp_expect_reject(input logic [2:0] code, input string what);
    logic [2:0] code_first;
    begin
      #1;
      check_bit(cp_ready_o, 1'b0, {what, ": rejected cp not accepted"});
      check_bit(cp_rej_o, 1'b1, {what, ": cp reject flag"});
      check_val({61'd0, cp_rej_code_o}, {61'd0, code}, {what, ": cp fault code"});
      code_first = cp_rej_code_o;
      check_bit(cp_busy_o, 1'b0, {what, ": reject has no side effect (busy)"});
      check_bit(cp_wdata_ready_o, 1'b0, {what, ": reject opens no write stream"});
      @(negedge clk);
      #1;
      check_bit(cp_rej_o, 1'b1, {what, ": cp reject stable"});
      check_val({61'd0, cp_rej_code_o}, {61'd0, code_first}, {what, ": cp code stable"});
      cp_valid_i = 1'b0;
      #1;
    end
  endtask

  // Push one cp write beat (ready is high while a cp write is active).
  task automatic cp_wbeat(input logic [DW-1:0] wd, input logic [DATA_BYTES-1:0] ben);
    begin
      @(negedge clk);
      cp_wdata_valid_i = 1'b1;
      cp_wdata_i = wd;
      cp_wben_i = ben;
      #1;
      check_bit(cp_wdata_ready_o, 1'b1, "cp wdata ready while write active");
      @(posedge clk);
      #1;
      cp_wdata_valid_i = 1'b0;
    end
  endtask

  // Pop one cp read beat; stall_cycles checks valid/payload stability.
  task automatic cp_rpop(input int stall, output logic [DW-1:0] d,
                         output logic [DATA_BYTES-1:0] en);
    begin
      @(negedge clk);
      #1;
      while (!cp_rdata_valid_o) begin
        @(negedge clk);
        #1;
      end
      d = cp_rdata_o;
      en = cp_rben_o;
      for (int i = 0; i < stall; i++) begin
        @(negedge clk);
        #1;
        check_bit(cp_rdata_valid_o, 1'b1, "cp rdata valid held under backpressure");
        check_bit((cp_rdata_o == d) && (cp_rben_o == en), 1'b1,
                  "cp rdata payload stable under backpressure");
      end
      cp_rdata_ready_i = 1'b1;
      #1;
      check_bit(cp_rdata_valid_o, 1'b1, "cp rdata valid at accept");
      @(posedge clk);
      #1;
      cp_rdata_ready_i = 1'b0;
    end
  endtask

  task automatic cp_rpop_check(input int stall, input logic [DW-1:0] ed,
                               input logic [DATA_BYTES-1:0] een, input string what);
    logic [DW-1:0] d;
    logic [DATA_BYTES-1:0] en;
    begin
      cp_rpop(stall, d, en);
      check_beat(d, en, ed, een, what);
    end
  endtask

  // Wait for the one-cycle done pulse; busy must have cleared at the
  // release edge and the pulse must not repeat.
  task automatic cp_wait_done(input int max_wait, input string what);
    int n;
    begin
      n = 0;
      @(negedge clk);
      #1;
      while (!cp_done_o && n <= max_wait) begin
        @(negedge clk);
        #1;
        n = n + 1;
      end
      check_bit(cp_done_o, 1'b1, {what, ": cp done pulse"});
      check_bit(n <= max_wait, 1'b1, {what, ": cp done bounded"});
      check_bit(cp_busy_o, 1'b0, {what, ": cp busy cleared at release edge"});
      check_bit(cp_wdata_ready_o, 1'b0, {what, ": cp write stream closed at done"});
      @(negedge clk);
      #1;
      check_bit(cp_done_o, 1'b0, {what, ": cp done is a one-cycle pulse"});
      check_bit(cp_busy_o, 1'b0, {what, ": cp stays idle after done"});
    end
  endtask
  // ------------------------------------------------------------ fixtures
  localparam logic [DW-1:0] W_A0  = {64{4'h1}};
  localparam logic [DW-1:0] W_A1  = {64{4'h2}};
  localparam logic [DW-1:0] W_A2  = {64{4'h3}};
  localparam logic [DW-1:0] W_C   = {64{4'h4}};
  localparam logic [DW-1:0] W_P0  = {64{4'h5}};
  localparam logic [DW-1:0] W_P1  = {64{4'h6}};
  localparam logic [DW-1:0] W_D10 = {64{4'h7}};
  localparam logic [DW-1:0] W_D11 = {64{4'h8}};
  localparam logic [DW-1:0] W_A2B = {64{4'h9}};
  localparam logic [DW-1:0] W_B2B = {64{4'hA}};
  localparam logic [DW-1:0] W_F14 = {64{4'hB}};
  localparam logic [DW-1:0] W_15  = {64{4'hC}};
  localparam logic [DW-1:0] W_G7  = {64{4'hD}};
  localparam logic [DW-1:0] W_B9  = {64{4'hF}};
  localparam logic [DW-1:0] JUNK  = {64{4'hE}};
  localparam logic [DW-1:0] W_VA  = {64{4'h3}};
  localparam logic [DW-1:0] W_VB  = {64{4'h4}};
  localparam logic [63:0]  PL    = 64'h1122_3344_5566_7788;
  localparam logic [63:0]  PH    = 64'h99AA_BBCC_DDEE_FF00;
  localparam logic [31:0]  PX    = 32'hCAFE_F00D;
  localparam logic [31:0]  PY    = 32'h0DDB_A11C;
  localparam logic [DATA_BYTES-1:0] EN_LO16  = 32'h0000_FFFF;
  localparam logic [DATA_BYTES-1:0] EN_HI16  = 32'hFFFF_0000;
  localparam logic [DATA_BYTES-1:0] EN_FULL  = 32'hFFFF_FFFF;
  localparam logic [DATA_BYTES-1:0] EN_B2431 = 32'hFF00_0000;
  localparam logic [DATA_BYTES-1:0] EN_B07   = 32'h0000_00FF;
  localparam logic [DATA_BYTES-1:0] EN_B2831 = 32'hF000_0000;
  localparam logic [DATA_BYTES-1:0] EN_B31   = 32'h8000_0000;
  localparam logic [DATA_BYTES-1:0] EN_B8    = 32'h0000_0100;
  localparam logic [DATA_BYTES-1:0] EN_B47   = 32'h0000_00F0;
  localparam logic [DW-1:0] W_CP1F = {192'd0, 32'h2222_2222, 32'h1111_1111};

  localparam logic [63:0] TXN_S0  = 64'h0000_1000_0000_0001;
  localparam logic [63:0] TXN_DW  = 64'h0000_1000_0000_0002;
  localparam logic [63:0] TXN_G   = 64'h0000_1000_0000_0003;
  localparam logic [63:0] TXN_SC  = 64'h0000_1000_0000_0004;
  localparam logic [63:0] TXN_RB  = 64'h0000_1000_0000_0005;
  localparam logic [63:0] TXN_A   = 64'h0000_2000_0000_0001;
  localparam logic [63:0] TXN_B   = 64'h0000_2000_0000_0002;
  localparam logic [63:0] TXN_C   = 64'h0000_2000_0000_0003;
  localparam logic [63:0] TXN_Q0  = 64'h0000_2000_0000_0004;
  localparam logic [63:0] TXN_Q1  = 64'h0000_2000_0000_0005;
  localparam logic [63:0] TXN_R0  = 64'h0000_2000_0000_0006;
  localparam logic [63:0] TXN_R1  = 64'h0000_2000_0000_0007;
  localparam logic [63:0] TXN_A2  = 64'h0000_2000_0000_0008;
  localparam logic [63:0] TXN_B2  = 64'h0000_2000_0000_0009;
  localparam logic [63:0] TXN_F14 = 64'h0000_2000_0000_000A;
  localparam logic [63:0] TXN_FR  = 64'h0000_2000_0000_000B;
  localparam logic [63:0] TXN_WR  = 64'h0000_2000_0000_000C;
  localparam logic [63:0] TXN_BP  = 64'h0000_2000_0000_000D;
  localparam logic [63:0] TXN_15  = 64'h0000_2000_0000_000E;
  localparam logic [63:0] TXN_LST = 64'h0000_2000_0000_000F;
  localparam logic [63:0] TXN_G7  = 64'h0000_2000_0000_0010;
  localparam logic [63:0] TXN_H8  = 64'h0000_3000_0000_0001;
  localparam logic [63:0] TXN_BND = 64'h0000_3000_0000_0002;
  localparam logic [63:0] TXN_P9  = 64'h0000_3000_0000_0003;
  localparam logic [63:0] TXN_X4  = 64'h0000_3000_0000_0004;
  localparam logic [63:0] TXN_X10 = 64'h0000_3000_0000_0005;
  localparam logic [63:0] TXN_Y11 = 64'h0000_3000_0000_0006;
  localparam logic [63:0] TXN_R4  = 64'h0000_3000_0000_0007;
  localparam logic [63:0] TXN_8A  = 64'h0000_4000_0000_0001;
  localparam logic [63:0] TXN_8B  = 64'h0000_4000_0000_0002;
  localparam logic [63:0] TXN_8C  = 64'h0000_4000_0000_0003;
  localparam logic [63:0] TXN_8D  = 64'h0000_4000_0000_0004;
  localparam logic [63:0] TXN_CP2  = 64'h0000_6000_0000_0000;
  localparam logic [63:0] TXN_CP3A = 64'h0000_6000_0000_0001;
  localparam logic [63:0] TXN_CP3B = 64'h0000_6000_0000_0002;
  localparam logic [63:0] TXN_CP3B2 = 64'h0000_6000_0000_0003;
  localparam logic [63:0] TXN_CP3C = 64'h0000_6000_0000_0004;
  localparam logic [63:0] TXN_CP3D = 64'h0000_6000_0000_0005;
  localparam logic [63:0] TXN_CP3E = 64'h0000_6000_0000_0006;
  localparam logic [63:0] TXN_CP4  = 64'h0000_6000_0000_0010;
  localparam logic [63:0] TXN_CP4M = 64'h0000_6000_0000_0011;
  localparam logic [63:0] TXN_CP5  = 64'h0000_6000_0000_0020;

  int w;
  int cw;
  int unsigned t_a, t_b, t_cp0, t_cp1, drain_q, lat_q;
  bit cp_got;

  // ------------------------------------------------------------- sequence
  initial begin
    errors = 0;
    checks = 0;
    wait_obs = 0;
    latency_obs = 0;
    clk = 1'b0;
    rstn = 1'b0;
    req_valid_i = '0;
    req_txn_i = '0;
    req_write_i = '0;
    req_seg_count_i = '0;
    req_bank_i = '0;
    req_addr_i = '0;
    req_len_i = '0;
    wdata_valid_i = '0;
    wdata_i = '0;
    rdata_ready_i = '0;
    fin_valid_i = '0;
    fin_txn_i = '0;
    fin_generation_i = '0;
    inv_valid_i = '0;
    inv_bank_i = '0;
    inv_addr_i = '0;
    inv_len_i = '0;
    profile_set_valid_i = 1'b0;
    profile_index_i = '0;
    profile_generation_i = '0;
    cp_valid_i = 1'b0;
    cp_write_i = 1'b0;
    cp_bank_i = '0;
    cp_addr_i = '0;
    cp_beats_i = '0;
    cp_wmode_i = 1'b0;
    cp_wdata_valid_i = 1'b0;
    cp_wdata_i = '0;
    cp_wben_i = '0;
    cp_rdata_ready_i = 1'b0;
    cache_quiescent_i = 1'b1;
    for (int pc = 0; pc < 2; pc++) begin
      s_txn[pc] = '0;
      s_write[pc] = 1'b0;
      s_cnt[pc] = '0;
      for (int s = 0; s < MAX_SEG; s++) begin
        s_bank[pc][s] = '0;
        s_addr[pc][s] = '0;
        s_len[pc][s] = '0;
      end
    end
    repeat (4) @(negedge clk);
    rstn = 1'b1;
    @(negedge clk);

    // ---- S0: reset state — never-written user bytes are invalid, not zeros.
    $display("TB_L1_BANK: S0 uninitialized validity");
    req_stage(P0, 1, 0, TXN_S0);
    seg_set(P0, 0, 2, USER0, 32);
    req_push(P0, w);
    check_bit(w == 0, 1'b1, "S0 idle accept no wait");
    rpop_check(P0, 0, '0, '0, TXN_S0, "S0 unwritten bank2 beat");
    wait_done(P0, 8, "S0");
    fin_push(P0, TXN_S0, 1, 0, "S0 release");

    // ---- S1: indexed fixture — DATA rows, gather (3,0,3), scatter (1,1,2).
    $display("TB_L1_BANK: S1 indexed gather/scatter fixture");
    // DATA: bank0, rows 0..3 = 64 bytes at USER0 (words 64,65).
    req_stage(P0, 1, 1, TXN_DW);
    seg_set(P0, 0, 0, USER0, 64);
    req_push(P0, w);
    wbeat_push(P0, mkword(row128(0), row128(1)));
    wbeat_push(P0, mkword(row128(2), row128(3)));
    wait_done(P0, 8, "S1 DATA write");
    fin_push(P0, TXN_DW, 1, 0, "S1 DATA release");

    // Gather: read DATA rows 3,0,3 (16B each) in ordinal order.
    req_stage(P0, 3, 0, TXN_G);
    seg_set(P0, 0, 0, USER0 + 48, 16);
    seg_set(P0, 1, 0, USER0, 16);
    seg_set(P0, 2, 0, USER0 + 48, 16);
    req_push(P0, w);
    rpop_check(P0, 0, mkword(128'd0, row128(3)), EN_HI16, TXN_G, "S1 gather seg0 row3");
    rpop_check(P0, 0, mkword(row128(0), 128'd0), EN_LO16, TXN_G, "S1 gather seg1 row0");
    rpop_check(P0, 0, mkword(128'd0, row128(3)), EN_HI16, TXN_G, "S1 gather seg2 row3");
    wait_done(P0, 8, "S1 gather");
    fin_push(P0, TXN_G, 1, 0, "S1 gather release");

    // Scatter: OUT bank1 rows 1,1,2 receive gathered segments in ordinal
    // order — ordinal 1 overwrites ordinal 0 on OUT row 1.
    req_stage(P0, 3, 1, TXN_SC);
    seg_set(P0, 0, 1, USER0 + 16, 16);
    seg_set(P0, 1, 1, USER0 + 16, 16);
    seg_set(P0, 2, 1, USER0 + 32, 16);
    req_push(P0, w);
    wbeat_push(P0, mkword(128'd0, row128(3)));  // ordinal 0: row1 <= [40..43]
    wbeat_push(P0, mkword(128'd0, row128(0)));  // ordinal 1: row1 <= [10..13] wins
    wbeat_push(P0, mkword(row128(3), 128'd0));  // ordinal 2: row2 <= [40..43]
    wait_done(P0, 8, "S1 scatter");
    fin_push(P0, TXN_SC, 1, 0, "S1 scatter release");

    // Readback: OUT row0/row3 invalid, row1 == [10..13], row2 == [40..43].
    req_stage(P0, 1, 0, TXN_RB);
    seg_set(P0, 0, 1, USER0, 64);
    req_push(P0, w);
    rpop_check(P0, 0, mkword(128'd0, row128(0)), EN_HI16, TXN_RB, "S1 OUT word64 rows0/1");
    rpop_check(P0, 0, mkword(row128(3), 128'd0), EN_LO16, TXN_RB, "S1 OUT word65 rows2/3");
    wait_done(P0, 8, "S1 readback");
    fin_push(P0, TXN_RB, 1, 0, "S1 readback release");

    // ---- S2: stalled compound holder, conflicting wait, no partial occupancy.
    $display("TB_L1_BANK: S2 compound holder and conflict");
    // A holds banks {2,5}: seg0 = b2 64B (2 beats), seg1 = b5 32B (1 beat).
    req_stage(P0, 2, 1, TXN_A);
    seg_set(P0, 0, 2, USER0, 64);
    seg_set(P0, 1, 5, USER0, 32);
    req_push(P0, w);
    check_bit(w == 0, 1'b1, "S2 A accepted immediately");
    wbeat_push(P0, W_A0);
    wbeat_push(P0, W_A1);
    // A now has one beat left on bank5 and keeps both banks busy.

    if (CONTEXT_COUNT >= 2) begin
      // B (port1): seg0 on free bank7, seg1 on busy bank5 -> must wait, and
      // must not take bank7 partially.
      req_stage(P1, 2, 1, TXN_B);
      seg_set(P1, 0, 7, USER0, 32);
      seg_set(P1, 1, 5, USER0, 32);
      @(negedge clk);
      req_drive(P1);
      #1;
      check_ready(P1, 1'b0, "S2 B waits on busy bank5");
      check_bit(req_rej_o[P1], 1'b0, "S2 B is not rejected (valid, just busy)");
      // Junk write beats from the waiting request must never be accepted.
      wdata_valid_i[P1] = 1'b1;
      wdata_i[P1] = JUNK;
      #1;
      check_bit(wdata_ready_o[P1], 1'b0, "S2 waiting request cannot present beats");
      @(posedge clk);
      #1;
      wdata_valid_i[P1] = 1'b0;
      @(negedge clk);
      #1;
      check_ready(P1, 1'b0, "S2 B still waiting");
      // Withdraw B; bank7 must be immediately usable (no partial occupancy).
      req_undrive(P1);
      req_stage(P1, 1, 1, TXN_C);
      seg_set(P1, 0, 7, USER0 + 32, 32);
      req_push(P1, w);
      check_bit(w == 0, 1'b1, "S2 C takes bank7 with zero wait after B withdrawn");
      wbeat_push(P1, W_C);
      wait_done(P1, 8, "S2 C write");
      fin_push(P1, TXN_C, 1, 0, "S2 C release (disjoint progress while A stalled)");
      wait_obs = wait_obs + 1;
    end else begin
      // C1 adaptation: with one port the conflict is the port itself being
      // busy; the compound request re-issues cleanly after release.
      $display("TB_L1_BANK: S2 C1 adaptation (single port)");
    end

    // Resume A, complete and release.
    wbeat_push(P0, W_A2);
    wait_done(P0, 8, "S2 A write");
    fin_push(P0, TXN_A, 1, 0, "S2 A release");
    if (CONTEXT_COUNT == 1) begin
      req_stage(P0, 1, 1, TXN_C);
      seg_set(P0, 0, 7, USER0 + 32, 32);
      req_push(P0, w);
      wbeat_push(P0, W_C);
      wait_done(P0, 8, "S2 C1 sequential C write");
      fin_push(P0, TXN_C, 1, 0, "S2 C1 C release");
    end

    // Readbacks: A's banks intact, bank7 has only C's word (no JUNK).
    req_stage(P0, 1, 0, TXN_R0);
    seg_set(P0, 0, 2, USER0, 64);
    req_push(P0, w);
    rpop_check(P0, 0, W_A0, EN_FULL, TXN_R0, "S2 b2 word64 intact");
    rpop_check(P0, 0, W_A1, EN_FULL, TXN_R0, "S2 b2 word65 intact");
    wait_done(P0, 8, "S2 b2 readback");
    fin_push(P0, TXN_R0, 1, 0, "S2 b2 release");

    req_stage(P0, 2, 0, TXN_R1);
    seg_set(P0, 0, 5, USER0, 32);
    seg_set(P0, 1, 7, USER0 + 32, 32);
    req_push(P0, w);
    rpop_check(P0, 0, W_A2, EN_FULL, TXN_R1, "S2 b5 word64 intact");
    rpop_check(P0, 0, W_C, EN_FULL, TXN_R1, "S2 b7 word65 == C data");
    wait_done(P0, 8, "S2 b5/b7 readback");
    fin_push(P0, TXN_R1, 1, 0, "S2 b5/b7 release");

    // Bank7 word64 must be entirely invalid: the junk beat never landed.
    req_stage(P0, 1, 0, TXN_FR);
    seg_set(P0, 0, 7, USER0, 32);
    req_push(P0, w);
    rpop_check(P0, 0, '0, '0, TXN_FR, "S2 b7 word64 untouched by waiting junk");
    wait_done(P0, 8, "S2 junk readback");
    fin_push(P0, TXN_FR, 1, 0, "S2 junk readback release");

    if (CONTEXT_COUNT >= 2) begin
      // ---- S3: same-cycle grants, fin identity battery, fin-before-done.
      $display("TB_L1_BANK: S3 dual-port arbitration and fin identity");
      // Same-cycle overlapping grants: port0 wins, port1 waits.
      req_stage(P0, 1, 1, TXN_Q0);
      seg_set(P0, 0, 3, USER0, 32);
      req_stage(P1, 1, 1, TXN_Q1);
      seg_set(P1, 0, 3, USER0, 32);
      @(negedge clk);
      req_drive(P0);
      req_drive(P1);
      #1;
      check_ready(P0, 1'b1, "S3 overlap: port0 granted");
      check_ready(P1, 1'b0, "S3 overlap: port1 not granted same cycle");
      @(posedge clk);
      #1;
      req_undrive(P0);
      // port1 keeps waiting through port0's data phase and release.
      @(negedge clk);
      #1;
      check_ready(P1, 1'b0, "S3 port1 waits while port0 holds bank3");
      wait_obs = wait_obs + 1;
      wbeat_push(P0, W_P0);
      wait_done(P0, 8, "S3 port0 write");
      fin_push(P0, TXN_Q0, 1, 0, "S3 port0 release");
      @(negedge clk);
      #1;
      check_ready(P1, 1'b1, "S3 port1 accepted after release");
      @(posedge clk);
      #1;
      req_undrive(P1);
      wbeat_push(P1, W_P1);
      wait_done(P1, 8, "S3 port1 write");
      fin_push(P1, TXN_Q1, 1, 0, "S3 port1 release");

      // Same-cycle disjoint grants: both accepted at the same edge.
      req_stage(P0, 1, 1, TXN_A2);
      seg_set(P0, 0, 10, USER0, 32);
      req_stage(P1, 1, 1, TXN_B2);
      seg_set(P1, 0, 11, USER0, 32);
      @(negedge clk);
      req_drive(P0);
      req_drive(P1);
      #1;
      check_ready(P0, 1'b1, "S3 disjoint: port0 ready");
      check_ready(P1, 1'b1, "S3 disjoint: port1 ready same cycle");
      @(posedge clk);
      #1;
      req_undrive(P0);
      req_undrive(P1);
      wbeat_push(P0, W_D10);
      wbeat_push(P1, W_D11);
      wait_done(P0, 8, "S3 b10 write");
      wait_done(P1, 8, "S3 b11 write");

      // A2 (port0, done, still holding) and B2 (port1, done, holding):
      // identity-checked release battery. No fin may touch the other owner.
      fin_push(P1, TXN_A2, 0, 1, "S3 wrong-port fin (A2 id on port1)");
      // B2 must be untouched: its bank still busy and releasable only by B2.
      fin_push(P0, TXN_A2 + 64'd7, 0, 1, "S3 stale fin id");
      fin_push(P0, TXN_A2, 1, 0, "S3 correct A2 release");
      fin_push(P0, TXN_A2, 0, 1, "S3 duplicate A2 release");
      fin_push(P1, TXN_B2, 1, 0, "S3 B2 release after foreign fin attempts");

      // fin before data completion: write with no beats yet.
      req_stage(P0, 1, 1, TXN_F14);
      seg_set(P0, 0, 14, USER0, 32);
      req_push(P0, w);
      fin_push(P0, TXN_F14, 0, 2, "S3 fin before write completion");
      wbeat_push(P0, W_F14);
      wait_done(P0, 8, "S3 b14 write");
      fin_push(P0, TXN_F14, 1, 0, "S3 b14 release after completion");
      // fin before data completion: read with undelivered beat.
      req_stage(P1, 1, 0, TXN_WR);
      seg_set(P1, 0, 14, USER0, 32);
      req_push(P1, w);
      fin_push(P1, TXN_WR, 0, 2, "S3 fin before read delivery");
      rpop_check(P1, 0, W_F14, EN_FULL, TXN_WR, "S3 b14 readback");
      wait_done(P1, 8, "S3 b14 read");
      fin_push(P1, TXN_WR, 1, 0, "S3 read release after delivery");
    end else begin
      $display("TB_L1_BANK: S3 skipped for C1 (dual-port only)");
    end

    // ---- S4: read latency + backpressure.
    $display("TB_L1_BANK: S4 latency and backpressure");
    req_stage(P0, 1, 1, TXN_BP);
    seg_set(P0, 0, 4, USER0, 64);
    req_push(P0, w);
    wbeat_push(P0, W_A0);
    wbeat_push(P0, W_A1);
    wait_done(P0, 8, "S4 bank4 write");
    fin_push(P0, TXN_BP, 1, 0, "S4 bank4 release");

    req_stage(P0, 1, 0, TXN_BP);
    seg_set(P0, 0, 4, USER0, 64);
    req_push(P0, w);
    // Sourced SRAM latency: no beat in the accept cycle itself...
    #1;
    check_bit(rdata_valid_o[P0], 1'b0, "S4 no combinational read beat");
    // ...and exactly one BANK_LATENCY_CYCLES stage to the first beat.
    @(posedge clk);
    #1;
    check_bit(rdata_valid_o[P0], 1'b1, "S4 first beat exactly 1 cycle after accept");
    latency_obs = latency_obs + 1;
    rpop_check(P0, 3, W_A0, EN_FULL, TXN_BP, "S4 beat0 with 3-cycle backpressure");
    rpop_check(P0, 0, W_A1, EN_FULL, TXN_BP, "S4 beat1");
    wait_done(P0, 8, "S4 read");
    fin_push(P0, TXN_BP, 1, 0, "S4 read release");

    if (CONTEXT_COUNT >= 2) begin
      // Disjoint port progresses while a read is backpressured.
      req_stage(P0, 1, 1, TXN_15);
      seg_set(P0, 0, 15, USER0, 32);
      req_push(P0, w);
      wbeat_push(P0, W_15);
      wait_done(P0, 8, "S4 b15 write");
      fin_push(P0, TXN_15, 1, 0, "S4 b15 release");
      req_stage(P0, 1, 0, TXN_15);
      seg_set(P0, 0, 15, USER0, 32);
      req_push(P0, w);
      @(posedge clk);
      #1;
      check_bit(rdata_valid_o[P0], 1'b1, "S4 b15 beat ready");
      // Port0 read now stalls (ready low); port1 completes a full txn.
      req_stage(P1, 1, 1, TXN_G7);
      seg_set(P1, 0, 2, USER0 + 32, 32);
      req_push(P1, w);
      wbeat_push(P1, W_G7);
      wait_done(P1, 8, "S4 b2 word65 write while port0 backpressured");
      fin_push(P1, TXN_G7, 1, 0, "S4 release while port0 backpressured");
      rpop_check(P0, 0, W_15, EN_FULL, TXN_15, "S4 b15 readback after stall");
      wait_done(P0, 8, "S4 b15 read");
      fin_push(P0, TXN_15, 1, 0, "S4 b15 read release");
    end

    // ---- S5: fault battery (profile 0: user SPM [2048, 65536)).
    $display("TB_L1_BANK: S5 reserved/overflow/shape faults");
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, 0, 4);
    req_expect_reject(P0, 3'd4, "S5 reserved byte 0");
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, 2044, 8);
    req_expect_reject(P0, 3'd4, "S5 reserved crossing at 2048");
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, USER0, 0);
    req_expect_reject(P0, 3'd2, "S5 zero length");
    req_stage(P0, 0, 1, TXN_BND);
    seg_set(P0, 0, 0, USER0, 32);
    req_expect_reject(P0, 3'd1, "S5 zero segment count");
    req_stage(P0, MAX_SEG + 1, 1, TXN_BND);
    seg_set(P0, 0, 0, USER0, 32);
    req_expect_reject(P0, 3'd1, "S5 segment count above MAX_SEG");
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, BANK_BYTES - 4, 8);
    req_expect_reject(P0, 3'd3, "S5 overflow past bank bytes");
    req_stage(P0, 2, 1, TXN_BND);
    seg_set(P0, 0, 0, USER0, 32);
    seg_set(P0, 1, 1, BANK_BYTES - 2, 4);
    req_expect_reject(P0, 3'd3, "S5 overflow on second segment, all-or-none");

    if (CONTEXT_COUNT >= 2) begin
      // Rejects are side-effect free even while another port holds banks.
      req_stage(P1, 1, 1, TXN_H8);
      seg_set(P1, 0, 8, USER0, 32);
      req_push(P1, w);
      req_stage(P1, 1, 1, TXN_BND);
      seg_set(P1, 0, 0, 0, 4);
      req_expect_reject(P1, 3'd4, "S5 concurrent reserved reject");
      wbeat_push(P1, W_A0);
      wait_done(P1, 8, "S5 b8 write");
      fin_push(P1, TXN_H8, 1, 0, "S5 b8 release");
    end

    // Rejections leave the module usable: last user byte of the bank works.
    req_stage(P0, 1, 1, TXN_LST);
    seg_set(P0, 0, 0, BANK_BYTES - 1, 1);
    req_push(P0, w);
    check_bit(w == 0, 1'b1, "S5 top-of-bank byte accepted after rejects");
    prior_holder_generation = accepted_generation[P0];
    wbeat_push(P0, {8'hAB, 248'd0});
    wait_done(P0, 8, "S5 top byte write");
    fin_push(P0, TXN_LST, 1, 0, "S5 top byte release");
    req_stage(P0, 1, 0, TXN_LST);
    seg_set(P0, 0, 0, BANK_BYTES - 1, 1);
    req_push(P0, w);
    rpop_check(P0, 0, {8'hAB, 248'd0}, EN_B31, TXN_LST, "S5 top byte readback");
    wait_done(P0, 8, "S5 top byte read");
    // The numeric caller txn is reused, but an old fin cannot release this
    // new incarnation even after the new data phase has completed.
    @(negedge clk);
    fin_valid_i[P0] = 1'b1;
    fin_txn_i[P0] = TXN_LST;
    fin_generation_i[P0] = prior_holder_generation;
    @(posedge clk); #1;
    fin_valid_i[P0] = 1'b0;
    check_bit(fin_rej_o[P0], 1'b1, "S5 reused-txn old incarnation rejected");
    check_bit(fin_ack_o[P0], 1'b0, "S5 old fin did not release new holder");
    check_bit(req_done_o[P0], 1'b1, "S5 new holder remains retained");
    fin_push(P0, TXN_LST, 1, 0, "S5 top byte read release");

    // ---- S6: profile transitions.
    $display("TB_L1_BANK: S6 profile transition gate");
    // Write across the future profile-3 SPM boundary (32768):
    // word1023 bytes 24..31 = PL, word1024 bytes 0..7 = PH.
    req_stage(P0, 1, 1, TXN_X4);
    seg_set(P0, 0, 4, 32760, 16);
    req_push(P0, w);
    wbeat_push(P0, {PL, 192'd0});
    wbeat_push(P0, {192'd0, PH});
    wait_done(P0, 8, "S6 boundary-straddling write");
    fin_push(P0, TXN_X4, 1, 0, "S6 boundary-straddling release");

    // A busy txn blocks the profile switch and leaves the boundary unchanged.
    req_stage(P0, 1, 1, TXN_H8);
    seg_set(P0, 0, 8, USER0, 32);
    req_push(P0, w);
    profile_push(PROFILE_IDX_W'(3), 64'd1, 0, 2'd2, "S6 switch blocked while busy");
    // C2 probes on the peer port; C1 must first finish its only active port.
    if (CONTEXT_COUNT == 1) begin
      wbeat_push(P0, W_A1);
      wait_done(P0, 8, "S6 C1 b8 write");
      fin_push(P0, TXN_H8, 1, 0, "S6 C1 b8 release");
    end
    probe_port = (CONTEXT_COUNT >= 2) ? P1 : P0;
    req_stage(probe_port, 1, 1, TXN_P9);
    seg_set(probe_port, 0, 9, 33000, 4);
    req_push(probe_port, w);
    check_bit(w == 0, 1'b1, "S6 old boundary remains active after rejected switch");
    wbeat_push(probe_port, {128'd0, W_B9[63:0], 64'd0});
    wait_done(probe_port, 8, "S6 b9 probe write");
    fin_push(probe_port, TXN_P9, 1, 0, "S6 b9 probe release");
    if (CONTEXT_COUNT >= 2) begin
      wbeat_push(P0, W_A1);
      wait_done(P0, 8, "S6 b8 write");
      fin_push(P0, TXN_H8, 1, 0, "S6 b8 release");
    end

    // The authorized profile edge cannot also admit old-profile data/clear.
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 9, 50000, 4);
    @(negedge clk);
    req_drive(P0);
    inv_valid_i[P0] = 1'b1;
    inv_bank_i[P0] = BANK_IDX_W'(9);
    inv_addr_i[P0] = BANK_ADDR_W'(50000);
    inv_len_i[P0] = BANK_LEN_W'(4);
    profile_set_valid_i = 1'b1;
    profile_index_i = PROFILE_IDX_W'(3);
    profile_generation_i = 64'd1;
    #1;
    check_ready(P0, 1'b0, "S6 profile edge excludes new data");
    check_bit(inv_ready_o[P0], 1'b0, "S6 profile edge excludes invalidation");
    @(posedge clk); #1;
    profile_set_valid_i = 1'b0;
    req_undrive(P0);
    inv_valid_i[P0] = 1'b0;
    check_bit(profile_ack_o, 1'b1, "S6 quiescent profile commit succeeded");
    check_bit(wdata_ready_o[P0], 1'b0, "S6 no old-profile transaction was admitted");

    // Post-switch geometry: user SPM is [2048, 32768).
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, 33000, 4);
    req_expect_reject(P0, 3'd5, "S6 cache partition fault");
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, 32767, 2);
    req_expect_reject(P0, 3'd5, "S6 crossing new SPM bound");
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, 2044, 8);
    req_expect_reject(P0, 3'd4, "S6 reserved bound unchanged");
    // Valid writes still work at/under the new bound.
    req_stage(P0, 1, 1, TXN_X10);
    seg_set(P0, 0, 10, 32764, 4);
    req_push(P0, w);
    wbeat_push(P0, {PX, 224'd0});
    wait_done(P0, 8, "S6 b10 under-bound write");
    fin_push(P0, TXN_X10, 1, 0, "S6 b10 release");
    // Validity in the still-user region is retained across the switch.
    req_stage(P0, 1, 0, TXN_R4);
    seg_set(P0, 0, 4, 32736, 32);
    req_push(P0, w);
    rpop_check(P0, 0, {PL, 192'd0}, EN_B2431, TXN_R4, "S6 b4 word1023 retained");
    wait_done(P0, 8, "S6 b4 readback");
    fin_push(P0, TXN_R4, 1, 0, "S6 b4 readback release");
    req_stage(P0, 1, 0, TXN_X10);
    seg_set(P0, 0, 10, 32736, 32);
    req_push(P0, w);
    rpop_check(P0, 0, {PX, 224'd0}, EN_B2831, TXN_X10, "S6 b10 word1023 readback");
    wait_done(P0, 8, "S6 b10 readback");
    fin_push(P0, TXN_X10, 1, 0, "S6 b10 readback release");

    // Generation must strictly increase; equal/lower is rejected.
    profile_push(PROFILE_IDX_W'(1), 64'd1, 0, 2'd1, "S6 same generation rejected");
    profile_push(PROFILE_IDX_W'(1), 64'd0, 0, 2'd1, "S6 lower generation rejected");
    profile_push(PROFILE_IDX_W'(1), 64'd6, 1, 0, "S6 switch to profile 1");
    // Profile 1: user SPM [2048, 57344).
    req_stage(P0, 1, 1, TXN_BND);
    seg_set(P0, 0, 0, 57344, 4);
    req_expect_reject(P0, 3'd5, "S6 profile-1 cache fault");
    req_stage(P0, 1, 1, TXN_Y11);
    seg_set(P0, 0, 11, 57340, 4);
    req_push(P0, w);
    wbeat_push(P0, {PY, 224'd0});
    wait_done(P0, 8, "S6 b11 top write");
    fin_push(P0, TXN_Y11, 1, 0, "S6 b11 release");

    // ---- S7: gated data issue does not block completion/release elsewhere.
    $display("TB_L1_BANK: S7 gated data issue");
    req_stage(P0, 1, 1, TXN_G7);
    seg_set(P0, 0, 2, USER0 + 32, 32);
    req_push(P0, w);
    // Gate: no write beats presented for several cycles; done must not fire.
    repeat (3) begin
      @(negedge clk);
      #1;
      check_bit(req_done_o[P0], 1'b0, "S7 gated write not done");
      check_bit(wdata_ready_o[P0], 1'b1, "S7 module still ready for beats");
    end
    if (CONTEXT_COUNT >= 2) begin
      // port1 completes and releases while port0's data issue is gated.
      req_stage(P1, 1, 0, TXN_WR);
      seg_set(P1, 0, 3, USER0, 32);
      req_push(P1, w);
      rpop_check(P1, 0, W_P1, EN_FULL, TXN_WR, "S7 b3 read during gate");
      wait_done(P1, 8, "S7 b3 read");
      fin_push(P1, TXN_WR, 1, 0, "S7 release during gate");
    end else begin
      // C1: the control plane stays alive while data issue is gated.
      profile_push(PROFILE_IDX_W'(2), 64'd7, 0, 2'd2, "S7 C1 profile gate check");
    end
    wbeat_push(P0, W_G7);
    wait_done(P0, 8, "S7 gated write completes after resume");
    fin_push(P0, TXN_G7, 1, 0, "S7 gated write release");

    // ---- S8: checked exact-range invalidation (agreed lifecycle hook).
    $display("TB_L1_BANK: S8 range invalidation and reuse");
    // bank13 words 64+65 written; the exact-range clear must not round.
    req_stage(P0, 1, 1, TXN_8A);
    seg_set(P0, 0, 13, USER0, 64);
    req_push(P0, w);
    wbeat_push(P0, W_VA);
    wbeat_push(P0, W_VB);
    wait_done(P0, 8, "S8 bank13 write");
    fin_push(P0, TXN_8A, 1, 0, "S8 bank13 release");

    // Invalidate exactly the low 16 bytes of word64: byte-exact, neighbours
    // in the same word and the next word keep their validity.
    inv_push(P0, 13, USER0, 16, 1, 0, "S8 partial-word invalidation");
    req_stage(P0, 1, 0, TXN_8B);
    seg_set(P0, 0, 13, USER0, 64);
    req_push(P0, w);
    rpop_check(P0, 0, mkword(128'd0, W_VA[255:128]), EN_HI16, TXN_8B,
               "S8 word64 high half still valid");
    rpop_check(P0, 0, W_VB, EN_FULL, TXN_8B, "S8 word65 untouched");
    wait_done(P0, 8, "S8 partial readback");
    fin_push(P0, TXN_8B, 1, 0, "S8 partial readback release");

    // Invalidate word65 entirely; word64's high half is outside the range.
    inv_push(P0, 13, USER0 + 32, 32, 1, 0, "S8 word65 invalidation");
    req_stage(P0, 1, 0, TXN_8B);
    seg_set(P0, 0, 13, USER0, 64);
    req_push(P0, w);
    rpop_check(P0, 0, mkword(128'd0, W_VA[255:128]), EN_HI16, TXN_8B,
               "S8 word64 high half survives word65 clear");
    rpop_check(P0, 0, '0, '0, TXN_8B, "S8 word65 fully invalid");
    wait_done(P0, 8, "S8 word65 readback");
    fin_push(P0, TXN_8B, 1, 0, "S8 word65 readback release");

    // Full-range clear, then idempotent repeat clear (still acks).
    inv_push(P0, 13, USER0, 64, 1, 0, "S8 full-range invalidation");
    inv_push(P0, 13, USER0, 64, 1, 0, "S8 idempotent repeat invalidation");
    req_stage(P0, 1, 0, TXN_8B);
    seg_set(P0, 0, 13, USER0, 64);
    req_push(P0, w);
    rpop_check(P0, 0, '0, '0, TXN_8B, "S8 word64 fully invalid");
    rpop_check(P0, 0, '0, '0, TXN_8B, "S8 word65 fully invalid (repeat)");
    wait_done(P0, 8, "S8 full-invalid readback");
    fin_push(P0, TXN_8B, 1, 0, "S8 full-invalid release");

    // Reuse safety: after the clear, rewriting ONLY the low half makes only
    // those bytes valid again — retained physical bytes are not fabricated
    // back into validity (validity and bytes are separate).
    req_stage(P0, 1, 1, TXN_8C);
    seg_set(P0, 0, 13, USER0, 16);
    req_push(P0, w);
    wbeat_push(P0, mkword(W_VB[127:0], 128'd0));
    wait_done(P0, 8, "S8 reuse low-half write");
    fin_push(P0, TXN_8C, 1, 0, "S8 reuse low-half release");
    req_stage(P0, 1, 0, TXN_8D);
    seg_set(P0, 0, 13, USER0, 64);
    req_push(P0, w);
    rpop_check(P0, 0, mkword(W_VB[127:0], 128'd0), EN_LO16, TXN_8D,
               "S8 only rewritten bytes valid again");
    rpop_check(P0, 0, '0, '0, TXN_8D, "S8 word65 stays invalid after reuse");
    wait_done(P0, 8, "S8 reuse readback");
    fin_push(P0, TXN_8D, 1, 0, "S8 reuse readback release");

    // Invalidation range faults share the request code space (profile 1:
    // user SPM ends at 57344, so 60000 is the cache partition).
    inv_push(P0, 13, USER0, 0, 0, 3'd2, "S8 inv zero length");
    inv_push(P0, 13, 0, 8, 0, 3'd4, "S8 inv reserved range");
    inv_push(P0, 13, BANK_BYTES - 2, 4, 0, 3'd3, "S8 inv overflow");
    inv_push(P0, 13, 60000, 4, 0, 3'd5, "S8 inv cache partition");
    // The fault battery cleared nothing: previous validity unchanged.
    req_stage(P0, 1, 0, TXN_8D);
    seg_set(P0, 0, 13, USER0, 16);
    req_push(P0, w);
    rpop_check(P0, 0, mkword(W_VB[127:0], 128'd0), EN_LO16, TXN_8D,
               "S8 rejected invs cleared nothing");
    wait_done(P0, 8, "S8 post-fault readback");
    fin_push(P0, TXN_8D, 1, 0, "S8 post-fault release");

    // ---- S9: integration regressions for arbitration and bank ownership.
    req_stage(P0, 1, 1, 64'h9000);
    seg_set(P0, 0, 14, USER0 + 96, DATA_BYTES);
    req_push(P0, w);
    wbeat_push(P0, W_A0);
    wait_done(P0, 8, "S9 seed");
    fin_push(P0, 64'h9000, 1, 0, "S9 seed release");
    if (CONTEXT_COUNT >= 2) begin
      // Port0 retains a legal payload with valid=0. It owns nothing.
      req_stage(P1, 1, 0, 64'h9001);
      seg_set(P1, 0, 14, USER0 + 96, DATA_BYTES);
      @(negedge clk); req_drive(P1); #1;
      check_ready(P1, 1'b1, "S9 invalid lower port cannot reserve a bank");
      @(posedge clk); #1; req_undrive(P1);
      rpop_check(P1, 0, W_A0, EN_FULL, 64'h9001, "S9 peer reads without phantom wait");
      wait_done(P1, 8, "S9 peer read");
      fin_push(P1, 64'h9001, 1, 0, "S9 peer release");
    end

    // All sourced banks fit in one compound reservation, not just four.
    req_stage(P0, BANK_COUNT, 1, 64'h9010);
    for (int b = 0; b < BANK_COUNT; b++)
      seg_set(P0, b, b, USER0 + 256, DATA_BYTES);
    req_push(P0, w);
    if (CONTEXT_COUNT >= 2) begin
      req_stage(P1, 1, 0, 64'h9011);
      seg_set(P1, 0, BANK_COUNT - 1, USER0 + 256, DATA_BYTES);
      @(negedge clk); req_drive(P1); #1;
      check_ready(P1, 1'b0, "S9 compound request holds its last bank immediately");
      @(posedge clk); #1; req_undrive(P1);
    end
    for (int b = 0; b < BANK_COUNT; b++) wbeat_push(P0, W_A0 ^ DW'(b));
    wait_done(P0, BANK_COUNT + 4, "S9 all-bank write");
    fin_push(P0, 64'h9010, 1, 0, "S9 all-bank write release");
    req_stage(P0, BANK_COUNT, 0, 64'h9012);
    for (int b = 0; b < BANK_COUNT; b++)
      seg_set(P0, b, b, USER0 + 256, DATA_BYTES);
    req_push(P0, w);
    for (int b = 0; b < BANK_COUNT; b++)
      rpop_check(P0, 0, W_A0 ^ DW'(b), EN_FULL, 64'h9012, "S9 all-bank byte readback");
    wait_done(P0, BANK_COUNT + 4, "S9 all-bank read");
    fin_push(P0, 64'h9012, 1, 0, "S9 all-bank read release");

    // A backpressured read retains its bank; invalidation cannot bypass it.
    req_stage(P0, 1, 0, 64'h9020);
    seg_set(P0, 0, 14, USER0 + 96, DATA_BYTES);
    req_push(P0, w);
    probe_port = (CONTEXT_COUNT >= 2) ? P1 : P0;
    @(negedge clk);
    inv_valid_i[probe_port] = 1'b1;
    inv_bank_i[probe_port] = BANK_IDX_W'(14);
    inv_addr_i[probe_port] = BANK_ADDR_W'(USER0 + 96);
    inv_len_i[probe_port] = BANK_LEN_W'(16);
    repeat (3) begin
      #1;
      check_bit(inv_ready_o[probe_port], 1'b0, "S9 invalidation waits for holder");
      check_bit(inv_ack_o[probe_port], 1'b0, "S9 no premature clear ACK");
      @(negedge clk);
    end
    rpop_check(P0, 0, W_A0, EN_FULL, 64'h9020, "S9 retained read survives pending clear");
    wait_done(P0, 8, "S9 holder read");
    fin_push(P0, 64'h9020, 1, 0, "S9 fin progresses while clear waits");
    inv_push(probe_port, 14, USER0 + 96, 16, 1, 0, "S9 clear after holder release");
    req_stage(P0, 1, 0, 64'h9021);
    seg_set(P0, 0, 14, USER0 + 96, DATA_BYTES);
    req_push(P0, w);
    rpop_check(P0, 0, W_A0, EN_HI16, 64'h9021, "S9 exact clear preserves neighbors");
    wait_done(P0, 8, "S9 clear readback");
    fin_push(P0, 64'h9021, 1, 0, "S9 clear readback release");

    // Same-edge clear wins over a new data request; a profile commit must
    // also wait for the real clear walker, not just the data-holder table.
    req_stage(P0, 1, 0, 64'h9030);
    seg_set(P0, 0, 14, USER0 + 96, DATA_BYTES);
    @(negedge clk);
    req_drive(P0);
    inv_valid_i[P0] = 1'b1;
    inv_bank_i[P0] = BANK_IDX_W'(14);
    inv_addr_i[P0] = BANK_ADDR_W'(USER0 + 112);
    inv_len_i[P0] = BANK_LEN_W'(16);
    #1;
    check_bit(inv_ready_o[P0], 1'b1, "S9 free bank accepts clear");
    check_ready(P0, 1'b0, "S9 clear and data cannot acquire the same bank");
    @(posedge clk); #1; inv_valid_i[P0] = 1'b0;
    profile_push(PROFILE_IDX_W'(0), 64'd7, 0, 2'd2, "S9 in-progress clear blocks profile");
    check_bit(inv_ack_o[P0], 1'b1, "S9 one-word clear completed");
    // The held data request has not fired on the profile-command edge.
    @(posedge clk); #1; req_undrive(P0);
    rpop_check(P0, 0, '0, '0, 64'h9030, "S9 read starts only after full invalidation");
    wait_done(P0, 8, "S9 post-clear read");
    fin_push(P0, 64'h9030, 1, 0, "S9 post-clear release");

    // ================= C030 cache data port scenarios (profile 1 after S6:
    // cache window [57344, 65536) per bank) =================

    // ---- CP1: real bytes — write-mode byte/validity semantics.
    $display("TB_L1_BANK: CP1 cp write-mode byte semantics");
    @(negedge clk);
    cp_drive(1, 6, 57344, 2, 0);
    cp_push(cw);
    check_bit(cw == 0, 1'b1, "CP1 cp accepted on idle cache bank");
    check_bit(cp_busy_o, 1'b1, "CP1 cp busy from accept edge");
    cp_wbeat(W_VA, EN_LO16);
    check_bit(cp_done_o, 1'b0, "CP1 no done before final beat");
    cp_wbeat(W_VB, EN_B2431);
    cp_wait_done(4, "CP1 refill write");
    @(negedge clk);
    cp_drive(0, 6, 57344, 2, 0);
    cp_push(cw);
    check_bit(cp_wdata_ready_o, 1'b0, "CP1 no wdata_ready during cp read");
    cp_rpop_check(0, W_VA, EN_LO16, "CP1 refill beat0 validity == wben");
    cp_rpop_check(0, W_VB, EN_B2431, "CP1 refill beat1 validity == wben");
    cp_wait_done(4, "CP1 refill readback");
    // Same slot rewritten with FEWER valid bytes (wmode 0): stale validity
    // must clear.
    @(negedge clk);
    cp_drive(1, 6, 57344, 1, 0);
    cp_push(cw);
    cp_wbeat(W_A0, EN_B07);
    cp_wait_done(4, "CP1 narrower refill");
    @(negedge clk);
    cp_drive(0, 6, 57344, 1, 0);
    cp_push(cw);
    cp_rpop_check(0, W_A0, EN_B07, "CP1 wmode0 clears stale validity");
    cp_wait_done(4, "CP1 narrower readback");
    // wmode 1 (data-only): only wben bytes change, validity untouched.
    @(negedge clk);
    cp_drive(1, 6, 57344, 1, 1);
    cp_push(cw);
    cp_wbeat(W_A1, EN_B47);
    cp_wait_done(4, "CP1 data-only write");
    @(negedge clk);
    cp_drive(0, 6, 57344, 1, 0);
    cp_push(cw);
    cp_rpop_check(0, W_CP1F, EN_B07, "CP1 wmode1 keeps validity, updates bytes");
    cp_wait_done(4, "CP1 data-only readback");

    // ---- CP2: window bounds and fault codes.
    $display("TB_L1_BANK: CP2 cp window bounds");
    req_stage(P0, 1, 1, TXN_CP2);
    seg_set(P0, 0, 6, 4096, 32);
    req_push(P0, w);
    wbeat_push(P0, W_P0);
    wait_done(P0, 8, "CP2 spm user write");
    fin_push(P0, TXN_CP2, 1, 0, "CP2 spm user release");
    // Below the window: reserved, user SPM, one word below the bound -> 7.
    @(negedge clk);
    cp_drive(1, 6, SYSTEM_RESERVED_BYTES, 1, 0);
    cp_expect_reject(3'd7, "CP2 reserved below window");
    @(negedge clk);
    cp_drive(1, 6, 4096, 1, 0);
    cp_expect_reject(3'd7, "CP2 user spm below window");
    @(negedge clk);
    cp_drive(1, 6, 57344 - DATA_BYTES, 1, 0);
    cp_expect_reject(3'd7, "CP2 word below bound");
    // One byte below the bound is misaligned: SHAPE outranks the window.
    @(negedge clk);
    cp_drive(0, 6, 57344 - 1, 1, 0);
    cp_expect_reject(3'd1, "CP2 byte below bound is misaligned");
    // SHAPE battery: zero beats, misaligned window word. (A bad bank index
    // is not encodable: BANK_IDX_W covers exactly 0..BANK_COUNT-1.)
    @(negedge clk);
    cp_drive(1, 6, 57344, 0, 0);
    cp_expect_reject(3'd1, "CP2 zero beats");
    @(negedge clk);
    cp_drive(1, 6, 57344 + 1, 1, 0);
    cp_expect_reject(3'd1, "CP2 misaligned window word");
    // OVERFLOW: last beat past the bank end.
    @(negedge clk);
    cp_drive(1, 6, 65536 - DATA_BYTES, 2, 0);
    cp_expect_reject(3'd3, "CP2 overflow past bank bytes");
    @(negedge clk);
    cp_drive(0, 6, 60000, 2048, 0);
    cp_expect_reject(3'd3, "CP2 huge beat count overflow");
    // Exactly the first cache address and the last beats of the bank work.
    @(negedge clk);
    cp_drive(1, 6, 65536 - 2*DATA_BYTES, 2, 0);
    cp_push(cw);
    check_bit(cw == 0, 1'b1, "CP2 accept immediate after rejects (no state)");
    cp_wbeat(W_A2B, EN_FULL);
    cp_wbeat(W_B2B, EN_FULL);
    cp_wait_done(4, "CP2 bank-top write");
    @(negedge clk);
    cp_drive(0, 6, 65536 - 2*DATA_BYTES, 2, 0);
    cp_push(cw);
    cp_rpop_check(0, W_A2B, EN_FULL, "CP2 bank-top beat0");
    cp_rpop_check(0, W_B2B, EN_FULL, "CP2 bank-top final beat");
    cp_wait_done(4, "CP2 bank-top readback");
    // SPM into the cache window is STILL rejected (bit-identical C029 rule).
    req_stage(P0, 1, 1, TXN_CP2);
    seg_set(P0, 0, 6, 57344, 32);
    req_expect_reject(P0, 3'd5, "CP2 spm into cache window still REJ_CACHE");
    req_stage(P0, 1, 1, TXN_CP2);
    seg_set(P0, 0, 6, 57344 - DATA_BYTES, 33);
    req_expect_reject(P0, 3'd5, "CP2 spm crossing bound still REJ_CACHE");
    // In-window traffic and SPM user bytes never disturb each other.
    @(negedge clk);
    cp_drive(1, 6, 57440, 1, 0);
    cp_push(cw);
    cp_wbeat(W_B9, EN_FULL);
    cp_wait_done(4, "CP2 in-window write");
    req_stage(P0, 1, 0, TXN_CP2);
    seg_set(P0, 0, 6, 4096, 32);
    req_push(P0, w);
    rpop_check(P0, 0, W_P0, EN_FULL, TXN_CP2, "CP2 spm user intact after cp traffic");
    wait_done(P0, 8, "CP2 spm readback");
    fin_push(P0, TXN_CP2, 1, 0, "CP2 spm readback release");
    @(negedge clk);
    cp_drive(0, 6, 57440, 1, 0);
    cp_push(cw);
    cp_rpop_check(0, W_B9, EN_FULL, "CP2 cp window intact after spm traffic");
    cp_wait_done(4, "CP2 cp readback");

    // ---- CP3: arbitration against SPM holders and invalidations.
    $display("TB_L1_BANK: CP3 cp/SPM arbitration");
    // (a) stalled SPM holder on bank 3: cp waits; a cp on a free bank
    // proceeds while the holder stalls; after the fin the cp is accepted.
    req_stage(P0, 1, 1, TXN_CP3A);
    seg_set(P0, 0, 3, USER0, 64);
    req_push(P0, w);
    wbeat_push(P0, W_A0);
    @(negedge clk);
    cp_drive(0, 3, 57344, 1, 0);
    #1;
    check_bit(cp_ready_o, 1'b0, "CP3a cp waits behind stalled SPM holder");
    cp_valid_i = 1'b0;  // single-outstanding port: withdraw first
    @(negedge clk);
    cp_drive(0, 10, 57344, 1, 0);
    cp_push(cw);
    check_bit(cw == 0, 1'b1, "CP3a cp on free bank proceeds while SPM stalls");
    cp_rpop_check(0, '0, '0, "CP3a unwritten window reads validity 0");
    cp_wait_done(4, "CP3a other-bank cp read");
    @(negedge clk);
    cp_drive(0, 3, 57344, 1, 0);
    wbeat_push(P0, W_A1);
    wait_done(P0, 8, "CP3a spm resume");
    fin_push(P0, TXN_CP3A, 1, 0, "CP3a spm release");
    cp_push(cw);
    check_bit(cw == 0, 1'b1, "CP3a cp accepted the cycle after SPM fin");
    cp_rpop_check(0, '0, '0, "CP3a bank3 window beat");
    cp_wait_done(4, "CP3a bank3 cp read");

    // (b) cp holds bank 3: SPM naming bank 3 waits (compound requests leave
    // the other bank free), invalidation naming bank 3 waits, other banks
    // proceed.
    @(negedge clk);
    cp_drive(1, 3, 57344, 2, 0);
    cp_push(cw);
    cp_wbeat(W_A0, EN_FULL);
    check_bit(cp_busy_o, 1'b1, "CP3b cp busy while holding");
    req_stage(P0, 1, 1, TXN_CP3B);
    seg_set(P0, 0, 3, USER0, 32);
    @(negedge clk);
    req_drive(P0);
    #1;
    check_ready(P0, 1'b0, "CP3b SPM request waits on cp-held bank");
    req_stage(P0, 2, 1, TXN_CP3B2);
    seg_set(P0, 0, 3, USER0, 32);
    seg_set(P0, 1, 10, USER0, 32);
    @(negedge clk);
    req_drive(P0);
    #1;
    check_ready(P0, 1'b0, "CP3b compound request waits, no partial occupancy");
    req_undrive(P0);
    @(negedge clk);
    inv_valid_i[P0] = 1'b1;
    inv_bank_i[P0] = BANK_IDX_W'(3);
    inv_addr_i[P0] = BANK_ADDR_W'(USER0);
    inv_len_i[P0] = BANK_LEN_W'(32);
    #1;
    check_bit(inv_ready_o[P0], 1'b0, "CP3b invalidation waits on cp-held bank");
    @(posedge clk);
    #1;
    inv_valid_i[P0] = 1'b0;
    if (CONTEXT_COUNT >= 2) begin
      req_stage(P1, 1, 1, TXN_CP3E);
      seg_set(P1, 0, 10, USER0, 32);
      req_push(P1, w);
      wbeat_push(P1, W_C);
      wait_done(P1, 8, "CP3b other-bank write while cp holds");
      fin_push(P1, TXN_CP3E, 1, 0, "CP3b other-bank release");
    end else begin
      req_stage(P0, 1, 1, TXN_CP3E);
      seg_set(P0, 0, 10, USER0, 32);
      req_push(P0, w);
      check_bit(w == 0, 1'b1, "CP3b other bank free for SPM while cp holds");
      wbeat_push(P0, W_C);
      wait_done(P0, 8, "CP3b other-bank write while cp holds");
      fin_push(P0, TXN_CP3E, 1, 0, "CP3b other-bank release");
    end
    cp_wbeat(W_A1, EN_FULL);
    cp_wait_done(4, "CP3b cp write done");
    req_stage(P0, 1, 1, TXN_CP3B);
    seg_set(P0, 0, 3, USER0, 32);
    req_push(P0, w);
    check_bit(w <= 1, 1'b1, "CP3b SPM accepted right after cp done");
    wbeat_push(P0, W_P0);
    wait_done(P0, 8, "CP3b post-cp SPM write");
    fin_push(P0, TXN_CP3B, 1, 0, "CP3b post-cp SPM release");

    // (c) same-cycle SPM accept vs cp request on the same free bank: SPM
    // wins, the cache request completes afterwards.
    req_stage(P0, 1, 1, TXN_CP3C);
    seg_set(P0, 0, 3, USER0 + 32, 32);
    @(negedge clk);
    req_drive(P0);
    cp_drive(1, 3, 57344 + 32, 1, 0);
    #1;
    check_ready(P0, 1'b1, "CP3c same-cycle SPM wins the free bank");
    check_bit(cp_ready_o, 1'b0, "CP3c cp waits one more cycle");
    @(posedge clk);
    #1;
    req_undrive(P0);
    wbeat_push(P0, W_D10);
    wait_done(P0, 8, "CP3c spm write");
    fin_push(P0, TXN_CP3C, 1, 0, "CP3c spm release");
    cp_push(cw);
    check_bit(cw == 0, 1'b1, "CP3c cp accepted after the SPM txn");
    cp_wbeat(W_B9, EN_FULL);
    cp_wait_done(4, "CP3c cp write");

    // (d) a port-0 fin with numeric txn 0 on ANOTHER bank must not free a
    // bank held by the cp (the hold lives outside busy[]/holder_*).
    @(negedge clk);
    cp_drive(1, 3, 57344 + 64, 2, 0);
    cp_push(cw);
    cp_wbeat(W_A2B, EN_FULL);
    req_stage(P0, 1, 1, 64'd0);
    seg_set(P0, 0, 10, USER0 + 64, 32);
    req_push(P0, w);
    wbeat_push(P0, W_D11);
    wait_done(P0, 8, "CP3d txn0 write");
    fin_push(P0, 64'd0, 1, 0, "CP3d txn0 fin on another bank");
    req_stage(P0, 1, 1, TXN_CP3D);
    seg_set(P0, 0, 3, USER0, 32);
    @(negedge clk);
    req_drive(P0);
    #1;
    check_ready(P0, 1'b0, "CP3d port0/txn0 fin does not free the cp bank");
    req_undrive(P0);
    cp_wbeat(W_B2B, EN_FULL);
    cp_wait_done(4, "CP3d cp write done");
    req_push(P0, w);
    wbeat_push(P0, W_P1);
    wait_done(P0, 8, "CP3d post-cp write");
    fin_push(P0, TXN_CP3D, 1, 0, "CP3d post-cp release");

    // ---- CP4: anti-starvation under sustained back-to-back SPM pressure.
    $display("TB_L1_BANK: CP4 cp fairness bound");
    // Measure one quick SPM txn (accept -> busy clear) for the bound term.
    req_stage(P0, 1, 1, TXN_CP4M);
    seg_set(P0, 0, 5, USER0, DATA_BYTES);
    req_push(P0, w);
    t_a = cyc;
    wbeat_push(P0, W_A0);
    wait_done(P0, 8, "CP4 measure write");
    fin_push(P0, TXN_CP4M, 1, 0, "CP4 measure release");
    t_b = cyc;
    drain_q = t_b - t_a;
    // Sustained pressure: re-issue the instant the bank frees.
    @(negedge clk);
    cp_drive(1, 5, 57344, 1, 0);
    req_stage(P0, 1, 1, TXN_CP4);
    seg_set(P0, 0, 5, USER0, DATA_BYTES);
    req_drive(P0);
    #1;
    check_ready(P0, 1'b1, "CP4 same-cycle free bank goes to SPM");
    check_bit(cp_ready_o, 1'b0, "CP4 cp waits when SPM claims the bank");
    @(posedge clk);
    #1;
    req_undrive(P0);
    t_cp0 = cyc;
    wbeat_push(P0, W_A1);
    wait_done(P0, 8, "CP4 spm txn");
    fin_push(P0, TXN_CP4, 1, 0, "CP4 spm release");
    cp_got = 1'b0;
    for (int i = 1; i <= 24 && !cp_got; i++) begin
      req_stage(P0, 1, 1, TXN_CP4 + 64'(i));
      seg_set(P0, 0, 5, USER0, DATA_BYTES);
      @(negedge clk);
      req_drive(P0);
      #1;
      if (req_ready_o[P0]) begin
        check_bit(cp_ready_o, 1'b0, "CP4 SPM granted only while cp not granted");
        @(posedge clk);
        #1;
        req_undrive(P0);
        wbeat_push(P0, W_A2);
        wait_done(P0, 8, "CP4 spm pressure write");
        fin_push(P0, TXN_CP4 + 64'(i), 1, 0, "CP4 spm pressure release");
      end else begin
        // Urgent window: new SPM traffic on the cp bank is held off and
        // the cache request is granted instead.
        check_ready(P0, 1'b0, "CP4 urgent window blocks new SPM on cp bank");
        check_bit(cp_ready_o, 1'b1, "CP4 urgent window grants cp");
        @(posedge clk);
        #1;
        req_undrive(P0);
        cp_valid_i = 1'b0;
        cp_got = 1'b1;
      end
    end
    t_cp1 = cyc;
    lat_q = t_cp1 - t_cp0;
    cp_valid_i = 1'b0;
    check_bit(cp_got, 1'b1, "CP4 cp accepted under sustained SPM pressure");
    check_bit(lat_q <= CP_MAX_WAIT + drain_q + 2, 1'b1,
              "CP4 fairness bound CP_MAX_WAIT + drain + 2");
    $display("TB_L1_BANK: CP4 cp accept latency=%0d cycles, bound=%0d (CP_MAX_WAIT=%0d drain=%0d)",
             lat_q, CP_MAX_WAIT + drain_q + 2, CP_MAX_WAIT, drain_q);
    cp_wbeat(W_A2B, EN_FULL);
    cp_wait_done(4, "CP4 cp write");

    // ---- CP5: profile switch vs the cache port.
    $display("TB_L1_BANK: CP5 profile switch gate");
    @(negedge clk);
    cp_drive(1, 6, 57344 + 64, 2, 0);
    cp_push(cw);
    cp_wbeat(W_A0, EN_FULL);
    profile_push(PROFILE_IDX_W'(0), 64'd20, 0, 2'd2, "CP5 switch blocked while cp busy");
    cp_wbeat(W_A1, EN_FULL);
    cp_wait_done(4, "CP5 cp write done");
    cache_quiescent_i = 1'b0;
    profile_push(PROFILE_IDX_W'(0), 64'd20, 0, 2'd2,
                 "CP5 switch blocked while cache not quiescent");
    cache_quiescent_i = 1'b1;
    profile_push(PROFILE_IDX_W'(0), 64'd20, 1, 0, "CP5 switch accepted when idle");
    check_val(64'(active_profile_o), 64'd0, "CP5 active_profile_o tracks switch");
    check_val(profile_generation_o, 64'd20, "CP5 profile_generation_o tracks switch");
    // Profile 0: the whole bank is SPM — the old window addresses became
    // user SPM, so the cp is refused there and the SPM port may write them.
    @(negedge clk);
    cp_drive(0, 6, 57344, 1, 0);
    cp_expect_reject(3'd7, "CP5 old first window address now spm");
    @(negedge clk);
    cp_drive(0, 6, 40000, 1, 0);
    cp_expect_reject(3'd7, "CP5 mid-bank address now spm");
    req_stage(P0, 1, 1, TXN_CP5);
    seg_set(P0, 0, 6, 57344, 32);
    req_push(P0, w);
    wbeat_push(P0, W_P1);
    wait_done(P0, 8, "CP5 old window byte writable via spm");
    fin_push(P0, TXN_CP5, 1, 0, "CP5 old window spm release");
    // And vice versa: profile 3 (bound 32768) turns 40000 into cache window.
    profile_push(PROFILE_IDX_W'(3), 64'd21, 1, 0, "CP5 switch to profile 3");
    check_val(64'(active_profile_o), 64'd3, "CP5 active_profile_o profile 3");
    check_val(profile_generation_o, 64'd21, "CP5 profile_generation_o 21");
    @(negedge clk);
    cp_drive(1, 6, 40000, 1, 0);
    cp_push(cw);
    cp_wbeat(W_B9, EN_FULL);
    cp_wait_done(4, "CP5 new window write");
    @(negedge clk);
    cp_drive(0, 6, 40000, 1, 0);
    cp_push(cw);
    cp_rpop_check(0, W_B9, EN_FULL, "CP5 new window readback");
    cp_wait_done(4, "CP5 new window readback");
    req_stage(P0, 1, 1, TXN_CP5);
    seg_set(P0, 0, 6, 57344, 32);
    req_expect_reject(P0, 3'd5, "CP5 57344 back to cache partition for spm");
    @(negedge clk);
    cp_drive(0, 6, 32768 - DATA_BYTES, 1, 0);
    cp_expect_reject(3'd7, "CP5 word below profile-3 bound");
    @(negedge clk);
    cp_drive(0, 6, 32768, 1, 0);
    cp_push(cw);
    cp_rpop_check(0, '0, '0, "CP5 first profile-3 window word");
    cp_wait_done(4, "CP5 profile-3 window read");

    // ---- CP6: protocol shape.
    $display("TB_L1_BANK: CP6 cp protocol");
    check_bit(cp_busy_o, 1'b0, "CP6 cp idle between transactions");
    check_bit(cp_wdata_ready_o, 1'b0, "CP6 no wdata_ready when idle");
    check_bit(cp_rdata_valid_o, 1'b0, "CP6 no rdata when idle");
    check_bit(cp_done_o, 1'b0, "CP6 no done when idle");
    @(negedge clk);
    cp_drive(1, 6, 57344 + 128, 2, 0);
    cp_push(cw);
    cp_wbeat(W_A0, EN_LO16);
    repeat (3) begin
      @(negedge clk);
      #1;
      check_bit(cp_busy_o, 1'b1, "CP6 busy held between write beats");
      check_bit(cp_wdata_ready_o, 1'b1, "CP6 wdata_ready while write active");
      check_bit(cp_done_o, 1'b0, "CP6 no done before final beat");
      check_bit(cp_rdata_valid_o, 1'b0, "CP6 no read beat during write");
    end
    cp_wbeat(W_A1, EN_HI16);
    cp_wait_done(4, "CP6 stalled write done");
    @(negedge clk);
    cp_drive(0, 6, 57344 + 128, 2, 0);
    cp_push(cw);
    check_bit(cp_wdata_ready_o, 1'b0, "CP6 no wdata_ready during read");
    cp_rpop_check(3, W_A0, EN_LO16, "CP6 beat0 stable under backpressure");
    cp_rpop_check(2, W_A1, EN_HI16, "CP6 beat1 stable under backpressure");
    cp_wait_done(4, "CP6 backpressured read done");
    // A rejected request leaves the port idle and immediately usable.
    @(negedge clk);
    cp_drive(1, 6, 32768 - DATA_BYTES, 1, 0);
    cp_expect_reject(3'd7, "CP6 reject changes no state");
    @(negedge clk);
    cp_drive(1, 6, 57344 + 128, 1, 0);
    cp_push(cw);
    check_bit(cw == 0, 1'b1, "CP6 accept immediate after reject");
    cp_wbeat(W_15, EN_FULL);
    cp_wait_done(4, "CP6 post-reject write");
    @(negedge clk);
    cp_drive(0, 6, 57344 + 128, 1, 0);
    cp_push(cw);
    cp_rpop_check(0, W_15, EN_FULL, "CP6 last-beat data observable after done");
    cp_wait_done(4, "CP6 post-reject readback");

    // A reset in the middle of a cp write must clear the cp hold, the done
    // pulse and the wait counter (X-safety for 4-state simulation), and the
    // bank must be free for SPM immediately afterwards.
    @(negedge clk);
    cp_drive(1, 6, 57344 + 160, 2, 0);
    cp_push(cw);
    cp_wbeat(W_A0, EN_FULL);  // mid-write: hold active, one beat outstanding
    @(negedge clk);
    rstn = 1'b0;
    cp_valid_i = 1'b0;
    @(posedge clk);
    #1;
    check_bit(cp_busy_o, 1'b0, "CP6 reset clears the cp hold");
    check_bit(cp_done_o, 1'b0, "CP6 reset clears cp_done");
    repeat (2) @(negedge clk);
    rstn = 1'b1;
    @(negedge clk);
    #1;
    check_bit(cp_busy_o, 1'b0, "CP6 hold stays clear after reset release");
    check_val(64'(active_profile_o), 64'd0, "CP6 reset returns profile 0");
    req_stage(P0, 1, 1, 64'h0000_6000_0000_0007);
    seg_set(P0, 0, 6, USER0, 32);
    req_push(P0, w);
    check_bit(w == 0, 1'b1, "CP6 cp bank free for SPM after reset");
    wbeat_push(P0, W_P0);
    wait_done(P0, 8, "CP6 post-reset write");
    fin_push(P0, 64'h0000_6000_0000_0007, 1, 0, "CP6 post-reset release");

    // ---------------------------------------------------------------- done
    if (errors == 0) begin
      $display("C029_PASS tb_l1_bank checks=%0d conflict_waits=%0d latency_proofs=%0d",
               checks, wait_obs, latency_obs);
      $display("TB_L1_BANK: first read beat latency = 1 cycle (sourced BANK_LATENCY_CYCLES)");
    end else begin
      $display("C029_FAIL tb_l1_bank errors=%0d checks=%0d", errors, checks);
      $fatal(1, "C029_ASSERT tb_l1_bank %0d failing checks", errors);
    end
    $finish;
  end

endmodule
