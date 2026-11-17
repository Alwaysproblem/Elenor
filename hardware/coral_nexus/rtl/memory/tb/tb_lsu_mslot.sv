// tb_lsu_mslot.sv — C031 / W4-03 unit bench for rtl/memory/lsu_mslot.sv.
//
// Proves the real scalar LSU owner over the real C029 l1_bank bytes with the
// miss owner (lsu_cache_path) REPLACED BY PORT-LEVEL STIMULUS ONLY: cached
// requests are never issued into a real cache engine here (the parent
// integration bench tb_c031_memory.sv owns that path); this bench drives the
// cache_req/cache_rsp channels directly, so every cached-window check is a
// protocol check of THIS module (payload, retention, cancellation, settle
// forwarding, multi-record and out-of-order completion), not a claim about
// the miss engine.
//
// Scenario map (all on C1 and C2 unless noted; two-Context cases adapt at
// C1 without weakening C2 coverage):
//   S0  reset state, no installed table: identity fault ERR_OWNER, no bank
//       activity, counters zero.
//   S1  config battery: bad table rejects (cached+writable, misaligned
//       cached binding, stripe 0), good installs, quiescent same-generation
//       reinstall, out-of-range context (C2), second Context install (C2).
//   S2  real bank bytes through the LSU: aligned/unaligned word/half/byte
//       stores and loads, never-written byte -> ERR_UNINITIALIZED.
//   S3  sign/zero extension: 0x80 byte, 0x8000 half.
//   S4  unaligned access structure: in-word unaligned, cross-word two-beat,
//       cross-stripe cross-bank two-segment store with exact bank_busy_o
//       masks mid-flight; WO/RW window alias of the same bank bytes.
//   S5  serial multi-record issue and oldest-ready completion order.
//   S6  16-credit backpressure and retention: 16 records outstanding,
//       17th refused, payload stability under response stall, in-order
//       drain, accepted == terminal.
//   S7  (C2) same-cycle conflicting bank grants (fixed port priority, loser
//       waits with wait_bank attribution), response-stall independence.
//   S8  owner faults: duplicate key, stale generation, future generation,
//       foreign ctx_slot; stale_o accounting.
//   S9  cancel battery: unissued -> ISOLATED (no bank activity), mid-write
//       -> COMPLETED with effects readable, mid-read drain -> ISOLATED,
//       same-edge request+cancel admission bubble (no handshake, no leaked
//       accepted count), same-generation work continues WITHOUT reconfigure,
//       config rejected until drain then accepted, stale cancel no-op,
//       double cancel harmless.
//   S10 address/permission faults: outside all windows, window-end crossing,
//       byte_count 0/3, RO write, WO read, reserved-bytes window (real bank
//       reject), fence with byte_count!=0; no side effects.
//   S11 MEM-06 fence: older stalled record + queued fence + younger request:
//       younger never launches past the fence (bank bit proof), fence
//       completes only after older terminal, peer Context unblocked (C2),
//       fence cancel -> ISOLATED.
//   S12 cache-channel protocol without the miss owner: payload fields and
//       retention under !ready, cancel of unlaunched -> ISOLATED, served
//       responses with LSU-side extension, cancelled waiter forwards the
//       settled final state, same-Context out-of-order completion replay
//       (miss waits; younger SPM store/load complete; second miss completes
//       before the first), 16 cached records in flight with reverse-order
//       service, credit refusal at 16.
//   S13 profile passthrough (ack, non-increasing reject, reject-then-ack
//       across a bank hold) and the cp data-port bridge (write/read beats,
//       done pulse, bank_busy_o untouched, SPM unaffected).
//   S14 (C2) interleaved Context stress with byte integrity and final
//       conservation per Context.
//   S15 LAST: wrong record_id cache response -> not accepted + sticky
//       invariant_error_o (deliberate injection, expected at the summary).
//
// Builds: packages, l1_bank.sv, lsu_mslot.sv, then this bench; Verilator
// --assert with -GCONTEXT_COUNT=1 and =2. Prints C031_PASS tb_lsu_mslot only
// after every check passed; failures print C031_ASSERT markers and end
// non-zero. invariant_error_o is expected 0 everywhere except after the
// deliberate S15 injection.
`timescale 1ns/1ps

module tb_lsu_mslot
  import c029_params_pkg::*;
  import c031_lsu_pkg::*;
#(
  parameter int CONTEXT_COUNT = C2_CONTEXTS
);

  localparam int unsigned USER0 = SYSTEM_RESERVED_BYTES;

  // Known word rewritten at W0 rel 0 before the cancel battery; every later
  // rel-0 load expects it (C2 parked stores in S7 overwrite the S2 value).
  localparam logic [31:0] REL0_WORD = 32'h0BB0_0000;

  // ------------------------------------------------------------- DUT I/O
  logic clk;
  logic rst_n;

  logic [CONTEXT_COUNT-1:0] req_valid_i, req_ready_o;
  request_t req_i [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] rsp_valid_o, rsp_ready_i;
  completion_t rsp_o [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] cancel_i;
  logic [15:0] cancel_generation_i [CONTEXT_COUNT];
  logic config_valid_i, config_ready_o, config_reject_o;
  logic [31:0] config_context_i;
  logic [15:0] config_generation_i;
  window_t config_windows_i [SLOT_COUNT];
  logic issue_enable_i;
  cache_port_request_t cp_i;
  cache_port_response_t cp_o;
  logic cache_quiescent_i;
  logic profile_valid_i;
  logic [$clog2(PROFILE_COUNT)-1:0] profile_index_i;
  logic [31:0] profile_generation_i;
  logic profile_ack_o, profile_reject_o;
  logic [$clog2(PROFILE_COUNT)-1:0] bank_profile_o;
  logic [CONTEXT_COUNT-1:0] cache_req_valid_o, cache_req_ready_i;
  cache_request_t cache_req_o [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] cache_rsp_valid_i, cache_rsp_ready_o;
  cache_response_t cache_rsp_i [CONTEXT_COUNT];
  logic [31:0] outstanding_o [CONTEXT_COUNT];
  logic [63:0] accepted_o [CONTEXT_COUNT], terminal_o [CONTEXT_COUNT];
  logic [63:0] wait_bank_o [CONTEXT_COUNT], wait_mshr_o [CONTEXT_COUNT];
  logic [63:0] stale_o [CONTEXT_COUNT];
  logic [BANK_COUNT-1:0] bank_busy_o;
  logic invariant_error_o;

  lsu_mslot #(
    .CONTEXT_COUNT (CONTEXT_COUNT),
    .WINDOW_COUNT  (SLOT_COUNT)
  ) dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .req_valid_i       (req_valid_i),
    .req_ready_o       (req_ready_o),
    .req_i             (req_i),
    .rsp_valid_o       (rsp_valid_o),
    .rsp_ready_i       (rsp_ready_i),
    .rsp_o             (rsp_o),
    .cancel_i          (cancel_i),
    .cancel_generation_i (cancel_generation_i),
    .config_valid_i    (config_valid_i),
    .config_ready_o    (config_ready_o),
    .config_reject_o   (config_reject_o),
    .config_context_i  (config_context_i),
    .config_generation_i (config_generation_i),
    .config_windows_i  (config_windows_i),
    .issue_enable_i    (issue_enable_i),
    .cp_i              (cp_i),
    .cp_o              (cp_o),
    .cache_quiescent_i (cache_quiescent_i),
    .profile_valid_i   (profile_valid_i),
    .profile_index_i   (profile_index_i),
    .profile_generation_i (profile_generation_i),
    .profile_ack_o     (profile_ack_o),
    .profile_reject_o  (profile_reject_o),
    .bank_profile_o    (bank_profile_o),
    .cache_req_valid_o (cache_req_valid_o),
    .cache_req_ready_i (cache_req_ready_i),
    .cache_req_o       (cache_req_o),
    .cache_rsp_valid_i (cache_rsp_valid_i),
    .cache_rsp_ready_o (cache_rsp_ready_o),
    .cache_rsp_i       (cache_rsp_i),
    .outstanding_o     (outstanding_o),
    .accepted_o        (accepted_o),
    .terminal_o        (terminal_o),
    .wait_bank_o       (wait_bank_o),
    .wait_mshr_o       (wait_mshr_o),
    .stale_o           (stale_o),
    .bank_busy_o       (bank_busy_o),
    .invariant_error_o (invariant_error_o)
  );

  always #5 clk = ~clk;

  int errors;
  int checks;
  logic [63:0] cnt_snap;

  initial begin
    #8_000_000;
    $error("C031_ASSERT tb_lsu_mslot progress: timeout");
    $display("C031_FAIL tb_lsu_mslot timeout");
    $fatal(1);
  end

  // -------------------------------------------------------------- helpers
  task automatic check_bit(input logic got, input logic exp, input string what);
    begin
      checks = checks + 1;
      if (got !== exp) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: got %0b expected %0b", what, got, exp);
      end
    end
  endtask

  task automatic check_val(input logic [31:0] got, input logic [31:0] exp,
                           input string what);
    begin
      checks = checks + 1;
      if (got !== exp) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: got 0x%0h expected 0x%0h",
               what, got, exp);
      end
    end
  endtask

  task automatic check_val64(input logic [63:0] got, input logic [63:0] exp,
                             input string what);
    begin
      checks = checks + 1;
      if (got !== exp) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: got 0x%0h expected 0x%0h",
               what, got, exp);
      end
    end
  endtask

  task automatic check_val_wide(input logic [DATA_BYTES*8-1:0] got,
                                input logic [DATA_BYTES*8-1:0] exp,
                                input string what);
    begin
      checks = checks + 1;
      if (got !== exp) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: wide data mismatch", what);
      end
    end
  endtask

  function automatic request_t mk_req(input int c, input logic [15:0] gen,
                                      input logic [31:0] seq_,
                                      input logic [31:0] addr,
                                      input logic [31:0] data,
                                      input logic [2:0] bc, input logic wr,
                                      input logic uns, input logic [4:0] rd,
                                      input logic flt, input logic fen);
    request_t r;
    r.owner.ctx_slot = 32'(c);
    r.owner.ctx_generation = gen;
    r.owner.owner_op_seq = seq_;
    r.owner.subop_id = 8'd0;
    r.pc = 32'h0000_1000 + seq_;
    r.address = addr;
    r.data = data;
    r.rd = rd;
    r.byte_count = bc;
    r.write = wr;
    r.unsigned_load = uns;
    r.float_reg = flt;
    r.fence = fen;
    return r;
  endfunction

  task automatic win_set(input int idx, input logic valid, input logic rd_,
                         input logic wr_, input logic cached,
                         input logic [31:0] base, input logic [31:0] size,
                         input logic [31:0] spm, input logic [31:0] stripe,
                         input logic [31:0] bid, input logic [31:0] bgen,
                         input logic [31:0] boff, input logic [31:0] arena,
                         input logic [31:0] scope);
    begin
      config_windows_i[idx].valid = valid;
      config_windows_i[idx].readable = rd_;
      config_windows_i[idx].writable = wr_;
      config_windows_i[idx].cached = cached;
      config_windows_i[idx].base = base;
      config_windows_i[idx].size = size;
      config_windows_i[idx].spm_base = spm;
      config_windows_i[idx].stripe_bytes = stripe;
      config_windows_i[idx].binding_id = bid;
      config_windows_i[idx].binding_generation = bgen;
      config_windows_i[idx].binding_offset = boff;
      config_windows_i[idx].arena_generation = arena;
      config_windows_i[idx].scope_epoch = scope;
    end
  endtask

  task automatic cfg_table_ctx0;
    begin
      for (int w = 0; w < SLOT_COUNT; w++)
        win_set(w, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
      // W0: read-write SPM, stripe 64.
      win_set(0, 1, 1, 1, 0, 32'h0001_0000, 32'h1000, 32'(USER0), 64,
              32'h1111_0000, 1, 32'h0010_0000, 3, 9);
      // W1: read-only.
      win_set(1, 1, 1, 0, 0, 32'h0002_0000, 32'h0800, 32'(USER0 + 4096), 128,
              32'h2222_0000, 1, 0, 3, 9);
      // W2: write-only.
      win_set(2, 1, 0, 1, 0, 32'h0003_0000, 32'h0800, 32'(USER0 + 8192), 128,
              32'h3333_0000, 1, 0, 3, 9);
      // W3: cached read-only, line-aligned binding (whole-line refill).
      win_set(3, 1, 1, 0, 1, 32'h0004_0000, 32'h1000, 0, 64,
              32'h4444_0000, 2, 32'h0200_0000, 5, 11);
      // W4: read-write alias over W2's SPM bytes.
      win_set(4, 1, 1, 1, 0, 32'h0005_0000, 32'h0800, 32'(USER0 + 8192), 128,
              32'h5555_0000, 1, 0, 3, 9);
      // W5: reserved-bytes window (spm_base 0): legal config, bank rejects.
      win_set(5, 1, 1, 1, 0, 32'h0006_0000, 32'h0100, 0, 64,
              32'h6666_0000, 1, 0, 3, 9);
    end
  endtask

  task automatic cfg_table_ctx1;
    begin
      for (int w = 0; w < SLOT_COUNT; w++)
        win_set(w, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0);
      win_set(0, 1, 1, 1, 0, 32'h0001_0000, 32'h1000, 32'(USER0 + 12288), 64,
              32'h9999_0000, 1, 32'h0030_0000, 4, 10);
    end
  endtask

  task automatic cfg_go(input int c, input logic [15:0] gen,
                        input logic exp_rej, input string what);
    begin
      @(negedge clk);
      config_valid_i = 1'b1;
      config_context_i = 32'(c);
      config_generation_i = gen;
      @(negedge clk);
      config_valid_i = 1'b0;
      check_bit(config_reject_o, exp_rej, what);
      @(posedge clk);
      #1;
      check_bit(config_reject_o, 1'b0, {what, ": pulse clears"});
    end
  endtask

  task automatic req_go(input int c, input request_t r, input string what);
    int guard;
    begin
      @(negedge clk);
      req_valid_i[c] = 1'b1;
      req_i[c] = r;
      guard = 0;
      while (req_ready_o[c] !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: req never ready ctx %0d", what, c);
      end
      @(posedge clk);
      @(negedge clk);
      req_valid_i[c] = 1'b0;
    end
  endtask

  // Drive a request that must NOT be accepted (credit full), then withdraw.
  task automatic req_stall_check(input int c, input request_t r,
                                 input string what);
    logic [63:0] acc_before;
    begin
      acc_before = accepted_o[c];
      @(negedge clk);
      req_valid_i[c] = 1'b1;
      req_i[c] = r;
      repeat (2) begin
        @(posedge clk);
        #1;
        check_bit(req_ready_o[c], 1'b0, {what, ": ready low while full"});
      end
      check_val64(accepted_o[c], acc_before, {what, ": no acceptance"});
      @(negedge clk);
      req_valid_i[c] = 1'b0;
    end
  endtask

  task automatic rsp_take(input int c, input logic [1:0] fexp,
                          input logic [3:0] eexp, input logic [31:0] dexp,
                          input logic [31:0] seq_exp, input string what,
                          input int stall);
    int guard;
    completion_t cap;
    begin
      guard = 0;
      while (rsp_valid_o[c] !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: completion never came ctx %0d",
               what, c);
      end
      if (stall > 0) begin
        cap = rsp_o[c];
        repeat (stall) begin
          @(negedge clk);
          checks = checks + 1;
          if (rsp_o[c] != cap) begin
            errors = errors + 1;
            $error("C031_ASSERT tb_lsu_mslot %s: payload moved under stall",
                   what);
          end
        end
      end
      check_val(32'(rsp_o[c].final_state), 32'(fexp), {what, ": final_state"});
      check_val(32'(rsp_o[c].error), 32'(eexp), {what, ": error"});
      check_val(rsp_o[c].data, dexp, {what, ": data"});
      check_val(rsp_o[c].owner.owner_op_seq, seq_exp, {what, ": owner seq"});
      rsp_ready_i[c] = 1'b1;
      @(posedge clk);
      @(negedge clk);
      rsp_ready_i[c] = 1'b0;
    end
  endtask

  task automatic cancel_go(input int c, input logic [15:0] gen);
    begin
      @(negedge clk);
      cancel_i[c] = 1'b1;
      cancel_generation_i[c] = gen;
      @(negedge clk);
      cancel_i[c] = 1'b0;
    end
  endtask

  task automatic wait_out(input int c, input int n, input string what);
    int guard;
    begin
      guard = 0;
      while (int'(outstanding_o[c]) !== n && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: outstanding stuck at %0d want %0d",
               what, outstanding_o[c], n);
      end
    end
  endtask

  task automatic prof_cmd(input logic [$clog2(PROFILE_COUNT)-1:0] idx,
                          input logic [31:0] gen, input logic exp_ack,
                          input logic exp_rej, input string what);
    int guard;
    begin
      @(negedge clk);
      profile_valid_i = 1'b1;
      profile_index_i = idx;
      profile_generation_i = gen;
      // Exactly one asserted edge: drop at the FOLLOWING negedge. Dropping
      // in the same delta as the posedge races the bank's edge sample of
      // profile_set_valid_i (Verilator may sample the already-dropped 0).
      @(posedge clk);
      @(negedge clk);
      profile_valid_i = 1'b0;
      guard = 0;
      while (profile_ack_o !== 1'b1 && profile_reject_o !== 1'b1
             && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: no profile response", what);
      end
      check_bit(profile_ack_o, exp_ack, {what, ": ack"});
      check_bit(profile_reject_o, exp_rej, {what, ": reject"});
      @(negedge clk);
    end
  endtask

  // Push one cached-window read and capture its record_id at the launch
  // handshake (cache_req_ready_i must already be 1).
  task automatic cache_push(input int c, input request_t r,
                            output logic [RECORD_W-1:0] rid,
                            input string what);
    int guard;
    begin
      @(negedge clk);
      req_valid_i[c] = 1'b1;
      req_i[c] = r;
      guard = 0;
      while (req_ready_o[c] !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: cached req never accepted", what);
      end
      @(posedge clk);
      @(negedge clk);
      req_valid_i[c] = 1'b0;
      guard = 0;
      while (!(cache_req_valid_o[c] === 1'b1 && cache_req_ready_i[c] === 1'b1)
             && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: cache launch never came", what);
      end
      rid = cache_req_o[c].record_id;
      check_val(cache_req_o[c].owner.owner_op_seq, r.owner.owner_op_seq,
                {what, ": cache launch owner echo"});
      @(posedge clk);
      @(negedge clk);
    end
  endtask

  task automatic cache_serve(input int c, input logic [RECORD_W-1:0] rid,
                             input logic [1:0] fin, input logic [3:0] err,
                             input logic [31:0] data, input string what);
    int guard;
    begin
      @(negedge clk);
      cache_rsp_valid_i[c] = 1'b1;
      cache_rsp_i[c].owner = '0;
      cache_rsp_i[c].record_id = rid;
      cache_rsp_i[c].data = data;
      cache_rsp_i[c].final_state = fin;
      cache_rsp_i[c].error = err;
      // cache_rsp_ready_o depends combinationally on cache_rsp_valid_i; let
      // the match settle within the SAME low phase before the first poll
      // (#1, no posedge crossed): otherwise the poll reads the pre-drive
      // value while the handshake fires at the coming posedge and the
      // record leaves S_CWAIT unseen (dead-channel spin).
      #1;
      guard = 0;
      while (cache_rsp_ready_o[c] !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot %s: cache rsp never accepted", what);
      end
      @(posedge clk);
      @(negedge clk);
      cache_rsp_valid_i[c] = 1'b0;
    end
  endtask

  // ------------------------------------------------------------- stimulus
  initial begin : main
    errors = 0;
    checks = 0;
    clk = 1'b0;
    rst_n = 1'b0;
    req_valid_i = '0;
    for (int c = 0; c < CONTEXT_COUNT; c++) req_i[c] = '0;
    rsp_ready_i = '0;
    cancel_i = '0;
    for (int c = 0; c < CONTEXT_COUNT; c++) cancel_generation_i[c] = '0;
    config_valid_i = 1'b0;
    config_context_i = '0;
    config_generation_i = '0;
    for (int w = 0; w < SLOT_COUNT; w++) config_windows_i[w] = '0;
    issue_enable_i = 1'b1;
    cp_i = '0;
    cache_quiescent_i = 1'b1;
    profile_valid_i = 1'b0;
    profile_index_i = '0;
    profile_generation_i = '0;
    cache_req_ready_i = '0;
    cache_rsp_valid_i = '0;
    for (int c = 0; c < CONTEXT_COUNT; c++) cache_rsp_i[c] = '0;

    repeat (4) @(negedge clk);
    rst_n = 1'b1;
    @(negedge clk);

    // ================================================= S0 reset / no table
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      check_bit(req_ready_o[c], 1'b1, "S0 credit available after reset");
      check_bit(rsp_valid_o[c], 1'b0, "S0 no completion at reset");
      check_val(outstanding_o[c], 32'd0, "S0 outstanding zero");
      check_val64(accepted_o[c], 64'd0, "S0 accepted zero");
      check_val64(terminal_o[c], 64'd0, "S0 terminal zero");
    end
    check_val({16'd0, bank_busy_o}, '0, "S0 bank idle");
    check_bit(invariant_error_o, 1'b0, "S0 invariant clean");
    req_go(0, mk_req(0, 16'd5, 32'd1, 32'h0001_0000, 32'h0, 3'd4, 1'b0, 1'b0,
                     5'd1, 1'b0, 1'b0), "S0 no-table request");
    rsp_take(0, FAULTED, ERR_OWNER, 32'd0, 32'd1, "S0 no-table fault", 0);
    check_val64(stale_o[0], 64'd1, "S0 stale owner counted");
    check_val({16'd0, bank_busy_o}, '0, "S0 no bank activity for fault");
    check_val(outstanding_o[0], 32'd0, "S0 fault released credit");

    // ================================================= S1 config battery
    cfg_table_ctx0;
    // cached+writable is rejected (line-aligned or not).
    win_set(3, 1, 1, 1, 1, 32'h0004_0000, 32'h1000, 0, 64,
            32'h4444_0000, 2, 32'h0200_0000, 5, 11);
    cfg_go(0, 16'd5, 1'b1, "S1 cached+writable rejected");
    cfg_table_ctx0;
    // non-line-aligned cached binding rejected (whole-line authorization).
    win_set(3, 1, 1, 0, 1, 32'h0004_0000, 32'h1000, 0, 64,
            32'h4444_0000, 2, 32'h0200_0024, 5, 11);
    cfg_go(0, 16'd5, 1'b1, "S1 cached misaligned binding rejected");
    cfg_table_ctx0;
    // stripe 0 rejected.
    win_set(6, 1, 1, 1, 0, 32'h0007_0000, 32'h100, 32'(USER0), 0,
            0, 0, 0, 0, 0);
    cfg_go(0, 16'd5, 1'b1, "S1 stripe0 rejected");
    cfg_table_ctx0;
    cfg_go(0, 16'd5, 1'b0, "S1 good table installed");
    cfg_table_ctx0;
    cfg_go(0, 16'd5, 1'b0, "S1 quiescent same-gen reinstall");
    if (CONTEXT_COUNT == 2) begin
      cfg_go(5, 16'd7, 1'b1, "S1 out-of-range context rejected");
      cfg_table_ctx1;
      cfg_go(1, 16'd7, 1'b0, "S1 ctx1 table installed");
    end

    // ================================================= S2 real bank bytes
    req_go(0, mk_req(0, 16'd5, 32'd10, 32'h0001_0000, 32'hDEAD_BEEF, 3'd4,
                     1'b1, 1'b0, 5'd2, 1'b0, 1'b0), "S2 store word");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd10, "S2 store word", 0);
    req_go(0, mk_req(0, 16'd5, 32'd11, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd3, 1'b0, 1'b0), "S2 load word");
    rsp_take(0, COMPLETED, ERR_NONE, 32'hDEAD_BEEF, 32'd11, "S2 load word", 0);
    req_go(0, mk_req(0, 16'd5, 32'd12, 32'h0001_0006, 32'h0000_BEEF, 3'd2,
                     1'b1, 1'b0, 5'd4, 1'b0, 1'b0), "S2 unaligned half store");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd12,
             "S2 unaligned half store", 0);
    req_go(0, mk_req(0, 16'd5, 32'd13, 32'h0001_0006, 32'h0, 3'd2, 1'b0,
                     1'b1, 5'd5, 1'b0, 1'b0), "S2 unaligned half load");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h0000_BEEF, 32'd13,
             "S2 unaligned half load", 0);
    req_go(0, mk_req(0, 16'd5, 32'd14, 32'h0001_000B, 32'h0000_007A, 3'd1,
                     1'b1, 1'b0, 5'd6, 1'b0, 1'b0), "S2 byte store");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd14, "S2 byte store", 0);
    req_go(0, mk_req(0, 16'd5, 32'd15, 32'h0001_000B, 32'h0, 3'd1, 1'b0,
                     1'b0, 5'd7, 1'b0, 1'b0), "S2 byte load");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h0000_007A, 32'd15, "S2 byte load", 0);
    req_go(0, mk_req(0, 16'd5, 32'd16, 32'h0001_0010, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd8, 1'b0, 1'b0), "S2 never-written word");
    rsp_take(0, FAULTED, ERR_UNINITIALIZED, 32'd0, 32'd16,
             "S2 uninit fault", 0);
    // Directed ben-gating kill: a partially-written word (1 of 4 bytes
    // valid) must FAULT with ERR_UNINITIALIZED and zero data; filling the
    // remaining bytes must then read COMPLETED with the exact bytes.
    req_go(0, mk_req(0, 16'd5, 32'd17, 32'h0001_0034, 32'h0000_005C, 3'd1,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S2 partial byte store");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd17,
             "S2 partial byte store", 0);
    req_go(0, mk_req(0, 16'd5, 32'd18, 32'h0001_0034, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S2 partial-uninit word");
    rsp_take(0, FAULTED, ERR_UNINITIALIZED, 32'd0, 32'd18,
             "S2 partial-uninit fault", 3);
    req_go(0, mk_req(0, 16'd5, 32'd19, 32'h0001_0034, 32'h1122_335C, 3'd4,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S2 fill remaining bytes");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd19,
             "S2 fill remaining bytes", 0);
    req_go(0, mk_req(0, 16'd5, 32'd22, 32'h0001_0034, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S2 filled word readback");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h1122_335C, 32'd22,
             "S2 filled word readback", 0);
    // Cancelled partially-uninit read: ISOLATED carries only the valid
    // byte; missing lanes contribute zero to the assembled data.
    req_go(0, mk_req(0, 16'd5, 32'd23, 32'h0001_0038, 32'h0000_007D, 3'd1,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S2 cancel partial store");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd23,
             "S2 cancel partial store", 0);
    req_go(0, mk_req(0, 16'd5, 32'd24, 32'h0001_0038, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S2 cancel partial read");
    cancel_go(0, 16'd5);
    rsp_take(0, ISOLATED, ERR_NONE, 32'h0000_007D, 32'd24,
             "S2 cancelled partial assembly", 0);
    if (CONTEXT_COUNT == 2) begin
      req_go(1, mk_req(1, 16'd7, 32'd20, 32'h0001_0000, 32'h1234_5678, 3'd4,
                       1'b1, 1'b0, 5'd2, 1'b0, 1'b0), "S2 ctx1 store");
      rsp_take(1, COMPLETED, ERR_NONE, 32'd0, 32'd20, "S2 ctx1 store", 0);
      req_go(1, mk_req(1, 16'd7, 32'd21, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd3, 1'b0, 1'b0), "S2 ctx1 load");
      rsp_take(1, COMPLETED, ERR_NONE, 32'h1234_5678, 32'd21,
               "S2 ctx1 load", 0);
    end

    // ================================================= S3 extension
    req_go(0, mk_req(0, 16'd5, 32'd30, 32'h0001_0020, 32'h0000_0080, 3'd1,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S3 store 0x80");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd30, "S3 store 0x80", 0);
    req_go(0, mk_req(0, 16'd5, 32'd31, 32'h0001_0020, 32'h0, 3'd1, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S3 lb");
    rsp_take(0, COMPLETED, ERR_NONE, 32'hFFFF_FF80, 32'd31, "S3 lb signed", 0);
    req_go(0, mk_req(0, 16'd5, 32'd32, 32'h0001_0020, 32'h0, 3'd1, 1'b0,
                     1'b1, 5'd1, 1'b0, 1'b0), "S3 lbu");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h0000_0080, 32'd32, "S3 lbu", 0);
    req_go(0, mk_req(0, 16'd5, 32'd33, 32'h0001_0028, 32'h0000_8000, 3'd2,
                     1'b1, 1'b0, 5'd2, 1'b0, 1'b0), "S3 store 0x8000");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd33, "S3 store 0x8000", 0);
    req_go(0, mk_req(0, 16'd5, 32'd34, 32'h0001_0028, 32'h0, 3'd2, 1'b0,
                     1'b0, 5'd2, 1'b0, 1'b0), "S3 lh");
    rsp_take(0, COMPLETED, ERR_NONE, 32'hFFFF_8000, 32'd34, "S3 lh signed", 0);
    req_go(0, mk_req(0, 16'd5, 32'd35, 32'h0001_0028, 32'h0, 3'd2, 1'b0,
                     1'b1, 5'd2, 1'b0, 1'b0), "S3 lhu");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h0000_8000, 32'd35, "S3 lhu", 0);

    // ================================================= S4 unaligned structure
    // Cross-word two-beat store at rel 222 (bank 3 bytes 30..33).
    req_go(0, mk_req(0, 16'd5, 32'd40, 32'h0001_00DE, 32'h1122_3344, 3'd4,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S4 cross-word store");
    @(negedge clk);
    check_val({16'd0, bank_busy_o}, 32'(1 << 3), "S4 cross-word holds bank 3");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd40,
             "S4 cross-word store done", 0);
    check_val({16'd0, bank_busy_o}, '0, "S4 bank freed after fin");
    req_go(0, mk_req(0, 16'd5, 32'd41, 32'h0001_00DE, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S4 cross-word load");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h1122_3344, 32'd41,
             "S4 cross-word load", 0);
    // Cross-stripe cross-bank store at rel 254 (banks 3 and 4).
    req_go(0, mk_req(0, 16'd5, 32'd42, 32'h0001_00FE, 32'hAABB_CCDD, 3'd4,
                     1'b1, 1'b0, 5'd2, 1'b0, 1'b0), "S4 cross-bank store");
    @(negedge clk);
    check_val({16'd0, bank_busy_o}, 32'((1 << 3) | (1 << 4)),
              "S4 cross-bank holds banks 3 and 4");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd42,
             "S4 cross-bank store done", 0);
    check_val({16'd0, bank_busy_o}, '0, "S4 both banks freed");
    req_go(0, mk_req(0, 16'd5, 32'd43, 32'h0001_00FE, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd2, 1'b0, 1'b0), "S4 cross-bank load");
    rsp_take(0, COMPLETED, ERR_NONE, 32'hAABB_CCDD, 32'd43,
             "S4 cross-bank load", 0);
    // WO window store, RW alias readback (same bank bytes, two windows).
    req_go(0, mk_req(0, 16'd5, 32'd44, 32'h0003_0008, 32'h0000_CAFE, 3'd2,
                     1'b1, 1'b0, 5'd3, 1'b0, 1'b0), "S4 WO store");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd44, "S4 WO store", 0);
    req_go(0, mk_req(0, 16'd5, 32'd45, 32'h0005_0008, 32'h0, 3'd2, 1'b0,
                     1'b1, 5'd4, 1'b0, 1'b0), "S4 alias readback");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h0000_CAFE, 32'd45,
             "S4 alias readback", 0);

    // ================================================= S5 serial records
    for (int k = 0; k < 4; k++) begin
      req_go(0, mk_req(0, 16'd5, 32'd50 + k, 32'h0001_0200 + 4 * k,
                       32'hC0DE_0000 + k, 3'd4, 1'b1, 1'b0, 5'd1, 1'b0, 1'b0),
             "S5 queued store");
    end
    for (int k = 0; k < 4; k++) begin
      rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd50 + k,
               "S5 store completion order", 0);
    end
    for (int k = 0; k < 4; k++) begin
      req_go(0, mk_req(0, 16'd5, 32'd60 + k, 32'h0001_0200 + 4 * k, 32'h0,
                       3'd4, 1'b0, 1'b0, 5'd2, 1'b0, 1'b0), "S5 queued load");
    end
    for (int k = 0; k < 4; k++) begin
      rsp_take(0, COMPLETED, ERR_NONE, 32'hC0DE_0000 + k, 32'd60 + k,
               "S5 load completion order", 0);
    end

    // ================================================= S6 16-credit stress
    for (int k = 0; k < 16; k++) begin
      req_go(0, mk_req(0, 16'd5, 32'd100 + k, 32'h0001_0080 + 4 * k,
                       32'hF00D_0000 + k, 3'd4, 1'b1, 1'b0, 5'd1, 1'b0, 1'b0),
             "S6 prestore");
      rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd100 + k,
               "S6 prestore done", 0);
    end
    for (int k = 0; k < 16; k++) begin
      req_go(0, mk_req(0, 16'd5, 32'd200 + k, 32'h0001_0080 + 4 * k, 32'h0,
                       3'd4, 1'b0, 1'b0, 5'd2, 1'b0, 1'b0),
             "S6 retained load");
    end
    wait_out(0, 16, "S6 sixteen records retained");
    check_bit(req_ready_o[0], 1'b0, "S6 credit exhausted");
    req_stall_check(0, mk_req(0, 16'd5, 32'd299, 32'h0001_0000, 32'h0, 3'd4,
                              1'b0, 1'b0, 5'd3, 1'b0, 1'b0), "S6 17th load");
    check_val64(accepted_o[0] - terminal_o[0], 64'd16,
                "S6 conservation while stalled");
    for (int k = 0; k < 16; k++) begin
      rsp_take(0, COMPLETED, ERR_NONE, 32'hF00D_0000 + k, 32'd200 + k,
               "S6 drain order", (k == 0) ? 3 : 0);
    end
    check_val64(accepted_o[0], terminal_o[0], "S6 accepted == terminal");
    check_val(outstanding_o[0], 32'd0, "S6 outstanding drained");
    check_bit(req_ready_o[0], 1'b1, "S6 credit returned");

    // ================================================= S7 (C2) conflicts
    if (CONTEXT_COUNT == 2) begin : s7_conflicts
      logic [63:0] wb1_before;
      localparam int PEER = (CONTEXT_COUNT == 2) ? 1 : 0;
      wb1_before = wait_bank_o[PEER];
      issue_enable_i = 1'b0;
      // Both present bank 3 in the same cycle once issue re-enables.
      req_go(0, mk_req(0, 16'd5, 32'd70, 32'h0001_00DE, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd1, 1'b0, 1'b0), "S7 ctx0 conflicting read");
      req_go(1, mk_req(1, 16'd7, 32'd71, 32'h0001_04C0, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd1, 1'b0, 1'b0), "S7 ctx1 conflicting read");
      issue_enable_i = 1'b1;
      @(posedge clk);
      @(negedge clk);
      rsp_take(0, COMPLETED, ERR_NONE, 32'h1122_3344, 32'd70,
               "S7 ctx0 read wins port", 0);
      rsp_take(1, FAULTED, ERR_UNINITIALIZED, 32'd0, 32'd71,
               "S7 ctx1 read uninit", 0);
      checks = checks + 1;
      if (wait_bank_o[PEER] <= wb1_before) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot S7: wait_bank never attributed");
      end
      // Response-stall independence: ctx0 completion parked, ctx1 flows.
      req_go(0, mk_req(0, 16'd5, 32'd72, 32'h0001_0000, 32'hEEDD_0011, 3'd4,
                       1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S7 ctx0 parked store");
      req_go(1, mk_req(1, 16'd7, 32'd73, 32'h0001_0004, 32'h0BAD_0002, 3'd4,
                       1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S7 ctx1 flowing store");
      rsp_take(1, COMPLETED, ERR_NONE, 32'd0, 32'd73, "S7 ctx1 unblocked", 0);
      check_bit(rsp_valid_o[0], 1'b1, "S7 ctx0 completion parked");
      rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd72, "S7 ctx0 unparked", 0);
    end

    // ================================================= S8 owner faults
    // All six records are pushed first (the original store stays outstanding
    // while the duplicate is pushed — that is what makes it a duplicate);
    // completions are then consumed in PRESENTATION order below.
    cnt_snap = stale_o[0];
    req_go(0, mk_req(0, 16'd5, 32'd80, 32'h0001_0000, 32'h1111_2222, 3'd4,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S8 original store");
    req_go(0, mk_req(0, 16'd5, 32'd80, 32'h0001_0000, 32'h0, 3'd4, 1'b1,
                     1'b0, 5'd1, 1'b0, 1'b0), "S8 duplicate rejected");
    req_go(0, mk_req(0, 16'd4, 32'd81, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S8 stale gen");
    req_go(0, mk_req(0, 16'd9, 32'd82, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S8 future gen");
    // Foreign identity: mutate BEFORE the offer so the DUT samples it.
    begin
      int guard;
      @(negedge clk);
      req_valid_i[0] = 1'b1;
      req_i[0] = mk_req(0, 16'd5, 32'd83, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                        1'b0, 5'd1, 1'b0, 1'b0);
      req_i[0].owner.ctx_slot = 32'd1;
      guard = 0;
      while (req_ready_o[0] !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      @(posedge clk);
      @(negedge clk);
      req_valid_i[0] = 1'b0;
    end
    req_go(0, mk_req(0, 16'd5, 32'd84, 32'h0001_0000, 32'h0, 3'd3, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S8 bad byte_count");
    // Consume in PRESENTATION order (terminal-reach order — the parked
    // presenter never re-orders and never preempts mid-stall): classify
    // each completion against the expected set and require every class
    // exactly once. The dup fault parks before the original store settles.
    begin : s8_consume
      int guard;
      int got_orig, got_dup, got_stale, got_future, got_foreign, got_bc3;
      got_orig = 0;
      got_dup = 0;
      got_stale = 0;
      got_future = 0;
      got_foreign = 0;
      got_bc3 = 0;
      repeat (6) begin
        guard = 0;
        while (rsp_valid_o[0] !== 1'b1 && guard < 4000) begin
          @(negedge clk);
          guard = guard + 1;
        end
        if (guard >= 4000) begin
          errors = errors + 1;
          $error("C031_ASSERT tb_lsu_mslot S8: completion never came");
        end
        checks = checks + 1;
        if (rsp_o[0].final_state == COMPLETED && rsp_o[0].error == ERR_NONE
            && rsp_o[0].owner.owner_op_seq == 32'd80) begin
          got_orig = got_orig + 1;
          check_val(rsp_o[0].data, 32'd0, "S8 original store data path");
        end else if (rsp_o[0].final_state == FAULTED
                     && rsp_o[0].error == ERR_OWNER
                     && rsp_o[0].owner.owner_op_seq == 32'd80) begin
          got_dup = got_dup + 1;
        end else if (rsp_o[0].final_state == FAULTED
                     && rsp_o[0].error == ERR_OWNER
                     && rsp_o[0].owner.owner_op_seq == 32'd81) begin
          got_stale = got_stale + 1;
        end else if (rsp_o[0].final_state == FAULTED
                     && rsp_o[0].error == ERR_OWNER
                     && rsp_o[0].owner.owner_op_seq == 32'd82) begin
          got_future = got_future + 1;
        end else if (rsp_o[0].final_state == FAULTED
                     && rsp_o[0].error == ERR_OWNER
                     && rsp_o[0].owner.owner_op_seq == 32'd83) begin
          got_foreign = got_foreign + 1;
        end else if (rsp_o[0].final_state == FAULTED
                     && rsp_o[0].error == ERR_RANGE
                     && rsp_o[0].owner.owner_op_seq == 32'd84) begin
          got_bc3 = got_bc3 + 1;
        end else begin
          errors = errors + 1;
          $error("C031_ASSERT tb_lsu_mslot S8: unexpected class final=%0d err=%0d seq=%0d",
                 rsp_o[0].final_state, rsp_o[0].error,
                 rsp_o[0].owner.owner_op_seq);
        end
        rsp_ready_i[0] = 1'b1;
        @(posedge clk);
        @(negedge clk);
        rsp_ready_i[0] = 1'b0;
      end
      check_val(32'(got_orig), 32'd1, "S8 original consumed once");
      check_val(32'(got_dup), 32'd1, "S8 duplicate consumed once");
      check_val(32'(got_stale), 32'd1, "S8 stale consumed once");
      check_val(32'(got_future), 32'd1, "S8 future consumed once");
      check_val(32'(got_foreign), 32'd1, "S8 foreign consumed once");
      check_val(32'(got_bc3), 32'd1, "S8 bc3 consumed once");
    end
    check_val64(stale_o[0] - cnt_snap, 64'd4, "S8 stale accounting");

    // ================================================= S9 cancel battery
    // Rel 0 gets a known word so later loads are C1/C2 independent.
    req_go(0, mk_req(0, 16'd5, 32'd89, 32'h0001_0000, REL0_WORD, 3'd4,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S9 rel0 rewrite");
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd89, "S9 rel0 rewrite done", 0);
    // (a) unissued -> ISOLATED with zero bank activity.
    issue_enable_i = 1'b0;
    req_go(0, mk_req(0, 16'd5, 32'd90, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9a unissued load");
    check_val({16'd0, bank_busy_o}, '0, "S9a no bank hold while gated");
    cancel_go(0, 16'd5);
    issue_enable_i = 1'b1;
    rsp_take(0, ISOLATED, ERR_NONE, 32'd0, 32'd90, "S9a isolated", 0);
    check_val({16'd0, bank_busy_o}, '0, "S9a bank still idle");
    // (b) mid-write cancel -> COMPLETED with effects.
    req_go(0, mk_req(0, 16'd5, 32'd91, 32'h0001_00DE, 32'h5566_7788, 3'd4,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S9b mid-write store");
    cancel_go(0, 16'd5);
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd91, "S9b store completed", 0);
    req_go(0, mk_req(0, 16'd5, 32'd92, 32'h0001_00DE, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9b readback");
    rsp_take(0, COMPLETED, ERR_NONE, 32'h5566_7788, 32'd92,
             "S9b effects landed", 0);
    // (c) mid-read cancel -> drain then ISOLATED.
    req_go(0, mk_req(0, 16'd5, 32'd93, 32'h0001_00DE, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9c mid-read load");
    cancel_go(0, 16'd5);
    rsp_take(0, ISOLATED, ERR_NONE, 32'h5566_7788, 32'd93,
             "S9c drained then isolated", 0);
    check_val({16'd0, bank_busy_o}, '0, "S9c bank freed after drain");
    // (d) same-edge request + effective cancel: admission bubble.
    cnt_snap = accepted_o[0];
    @(negedge clk);
    req_valid_i[0] = 1'b1;
    req_i[0] = mk_req(0, 16'd5, 32'd94, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                      1'b0, 5'd1, 1'b0, 1'b0);
    cancel_i[0] = 1'b1;
    cancel_generation_i[0] = 16'd5;
    @(posedge clk);
    #1;
    check_bit(req_ready_o[0], 1'b0, "S9d bubble: no admission at cancel edge");
    @(negedge clk);
    cancel_i[0] = 1'b0;
    check_val64(accepted_o[0], cnt_snap, "S9d no leaked acceptance");
    @(posedge clk);
    @(negedge clk);
    req_valid_i[0] = 1'b0;
    rsp_take(0, COMPLETED, ERR_NONE, REL0_WORD, 32'd94,
             "S9d admitted after bubble", 0);
    check_val(outstanding_o[0], 32'd0, "S9d no leaked record");
    // (e) same-generation work continues WITHOUT reconfigure.
    req_go(0, mk_req(0, 16'd5, 32'd95, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9e same-gen after cancel");
    rsp_take(0, COMPLETED, ERR_NONE, REL0_WORD, 32'd95,
             "S9e no sticky generation disable", 0);
    // (f) config refused until drain, then accepted; old gen dies.
    req_go(0, mk_req(0, 16'd5, 32'd96, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9f blocking load");
    cfg_table_ctx0;
    cfg_go(0, 16'd6, 1'b1, "S9f config rejected while outstanding");
    rsp_take(0, COMPLETED, ERR_NONE, REL0_WORD, 32'd96, "S9f drain", 0);
    cfg_go(0, 16'd6, 1'b0, "S9f config accepted after drain");
    req_go(0, mk_req(0, 16'd5, 32'd97, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9f old gen dead");
    rsp_take(0, FAULTED, ERR_OWNER, 32'd0, 32'd97, "S9f old gen fault", 0);
    req_go(0, mk_req(0, 16'd6, 32'd98, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9f new gen live");
    rsp_take(0, COMPLETED, ERR_NONE, REL0_WORD, 32'd98,
             "S9f new gen completes", 0);
    // (g) stale cancel counted no-op; double cancel harmless.
    cnt_snap = stale_o[0];
    cancel_go(0, 16'd999);
    check_val64(stale_o[0] - cnt_snap, 64'd1, "S9g stale cancel counted");
    cancel_go(0, 16'd6);
    cancel_go(0, 16'd6);
    req_go(0, mk_req(0, 16'd6, 32'd99, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S9g ingress after cancels");
    rsp_take(0, COMPLETED, ERR_NONE, REL0_WORD, 32'd99,
             "S9g ingress unaffected", 0);

    // ================================================= S10 address faults
    req_go(0, mk_req(0, 16'd6, 32'd110, 32'h0007_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S10 outside");
    rsp_take(0, FAULTED, ERR_RANGE, 32'd0, 32'd110, "S10 outside fault", 0);
    req_go(0, mk_req(0, 16'd6, 32'd111, 32'h0001_0FFE, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S10 window-end crossing");
    rsp_take(0, FAULTED, ERR_RANGE, 32'd0, 32'd111, "S10 window-end fault", 0);
    req_go(0, mk_req(0, 16'd6, 32'd112, 32'h0001_0000, 32'h0, 3'd0, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S10 zero byte_count");
    rsp_take(0, FAULTED, ERR_RANGE, 32'd0, 32'd112, "S10 bc0 fault", 0);
    req_go(0, mk_req(0, 16'd6, 32'd113, 32'h0002_0000, 32'h0, 3'd4, 1'b1,
                     1'b0, 5'd1, 1'b0, 1'b0), "S10 RO write");
    rsp_take(0, FAULTED, ERR_PERMISSION, 32'd0, 32'd113, "S10 RO fault", 0);
    req_go(0, mk_req(0, 16'd6, 32'd114, 32'h0003_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S10 WO read");
    rsp_take(0, FAULTED, ERR_PERMISSION, 32'd0, 32'd114, "S10 WO fault", 0);
    req_go(0, mk_req(0, 16'd6, 32'd115, 32'h0006_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S10 reserved window");
    rsp_take(0, FAULTED, ERR_RANGE, 32'd0, 32'd115, "S10 reserved fault", 0);
    check_val({16'd0, bank_busy_o}, '0, "S10 faults left no bank hold");
    req_go(0, mk_req(0, 16'd6, 32'd116, 32'h0001_0000, 32'h0, 3'd2, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b1), "S10 fence with bc");
    rsp_take(0, FAULTED, ERR_RANGE, 32'd0, 32'd116, "S10 fence bc fault", 0);

    // ================================================= S11 fence semantics
    begin
      // Older load (bank 2 via rel 128) mid-flight, fence, younger store to
      // bank 10 (rel 640) that must NOT launch past the fence.
      issue_enable_i = 1'b0;
      req_go(0, mk_req(0, 16'd6, 32'd120, 32'h0001_0080, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd1, 1'b0, 1'b0), "S11 older load");
      req_go(0, mk_req(0, 16'd6, 32'd121, 32'h0, 32'h0, 3'd0, 1'b0, 1'b0,
                       5'd0, 1'b0, 1'b1), "S11 fence");
      req_go(0, mk_req(0, 16'd6, 32'd122, 32'h0001_0280, 32'h5EED_0001, 3'd4,
                       1'b1, 1'b0, 5'd2, 1'b0, 1'b0), "S11 younger store");
      if (CONTEXT_COUNT == 2) begin
        req_go(1, mk_req(1, 16'd7, 32'd123, 32'h0001_0000, 32'h0BAD_0003,
                         3'd4, 1'b1, 1'b0, 5'd1, 1'b0, 1'b0),
               "S11 peer store");
      end
      issue_enable_i = 1'b1;
      @(negedge clk);
      // Only the older load holds a bank; the younger store (bank 10) must
      // not hold its bank while the fence is queued behind the load.
      check_bit(bank_busy_o[2], 1'b1, "S11 older load holds its bank");
      check_bit(bank_busy_o[10], 1'b0, "S11 younger store gated by fence");
      if (CONTEXT_COUNT == 2) begin
        rsp_take(1, COMPLETED, ERR_NONE, 32'd0, 32'd123,
                 "S11 peer unblocked by fence", 0);
      end
      rsp_take(0, COMPLETED, ERR_NONE, 32'hF00D_0000, 32'd120,
               "S11 older load", 0);
      rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd121, "S11 fence", 0);
      rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd122,
               "S11 younger after fence", 0);
      // Fence and older op cancelled together: load drains, fence isolated.
      req_go(0, mk_req(0, 16'd6, 32'd124, 32'h0001_00DE, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd1, 1'b0, 1'b0), "S11 cancel older");
      req_go(0, mk_req(0, 16'd6, 32'd125, 32'h0, 32'h0, 3'd0, 1'b0, 1'b0,
                       5'd0, 1'b0, 1'b1), "S11 cancel fence");
      cancel_go(0, 16'd6);
      // Presentation order is terminal-reach order: the queued fence
      // isolates at the cancel edge, the issued read drains first.
      rsp_take(0, ISOLATED, ERR_NONE, 32'd0, 32'd125,
               "S11 cancelled fence isolated", 0);
      rsp_take(0, ISOLATED, ERR_NONE, 32'h5566_7788, 32'd124,
              "S11 cancelled load isolated", 0);
      // Structural fence kill (order-independent): an unserved miss keeps
      // the fence queued forever; the younger SPM store must NEVER launch
      // (bank bit) and NO completion may present during the whole window.
      begin
        logic [3:0] rid_k;
        cache_req_ready_i[0] = 1'b1;
        cache_push(0, mk_req(0, 16'd6, 32'd126, 32'h0004_0028, 32'h0, 3'd4,
                             1'b0, 1'b0, 5'd3, 1'b0, 1'b0), rid_k,
                   "S11 kill miss");
        cache_req_ready_i[0] = 1'b0;
        req_go(0, mk_req(0, 16'd6, 32'd127, 32'h0, 32'h0, 3'd0, 1'b0, 1'b0,
                         5'd0, 1'b0, 1'b1), "S11 kill fence");
        req_go(0, mk_req(0, 16'd6, 32'd128, 32'h0001_0280, 32'hF1F1_0002,
                         3'd4, 1'b1, 1'b0, 5'd2, 1'b0, 1'b0),
               "S11 kill younger store");
        repeat (20) begin
          @(negedge clk);
          check_bit(bank_busy_o[10], 1'b0,
                    "S11 kill: younger never launches past queued fence");
          check_bit(rsp_valid_o[0], 1'b0,
                    "S11 kill: no completion presents before the fence");
        end
        cache_serve(0, rid_k, COMPLETED, ERR_NONE, 32'h0DD0_0126,
                    "S11 kill serve");
        rsp_take(0, COMPLETED, ERR_NONE, 32'h0DD0_0126, 32'd126,
                 "S11 kill miss done", 0);
        rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd127,
                 "S11 kill fence done", 0);
        rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd128,
                 "S11 kill store done", 0);
      end
    end

    // ================================================= S12 cache channel
    begin
      logic [3:0] rid_a, rid_b, rid_c, rid_d;
      logic [3:0] rids [16];
      logic [63:0] wm_before;
      // Payload check under !ready, then cancel of the unlaunched record.
      req_go(0, mk_req(0, 16'd6, 32'd130, 32'h0004_0008, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd3, 1'b0, 1'b0), "S12 retained cached read");
      repeat (2) begin
        @(negedge clk);
        check_bit(cache_req_valid_o[0], 1'b1, "S12 cache launch offered");
        check_val(cache_req_o[0].owner.owner_op_seq, 32'd130,
                  "S12 payload owner");
        check_val(32'(cache_req_o[0].binding_id), 32'h4444_0000,
                  "S12 payload binding_id");
        check_val(32'(cache_req_o[0].binding_generation), 32'd2,
                  "S12 payload binding_generation");
        check_val(cache_req_o[0].offset, 32'h0200_0008, "S12 payload offset");
        check_val(32'(cache_req_o[0].arena_generation), 32'd5,
                  "S12 payload arena");
        check_val(32'(cache_req_o[0].scope_epoch), 32'd11,
                  "S12 payload scope");
        check_val(32'(cache_req_o[0].profile_generation), 32'd0,
                  "S12 payload profile generation");
        check_val(32'(cache_req_o[0].byte_count), 32'd4,
                  "S12 payload byte_count");
      end
      rid_a = cache_req_o[0].record_id;
      cancel_go(0, 16'd6);
      rsp_take(0, ISOLATED, ERR_NONE, 32'd0, 32'd130,
               "S12 unlaunched cancel isolated", 0);
      check_bit(cache_req_valid_o[0], 1'b0, "S12 launch retracted");
      // Served response with LSU-side extension.
      cache_req_ready_i[0] = 1'b1;
      cache_push(0, mk_req(0, 16'd6, 32'd131, 32'h0004_000C, 32'h0, 3'd4,
                           1'b0, 1'b0, 5'd3, 1'b0, 1'b0), rid_b,
                 "S12 served push");
      wm_before = wait_mshr_o[0];
      repeat (2) @(negedge clk);
      checks = checks + 1;
      if (wait_mshr_o[0] <= wm_before) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot S12: wait_mshr never attributed");
      end
      cache_serve(0, rid_b, COMPLETED, ERR_NONE, 32'h9ABC_DEF0,
                  "S12 serve word");
      rsp_take(0, COMPLETED, ERR_NONE, 32'h9ABC_DEF0, 32'd131,
               "S12 served word", 0);
      cache_push(0, mk_req(0, 16'd6, 32'd132, 32'h0004_0010, 32'h0, 3'd2,
                           1'b0, 1'b0, 5'd4, 1'b0, 1'b0), rid_c,
                 "S12 served half signed");
      cache_serve(0, rid_c, COMPLETED, ERR_NONE, 32'h0000_AABB,
                  "S12 serve half");
      rsp_take(0, COMPLETED, ERR_NONE, 32'hFFFF_AABB, 32'd132,
               "S12 half signed", 0);
      cache_push(0, mk_req(0, 16'd6, 32'd133, 32'h0004_0010, 32'h0, 3'd2,
                           1'b0, 1'b1, 5'd4, 1'b0, 1'b0), rid_d,
                 "S12 served half unsigned");
      cache_serve(0, rid_d, COMPLETED, ERR_NONE, 32'h0000_AABB,
                  "S12 serve half unsigned");
      rsp_take(0, COMPLETED, ERR_NONE, 32'h0000_AABB, 32'd133,
               "S12 half unsigned", 0);
      // Cancelled waiter forwards the settled final state verbatim.
      cache_push(0, mk_req(0, 16'd6, 32'd134, 32'h0004_0014, 32'h0, 3'd4,
                           1'b0, 1'b0, 5'd3, 1'b0, 1'b0), rid_b,
                 "S12 cancelled waiter push");
      cancel_go(0, 16'd6);
      cache_serve(0, rid_b, ISOLATED, ERR_NONE, 32'h0, "S12 serve isolated");
      rsp_take(0, ISOLATED, ERR_NONE, 32'd0, 32'd134,
               "S12 cancelled waiter isolated", 0);
      // Same-Context out-of-order replay (parent C1 acceptance shape):
      // miss waits; younger SPM ops complete; second miss completes first.
      cache_push(0, mk_req(0, 16'd6, 32'd135, 32'h0004_0018, 32'h0, 3'd4,
                           1'b0, 1'b0, 5'd3, 1'b0, 1'b0), rid_b,
                 "S12 replay miss seq3");
      req_go(0, mk_req(0, 16'd6, 32'd136, 32'h0001_0300, 32'hD00D_0004, 3'd4,
                       1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S12 replay store seq4");
      rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd136,
               "S12 younger store completes first", 0);
      req_go(0, mk_req(0, 16'd6, 32'd137, 32'h0001_0300, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd2, 1'b0, 1'b0), "S12 replay load seq5");
      rsp_take(0, COMPLETED, ERR_NONE, 32'hD00D_0004, 32'd137,
               "S12 younger load completes", 0);
      cache_push(0, mk_req(0, 16'd6, 32'd138, 32'h0004_001C, 32'h0, 3'd4,
                           1'b0, 1'b0, 5'd3, 1'b0, 1'b0), rid_c,
                 "S12 replay miss seq6");
      cache_serve(0, rid_c, COMPLETED, ERR_NONE, 32'h600D_0006,
                  "S12 serve seq6 first");
      rsp_take(0, COMPLETED, ERR_NONE, 32'h600D_0006, 32'd138,
               "S12 seq6 before seq3", 0);
      cache_serve(0, rid_b, COMPLETED, ERR_NONE, 32'h3000_0003,
                  "S12 serve seq3 last");
      rsp_take(0, COMPLETED, ERR_NONE, 32'h3000_0003, 32'd135,
               "S12 seq3 completes", 0);
      // 16 cached records in flight, reverse-order service.
      for (int k = 0; k < 16; k++) begin
        cache_push(0, mk_req(0, 16'd6, 32'd140 + k, 32'h0004_0400 + 4 * k,
                             32'h0, 3'd4, 1'b0, 1'b0, 5'd3, 1'b0, 1'b0),
                   rids[k], "S12 stress push");
      end
      wait_out(0, 16, "S12 sixteen cached records in flight");
      check_bit(req_ready_o[0], 1'b0, "S12 credit exhausted (cached)");
      req_stall_check(0, mk_req(0, 16'd6, 32'd199, 32'h0004_0000, 32'h0,
                                3'd4, 1'b0, 1'b0, 5'd3, 1'b0, 1'b0),
                      "S12 17th cached");
      for (int k = 15; k >= 0; k--) begin
        cache_serve(0, rids[k], COMPLETED, ERR_NONE, 32'h57BE_5500 + k,
                    "S12 reverse serve");
        rsp_take(0, COMPLETED, ERR_NONE, 32'h57BE_5500 + k, 32'd140 + k,
                 "S12 reverse completion", (k == 15) ? 3 : 0);
      end
      cache_req_ready_i[0] = 1'b0;
      check_val(outstanding_o[0], 32'd0, "S12 cached stress drained");
    end

    // ================================================= S13 profile + cp
    prof_cmd(2'd1, 32'd1, 1'b1, 1'b0, "S13 profile 1 ack");
    check_val(32'(bank_profile_o), 32'd1, "S13 bank profile moved");
    prof_cmd(2'd1, 32'd1, 1'b0, 1'b1, "S13 non-increasing rejected");
    // Profile held across a bank hold: reject while busy, then ack.
    req_go(0, mk_req(0, 16'd6, 32'd150, 32'h0001_00DE, 32'hF0A0_00AA, 3'd4,
                     1'b1, 1'b0, 5'd1, 1'b0, 1'b0), "S13 hold store");
    begin : s13_hold
      int guard;
      int saw_rej;
      @(negedge clk);
      profile_valid_i = 1'b1;
      profile_index_i = 2'd2;
      profile_generation_i = 32'd2;
      saw_rej = 0;
      guard = 0;
      while (profile_ack_o !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        if (profile_reject_o === 1'b1) saw_rej = 1;
        guard = guard + 1;
      end
      if (guard >= 4000) begin
        errors = errors + 1;
        $error("C031_ASSERT tb_lsu_mslot S13: profile never acked");
      end
      checks = checks + 1;
      check_bit(saw_rej[0], 1'b1, "S13 profile rejected while bank busy");
      check_val(32'(bank_profile_o), 32'd2, "S13 profile 2 after ack");
      profile_valid_i = 1'b0;
    end
    rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd150, "S13 hold store done", 0);
    // Unlaunched cached read carries the acknowledged profile generation.
    begin
      req_go(0, mk_req(0, 16'd6, 32'd151, 32'h0004_0020, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd3, 1'b0, 1'b0), "S13 cached after profile");
      @(negedge clk);
      check_bit(cache_req_valid_o[0], 1'b1,
                "S13 cached launch offered (unlaunched)");
      check_val(32'(cache_req_o[0].profile_generation), 32'd2,
                "S13 payload carries profile generation");
      cancel_go(0, 16'd6);
      rsp_take(0, ISOLATED, ERR_NONE, 32'd0, 32'd151,
               "S13 cached cancelled", 0);
    end
    // cp data-port bridge: 2-beat refill write then readback, SPM untouched.
    begin : s13_cp
      logic [DATA_BYTES*8-1:0] beat0, beat1;
      int guard;
      beat0 = {DATA_BYTES{8'h5A}};
      beat1 = {DATA_BYTES{8'hA5}};
      @(negedge clk);
      cp_i.valid = 1'b1;
      cp_i.write = 1'b1;
      cp_i.bank = 4'd2;
      cp_i.address = 16'(57344 + 64);
      cp_i.beats = 12'd2;
      cp_i.wmode = 1'b0;
      cp_i.wben = {DATA_BYTES{1'b1}};
      // cp_o.ready depends combinationally on the just-driven cp_i fields;
      // settle within the same low phase before the first poll.
      #1;
      guard = 0;
      while (cp_o.ready !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      @(posedge clk);
      @(negedge clk);
      cp_i.valid = 1'b0;
      cp_i.wvalid = 1'b1;
      cp_i.wdata = beat0;
      @(posedge clk);
      @(negedge clk);
      cp_i.wdata = beat1;
      @(posedge clk);
      @(negedge clk);
      cp_i.wvalid = 1'b0;
      check_val({16'd0, bank_busy_o}, '0, "S13 cp never sets bank_busy_o");
      guard = 0;
      while (cp_o.done !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      check_bit(cp_o.done, 1'b1, "S13 cp write done pulse");
      // read back
      @(negedge clk);
      cp_i.valid = 1'b1;
      cp_i.write = 1'b0;
      cp_i.rready = 1'b1;
      @(posedge clk);
      @(negedge clk);
      cp_i.valid = 1'b0;
      guard = 0;
      while (cp_o.rvalid !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      check_val_wide(cp_o.rdata, beat0, "S13 cp read beat 0");
      @(negedge clk);
      guard = 0;
      while (cp_o.rvalid !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      check_val_wide(cp_o.rdata, beat1, "S13 cp read beat 1");
      @(negedge clk);
      cp_i.rready = 1'b0;
      guard = 0;
      while (cp_o.done !== 1'b1 && guard < 4000) begin
        @(negedge clk);
        guard = guard + 1;
      end
      check_bit(cp_o.done, 1'b1, "S13 cp read done pulse");
    end
    req_go(0, mk_req(0, 16'd6, 32'd152, 32'h0001_0000, 32'h0, 3'd4, 1'b0,
                     1'b0, 5'd1, 1'b0, 1'b0), "S13 SPM after profile");
    rsp_take(0, COMPLETED, ERR_NONE, REL0_WORD, 32'd152,
             "S13 SPM still serves", 0);

    // ================================================= S14 (C2) interleave
    if (CONTEXT_COUNT == 2) begin
      for (int k = 0; k < 8; k++) begin
        req_go(0, mk_req(0, 16'd6, 32'd160 + k, 32'h0001_0280 + 4 * k,
                         32'h1A00_0000 + k, 3'd4, 1'b1, 1'b0, 5'd1, 1'b0,
                         1'b0), "S14 ctx0 store");
        req_go(1, mk_req(1, 16'd7, 32'd180 + k, 32'h0001_0540 + 4 * k,
                         32'h1B01_0000 + k, 3'd4, 1'b1, 1'b0, 5'd1, 1'b0,
                         1'b0), "S14 ctx1 store");
      end
      for (int k = 0; k < 8; k++) begin
        rsp_take(0, COMPLETED, ERR_NONE, 32'd0, 32'd160 + k,
                 "S14 ctx0 completion", 0);
        rsp_take(1, COMPLETED, ERR_NONE, 32'd0, 32'd180 + k,
                 "S14 ctx1 completion", 0);
      end
      req_go(0, mk_req(0, 16'd6, 32'd170, 32'h0001_0280, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd2, 1'b0, 1'b0), "S14 ctx0 readback");
      rsp_take(0, COMPLETED, ERR_NONE, 32'h1A00_0000, 32'd170,
               "S14 ctx0 bytes intact", 0);
      req_go(1, mk_req(1, 16'd7, 32'd171, 32'h0001_0540, 32'h0, 3'd4, 1'b0,
                       1'b0, 5'd2, 1'b0, 1'b0), "S14 ctx1 readback");
      rsp_take(1, COMPLETED, ERR_NONE, 32'h1B01_0000, 32'd171,
               "S14 ctx1 bytes intact", 0);
    end

    // ================================================= S15 wrong record id
    begin
      logic [3:0] rid_w;
      cache_req_ready_i[0] = 1'b1;
      cache_push(0, mk_req(0, 16'd6, 32'd190, 32'h0004_0024, 32'h0, 3'd4,
                           1'b0, 1'b0, 5'd3, 1'b0, 1'b0), rid_w,
                 "S15 waiter push");
      cache_req_ready_i[0] = 1'b0;
      @(negedge clk);
      cache_rsp_valid_i[0] = 1'b1;
      cache_rsp_i[0].owner = '0;
      cache_rsp_i[0].record_id = rid_w ^ 4'h1;
      cache_rsp_i[0].data = 32'h0;
      cache_rsp_i[0].final_state = COMPLETED;
      cache_rsp_i[0].error = ERR_NONE;
      repeat (2) begin
        @(posedge clk);
        #1;
        check_bit(cache_rsp_ready_o[0], 1'b0,
                  "S15 mismatched record_id not accepted");
      end
      @(negedge clk);
      cache_rsp_valid_i[0] = 1'b0;
      check_bit(invariant_error_o, 1'b1, "S15 sticky invariant latched");
      // The cancelled waiter settles only through the miss owner: serve it.
      cancel_go(0, 16'd6);
      cache_serve(0, rid_w, ISOLATED, ERR_NONE, 32'h0,
                  "S15 cancelled waiter settle");
      rsp_take(0, ISOLATED, ERR_NONE, 32'd0, 32'd190,
               "S15 waiter isolated", 0);
      check_bit(invariant_error_o, 1'b1, "S15 invariant stays sticky");
    end

    // ================================================= final summary
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      check_val64(accepted_o[c], terminal_o[c],
                  "final accepted == terminal (conservation)");
      check_val(outstanding_o[c], 32'd0, "final outstanding zero");
    end
    check_val({16'd0, bank_busy_o}, '0, "final bank idle");
    if (errors == 0) begin
      $display("C031_PASS tb_lsu_mslot checks=%0d contexts=%0d",
               checks, CONTEXT_COUNT);
      $display("TB_LSU_MSLOT: unordered-oldest-ready completions; fence MEM-06; cancel per-record");
    end else begin
      $display("C031_FAIL tb_lsu_mslot errors=%0d checks=%0d", errors, checks);
      $fatal(1, "C031_ASSERT tb_lsu_mslot %0d failing checks", errors);
    end
    $finish;
  end

endmodule
