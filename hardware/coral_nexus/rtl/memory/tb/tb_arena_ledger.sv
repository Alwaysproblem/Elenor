// C029 end-to-end memory primitive bench: real ledger -> real Frames -> real
// bank bytes/validity. No fabricated clear ACKs or Python execution shadow.
`timescale 1ns/1ps
module tb_arena_ledger #(
  parameter int CONTEXT_COUNT = c029_params_pkg::C2_CONTEXTS
) ();
  localparam int BANKS = c029_params_pkg::BANK_COUNT;
  localparam int SLOTS = c029_params_pkg::SLOT_COUNT;
  localparam int DATA_BYTES = c029_params_pkg::DATA_BYTES;
  localparam int BANK_BYTES = c029_params_pkg::BANK_BYTES;
  localparam int BYTE_AW = $clog2(BANK_BYTES) + 1;
  localparam int ADDR_AW = $clog2(BANKS * BANK_BYTES) + 1;
  localparam int SLOT_AW = (SLOTS <= 1) ? 1 : $clog2(SLOTS);
  localparam int CTX_AW = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT);
  localparam int BANK_AW = (BANKS <= 1) ? 1 : $clog2(BANKS);
  localparam int SEG_AW = $clog2(BANKS + 1);
  localparam int CAPACITY = c029_params_pkg::SPM_BYTES_PER_BANK[0] -
                           c029_params_pkg::SYSTEM_RESERVED_BYTES;
  localparam int STRIPE = 4096;
  localparam int START = ((c029_params_pkg::SYSTEM_RESERVED_BYTES + STRIPE - 1) / STRIPE) * STRIPE;
  localparam int ROUND_BYTES = BANKS * STRIPE;
  localparam int TAIL_BYTES = DATA_BYTES + 5;
  localparam logic [3:0] HOLD = 0, PREPARE = 1, COMMIT = 2, COMPENSATE = 3,
    BIND = 4, FREE = 5, PIN = 6, UNPIN = 7, BEGIN_ACCESS = 8, END_ACCESS = 9,
    RETIRE = 10, COMPLETION_ACK = 11, FRAME_RELEASE = 12;
  localparam logic [6:0] OK = 0, COMPENSATED = 1, RETIRED = 2,
    BAD_STATE = 5, STALE = 6, BAD_BANK = 8, DUP_BANK = 9, BAD_SIZE = 10,
    PERMANENT = 11, WAIT_CAPACITY = 12, WAIT_FRAGMENTATION = 13, BAD_TOTAL = 14,
    POST_AUTH = 16, NOT_PREPARED = 17, BAD_SLOT = 20, SLOT_BUSY = 21,
    VIEW_OVERLAP = 22, VIEW_ESCAPE = 23, VIEW_LAYOUT = 24, UNKNOWN_VIEW = 25,
    DOUBLE_FREE = 26, LIVE_REFS = 27, STALE_VIEW = 28, NOT_DRAINED = 30,
    ALREADY_RETIRED = 31, BAD_COMPLETION = 32, UNDERFLOW = 35,
    BAD_PROFILE = 36, FRAME_FAULT = 38;
  localparam logic [DATA_BYTES*8-1:0] WORD_A = {DATA_BYTES{8'hA5}},
    WORD_B = {DATA_BYTES{8'h3C}}, WORD_LAST = {DATA_BYTES{8'h7E}};

  logic clk = 0, rst_n = 0;
  logic cmd_valid = 0, cmd_ready, rec_valid = 0, rec_ready, rsp_valid, rsp_ready = 0;
  logic [3:0] cmd_op = 0;
  logic [CTX_AW-1:0] cmd_ctx = 0, rsp_ctx;
  logic [63:0] cmd_ctx_generation = 0, cmd_ticket = 0, cmd_profile_generation = 7;
  logic [63:0] cmd_owner = 0, cmd_view_alloc_id = 0, cmd_arena_alloc_gen = 0;
  logic [ADDR_AW-1:0] cmd_reserved_bytes = 0, cmd_logical = 0, cmd_arena_offset = 0, cmd_base_addr = 0;
  logic [BYTE_AW-1:0] cmd_alignment = 0, cmd_stripe = 0;
  logic [BANK_AW:0] cmd_seg_count = 0, rec_bank = 0;
  logic [63:0] cmd_layout_count = 0;
  logic [SLOTS-1:0] cmd_slot_mask = 0;
  logic [SLOT_AW:0] cmd_slot = 0, rec_slot = 0;
  logic [BYTE_AW-1:0] rec_size = 0;
  logic [ADDR_AW-1:0] rec_arena_offset = 0, rec_logical = 0;
  logic [6:0] rsp_code;
  logic [63:0] rsp_latency, rsp_arena_gen, rsp_view_alloc_id, profile_generation_i = 7;
  logic [CONTEXT_COUNT-1:0] exec_drained = 0, completion_valid, retire_ready;
  logic [CONTEXT_COUNT*64-1:0] exec_drained_generation_i = 0;
  logic [CONTEXT_COUNT*64-1:0] completion_ctx_generation, completion_arena_gen, completion_ticket;
  logic [CONTEXT_COUNT-1:0] inv_valid_o, inv_ready_i, inv_ack_i, inv_rej_i;
  logic [CONTEXT_COUNT*64-1:0] inv_alloc_o, inv_gen_o;
  logic [CONTEXT_COUNT*(BANK_AW+1)-1:0] inv_bank_o;
  logic [CONTEXT_COUNT*BYTE_AW-1:0] inv_addr_o, inv_len_o;
  logic [CONTEXT_COUNT*(SLOT_AW+1)-1:0] query_slot_i = 0;
  logic [CONTEXT_COUNT-1:0] query_valid_o;
  logic [CONTEXT_COUNT*64-1:0] query_alloc_o, query_owner_o, query_arena_gen_o;
  logic [CONTEXT_COUNT*64-1:0] query_context_gen_o, query_profile_gen_o;
  logic [CONTEXT_COUNT*ADDR_AW-1:0] query_base_o, query_size_o;
  logic [BANKS*BYTE_AW-1:0] dbg_held_bytes, dbg_bound_bytes, dbg_free_bytes;
  logic [CONTEXT_COUNT*4-1:0] dbg_ctx_state, dbg_frame_state;
  logic [CONTEXT_COUNT*SLOTS-1:0] dbg_frame_live_mask, dbg_frame_shadow_mask;
  logic [CONTEXT_COUNT*BANKS-1:0] dbg_cres_valid;
  logic [CONTEXT_COUNT*BANKS*BYTE_AW-1:0] dbg_cres_start, dbg_cres_size;
  logic [CONTEXT_COUNT*SLOTS-1:0] dbg_view_live_mask, dbg_view_invalid_mask;
  logic [BANKS-1:0] dbg_place_valid;
  logic [BANKS*BYTE_AW-1:0] dbg_place_start;
  logic [63:0] dbg_arena_gen;
  logic [4:0] dbg_frame_fault;
  logic dbg_invariant_error;
  arena_ledger #(.CONTEXT_COUNT(CONTEXT_COUNT)) dut (.*);

  logic [CONTEXT_COUNT-1:0] b_req = 0, b_ready, b_reject, b_write = 0, b_done;
  logic [CONTEXT_COUNT-1:0][63:0] b_txn = 0, b_offer_gen, b_done_gen;
  logic [CONTEXT_COUNT-1:0][SEG_AW-1:0] b_count = 0;
  logic [CONTEXT_COUNT-1:0][BANKS-1:0][BANK_AW-1:0] b_bank = 0;
  logic [CONTEXT_COUNT-1:0][BANKS-1:0][BYTE_AW-2:0] b_addr = 0;
  logic [CONTEXT_COUNT-1:0][BANKS-1:0][BYTE_AW-1:0] b_len = 0;
  logic [CONTEXT_COUNT-1:0] b_wvalid = 0, b_wready, b_rvalid, b_rready = 0;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES*8-1:0] b_wdata = 0, b_rdata;
  logic [CONTEXT_COUNT-1:0][DATA_BYTES-1:0] b_ben;
  logic [CONTEXT_COUNT-1:0][63:0] b_rtxn, b_rgen, b_fin_txn = 0, b_fin_gen = 0;
  logic [CONTEXT_COUNT-1:0] b_fin = 0, b_fin_ready, b_fin_ack, b_fin_reject;
  logic [CONTEXT_COUNT-1:0][BANK_AW-1:0] b_inv_bank;
  logic [CONTEXT_COUNT-1:0][BYTE_AW-2:0] b_inv_addr;
  longint unsigned cycles = 0, accepted_cycle = 0, checks = 0, clear_acks = 0;
  logic [63:0] generations [CONTEXT_COUNT], tickets [CONTEXT_COUNT], arena_ids [CONTEXT_COUNT];
  logic [63:0] bank_generation [CONTEXT_COUNT];
  logic [63:0] next_ticket = 100, next_bank_txn = 1, last_arena = 0, last_view = 0;
  logic [63:0] view_a, view_b, rebound, old_generation, old_ticket, old_arena;
  logic [DATA_BYTES*8-1:0] sentinel;
  longint unsigned ack_before;
  int chunk, fragment_request;

  for (genvar c = 0; c < CONTEXT_COUNT; c++) begin : adapt
    assign b_inv_bank[c] = inv_bank_o[c*(BANK_AW+1) +: BANK_AW];
    assign b_inv_addr[c] = inv_addr_o[c*BYTE_AW +: BYTE_AW-1];
    always @(posedge clk) if (rst_n) begin
      if (inv_valid_o[c]) begin
        assert (64'(inv_bank_o[c*(BANK_AW+1) +: BANK_AW+1]) < 64'(BANKS) &&
                64'(inv_addr_o[c*BYTE_AW +: BYTE_AW]) < 64'(BANK_BYTES))
          else $fatal(1, "C029_ASSERT invalidation adapter may not truncate an out-of-range value");
      end
    end
  end
  l1_bank #(.CONTEXT_COUNT(CONTEXT_COUNT)) bank (
    .clk(clk), .rstn(rst_n),
    .req_valid_i(b_req), .req_ready_o(b_ready), .req_rej_o(b_reject), .req_rej_code_o(),
    .req_txn_i(b_txn), .req_generation_o(b_offer_gen), .req_write_i(b_write),
    .req_seg_count_i(b_count), .req_bank_i(b_bank), .req_addr_i(b_addr), .req_len_i(b_len),
    .req_done_o(b_done), .req_done_txn_o(), .req_done_generation_o(b_done_gen),
    .wdata_valid_i(b_wvalid), .wdata_ready_o(b_wready), .wdata_i(b_wdata),
    .rdata_valid_o(b_rvalid), .rdata_ready_i(b_rready), .rdata_txn_o(b_rtxn),
    .rdata_generation_o(b_rgen), .rdata_data_o(b_rdata), .rdata_ben_o(b_ben),
    .fin_valid_i(b_fin), .fin_txn_i(b_fin_txn), .fin_generation_i(b_fin_gen),
    .fin_ready_o(b_fin_ready), .fin_ack_o(b_fin_ack), .fin_rej_o(b_fin_reject), .fin_rej_code_o(),
    .inv_valid_i(inv_valid_o), .inv_ready_o(inv_ready_i), .inv_bank_i(b_inv_bank),
    .inv_addr_i(b_inv_addr), .inv_len_i(inv_len_o), .inv_ack_o(inv_ack_i),
    .inv_rej_o(inv_rej_i), .inv_rej_code_o(),
    .profile_set_valid_i(1'b0), .profile_index_i('0), .profile_generation_i(profile_generation_i),
    .profile_set_ready_o(), .profile_ack_o(), .profile_rej_o(), .profile_rej_code_o(),
    // C030 cache data port: the ledger bench never drives the cache master;
    // tie the request side off and report the cache idle/quiescent.
    .cp_valid_i(1'b0), .cp_write_i(1'b0), .cp_bank_i('0), .cp_addr_i('0),
    .cp_beats_i('0), .cp_wmode_i(1'b0), .cp_wdata_valid_i(1'b0), .cp_wdata_i('0),
    .cp_wben_i('0), .cp_rdata_ready_i(1'b0), .cache_quiescent_i(1'b1),
    .cp_ready_o(), .cp_rej_o(), .cp_rej_code_o(), .cp_wdata_ready_o(),
    .cp_rdata_valid_o(), .cp_rdata_o(), .cp_rben_o(), .cp_done_o(), .cp_busy_o(),
    .active_profile_o(), .profile_generation_o()
  );
  always #5 clk = ~clk;
  always @(posedge clk) begin
    cycles <= cycles + 1;
    if (rst_n) for (int c = 0; c < CONTEXT_COUNT; c++) if (inv_ack_i[c]) clear_acks++;
  end
  initial begin #100_000_000; $fatal(1, "C029_ASSERT integrated memory progress watchdog"); end

  task automatic require(input logic condition, input string label);
    checks++;
    if (condition !== 1'b1) $fatal(1, "C029_ASSERT tb_arena_ledger %s", label);
  endtask
  task automatic select_context(input int c);
    cmd_ctx = CTX_AW'(c); cmd_ctx_generation = generations[c]; cmd_ticket = tickets[c];
    cmd_arena_alloc_gen = arena_ids[c]; cmd_profile_generation = profile_generation_i;
    cmd_owner = 64'h700 + 64'(c);
  endtask
  task automatic header(input logic [3:0] op);
    @(negedge clk); cmd_op = op; cmd_valid = 1; #1;
    while (!cmd_ready) begin @(negedge clk); #1; end
    @(posedge clk); #1; accepted_cycle = cycles; cmd_valid = 0;
  endtask
  task automatic record_beat;
    @(negedge clk); rec_valid = 1; #1;
    while (!rec_ready && !rsp_valid) begin @(negedge clk); #1; end
    if (!rsp_valid) begin @(posedge clk); #1; end
    rec_valid = 0;
  endtask
  task automatic bank_record(input int number, input int bytes);
    rec_bank = (BANK_AW+1)'(number); rec_size = BYTE_AW'(bytes); record_beat();
  endtask
  task automatic layout_record(input int slot, input int offset, input int bytes);
    rec_slot = (SLOT_AW+1)'(slot); rec_arena_offset = ADDR_AW'(offset);
    rec_logical = ADDR_AW'(bytes); record_beat();
  endtask
  task automatic response(input logic [6:0] expected, input string label);
    logic [63:0] latency, arena, allocation;
    #1;
    while (!rsp_valid) begin @(posedge clk); #1; end
    require(rsp_code == expected, $sformatf("%s code expected=%0d got=%0d", label, expected, rsp_code));
    require(rsp_ctx == cmd_ctx, {label, ": response owner"});
    require(rsp_latency == cycles - accepted_cycle + 1, {label, ": measured latency"});
    latency = rsp_latency; arena = rsp_arena_gen; allocation = rsp_view_alloc_id;
    last_arena = arena; last_view = allocation;
    repeat (2) begin
      @(posedge clk); #1;
      require(rsp_valid && rsp_code == expected && rsp_latency == latency &&
              rsp_arena_gen == arena && rsp_view_alloc_id == allocation,
              {label, ": response retained"});
    end
    $display("C029_LATENCY op=%0d cycles=%0d %s", cmd_op, latency, label);
    @(negedge clk); rsp_ready = 1;
    @(posedge clk); #1; rsp_ready = 0;
  endtask
  task automatic operation(input logic [3:0] op, input logic [6:0] code, input string label);
    header(op); response(code, label);
  endtask
  task automatic hold_header(
    input int c, input int quantum, input int records, input int total, input logic [SLOTS-1:0] mask
  );
    select_context(c);
    cmd_ctx_generation = generations[c] + 1; cmd_ticket = next_ticket; next_ticket++;
    cmd_alignment = BYTE_AW'(quantum); cmd_stripe = BYTE_AW'(quantum);
    cmd_seg_count = (BANK_AW+1)'(records); cmd_reserved_bytes = ADDR_AW'(total);
    cmd_slot_mask = mask; header(HOLD);
  endtask
  task automatic hold_one(
    input int c, input int number, input int bytes, input logic [6:0] expected, input string label
  );
    hold_header(c, DATA_BYTES, 1, bytes, '0);
    if (!rsp_valid) bank_record(number, bytes);
    response(expected, label);
    if (expected == OK) begin
      generations[c] = cmd_ctx_generation; tickets[c] = cmd_ticket; arena_ids[c] = 0;
    end
  endtask
  task automatic account(input int number, input int held, input int bound, input string label);
    require(int'(dbg_held_bytes[number*BYTE_AW +: BYTE_AW]) == held &&
            int'(dbg_bound_bytes[number*BYTE_AW +: BYTE_AW]) == bound &&
            int'(dbg_free_bytes[number*BYTE_AW +: BYTE_AW]) == CAPACITY - held - bound,
            label);
  endtask
  task automatic query(input int c, input int slot, input logic [63:0] id, input int bytes, base);
    query_slot_i[c*(SLOT_AW+1) +: SLOT_AW+1] = (SLOT_AW+1)'(slot); #1;
    require(query_valid_o[c] && query_alloc_o[c*64 +: 64] == id &&
            query_arena_gen_o[c*64 +: 64] == arena_ids[c] &&
            query_context_gen_o[c*64 +: 64] == generations[c] &&
            query_profile_gen_o[c*64 +: 64] == profile_generation_i &&
            int'(query_base_o[c*ADDR_AW +: ADDR_AW]) == base &&
            int'(query_size_o[c*ADDR_AW +: ADDR_AW]) == bytes,
            "actual engine shadow lookup");
  endtask
  task automatic view_handle(input int slot, input logic [63:0] id, input int bytes, offset, base);
    cmd_slot = (SLOT_AW+1)'(slot); cmd_view_alloc_id = id;
    cmd_logical = ADDR_AW'(bytes); cmd_arena_offset = ADDR_AW'(offset);
    cmd_base_addr = ADDR_AW'(base);
  endtask

  // Single-word bank requests; bank's own bench proves compound/multibeat traffic.
  task automatic bank_start(input int c, input logic write_access, input int physical, bytes);
    require(bytes > 0 && (physical % DATA_BYTES) + bytes <= DATA_BYTES, "single-word stimulus");
    @(negedge clk);
    b_txn[c] = next_bank_txn; next_bank_txn++;
    b_write[c] = write_access; b_count[c] = SEG_AW'(1);
    b_bank[c][0] = BANK_AW'(physical / BANK_BYTES);
    b_addr[c][0] = (BYTE_AW-1)'(physical % BANK_BYTES); b_len[c][0] = BYTE_AW'(bytes);
    b_req[c] = 1; #1;
    while (!b_ready[c]) begin
      require(!b_reject[c], "legal Frame-derived bank request rejected");
      @(negedge clk); #1;
    end
    bank_generation[c] = b_offer_gen[c];
    @(posedge clk); #1; b_req[c] = 0;
  endtask
  task automatic bank_finish(input int c);
    while (!b_done[c]) begin @(negedge clk); #1; end
    require(b_done_gen[c] == bank_generation[c], "bank completion incarnation");
    @(negedge clk);
    b_fin[c] = 1; b_fin_txn[c] = b_txn[c]; b_fin_gen[c] = bank_generation[c];
    @(posedge clk); #1; b_fin[c] = 0;
    require(b_fin_ack[c] && !b_fin_reject[c], "bank holder released");
  endtask
  task automatic bank_write(input int c, input int physical, bytes,
                            input logic [DATA_BYTES*8-1:0] value);
    bank_start(c, 1, physical, bytes);
    @(negedge clk); b_wvalid[c] = 1; b_wdata[c] = value; #1;
    require(b_wready[c], "write holder ready");
    @(posedge clk); #1; b_wvalid[c] = 0;
    bank_finish(c);
  endtask
  task automatic bank_consume(input int c, input logic [DATA_BYTES*8-1:0] value,
                              input logic [DATA_BYTES-1:0] validity);
    while (!b_rvalid[c]) begin @(negedge clk); #1; end
    require(b_rtxn[c] == b_txn[c] && b_rgen[c] == bank_generation[c], "read provenance");
    require(b_ben[c] == validity, "physical byte validity after lifecycle operation");
    for (int k = 0; k < DATA_BYTES; k++)
      if (validity[k]) require(b_rdata[c][k*8 +: 8] == value[k*8 +: 8], "retained byte content");
    @(negedge clk); b_rready[c] = 1;
    @(posedge clk); #1; b_rready[c] = 0;
    bank_finish(c);
  endtask
  task automatic bank_read(input int c, input int physical,
                           input logic [DATA_BYTES*8-1:0] value,
                           input logic [DATA_BYTES-1:0] validity);
    bank_start(c, 0, physical, DATA_BYTES); bank_consume(c, value, validity);
  endtask

  initial begin
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      generations[c] = 0; tickets[c] = 0; arena_ids[c] = 0; bank_generation[c] = 0;
    end
    sentinel = '0; sentinel[5*8 +: 8] = 8'hA7;
    repeat (3) begin @(posedge clk); #1; end
    rst_n = 1;
    for (int b = 0; b < BANKS; b++) account(b, 0, 0, "initial capacity");

    // No partial reservations on malformed records or permanent failure.
    hold_header(0, DATA_BYTES, 1, DATA_BYTES, '0); bank_record(BANKS, DATA_BYTES);
    response(BAD_BANK, "invalid bank");
    hold_header(0, DATA_BYTES, 2, DATA_BYTES * 2, '0);
    bank_record(0, DATA_BYTES); bank_record(0, DATA_BYTES); response(DUP_BANK, "duplicate bank");
    hold_header(0, DATA_BYTES, 1, DATA_BYTES, '0); bank_record(0, DATA_BYTES - 1);
    response(BAD_SIZE, "non-stripe reservation");
    hold_header(0, DATA_BYTES, 1, DATA_BYTES + 1, '0); bank_record(0, DATA_BYTES);
    response(BAD_TOTAL, "reservation total mismatch");
    hold_one(0, 0, CAPACITY + DATA_BYTES, PERMANENT, "empty bank cannot fit request");
    account(0, 0, 0, "failed holds changed nothing");

    // A held physical Frame cannot be reserved by a second ticket (C1 and C2).
    hold_one(0, 0, DATA_BYTES, OK, "exclusive Frame hold");
    hold_one(0, 0, DATA_BYTES, BAD_STATE, "held Frame cannot be promised twice");
    select_context(0); operation(COMPENSATE, COMPENSATED, "preauth compensation");
    account(0, 0, 0, "compensation returns held capacity");

    if (CONTEXT_COUNT >= 2) begin
      hold_one(1, BANKS - 1, CAPACITY, OK, "peer holds last bank");
      hold_header(0, DATA_BYTES, 2, DATA_BYTES * 2, '0);
      bank_record(0, DATA_BYTES); bank_record(BANKS - 1, DATA_BYTES);
      response(WAIT_CAPACITY, "late bank failure is all-or-none");
      account(0, 0, 0, "failed compound hold did not reserve first bank");
      account(BANKS - 1, CAPACITY, 0, "peer hold remains intact");
      select_context(1); operation(COMPENSATE, COMPENSATED, "release peer hold");

      chunk = (CAPACITY / 3 / DATA_BYTES) * DATA_BYTES;
      hold_one(0, 0, chunk, OK, "fragment prefix");
      hold_one(1, 0, chunk, OK, "fragment middle");
      select_context(0); operation(COMPENSATE, COMPENSATED, "remove prefix");
      fragment_request = CAPACITY - 2 * chunk + DATA_BYTES;
      hold_one(0, 0, fragment_request, WAIT_FRAGMENTATION, "total free is not contiguous free");
      select_context(1); operation(COMPENSATE, COMPENSATED, "remove middle");
      hold_one(0, 0, fragment_request, OK, "released extents naturally coalesce");
      require(int'(dbg_cres_start[0 +: BYTE_AW]) == c029_params_pkg::SYSTEM_RESERVED_BYTES,
              "first-fit restarts at reserved floor");
      select_context(0); operation(COMPENSATE, COMPENSATED, "finish fragmentation case");
    end

    // Full source-shaped striped Arena, followed by real Frame installation.
    hold_header(0, STRIPE, BANKS, BANKS * 2 * STRIPE, SLOTS'(3));
    for (int b = 0; b < BANKS; b++) bank_record(b, 2 * STRIPE);
    response(OK, "striped Arena hold");
    generations[0] = cmd_ctx_generation; tickets[0] = cmd_ticket;
    for (int b = 0; b < BANKS; b++) account(b, 2 * STRIPE, 0, "held before authorization");
    select_context(0); operation(COMMIT, NOT_PREPARED, "no executable Frame before prepare");
    cmd_layout_count = 1;
    header(PREPARE); layout_record(2, 0, STRIPE); response(BAD_SLOT, "record outside held slot mask");
    require(dbg_frame_state[0 +: 4] == 0, "bad metadata released the staged Frame");
    header(PREPARE); layout_record(0, 1, STRIPE); response(FRAME_FAULT, "malformed immutable layout");
    operation(FRAME_RELEASE, OK, "recover staged Frame");
    cmd_layout_count = 64'(SLOTS + 3);
    header(PREPARE);
    for (int r = 0; r < SLOTS + 3; r++) begin
      if (r == 4) repeat (3) begin @(posedge clk); #1; end
      if (r >= SLOTS + 1) layout_record(1, ROUND_BYTES, TAIL_BYTES);
      else layout_record(0, 0, (r == 0) ? ROUND_BYTES : STRIPE);
    end
    response(OK, "program has more buffers than live Slots");
    operation(COMMIT, OK, "irrevocable final authorization");
    arena_ids[0] = last_arena; select_context(0);
    for (int b = 0; b < BANKS; b++) account(b, 0, 2 * STRIPE, "bound after authorization");
    operation(COMPENSATE, POST_AUTH, "authorized work cannot roll back");

    view_handle(0, 0, ROUND_BYTES, 0, 0);
    operation(BIND, OK, "bind full stripe round"); view_a = last_view;
    query(0, 0, view_a, ROUND_BYTES, START);
    bank_write(0, START, DATA_BYTES, WORD_A);
    bank_write(0, (BANKS - 1) * BANK_BYTES + START, DATA_BYTES, WORD_LAST);
    view_handle(1, 0, TAIL_BYTES, ROUND_BYTES, 0);
    operation(BIND, OK, "bind adjacent tail view"); view_b = last_view;
    require(view_b != view_a, "allocator issues distinct allocation IDs");
    query(0, 1, view_b, TAIL_BYTES, START + STRIPE);
    bank_write(0, START + STRIPE, DATA_BYTES, WORD_B);
    bank_write(0, START + STRIPE + DATA_BYTES, 5, WORD_B);
    // Privileged bank stimulus in padding detects a rounded-up over-clear.
    bank_write(0, START + STRIPE + TAIL_BYTES, 1, sentinel);

    view_handle(0, view_a, ROUND_BYTES, 0, START);
    operation(PIN, OK, "first pin"); operation(PIN, OK, "second pin");
    operation(BEGIN_ACCESS, OK, "accepted memory reference");
    operation(FREE, LIVE_REFS, "free blocked by references");
    operation(UNPIN, OK, "first unpin"); operation(UNPIN, OK, "second unpin");
    operation(UNPIN, UNDERFLOW, "duplicate unpin cannot underflow");
    cmd_profile_generation++;
    operation(END_ACCESS, STALE_VIEW, "wrong profile cannot close another reference");
    cmd_profile_generation--;
    operation(FREE, LIVE_REFS, "stale return did not drain live access");
    operation(END_ACCESS, OK, "real access returned");
    operation(END_ACCESS, UNDERFLOW, "duplicate access return");
    operation(FREE, OK, "free after real bank clear ACKs");
    for (int b = 0; b < BANKS; b++) account(b, 0, 2 * STRIPE, "view free does not return Arena");
    bank_read(0, START, '0, '0);
    bank_read(0, (BANKS - 1) * BANK_BYTES + START, '0, '0);
    bank_read(0, START + STRIPE, WORD_B, '1);
    operation(FREE, DOUBLE_FREE, "double view free");

    view_handle(0, 0, STRIPE, 0, 0);
    operation(BIND, OK, "new lifetime changes view shape"); rebound = last_view;
    require(rebound != view_a && rebound != view_b, "new lifetime gets fresh identity");
    query(0, 0, rebound, STRIPE, START);
    bank_read(0, START, '0, '0);
    view_handle(0, view_a, ROUND_BYTES, 0, START);
    operation(FREE, STALE_VIEW, "old handle cannot free rebound slot");
    view_handle(0, rebound, STRIPE, 0, START);
    operation(FREE, OK, "free rebound view");

    // A real bank read is still held/backpressured while FREE is attempted.
    view_handle(1, view_b, TAIL_BYTES, ROUND_BYTES, START + STRIPE);
    operation(BEGIN_ACCESS, OK, "retain tail access");
    bank_start(0, 0, START + STRIPE, DATA_BYTES);
    ack_before = clear_acks;
    operation(FREE, LIVE_REFS, "held read blocks cleanup");
    require(clear_acks == ack_before, "rejected free started no invalidation");
    bank_consume(0, WORD_B, '1);
    operation(END_ACCESS, OK, "bank phase settled before reference return");
    operation(FREE, OK, "exact tail invalidation");
    bank_read(0, START + STRIPE, '0, '0);
    bank_read(0, START + STRIPE + DATA_BYTES, sentinel, DATA_BYTES'(1) << 5);

    require(!retire_ready[0], "empty views alone do not imply execution drained");
    operation(RETIRE, NOT_DRAINED, "retire without execution drain");
    exec_drained[0] = 1;
    exec_drained_generation_i[0 +: 64] = generations[0] - 1;
    #1; require(!retire_ready[0], "old-generation drain is not current drainage");
    operation(RETIRE, NOT_DRAINED, "stale drain rejected");
    exec_drained_generation_i[0 +: 64] = generations[0];
    #1; require(retire_ready[0], "current-generation complete drainage");
    operation(RETIRE, RETIRED, "authorized local retirement");
    for (int b = 0; b < BANKS; b++) account(b, 0, 0, "retirement returns every reserved bank");
    bank_read(0, START + STRIPE + DATA_BYTES, '0, '0);
    old_generation = generations[0]; old_ticket = tickets[0]; old_arena = arena_ids[0];
    repeat (8) begin
      @(posedge clk); #1;
      require(completion_valid[0] && completion_ctx_generation[0 +: 64] == old_generation &&
              completion_ticket[0 +: 64] == old_ticket &&
              completion_arena_gen[0 +: 64] == old_arena,
              "completion retained without parent-resource release");
    end
    hold_one(0, 0, DATA_BYTES, BAD_STATE, "no Context reuse before completion consumption");
    select_context(0); cmd_ticket++;
    operation(COMPLETION_ACK, BAD_COMPLETION, "wrong completion identity");
    select_context(0); operation(COMPLETION_ACK, OK, "consume exact completion");
    require(!completion_valid[0], "completion consumed once");
    exec_drained[0] = 0;

    // Equal-generation replay must not reinstall a retired Context.
    select_context(0); cmd_ticket = next_ticket; next_ticket++;
    cmd_seg_count = 0; cmd_reserved_bytes = 0; cmd_slot_mask = 0;
    operation(HOLD, STALE, "consumed Context generation cannot be reused");
    hold_header(0, STRIPE, 0, 0, '0); response(OK, "empty Arena hold");
    generations[0] = cmd_ctx_generation; tickets[0] = cmd_ticket; select_context(0);
    cmd_layout_count = 0; header(PREPARE); response(OK, "empty Frame prepare");
    operation(COMMIT, OK, "empty Frame commit"); arena_ids[0] = last_arena;
    select_context(0);
    require(arena_ids[0] != old_arena, "new Arena identity even without views");
    cmd_arena_alloc_gen = old_arena;
    operation(RETIRE, STALE, "old Arena retirement isolated");
    select_context(0);
    exec_drained[0] = 1; exec_drained_generation_i[0 +: 64] = generations[0];
    ack_before = clear_acks;
    operation(RETIRE, RETIRED, "empty Arena retirement");
    require(clear_acks == ack_before, "empty Arena issues no fictitious clear");
    operation(RETIRE, ALREADY_RETIRED, "duplicate retirement");
    operation(COMPLETION_ACK, OK, "empty completion consumed");
    for (int b = 0; b < BANKS; b++) account(b, 0, 0, "final capacity");
    require(!dbg_invariant_error, "ledger and Frame invariants remain intact");
    $display("C029_PASS tb_arena_ledger checks=%0d context_count=%0d clear_acks=%0d",
             checks, CONTEXT_COUNT, clear_acks);
    $finish;
  end
endmodule
