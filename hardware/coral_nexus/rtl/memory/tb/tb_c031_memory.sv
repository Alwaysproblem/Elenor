// C031: real LSU + bank + cache/MSHR/outbridge + host C023 MemoryService.
// Transport delays/reordering below affect delivery, never fabricate bytes or
// terminal closure. The native phase uses a real generated full C026 ROB.
`timescale 1ns/1ps
module tb_c031_memory;
  import c029_params_pkg::*;
  import c031_lsu_pkg::*;
  parameter int CONTEXT_COUNT = C2_CONTEXTS;
  localparam int PEER = CONTEXT_COUNT == 2 ? 1 : 0;
  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;
  logic [CONTEXT_COUNT-1:0] req_valid, req_ready, rsp_valid, rsp_ready;
  request_t req [CONTEXT_COUNT];
  completion_t rsp [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] cancel;
  logic [15:0] cancel_gen [CONTEXT_COUNT];
  logic config_valid, config_ready, config_reject;
  logic [31:0] config_ctx;
  logic [15:0] config_gen;
  window_t windows [SLOT_COUNT];
  logic profile_valid, profile_ack, profile_reject;
  logic [$clog2(PROFILE_COUNT)-1:0] bank_profile;
  logic cache_config_valid, cache_config_ready, cache_config_done, cache_config_error;
  logic [CONTEXT_COUNT-1:0] cr_valid, cr_ready, cs_valid, cs_ready;
  cache_request_t cr [CONTEXT_COUNT];
  cache_response_t cs [CONTEXT_COUNT];
  cache_port_request_t cp_req;
  cache_port_response_t cp_rsp;
  logic cache_quiescent;
  logic mem_valid, mem_ready, mr_valid, mr_ready, term_valid, term_ready;
  memory_request_t mem;
  memory_response_t mr;
  memory_terminal_t term;
  logic [31:0] outstanding [CONTEXT_COUNT], cache_outstanding;
  logic [63:0] accepted [CONTEXT_COUNT], terminal [CONTEXT_COUNT];
  logic [63:0] wait_bank [CONTEXT_COUNT], wait_cache [CONTEXT_COUNT], stale [CONTEXT_COUNT];
  logic [63:0] merged;
  logic [63:0] mshr_wait [CONTEXT_COUNT];
  logic [BANK_COUNT-1:0] bank_busy;
  logic lsu_error, cache_error;
  logic issue_enable = 1;

  lsu_mslot #(.CONTEXT_COUNT(CONTEXT_COUNT)) dut (
    .clk, .rst_n, .req_valid_i(req_valid), .req_ready_o(req_ready), .req_i(req),
    .rsp_valid_o(rsp_valid), .rsp_ready_i(rsp_ready), .rsp_o(rsp),
    .cancel_i(cancel), .cancel_generation_i(cancel_gen),
    .config_valid_i(config_valid), .config_ready_o(config_ready), .config_reject_o(config_reject),
    .config_context_i(config_ctx), .config_generation_i(config_gen), .config_windows_i(windows),
    .issue_enable_i(issue_enable), .cp_i(cp_req), .cp_o(cp_rsp), .cache_quiescent_i(cache_quiescent),
    .profile_valid_i(profile_valid), .profile_index_i(2'd1), .profile_generation_i(32'd1),
    .profile_ack_o(profile_ack), .profile_reject_o(profile_reject), .bank_profile_o(bank_profile),
    .cache_req_valid_o(cr_valid), .cache_req_ready_i(cr_ready), .cache_req_o(cr),
    .cache_rsp_valid_i(cs_valid), .cache_rsp_ready_o(cs_ready), .cache_rsp_i(cs),
    .outstanding_o(outstanding), .accepted_o(accepted), .terminal_o(terminal),
    .wait_bank_o(wait_bank), .wait_mshr_o(wait_cache), .stale_o(stale),
    .bank_busy_o(bank_busy), .invariant_error_o(lsu_error)
  );
  lsu_cache_path #(.CONTEXT_COUNT(CONTEXT_COUNT)) path (
    .clk, .rst_n, .req_valid_i(cr_valid), .req_ready_o(cr_ready), .req_i(cr),
    .rsp_valid_o(cs_valid), .rsp_ready_i(cs_ready), .rsp_o(cs),
    .cancel_i(cancel), .cancel_generation_i(cancel_gen),
    .config_valid_i(cache_config_valid), .config_ready_o(cache_config_ready),
    .config_done_o(cache_config_done), .config_error_o(cache_config_error),
    .config_profile_i(2'd1), .config_generation_i(32'd1), .bank_profile_i(bank_profile),
    .cp_o(cp_req), .cp_i(cp_rsp), .cache_quiescent_o(cache_quiescent),
    .mem_req_valid_o(mem_valid), .mem_req_ready_i(mem_ready), .mem_req_o(mem),
    .mem_rsp_valid_i(mr_valid), .mem_rsp_ready_o(mr_ready), .mem_rsp_i(mr),
    .term_valid_o(term_valid), .term_ready_i(term_ready), .term_o(term),
    .outstanding_o(cache_outstanding), .wait_mshr_o(mshr_wait), .merged_o(merged),
    .invariant_error_o(cache_error)
  );

  logic native_mode = 0, rob_enq_valid = 0, rob_complete_valid;
  request_t rob_enq;
  completion_t rob_complete;
  logic inject_completion = 0;
  completion_t injected_completion;
  logic [15:0] generations [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] rob_empty;
  logic [3:0] rob_retired [CONTEXT_COUNT];
  logic [15:0] rob_stale;
  logic [CONTEXT_COUNT-1:0] response_enable;
  int rob_select;
  c031_rob_adapter #(.CONTEXT_COUNT(CONTEXT_COUNT)) rob (
    .clk, .rst_n, .enq_valid_i(rob_enq_valid && req_ready[rob_enq.owner.ctx_slot]), .enq_i(rob_enq),
    .completion_valid_i(rob_complete_valid), .completion_i(rob_complete),
    .generation_i(generations), .empty_o(rob_empty), .retired_o(rob_retired), .stale_o(rob_stale)
  );
  always_comb begin
    rsp_ready = response_enable;
    rob_complete_valid = 0; rob_complete = '0; rob_select = -1;
    if (native_mode) begin
      rsp_ready = '0;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (rob_select < 0 && rsp_valid[c] && response_enable[c]) rob_select = c;
      end
      if (rob_select >= 0) begin
        rsp_ready[rob_select] = 1;
        rob_complete_valid = 1;
        rob_complete = rsp[rob_select];
      end
    end
    if (inject_completion) begin
      rob_complete_valid = 1;
      rob_complete = injected_completion;
    end
  end

  int checks = 0, cycles = 0, got [CONTEXT_COUNT], retired [CONTEXT_COUNT];
  logic wanted [CONTEXT_COUNT][1024], seen [CONTEXT_COUNT][1024];
  logic [31:0] golden [CONTEXT_COUNT][1024];
  logic [1:0] final_expected [CONTEXT_COUNT][1024];
  logic write_expected [CONTEXT_COUNT][1024];
  task automatic require(input logic ok, input string why);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C031_ASSERT tb_c031_memory %s cycle=%0d", why, cycles);
  endtask
  function automatic logic [31:0] bytes_at(input int off, input int n, input int salt = 0);
    logic [31:0] value;
    value = 0;
    for (int b = 0; b < n; b++) value[b*8 +: 8] = 8'((off+b)*17 + ((off+b)/64)*13 + 3 + salt);
    return value;
  endfunction
  always @(posedge clk) begin
    cycles++;
    if (rst_n) begin
      require(!lsu_error && !cache_error, "live invariants");
      if (mem_held) require(mem_valid && mem == held_mem, "MEM_REQ stable under backpressure");
      if (term_held) require(term_valid && term == held_term, "terminal stable under backpressure");
      mem_held = mem_valid && !mem_ready; held_mem = mem;
      term_held = term_valid && !term_ready; held_term = term;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        retired[c] += int'(rob_retired[c]);
        if (rsp_held[c]) require(rsp_valid[c] && rsp[c] == held_rsp[c], "LSU response held stable");
        rsp_held[c] = rsp_valid[c] && !rsp_ready[c]; held_rsp[c] = rsp[c];
        if (rsp_valid[c] && rsp_ready[c]) begin
          require(rsp[c].owner.ctx_slot == 32'(c) && rsp[c].owner.ctx_generation == 16'd1,
                  "completion Context identity");
          require(rsp[c].owner.subop_id == 0 && rsp[c].pc == 32'h8000, "complete owner and PC");
          require(rsp[c].owner.owner_op_seq < 1024, "response seq range");
          require(wanted[c][rsp[c].owner.owner_op_seq] && !seen[c][rsp[c].owner.owner_op_seq],
                  "exactly one response per accepted instruction");
          require(rsp[c].final_state == final_expected[c][rsp[c].owner.owner_op_seq],
                  "real terminal class");
          require(rsp[c].write == write_expected[c][rsp[c].owner.owner_op_seq], "load/store kind");
          if (!rsp[c].write && rsp[c].final_state == COMPLETED)
            require(rsp[c].data == golden[c][rsp[c].owner.owner_op_seq], "actual result bytes");
          seen[c][rsp[c].owner.owner_op_seq] = 1;
          got[c]++;
        end
      end
    end
  end

  // Python C023 peer: synchronous function calls only at real external
  // handshakes. The bounded transport queue may reorder/delay responses.
  integer request_file, response_file, scan_status;
  string endpoint;
  logic [MSHR_ENTRIES-1:0] pending;
  logic deliver = 1, term_enable = 1, duplicate_response = 0, fault_next = 0;
  logic request_enable = 1;
  memory_response_t queued [MSHR_ENTRIES];
  int free_slot, pick_slot, sent = 0, acked = 0;
  int hold_oldest = -1;
  memory_request_t held_mem;
  memory_terminal_t held_term;
  logic mem_held = 0, term_held = 0;
  completion_t held_rsp [CONTEXT_COUNT];
  logic [CONTEXT_COUNT-1:0] rsp_held = '0;
  logic [31:0] service_status;
  logic [LINE_BYTES*8-1:0] service_data;
  logic request_allow = 0;
  always @(negedge clk) request_allow = cycles % 7 != 0 && request_enable;
  always_comb begin
    free_slot = -1;
    for (int i = 0; i < MSHR_ENTRIES; i++)
      if (!pending[i] && free_slot < 0) free_slot = i;
    mem_ready = free_slot >= 0 && request_allow;
    term_ready = term_enable;
  end
  always @(posedge clk) begin
    if (!rst_n) begin
      pending <= '0; mr_valid <= 0; mr <= '0;
    end else begin
      if (mem_valid && mem_ready) begin
        require(mem.ctx_slot < CONTEXT_COUNT && mem.ctx_generation == 1 &&
                mem.arena_generation == 99 && mem.scope_epoch == 77,
                "external request retains full provenance");
        $fdisplay(request_file, "R %h %h %h %h %h %h %h %h %h %h %h",
          mem.txn_id, mem.slice_id, mem.slice_count, mem.binding_id,
          mem.binding_generation, mem.binding_offset, mem.length,
          mem.ctx_slot, mem.ctx_generation, mem.arena_generation, mem.scope_epoch);
        $fflush(request_file);
        scan_status = $fscanf(response_file, "%h %h", service_status, service_data);
        require(scan_status == 2 && service_status == 0, "real MemoryService read accepted");
        queued[free_slot] <= '{txn_id:mem.txn_id, slice_id:mem.slice_id,
          binding_generation:mem.binding_generation, status:service_status, data:service_data};
        pending[free_slot] <= 1;
        sent++;
      end
      if (mr_valid && mr_ready) begin
        if (duplicate_response) duplicate_response = 0;
        else mr_valid <= 0;
      end else if (!mr_valid && deliver) begin
        pick_slot = -1;
        // newest-first, so older long-latency Contexts do not dictate completion
        for (int i = 0; i < MSHR_ENTRIES; i++)
          if (pending[i] && i != hold_oldest) pick_slot = i;
        if (pick_slot >= 0) begin
          mr <= queued[pick_slot];
          if (fault_next) begin mr.status <= 32'd1; fault_next = 0; end
          mr_valid <= 1; pending[pick_slot] <= 0;
        end
      end
      if (term_valid && term_ready) begin
        $fdisplay(request_file, "T %h %h %h %h", term.txn_id, term.slice_id,
                  term.slice_count, {30'd0, term.final_state});
        $fflush(request_file);
        scan_status = $fscanf(response_file, "%h", service_status);
        require(scan_status == 1 && service_status == 0, "real MemoryService terminal closed");
        acked++;
      end
    end
  end

  task automatic configure(input int c);
    @(negedge clk);
    for (int s = 0; s < SLOT_COUNT; s++) windows[s] = '0;
    windows[0] = '{valid:1, readable:1, writable:1, cached:0,
      base:32'h1000, size:32'd4096, spm_base:32'(4096+c*4096), stripe_bytes:32'd32,
      binding_id:0, binding_generation:0, binding_offset:0, arena_generation:99, scope_epoch:77};
    windows[1] = '{valid:1, readable:1, writable:0, cached:1,
      base:32'h2000, size:32'd4096, spm_base:0, stripe_bytes:32'd32,
      binding_id:7, binding_generation:1, binding_offset:0, arena_generation:99, scope_epoch:77};
    windows[2] = windows[1]; windows[2].base = 32'h3000; windows[2].binding_id = 8;
    windows[3] = windows[1]; windows[3].base = 32'h4000; windows[3].binding_generation = 2;
    config_ctx = 32'(c); config_gen = 1; config_valid = 1;
    #1; require(config_ready && !config_reject, "quiescent window install");
    @(posedge clk); @(negedge clk); config_valid = 0;
  endtask
  task automatic send(input int c, input int seq, input logic wr, input int addr,
                      input logic [31:0] data, input logic [31:0] expect_data,
                      input logic [1:0] final_state = COMPLETED, input logic fence_op = 0,
                      input logic float_op = 0);
    @(negedge clk);
    wanted[c][seq] = 1; golden[c][seq] = expect_data;
    final_expected[c][seq] = final_state; write_expected[c][seq] = wr;
    req[c] = '{owner:'{ctx_slot:32'(c), ctx_generation:16'd1,
      owner_op_seq:32'(seq), subop_id:8'd0}, pc:32'h8000, address:32'(addr), data:data,
      rd:5'd5, byte_count:fence_op ? 3'd0 : 3'd4, write:wr, unsigned_load:1'b1,
      float_reg:float_op, fence:fence_op};
    req_valid[c] = 1;
    if (native_mode) begin rob_enq = req[c]; rob_enq_valid = 1; end
    begin
      int bound;
      bound = 0;
      do begin @(posedge clk); bound++; end while (!req_ready[c] && bound < 10000);
      require(req_ready[c], "bounded ingress acceptance");
    end
    @(negedge clk); req_valid[c] = 0; rob_enq_valid = 0;
  endtask
  task automatic await_seen(input int c, input int seq);
    int bound;
    bound = 0;
    while (!seen[c][seq] && bound < 20000) begin @(negedge clk); bound++; end
    require(seen[c][seq], "bounded exact completion");
  endtask
  task automatic close_all;
    int bound;
    bound = 0;
    while ((cache_outstanding != 0 || outstanding[0] != 0 || outstanding[PEER] != 0 ||
            sent != acked || pending != 0 || mr_valid) && bound < 30000) begin
      @(negedge clk); bound++;
    end
    require(cache_outstanding == 0 && outstanding[0] == 0 && outstanding[PEER] == 0 &&
            sent == acked && pending == 0 && !mr_valid, "all real obligations closed");
    for (int c = 0; c < CONTEXT_COUNT; c++)
      require(accepted[c] == terminal[c], "per-context credit conservation");
  endtask

  initial begin : run
    int baseline, count0, count1;
    logic [63:0] merged_before;
    logic [63:0] waits_before;
    require($value$plusargs("SERVICE=%s", endpoint), "service endpoint provided");
    request_file = $fopen({endpoint, ".req"}, "w");
    response_file = $fopen({endpoint, ".rsp"}, "r");
    require(request_file != 0 && response_file != 0, "open real service FIFOs");
    req_valid = 0; cancel = 0; config_valid = 0; config_ctx = 0; config_gen = 1;
    profile_valid = 0; cache_config_valid = 0; response_enable = '1; rob_enq = '0;
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      req[c] = '0; cancel_gen[c] = 1; generations[c] = 1; got[c] = 0; retired[c] = 0;
      for (int s = 0; s < 1024; s++) begin
        wanted[c][s] = 0; seen[c][s] = 0; golden[c][s] = 0;
        final_expected[c][s] = COMPLETED; write_expected[c][s] = 0;
      end
    end
    repeat (4) @(negedge clk); rst_n = 1;
    @(negedge clk); profile_valid = 1;
    @(negedge clk); require(profile_ack && !profile_reject, "bank profile commit"); profile_valid = 0;
    cache_config_valid = 1;
    do @(posedge clk); while (!cache_config_ready);
    @(negedge clk); cache_config_valid = 0;
    begin
      int limit;
      limit = 0;
      while (!cache_config_done && limit < 100) begin @(negedge clk); limit++; end
      require(cache_config_done && !cache_config_error, "cache and MSHR profile commit");
    end
    for (int c = 0; c < CONTEXT_COUNT; c++) configure(c);

    // Cache miss and cross-line bytes come from real Python ByteStore.
    send(0, 1, 0, 'h2000 + 62, 0, bytes_at(62, 4)); await_seen(0, 1); close_all();
    baseline = sent;
    send(0, 2, 0, 'h2000 + 8, 0, bytes_at(8, 4)); await_seen(0, 2); close_all();
    require(sent == baseline, "warm cache hit issues no external request");

    // Withhold an old refill. Peer SPM load/store and another miss progress.
    deliver = 0; baseline = sent;
    send(0, 3, 0, 'h2000 + 128, 0, bytes_at(128, 4));
    while (sent == baseline) @(negedge clk);
    send(PEER, 4, 1, 'h1000 + 31, 32'hdeadbeef, 0); await_seen(PEER, 4);
    send(PEER, 5, 0, 'h1000 + 31, 0, 32'hdeadbeef); await_seen(PEER, 5);
    require(!seen[0][3], "old refill remains pending while SPM progresses");
    send(PEER, 6, 0, 'h2000 + 256, 0, bytes_at(256, 4));
    begin
      int limit;
      limit = 0;
      while (sent < baseline + 2 && limit < 2000) begin @(negedge clk); limit++; end
      require(sent == baseline + 2, "multiple physical refills in flight");
    end
    hold_oldest = 0; duplicate_response = 1; deliver = 1;
    await_seen(PEER, 6);
    require(!seen[0][3], "younger miss completes while oldest external response withheld");
    hold_oldest = -1; await_seen(0, 3); close_all();

    // Same-line peer merge: cancelling Context A cannot kill Context B's data.
    if (CONTEXT_COUNT == 2) begin
      deliver = 0; baseline = sent; merged_before = merged;
      send(0, 7, 0, 'h2000 + 512, 0, 0, ISOLATED);
      send(PEER, 7, 0, 'h2000 + 516, 0, bytes_at(516, 4));
      begin
        int limit;
        limit = 0;
        while (merged == merged_before && limit < 10000) begin @(negedge clk); limit++; end
        require(merged == merged_before + 1, "peer registered an independent retained waiter");
      end
      require(sent == baseline + 1, "same provenance merges exactly one refill");
      cancel[0] = 1;
      @(negedge clk); cancel = 0;
      repeat (10) @(negedge clk);
      require(!seen[0][7] && !seen[PEER][7], "cancel does not fabricate refill settlement");
      deliver = 1;
      await_seen(PEER, 7); await_seen(0, 7); close_all();
    end

    // External terminal backpressure retains local records and credits.
    term_enable = 0; baseline = sent;
    send(0, 8, 0, 'h2000 + 768, 0, bytes_at(768, 4));
    while (sent == baseline) @(negedge clk);
    repeat (1800) @(negedge clk);
    require(!seen[0][8] && outstanding[0] != 0, "response is not terminal ACK");
    term_enable = 1; await_seen(0, 8); close_all();

    // A cache request accepted on the same edge as its Context's cancel is
    // born cancelled: it must complete ISOLATED, never a fabricated refill.
    begin
      int b2;
      send(0, 84, 0, 'h2000 + 640, 0, 0, ISOLATED);
      b2 = 0;
      while (!dut.cache_req_valid_o[0] && b2 < 4000) begin @(negedge clk); b2++; end
      require(dut.cache_req_valid_o[0], "cancel edge scenario reaches cache launch");
      cancel[0] = 1;
      @(posedge clk);
      @(negedge clk); cancel[0] = 0;
      await_seen(0, 84); close_all();
    end

    // A presented external request keeps its record under backpressure: a
    // newly allocated lower-index bridge record cannot steal the wires.
    begin
      int b3;
      request_enable = 1; deliver = 0; baseline = sent;
      send(0, 85, 0, 'h2000 + 3136, 0, bytes_at(3136, 4));
      while (sent == baseline) @(negedge clk);
      request_enable = 0;
      send(0, 86, 0, 'h2000 + 3200, 0, bytes_at(3200, 4));
      b3 = 0;
      while (!(mem_valid && mem.binding_offset == 32'd3200) && b3 < 4000) begin
        @(negedge clk); b3++;
      end
      require(mem_valid && mem.binding_offset == 32'd3200 && !mem_ready,
              "stall scenario parks the second request");
      deliver = 1;
      await_seen(0, 85);
      send(0, 87, 0, 'h2000 + 3264, 0, bytes_at(3264, 4));
      b3 = 0;
      while (!(mem_valid && mem.binding_offset == 32'd3264) && b3 < 4000) begin
        @(negedge clk); b3++;
      end
      repeat (8) @(negedge clk);
      request_enable = 1;
      await_seen(0, 86); await_seen(0, 87); close_all();
    end

    // Two 16-entry Context pools fill one MSHR's retained waiter budget.
    response_enable = '0; deliver = 0; baseline = sent; merged_before = merged;
    for (int i = 0; i < LSU_CREDIT_CAP; i++)
      for (int c = 0; c < CONTEXT_COUNT; c++)
        send(c, 32+i, 0, 'h2000 + 1024, 0, bytes_at(1024, 4));
    begin
      int limit;
      limit = 0;
      while (merged < merged_before + 64'(CONTEXT_COUNT*LSU_CREDIT_CAP-1) && limit < 40000) begin
        @(negedge clk); limit++;
      end
      require(merged == merged_before + 64'(CONTEXT_COUNT*LSU_CREDIT_CAP-1),
              "all bounded waiters retained before refill delivery");
    end
    for (int c = 0; c < CONTEXT_COUNT; c++)
      require(outstanding[c] == LSU_CREDIT_CAP && !req_ready[c], "credit ceiling per Context");
    require(sent == baseline + 1, "all live waiters merge one physical line");
    deliver = 1;
    repeat (2000) @(negedge clk);
    response_enable = '1;
    for (int c = 0; c < CONTEXT_COUNT; c++)
      for (int i = 0; i < LSU_CREDIT_CAP; i++) await_seen(c, 32+i);
    close_all();

    // Fill every MSHR with a distinct line. An extra Context request must
    // wait without losing its queue entry, then wake when token state changes.
    if (CONTEXT_COUNT == 2) begin
      deliver = 0; baseline = sent; waits_before = mshr_wait[PEER];
      for (int i = 0; i < MSHR_ENTRIES; i++)
        send(0, 60+i, 0, 'h2000 + 2048+i*LINE_BYTES, 0, bytes_at(2048+i*LINE_BYTES, 4));
      begin
        int limit;
        limit = 0;
        while (sent < baseline + MSHR_ENTRIES && limit < 30000) begin @(negedge clk); limit++; end
        require(sent == baseline + MSHR_ENTRIES, "all physical MSHR entries issued");
      end
      send(PEER, 60, 0, 'h2000 + 3584, 0, bytes_at(3584, 4));
      repeat (1500) @(negedge clk);
      require(sent == baseline + MSHR_ENTRIES && mshr_wait[PEER] > waits_before && !seen[PEER][60],
              "WAIT_MSHR holds extra request without dropping it");
      deliver = 1;
      await_seen(PEER, 60);
      for (int i = 0; i < MSHR_ENTRIES; i++) await_seen(0, 60+i);
      close_all();
    end

    // Identical line offsets from different binding IDs/generations cannot
    // alias in either the MSHR merge key or the persistent cache tag.
    deliver = 0; baseline = sent;
    send(0, 77, 0, 'h2000 + 1792, 0, bytes_at(1792, 4));
    send(0, 78, 0, 'h3000 + 1792, 0, bytes_at(1792, 4, 41));
    send(0, 79, 0, 'h4000 + 1792, 0, bytes_at(1792, 4, 83));
    begin
      int limit;
      limit = 0;
      while (sent < baseline + 3 && limit < 6000) begin @(negedge clk); limit++; end
      require(sent == baseline + 3, "full binding-generation-line merge identity");
    end
    deliver = 1;
    await_seen(0, 77); await_seen(0, 78); await_seen(0, 79); close_all();

    // An injected transport fault is FAULTED, never a successful/cancelled load.
    fault_next = 1;
    send(0, 80, 0, 'h2000 + 1536, 0, 0, FAULTED); await_seen(0, 80); close_all();

    // Fence waits for an older external read but does not block peer Context.
    deliver = 0; baseline = sent;
    send(0, 81, 0, 'h2000 + 1600, 0, bytes_at(1600, 4));
    while (sent == baseline) @(negedge clk);
    send(0, 82, 0, 0, 0, 0, COMPLETED, 1);
    send(0, 83, 1, 'h1000 + 64, 32'h01020304, 0);
    if (CONTEXT_COUNT == 2) begin
      send(PEER, 81, 1, 'h1000 + 64, 32'h11121314, 0); await_seen(PEER, 81);
    end
    repeat (50) @(negedge clk);
    require(!seen[0][82] && !seen[0][83], "fence and younger store wait older terminal");
    deliver = 1; await_seen(0, 81); await_seen(0, 82); await_seen(0, 83); close_all();

    // Native C026 full ROB: identical PC/rd/seq across independent Contexts.
    native_mode = 1; count0 = retired[0]; count1 = retired[PEER];
    response_enable = '0;
    for (int c = 0; c < CONTEXT_COUNT; c++) send(c, 100, 1, 'h1000, 32'(c+19), 0);
    begin
      int limit;
      limit = 0;
      while ((!rsp_valid[0] || !rsp_valid[PEER]) && limit < 1000) begin @(negedge clk); limit++; end
      require(rsp_valid[0] && rsp_valid[PEER], "native completions retained before injection");
    end
    require(!rob_empty[0] && !rob_empty[PEER], "native stores wait for exact LSU owner");
    injected_completion = rsp[PEER];
    injected_completion.owner.owner_op_seq = 32'd999;
    inject_completion = 1;
    @(negedge clk); inject_completion = 0;
    repeat (5) @(negedge clk);
    require(!rob_empty[0] && !rob_empty[PEER], "wrong op_seq cannot retire native store");
    injected_completion = rsp[PEER];
    injected_completion.owner.ctx_generation = 0;
    inject_completion = 1;
    @(negedge clk); inject_completion = 0;
    repeat (5) @(negedge clk);
    require(!rob_empty[0] && !rob_empty[PEER] && rob_stale != 0,
            "old-generation completion rejected by real native ROB");
    response_enable[PEER] = 1; await_seen(PEER, 100);
    repeat (10) @(negedge clk);
    if (CONTEXT_COUNT == 2) begin
      require(rob_empty[PEER] && !rob_empty[0] && retired[PEER] == count1 + 1,
              "native peer retires without completing same-PC Context A");
    end
    response_enable = '1; await_seen(0, 100);
    repeat (10) @(negedge clk);
    require(rob_empty[0] && retired[0] == count0 + 1, "native store retires once");
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      send(c, 101, 0, 'h1000, 0, 32'(c+19)); await_seen(c, 101);
    end
    repeat (10) @(negedge clk);
    require(rob_empty[0] && rob_empty[PEER], "native load owner writeback closes partitions");
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      send(c, 102, 0, 'h1000, 0, 32'(c+19), COMPLETED, 0, 1); await_seen(c, 102);
    end
    repeat (10) @(negedge clk);
    require(rob_empty[0] && rob_empty[PEER], "native floating load owner writeback closes partitions");
    close_all();
    $fdisplay(request_file, "D"); $fflush(request_file);
    scan_status = $fscanf(response_file, "%h", service_status);
    require(scan_status == 1 && service_status == 0, "MemoryService zero leaks");
    $display("C031_PASS tb_c031_memory contexts=%0d checks=%0d external=%0d terminals=%0d",
             CONTEXT_COUNT, checks, sent, acked);
    $fclose(request_file); $fclose(response_file); $finish;
  end
  initial begin #100_000_000; $fatal(1, "C031_ASSERT tb_c031_memory watchdog"); end
endmodule
