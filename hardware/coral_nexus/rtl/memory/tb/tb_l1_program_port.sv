// tb_l1_program_port.sv — C030 program-port regression for
// rtl/memory/l1_program_port.sv (C2 target geometry by default).
//
// Geometry follows the REAL C028 frontend contract: fetch moves one
// 32-byte LINE (fetchDataBits = 256) with the full-width CtxFetchKey
// (ctx_generation = 16, fetch_epoch = fetch_reqid = ownerOpSeqBits = 32,
// fetchAddrBits = 32), while installs stay word-beat (WORD_BYTES = 4) sized.
//
// Scenarios:
//   A  install two full lines plus a partial one, then alignment/range/
//      zero-length install rejects (no side effects) and an install-response
//      backpressure hold; fetch validation faults (range/align/zero/len/
//      never-installed/partial-line) carry no bytes; installed lines read
//      back with real byte checks (line-wide).
//   B  response identity: keys are echoed and never replaced by an address;
//      two same-address requests under different fetch_epoch never alias
//      (same-PC ABA), also across contexts and addresses.
//   C  per-Context in-flight bound: with delivery stalled the (N+1)th request
//      is not accepted (ready low), bound hits are counted, the held request
//      is never dropped, and the other Context stays independent.
//   E  RES-12 fairness: sustained contention from every requester; grants per
//      requester stay balanced and nobody starves; the install beats really
//      land in the array.
//   D  FET-06: installs overlapping execution-referenced text are rejected
//      with I_EXEC_REF and bytes stay unchanged (identical bytes too); after
//      the reference drops, a warm re-install of identical resident text is
//      answered I_ALREADY (no-op, bytes not rewritten) and an unreferenced
//      install succeeds.
//   F  honest counting: module counters equal independent port-boundary
//      shadow counters; reference misuse latches invariant_error_o.
//
// Fetch/install responses default to ready-low: every response is held under
// backpressure until this bench explicitly samples and releases it.
`timescale 1ns/1ps
module tb_l1_program_port
  import c029_params_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT = C2_CONTEXTS
) ();
  localparam int unsigned FO = 2;
  localparam int unsigned CTX_W = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT);
  // Real-design key field widths (PerCtxFetchMux Parameters):
  // ctxGenerationBits = 16, ownerOpSeqBits = 32, fetchAddrBits = 32.
  localparam int unsigned GEN_W = 16, EPOCH_W = 32, SEQ_W = 32, PROG_ID_W = 8;
  localparam int unsigned KEY_W = CTX_W + GEN_W + EPOCH_W + SEQ_W;
  localparam int unsigned INF_W = $clog2(FO + 1);
  localparam int unsigned SRAM = 256;
  localparam int unsigned FB = 32;  // fetchDataBits/8: one fetch line
  localparam int unsigned WB = 4;   // install word beat
  localparam int unsigned ADDR_W = 32;
  localparam int unsigned REQS = CONTEXT_COUNT + 1;
  // Index of the peer Context: 1 with two Contexts; clamped to 0 with one so the constant
  // index is always inside the arrays (every use is additionally guarded by CONTEXT_COUNT == 2).
  localparam int unsigned PEER = (CONTEXT_COUNT == 2) ? 1 : 0;
  localparam int unsigned OQ = CONTEXT_COUNT*FO + 4;
  localparam logic [INF_W-1:0] FO_MAX = INF_W'(FO);

  localparam logic [3:0] F_OK = 4'd0, F_RANGE = 4'd1, F_ALIGN = 4'd2, F_ZERO = 4'd3,
                         F_LEN = 4'd4, F_UNWRITTEN = 4'd5;
  localparam logic [2:0] I_OK = 3'd0, I_RANGE = 3'd1, I_ALIGN = 3'd2, I_ZERO = 3'd3,
                         I_EXEC_REF = 3'd4, I_ALREADY = 3'd5;

  logic clk = 0, rst_n = 0;
  logic ins_valid = 0, ins_ready;
  logic [ADDR_W-1:0] ins_addr = 0;
  logic [WB*8-1:0] ins_data = 0;
  logic [WB-1:0] ins_be = 0;
  logic [PROG_ID_W-1:0] ins_prog_id = 0, ins_rsp_prog_id;
  logic [EPOCH_W-1:0] ins_epoch = 0, ins_rsp_epoch;
  logic ins_rsp_valid, ins_rsp_ready = 0;
  logic [2:0] ins_rsp_code;
  logic fet_valid [CONTEXT_COUNT];
  logic fet_ready [CONTEXT_COUNT];
  logic [KEY_W-1:0] fet_key [CONTEXT_COUNT];
  logic [ADDR_W-1:0] fet_addr [CONTEXT_COUNT];
  logic [7:0] fet_len [CONTEXT_COUNT];
  logic rsp_valid, rsp_ready = 0, rsp_fault;
  logic [KEY_W-1:0] rsp_key;
  logic [FB*8-1:0] rsp_data;
  logic [3:0] rsp_code;
  logic ref_inc_valid = 0, ref_dec_valid = 0;
  logic [PROG_ID_W-1:0] ref_prog_id = 0;
  logic [EPOCH_W-1:0] ref_epoch = 0;
  logic [63:0] fetch_accepted_o, rsp_delivered_o, install_beats_o, port_busy_o, bound_hits_o;
  logic [CONTEXT_COUNT*INF_W-1:0] inf_o;
  logic invariant_error_o;

  initial begin
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      fet_valid[c] = 1'b0;
      fet_key[c] = '0;
      fet_addr[c] = '0;
      fet_len[c] = '0;
    end
  end

  l1_program_port #(
    .CONTEXT_COUNT(CONTEXT_COUNT),
    .FETCH_OUTSTANDING(FO),
    .SRAM_BYTES(SRAM),
    .FETCH_BYTES(FB),
    .WORD_BYTES(WB),
    .INS_RSP_DEPTH(4),
    .REF_SLOTS(8),
    .PROG_ID_W(PROG_ID_W),
    .EPOCH_W(EPOCH_W),
    .GEN_W(GEN_W),
    .SEQ_W(SEQ_W),
    .ADDR_W(ADDR_W)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .ins_valid(ins_valid),
    .ins_ready(ins_ready),
    .ins_addr(ins_addr),
    .ins_data(ins_data),
    .ins_be(ins_be),
    .ins_prog_id(ins_prog_id),
    .ins_epoch(ins_epoch),
    .ins_rsp_valid(ins_rsp_valid),
    .ins_rsp_ready(ins_rsp_ready),
    .ins_rsp_code(ins_rsp_code),
    .ins_rsp_prog_id(ins_rsp_prog_id),
    .ins_rsp_epoch(ins_rsp_epoch),
    .fet_valid(fet_valid),
    .fet_ready(fet_ready),
    .fet_key(fet_key),
    .fet_addr(fet_addr),
    .fet_len(fet_len),
    .rsp_valid(rsp_valid),
    .rsp_ready(rsp_ready),
    .rsp_key(rsp_key),
    .rsp_data(rsp_data),
    .rsp_fault(rsp_fault),
    .rsp_code(rsp_code),
    .ref_inc_valid(ref_inc_valid),
    .ref_dec_valid(ref_dec_valid),
    .ref_prog_id(ref_prog_id),
    .ref_epoch(ref_epoch),
    .fetch_accepted_o(fetch_accepted_o),
    .rsp_delivered_o(rsp_delivered_o),
    .install_beats_o(install_beats_o),
    .port_busy_o(port_busy_o),
    .bound_hits_o(bound_hits_o),
    .inf_o(inf_o),
    .invariant_error_o(invariant_error_o)
  );

  always #5 clk = ~clk;
  initial begin #5_000_000; $fatal(1, "C030_ASSERT tb_l1_program_port watchdog"); end

  int checks = 0;
  task automatic require(input logic ok, input string label);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C030_ASSERT tb_l1_program_port %s", label);
  endtask

  function automatic logic [KEY_W-1:0] fkey(input int unsigned c, input logic [GEN_W-1:0] g,
                                            input logic [EPOCH_W-1:0] e, input logic [SEQ_W-1:0] r);
    return {CTX_W'(c), g, e, r};
  endfunction

  function automatic logic [WB*8-1:0] pattern(input logic [7:0] seed);
    logic [WB*8-1:0] p;
    for (int b = 0; b < WB; b++) p[b*8 +: 8] = seed + 8'(b);
    return p;
  endfunction

  // One full fetch line assembled from word patterns: word w carries bytes
  // seed+4w .. seed+4w+3, so the line covers the contiguous seed..seed+31.
  function automatic logic [FB*8-1:0] linepat(input logic [7:0] seed);
    logic [FB*8-1:0] l;
    for (int w = 0; w < FB/WB; w++) l[w*WB*8 +: WB*8] = pattern(seed + 8'(4*w));
    return l;
  endfunction

  function automatic logic [INF_W-1:0] inf_of(input int unsigned c);
    return inf_o[c*INF_W +: INF_W];
  endfunction

  // Independent port-boundary shadow bookkeeping.
  bit ins_done;
  bit fet_done [CONTEXT_COUNT];
  int unsigned shadow_fet [CONTEXT_COUNT];
  int unsigned shadow_ins, shadow_rsp, shadow_busy;
  logic [KEY_W-1:0] okey [OQ];
  int unsigned ohead, otail;

  initial begin
    for (int c = 0; c < CONTEXT_COUNT; c++) shadow_fet[c] = 0;
    shadow_ins = 0;
    shadow_rsp = 0;
    shadow_busy = 0;
    ohead = 0;
    otail = 0;
  end

  always @(posedge clk) begin
    if (rst_n) begin
      automatic bit any_fet;
      any_fet = 1'b0;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (fet_valid[c] && fet_ready[c]) begin
          any_fet = 1'b1;
          fet_done[c] = 1'b1;
          shadow_fet[c] = shadow_fet[c] + 1;
          okey[otail % OQ] = fet_key[c];
          otail = otail + 1;
        end
      end
      if (any_fet) shadow_busy = shadow_busy + 1;
      if (ins_valid && ins_ready) begin
        ins_done = 1'b1;
        shadow_ins = shadow_ins + 1;
        shadow_busy = shadow_busy + 1;
      end
      if (rsp_valid && rsp_ready) begin
        shadow_rsp = shadow_rsp + 1;
        require(rsp_key == okey[ohead % OQ], "response key equals its request key");
        ohead = ohead + 1;
      end
    end
  end

  // Offer one fetch request and hold it until the port accepts it.
  task automatic fet_send(input int unsigned c, input logic [KEY_W-1:0] key,
                          input logic [ADDR_W-1:0] addr, input logic [7:0] len);
    fet_done[c] = 1'b0;
    @(negedge clk);
    fet_valid[c] = 1'b1;
    fet_key[c] = key;
    fet_addr[c] = addr;
    fet_len[c] = len;
    do @(negedge clk); while (!fet_done[c]);
    fet_valid[c] = 1'b0;
  endtask

  // Expect the head fetch response, then release exactly one beat of ready.
  task automatic rsp_expect(input logic [KEY_W-1:0] ekey, input logic [FB*8-1:0] edata,
                            input logic efault, input logic [3:0] ecode, input string label);
    while (!rsp_valid) @(negedge clk);
    require(rsp_key == ekey, {label, ": response key equals request key"});
    require(rsp_fault == efault, {label, ": fault flag"});
    require(rsp_code == ecode, {label, ": code"});
    if (!efault) require(rsp_data == edata, {label, ": line bytes"});
    else require(rsp_data == '0, {label, ": fault carries no bytes"});
    rsp_ready = 1'b1;
    @(negedge clk);
    rsp_ready = 1'b0;
  endtask

  // Offer one install beat, expect its registered response.
  task automatic ins_beat(input logic [ADDR_W-1:0] addr, input logic [WB*8-1:0] data,
                          input logic [WB-1:0] be, input logic [PROG_ID_W-1:0] pid,
                          input logic [EPOCH_W-1:0] ep, input logic [2:0] ecode,
                          input string label);
    ins_done = 1'b0;
    @(negedge clk);
    ins_valid = 1'b1;
    ins_addr = addr;
    ins_data = data;
    ins_be = be;
    ins_prog_id = pid;
    ins_epoch = ep;
    do @(negedge clk); while (!ins_done);
    ins_valid = 1'b0;
    while (!ins_rsp_valid) @(negedge clk);
    require(ins_rsp_code == ecode, {label, ": install code"});
    require(ins_rsp_prog_id == pid && ins_rsp_epoch == ep, {label, ": identity echo"});
    ins_rsp_ready = 1'b1;
    @(negedge clk);
    ins_rsp_ready = 1'b0;
  endtask

  task automatic ref_op(input bit inc, input logic [PROG_ID_W-1:0] pid,
                        input logic [EPOCH_W-1:0] ep);
    @(negedge clk);
    ref_inc_valid = inc;
    ref_dec_valid = !inc;
    ref_prog_id = pid;
    ref_epoch = ep;
    @(negedge clk);
    ref_inc_valid = 1'b0;
    ref_dec_valid = 1'b0;
  endtask

  initial begin
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    require(!rsp_valid && !ins_rsp_valid, "reset: no responses pending");
    require(fetch_accepted_o == 64'd0 && rsp_delivered_o == 64'd0 && install_beats_o == 64'd0 &&
            port_busy_o == 64'd0 && bound_hits_o == 64'd0, "reset: honest counters at zero");
    for (int c = 0; c < CONTEXT_COUNT; c++) require(inf_of(c) == '0, "reset: nothing in flight");
    require(!invariant_error_o, "reset: no invariant error");

    // ---------------- A: install lines, rejects, install backpressure ----
    for (int w = 0; w < FB/WB; w++)
      ins_beat(ADDR_W'(4*w), pattern(8'(8'h40 + 4*w)), '1, 8'h11, 32'd1, I_OK,
               "install line 0 word");
    for (int w = 0; w < FB/WB; w++)
      ins_beat(ADDR_W'(8'h20 + 4*w), pattern(8'(8'h80 + 4*w)), '1, 8'h11, 32'd1, I_OK,
               "install line 1 word");
    ins_beat(32'h0000_0002, pattern(8'h90), '1, 8'h11, 32'd1, I_ALIGN,
             "misaligned install rejected");
    ins_beat(32'd256, pattern(8'h91), '1, 8'h11, 32'd1, I_RANGE,
             "out-of-range install rejected");
    ins_beat(32'h40, pattern(8'h92), '0, 8'h11, 32'd1, I_ZERO,
             "zero-length install rejected");
    // Install response is registered and held under backpressure, then drains.
    @(negedge clk);
    ins_valid = 1'b1;
    ins_addr = 32'h40;
    ins_data = pattern(8'h60);
    ins_be = '1;
    ins_prog_id = 8'h33;
    ins_epoch = 32'd1;
    ins_done = 1'b0;
    do @(negedge clk); while (!ins_done);
    ins_valid = 1'b0;
    require(ins_rsp_valid, "install response registered");
    begin
      automatic logic [2:0] held_code;
      held_code = ins_rsp_code;
      repeat (3) begin
        @(negedge clk);
        require(ins_rsp_valid && ins_rsp_code == held_code,
                "install response held under backpressure");
      end
      ins_rsp_ready = 1'b1;
      @(negedge clk);
      ins_rsp_ready = 1'b0;
      require(!ins_rsp_valid, "install response drained once");
    end
    // Fetch validation: fault responses carry no bytes, no side effects.
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'hD1), 32'd256, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'hD1), '0, 1'b1, F_RANGE,
               "out-of-range fetch faults");
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'hD2), 32'h0000_0002, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'hD2), '0, 1'b1, F_ALIGN,
               "misaligned fetch faults");
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'hD3), 32'h0000_0004, 8'd0);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'hD3), '0, 1'b1, F_ZERO,
               "zero-length fetch faults");
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'hD4), 32'h0000_0004, 8'd4);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'hD4), '0, 1'b1, F_LEN,
               "word-width fetch faults: lines only");
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'hD5), 32'h0000_0060, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'hD5), '0, 1'b1, F_UNWRITTEN,
               "never-installed fetch faults");
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'hD6), 32'h0000_0040, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'hD6), '0, 1'b1, F_UNWRITTEN,
               "partial-line fetch faults");
    // Real byte checks of the installed lines (rejects left them alone).
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'd1), 32'h0000_0000, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'd1), linepat(8'h40), 1'b0, F_OK,
               "line 0 bytes");
    fet_send(0, fkey(0, 16'h0001, 32'd1, 32'd2), 32'h0000_0020, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd1, 32'd2), linepat(8'h80), 1'b0, F_OK,
               "line 1 bytes");

    // ---------------- B: response identity, ABA across epochs -------------
    begin
      automatic logic [KEY_W-1:0] k5, k6, k7, k8;
      k5 = fkey(0, 16'h0001, 32'd5, 32'd1);
      k6 = fkey(0, 16'h0001, 32'd6, 32'd2);
      fet_send(0, k5, 32'h0000_0000, 8'd32);
      fet_send(0, k6, 32'h0000_0000, 8'd32);
      rsp_expect(k5, linepat(8'h40), 1'b0, F_OK, "epoch-5 key echoed (same PC)");
      rsp_expect(k6, linepat(8'h40), 1'b0, F_OK, "epoch-6 key echoed (no ABA alias)");
      k8 = fkey(0, 16'h0002, 32'd7, 32'd2);
      if (CONTEXT_COUNT == 2) begin
        k7 = fkey(1, 16'h0002, 32'd7, 32'd1);
        fet_send(1, k7, 32'h0000_0020, 8'd32);
        fet_send(0, k8, 32'h0000_0020, 8'd32);
        rsp_expect(k7, linepat(8'h80), 1'b0, F_OK, "ctx1 key echoed");
        rsp_expect(k8, linepat(8'h80), 1'b0, F_OK, "ctx0 cross-address key echoed");
      end else begin
        fet_send(0, k8, 32'h0000_0020, 8'd32);
        rsp_expect(k8, linepat(8'h80), 1'b0, F_OK, "ctx0 cross-address key echoed");
      end
    end

    // ---------------- C: per-Context in-flight bound ----------------------
    begin
      automatic logic [KEY_W-1:0] k1, k2, kc1, k3;
      k1 = fkey(0, 16'h0001, 32'd2, 32'd1);
      k2 = fkey(0, 16'h0001, 32'd2, 32'd2);
      kc1 = fkey(1, 16'h0001, 32'd9, 32'd1);
      k3 = fkey(0, 16'h0001, 32'd3, 32'd3);
      fet_send(0, k1, 32'h0000_0000, 8'd32);
      fet_send(0, k2, 32'h0000_0020, 8'd32);
      require(inf_of(0) == FO_MAX, "context 0 at its in-flight bound");
      // The (N+1)th request is offered continuously: never accepted, never lost.
      fet_done[0] = 1'b0;
      @(negedge clk);
      fet_valid[0] = 1'b1;
      fet_key[0] = k3;
      fet_addr[0] = 32'h0000_0040;
      fet_len[0] = 8'd32;
      repeat (4) begin
        @(negedge clk);
        require(fet_ready[0] == 1'b0, "bound: (N+1)th request not accepted");
        require(!fet_done[0], "bound: no handshake for (N+1)th");
      end
      require(bound_hits_o >= 64'd4, "bound hits counted");
      // The other Context is bounded independently and still flows.
      if (CONTEXT_COUNT == 2) begin
        fet_send(1, kc1, 32'h0000_0020, 8'd32);
        require(inf_of(PEER) == FO_MAX - INF_W'(1), "context 1 unaffected by context 0 bound");
      end
      // Drain: the first delivery frees a slot, the held request then lands.
      rsp_expect(k1, linepat(8'h40), 1'b0, F_OK, "drain context 0 request 1");
      do @(negedge clk); while (!fet_done[0]);
      fet_valid[0] = 1'b0;
      rsp_expect(k2, linepat(8'h80), 1'b0, F_OK, "drain context 0 request 2");
      if (CONTEXT_COUNT == 2) rsp_expect(kc1, linepat(8'h80), 1'b0, F_OK, "drain context 1 request");
      rsp_expect(k3, '0, 1'b1, F_UNWRITTEN, "held (N+1)th delivered after bound freed");
      for (int c = 0; c < CONTEXT_COUNT; c++) require(inf_of(c) == '0, "in-flight returns to zero");
    end

    // ---------------- E: RES-12 fairness under sustained contention -------
    begin
      automatic int unsigned f0s, f1s, fis, g0, g1, gi, mx, mn;
      @(negedge clk);
      f0s = shadow_fet[0];
      f1s = (CONTEXT_COUNT == 2) ? shadow_fet[PEER] : 0;
      fis = shadow_ins;
      ins_valid = 1'b1;
      ins_addr = 32'h0000_0080;
      ins_data = pattern(8'h70);
      ins_be = '1;
      ins_prog_id = 8'h33;
      ins_epoch = 32'd1;
      fet_valid[0] = 1'b1;
      fet_key[0] = fkey(0, 16'h00A0, 32'd9, 32'd7);
      fet_addr[0] = 32'h0000_0000;
      fet_len[0] = 8'd32;
      if (CONTEXT_COUNT == 2) begin
        fet_valid[PEER] = 1'b1;
        fet_key[PEER] = fkey(PEER, 16'h00A1, 32'd3, 32'd7);
        fet_addr[PEER] = 32'h0000_0020;
        fet_len[PEER] = 8'd32;
      end
      rsp_ready = 1'b1;
      ins_rsp_ready = 1'b1;
      repeat (300) @(negedge clk);
      ins_valid = 1'b0;
      fet_valid[0] = 1'b0;
      if (CONTEXT_COUNT == 2) fet_valid[PEER] = 1'b0;
      @(negedge clk);
      while (rsp_valid || ins_rsp_valid) @(negedge clk);
      rsp_ready = 1'b0;
      ins_rsp_ready = 1'b0;
      @(negedge clk);
      g0 = shadow_fet[0] - f0s;
      g1 = (CONTEXT_COUNT == 2) ? (shadow_fet[PEER] - f1s) : 0;
      gi = shadow_ins - fis;
      if (CONTEXT_COUNT == 2) begin
        mx = (g0 > g1) ? ((g0 > gi) ? g0 : gi) : ((g1 > gi) ? g1 : gi);
        mn = (g0 < g1) ? ((g0 < gi) ? g0 : gi) : ((g1 < gi) ? g1 : gi);
      end else begin
        mx = (g0 > gi) ? g0 : gi;
        mn = (g0 < gi) ? g0 : gi;
      end
      require(gi >= 300/REQS - 2, "install not starved under contention");
      require(g0 >= 300/REQS - 2, "context 0 not starved under contention");
      if (CONTEXT_COUNT == 2) require(g1 >= 300/REQS - 2, "context 1 not starved under contention");
      require(mx - mn <= 2, "round-robin grant balance");
      require(!invariant_error_o, "no invariant error through fairness");
      require(ohead == otail, "every fairness response delivered");
    end

    // ---------------- D: FET-06 rewrite protection + warm re-install -----
    ref_op(1'b1, 8'h11, 32'd1);
    ins_beat(32'h0000_0000, pattern(8'h80), '1, 8'h22, 32'd1, I_EXEC_REF,
             "rewrite of referenced text rejected");
    fet_send(0, fkey(0, 16'h0001, 32'd4, 32'd1), 32'h0000_0000, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd4, 32'd1), linepat(8'h40), 1'b0, F_OK,
               "bytes unchanged after reject");
    ins_beat(32'h0000_001C, pattern(8'h81), '1, 8'h22, 32'd1, I_EXEC_REF,
             "overlap with referenced tail rejected");
    ins_beat(32'h0000_0000, pattern(8'h40), '1, 8'h11, 32'd1, I_EXEC_REF,
             "identical bytes to referenced text still rejected");
    ref_op(1'b0, 8'h11, 32'd1);
    ins_beat(32'h0000_0000, pattern(8'h40), '1, 8'h11, 32'd1, I_ALREADY,
             "warm re-install of identical text is a no-op (I_ALREADY)");
    ins_beat(32'h0000_0000, pattern(8'hA0), '1, 8'h11, 32'd1, I_OK,
             "install to unreferenced text succeeds");
    begin
      automatic logic [FB*8-1:0] exp_line;
      exp_line = linepat(8'h40);
      exp_line[WB*8-1:0] = pattern(8'hA0);
      fet_send(0, fkey(0, 16'h0001, 32'd5, 32'd1), 32'h0000_0000, 8'd32);
      rsp_expect(fkey(0, 16'h0001, 32'd5, 32'd1), exp_line, 1'b0, F_OK,
                 "new bytes visible after unreferenced install");
    end
    // The fairness-phase install beat really landed in the array: complete
    // the line around it, then read the whole line back.
    ins_beat(32'h0000_0080, pattern(8'h70), '1, 8'h33, 32'd1, I_ALREADY,
             "fairness beat is identical resident text (I_ALREADY)");
    for (int w = 1; w < FB/WB; w++)
      ins_beat(ADDR_W'(8'h80 + 4*w), pattern(8'(8'h70 + 4*w)), '1, 8'h33, 32'd1, I_OK,
               "complete fairness line");
    fet_send(0, fkey(0, 16'h0001, 32'd5, 32'd2), 32'h0000_0080, 8'd32);
    rsp_expect(fkey(0, 16'h0001, 32'd5, 32'd2), linepat(8'h70), 1'b0, F_OK,
               "fairness install wrote real bytes");
    // Same-edge race: the FIRST acquisition of resident text and an
    // overlapping install settling on the same clock edge must still keep
    // the bytes (FET-06); a reference that goes live with the write must
    // never observe overwritten text.
    @(negedge clk);
    ref_inc_valid = 1'b1;
    ref_prog_id = 8'h11;
    ref_epoch = 32'd1;
    ins_valid = 1'b1;
    ins_addr = 32'h0000_0000;
    ins_data = pattern(8'hB0);
    ins_be = '1;
    ins_prog_id = 8'h22;
    ins_epoch = 32'd1;
    @(negedge clk);
    ref_inc_valid = 1'b0;
    ins_valid = 1'b0;
    require(ins_rsp_valid, "same-edge race: install answered");
    require(ins_rsp_code == I_EXEC_REF, "same-edge acquire protects overlapping text");
    ins_rsp_ready = 1'b1;
    @(negedge clk);
    ins_rsp_ready = 1'b0;
    begin
      automatic logic [FB*8-1:0] exp_line2;
      exp_line2 = linepat(8'h40);
      exp_line2[WB*8-1:0] = pattern(8'hA0);
      fet_send(0, fkey(0, 16'h0001, 32'd6, 32'd1), 32'h0000_0000, 8'd32);
      rsp_expect(fkey(0, 16'h0001, 32'd6, 32'd1), exp_line2, 1'b0, F_OK,
                 "bytes unchanged through the same-edge acquire/write race");
    end
    ref_op(1'b0, 8'h11, 32'd1);

    // ---------------- F: honest counting + reference misuse latch ---------
    ref_op(1'b0, 8'hEE, 32'd1);
    require(invariant_error_o, "reference misuse latches invariant_error_o");
    begin
      automatic logic [63:0] sf;
      sf = '0;
      for (int c = 0; c < CONTEXT_COUNT; c++) sf = sf + 64'(shadow_fet[c]);
      require(fetch_accepted_o == sf, "honest accepted-fetch counter");
      require(rsp_delivered_o == 64'(shadow_rsp), "honest delivered-response counter");
      require(install_beats_o == 64'(shadow_ins), "honest install-beat counter");
      require(port_busy_o == 64'(shadow_busy), "honest port-busy counter");
      require(port_busy_o == fetch_accepted_o + install_beats_o,
              "port busy = fetch + install grants");
      require(bound_hits_o >= 64'd4, "bound hits still counted at the end");
    end
    for (int c = 0; c < CONTEXT_COUNT; c++) require(inf_of(c) == '0, "no context left in flight");
    $display("C030_PASS tb_l1_program_port checks=%0d contexts=%0d fetch_outstanding=%0d fetch_bytes=%0d",
             checks, CONTEXT_COUNT, FO, FB);
    $finish;
  end
endmodule
