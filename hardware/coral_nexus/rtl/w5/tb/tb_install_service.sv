// tb_install_service.sv — C039 install-service regression.
//
// Wraps the real rtl/w5/install_service.sv against the real C030
// rtl/memory/l1_program_port.sv: the service owns the port's install
// channel, this bench owns the fetch channels (the fetch probe reads the
// actual port, never a shadow copy) and both sides of the sealed PROGRAM
// word stream (G2T frames + blob, T2G frames). Execution-reference pulses
// reach the service and the port on the same edge.
//
// Trusted setup metadata and the image text bytes are the REAL C038
// attempt-002 kernel texts (144 B each, executable PT_LOAD of the sealed
// images): the pytest service stages them into /out as $readmemh files
// (c039_img_{a,b,c}.hex one byte per line, c039_meta.hex one 64-bit word
// per line: per image {program_id, version, epoch, slot, text_base,
// text_len, pkg[4], digest[4]}), so the digest the service must measure is
// exactly sha256(the bytes it pulls). Image C is a host-derived third text
// (A with one byte flipped) used for fault-path installs; its trusted
// digest is sha256(its own bytes).
//
// Message ids and payload word counts come from the generated
// cn_offsets.svh (same sealed constants the DUT includes).
`timescale 1ns/1ps
module tb_install_service
  import c029_params_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT = C2_CONTEXTS
) ();
  /* verilator lint_off WIDTHEXPAND */
  `include "cn_offsets.svh"
  /* verilator lint_on WIDTHEXPAND */

  localparam int unsigned FO = 2;
  localparam int unsigned CTX_W = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT);
  localparam int unsigned GEN_W = 16, EPOCH_W = 32, SEQ_W = 32, PROG_ID_W = 8;
  localparam int unsigned KEY_W = CTX_W + GEN_W + EPOCH_W + SEQ_W;
  localparam int unsigned INF_W = $clog2(FO + 1);
  localparam int unsigned SRAM = 8192;
  localparam int unsigned FB = 32;
  localparam int unsigned WB = 4;
  localparam int unsigned ADDR_W = 32;
  localparam int unsigned SLOTS = 2;
  localparam int unsigned SLOT_BYTES = 4096;

  localparam logic [2:0]  I_OK = 3'd0;
  localparam logic [3:0]  F_OK = 4'd0, F_UNWRITTEN = 4'd5;
  // c034 wire fault codes.
  localparam logic [63:0] FC_INVALID_DESCRIPTOR = 64'd0;
  localparam logic [63:0] FC_ADDRESS_FAULT      = 64'd2;
  localparam logic [63:0] FC_STREAM_PROTO       = 64'd5;
  localparam logic [63:0] FC_L2_CAPACITY        = 64'd8;
  // service states (install_service.sv state_o encoding)
  localparam logic [3:0] ST_IDLE = 4'd0, ST_RECV = 4'd1, ST_HASH = 4'd2,
                         ST_COMMIT = 4'd3, ST_ACK = 4'd4, ST_WAIT_READY = 4'd5;

  logic clk = 0, rst_n = 0;

  // ---- trusted setup sideband ----
  logic         setup_valid = 0, setup_ready;
  logic [63:0]  setup_pid = 0, setup_ver = 0, setup_epoch = 0;
  logic [255:0] setup_pkg = 0, setup_digest = 0;
  logic [63:0]  setup_slot = 0, setup_base = 0, setup_len = 0;

  // ---- PROGRAM stream ----
  logic        g2t_valid = 0, g2t_ready;
  logic [63:0] g2t_word = 0;
  logic        t2g_valid, t2g_ready = 1;
  logic [63:0] t2g_word;
  logic        ack_handle_valid;
  logic [63:0] ack_handle;

  // ---- service <-> C030 install channel ----
  logic                    ins_valid, ins_ready;
  logic [ADDR_W-1:0]       ins_addr;
  logic [WB*8-1:0]         ins_data;
  logic [WB-1:0]           ins_be;
  logic [PROG_ID_W-1:0]    ins_pid, ins_rsp_pid;
  logic [EPOCH_W-1:0]      ins_epoch, ins_rsp_epoch;
  logic                    ins_rsp_valid, ins_rsp_ready;
  logic [2:0]              ins_rsp_code;

  // ---- exec refs: one pulse, seen by service and port on the same edge ----
  logic                   ref_inc = 0, ref_dec = 0;
  logic [PROG_ID_W-1:0]   ref_pid = 0;
  logic [EPOCH_W-1:0]     ref_ep = 0;

  // ---- fetch probe (bench owns the C030 fetch channels) ----
  logic                    fet_valid [CONTEXT_COUNT];
  logic                    fet_ready [CONTEXT_COUNT];
  logic [KEY_W-1:0]        fet_key [CONTEXT_COUNT];
  logic [ADDR_W-1:0]       fet_addr [CONTEXT_COUNT];
  logic [7:0]              fet_len [CONTEXT_COUNT];
  logic                    rsp_valid, rsp_ready = 0, rsp_fault;
  logic [KEY_W-1:0]        rsp_key;
  logic [FB*8-1:0]         rsp_data;
  logic [3:0]              rsp_code;
  logic [ADDR_W-1:0]       gate_addr = 0;
  logic                    fetch_allow;

  // ---- observability ----
  logic [3:0]      state_o;
  logic [SLOTS-1:0] resident_o, ready_o, slot_valid_o;
  logic            fault_o, fault_seen;
  logic [63:0]     fault_code;
  logic [63:0]     cnt_req, cnt_resp, cnt_beats, cnt_inst, cnt_warm, cnt_inv;
  logic [63:0]     p_fetch_acc, p_rsp_del, p_ins_beats, p_busy, p_bound;
  logic [CONTEXT_COUNT*INF_W-1:0] p_inf;
  logic            p_inv_err;

  install_service #(
    .SLOTS(SLOTS), .SLOT_BYTES(SLOT_BYTES), .SRAM_BYTES(SRAM),
    .SETUP_SLOTS(8), .WORD_BYTES(WB), .LINE_BYTES(FB),
    .ADDR_W(ADDR_W), .PROG_ID_W(PROG_ID_W), .EPOCH_W(EPOCH_W)
  ) svc (
    .clk(clk), .rst_n(rst_n),
    .setup_valid_i(setup_valid), .setup_ready_o(setup_ready),
    .setup_program_id_i(setup_pid), .setup_version_i(setup_ver),
    .setup_package_hash_i(setup_pkg), .setup_program_epoch_i(setup_epoch),
    .setup_slot_i(setup_slot), .setup_text_base_i(setup_base),
    .setup_text_len_i(setup_len), .setup_text_digest_i(setup_digest),
    .g2t_valid_i(g2t_valid), .g2t_ready_o(g2t_ready), .g2t_word_i(g2t_word),
    .t2g_valid_o(t2g_valid), .t2g_ready_i(t2g_ready), .t2g_word_o(t2g_word),
    .ack_handle_valid_o(ack_handle_valid), .ack_handle_o(ack_handle),
    .ins_valid_o(ins_valid), .ins_ready_i(ins_ready),
    .ins_addr_o(ins_addr), .ins_data_o(ins_data), .ins_be_o(ins_be),
    .ins_prog_id_o(ins_pid), .ins_epoch_o(ins_epoch),
    .ins_rsp_valid_i(ins_rsp_valid), .ins_rsp_ready_o(ins_rsp_ready),
    .ins_rsp_code_i(ins_rsp_code), .ins_rsp_prog_id_i(ins_rsp_pid),
    .ins_rsp_epoch_i(ins_rsp_epoch),
    .exec_ref_inc_i(ref_inc), .exec_ref_dec_i(ref_dec),
    .exec_ref_prog_id_i(ref_pid), .exec_ref_epoch_i(ref_ep),
    .fetch_addr_i(gate_addr), .fetch_allow_o(fetch_allow),
    .state_o(state_o), .resident_o(resident_o), .ready_o(ready_o),
    .slot_valid_o(slot_valid_o), .fault_o(fault_o), .fault_seen_o(fault_seen),
    .fault_code_o(fault_code), .read_req_count_o(cnt_req),
    .resp_bytes_o(cnt_resp), .ins_beats_o(cnt_beats),
    .install_count_o(cnt_inst), .warm_hit_count_o(cnt_warm),
    .inv_count_o(cnt_inv)
  );

  l1_program_port #(
    .CONTEXT_COUNT(CONTEXT_COUNT), .FETCH_OUTSTANDING(FO),
    .SRAM_BYTES(SRAM), .FETCH_BYTES(FB), .WORD_BYTES(WB),
    .INS_RSP_DEPTH(4), .REF_SLOTS(8),
    .PROG_ID_W(PROG_ID_W), .EPOCH_W(EPOCH_W), .GEN_W(GEN_W),
    .SEQ_W(SEQ_W), .ADDR_W(ADDR_W)
  ) port (
    .clk(clk), .rst_n(rst_n),
    .ins_valid(ins_valid), .ins_ready(ins_ready),
    .ins_addr(ins_addr), .ins_data(ins_data), .ins_be(ins_be),
    .ins_prog_id(ins_pid), .ins_epoch(ins_epoch),
    .ins_rsp_valid(ins_rsp_valid), .ins_rsp_ready(ins_rsp_ready),
    .ins_rsp_code(ins_rsp_code), .ins_rsp_prog_id(ins_rsp_pid),
    .ins_rsp_epoch(ins_rsp_epoch),
    .fet_valid(fet_valid), .fet_ready(fet_ready),
    .fet_key(fet_key), .fet_addr(fet_addr), .fet_len(fet_len),
    .rsp_valid(rsp_valid), .rsp_ready(rsp_ready),
    .rsp_key(rsp_key), .rsp_data(rsp_data),
    .rsp_fault(rsp_fault), .rsp_code(rsp_code),
    .ref_inc_valid(ref_inc), .ref_dec_valid(ref_dec),
    .ref_prog_id(ref_pid), .ref_epoch(ref_ep),
    .fetch_accepted_o(p_fetch_acc), .rsp_delivered_o(p_rsp_del),
    .install_beats_o(p_ins_beats), .port_busy_o(p_busy),
    .bound_hits_o(p_bound), .inf_o(p_inf), .invariant_error_o(p_inv_err)
  );

  always #5 clk = ~clk;
  initial begin #5_000_000; $fatal(1, "C039_ASSERT tb_install_service watchdog"); end

  int checks = 0;
  task automatic require(input logic ok, input string label);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C039_ASSERT tb_install_service %s", label);
  endtask

  // ---------------------------------------------------------------------
  // Staged trusted inputs ($readmemh, staged by the pytest service).
  logic [7:0]  img [0:3*4096-1];
  logic [63:0] meta [3][0:13];
  logic [63:0] meta_words [0:41];

  task automatic load_inputs();
    $readmemh("/out/c039_img_a.hex", img, 0*4096, 0*4096+4095);
    $readmemh("/out/c039_img_b.hex", img, 1*4096, 1*4096+4095);
    $readmemh("/out/c039_img_c.hex", img, 2*4096, 2*4096+4095);
    $readmemh("/out/c039_meta.hex", meta_words);
    for (int i = 0; i < 3; i++)
      for (int w = 0; w < 14; w++) meta[i][w] = meta_words[i*14 + w];
  endtask

  function automatic logic [255:0] meta_pkg(input int i);
    logic [255:0] r;
    for (int w = 0; w < 4; w++) r[w*64 +: 64] = meta[i][2 + w];
    return r;
  endfunction

  function automatic logic [255:0] meta_dig(input int i);
    logic [255:0] r;
    for (int w = 0; w < 4; w++) r[w*64 +: 64] = meta[i][10 + w];
    return r;
  endfunction

  // ---------------------------------------------------------------------
  // T2G frame monitor: every emitted word lands in the log; wait_frame
  // parses the next complete frame of an expected type out of it.
  logic [63:0] t2g_log [0:2047];
  int t2g_n, scan;

  always @(posedge clk) begin
    if (rst_n && t2g_valid && t2g_ready) begin
      t2g_log[t2g_n] <= t2g_word;
      t2g_n <= t2g_n + 1;
    end
  end

  function automatic logic frame_at(input int at, output logic [63:0] mtype,
                                    output logic [63:0] pw [16]);
    logic [63:0] nw;
    mtype = '0;
    for (int i = 0; i < 16; i++) pw[i] = '0;
    if (at + 4 > t2g_n) return 1'b0;
    nw = t2g_log[at + 1];
    if (at + 4 + int'(nw) > t2g_n) return 1'b0;
    mtype = t2g_log[at];
    for (int i = 0; i < int'(nw) && i < 16; i++) pw[i] = t2g_log[at + 4 + i];
    return 1'b1;
  endfunction

  function automatic int frame_words(input int at);
    if (at + 2 > t2g_n) return 0;
    return 4 + int'(t2g_log[at + 1]);
  endfunction

  // Wait for the frame of `want` at/after the scan cursor; everything
  // skipped is reported through `skipped` so a warm install can assert NO
  // READ_REQ appeared between its INSTALL and its ACK.
  task automatic wait_frame(input logic [63:0] want, output logic [63:0] pw [16],
                            output int nskipped, output logic [63:0] skipped [16]);
    logic [63:0] mt;
    logic [63:0] lpw [16];
    int guard;
    nskipped = 0;
    for (int i = 0; i < 16; i++) skipped[i] = '0;
    guard = 0;
    while (1) begin
      guard++;
      if (guard > 200000)
        $fatal(1, "C039_ASSERT tb_install_service timeout waiting for t2g frame %0d", want);
      if (frame_at(scan, mt, lpw)) begin
        if (mt == want) begin
          for (int i = 0; i < 16; i++) pw[i] = lpw[i];
          scan = scan + frame_words(scan);
          return;
        end else begin
          if (nskipped < 8) skipped[nskipped] = mt;
          nskipped = nskipped + 1;
          scan = scan + frame_words(scan);
        end
      end else @(negedge clk);
    end
  endtask

  // ---------------------------------------------------------------------
  // G2T drivers (one word per beat; send_word tolerates stream backpressure).
  task automatic send_word(input logic [63:0] w);
    @(negedge clk);
    g2t_valid = 1'b1;
    g2t_word = w;
    do @(negedge clk); while (!g2t_ready);
    g2t_valid = 1'b0;
  endtask

  // One sealed frame: {type, pwords, seq, 0, pw[0..pwords)}.
  task automatic send_frame(input logic [63:0] mtype,
                            input logic [63:0] pw [16], input int n);
    send_word(mtype);
    send_word(64'(n));
    send_word(64'd0);                 // message_seq (Group bookkeeping)
    send_word(64'd0);                 // header_reserved
    for (int i = 0; i < n; i++) send_word(pw[i]);
  endtask

  task automatic send_install(input logic [63:0] p, v, e, input logic [255:0] h,
                              input logic [63:0] slot);
    logic [63:0] pw [16];
    pw[0] = p; pw[1] = v;
    for (int i = 0; i < 4; i++) pw[2 + i] = h[i*64 +: 64];
    pw[6] = e; pw[7] = slot;
    send_frame(64'(CN_MSG_PROGRAM_INSTALL), pw, 8);
  endtask

  task automatic send_invalidate(input logic [63:0] p, v, e, input logic [255:0] h);
    logic [63:0] pw [16];
    pw[0] = p; pw[1] = v;
    for (int i = 0; i < 4; i++) pw[2 + i] = h[i*64 +: 64];
    pw[6] = e; pw[7] = '0;
    send_frame(64'(CN_MSG_PROGRAM_INVALIDATE), pw, 7);
  endtask

  task automatic send_ready(input logic [63:0] p, v, e, input logic [255:0] h,
                            input logic [63:0] handle);
    logic [63:0] pw [16];
    pw[0] = p; pw[1] = v;
    for (int i = 0; i < 4; i++) pw[2 + i] = h[i*64 +: 64];
    pw[6] = e; pw[7] = handle;
    send_frame(64'(CN_MSG_PROGRAM_READY), pw, 8);
  endtask

  // One READ_RESP chunk + its blob words: `clen` real bytes of img[i][off..],
  // optionally corrupted one byte (corrupt >= 0 flips that absolute index's
  // low bit); the tail blob word is zero-padded.
  task automatic send_resp(input logic [63:0] p, v, e, input logic [255:0] h,
                           input int i, input int off, input int clen,
                           input int corrupt);
    logic [63:0] pw [16];
    logic [63:0] w;
    int nw, bi;
    pw[0] = p; pw[1] = v;
    for (int k = 0; k < 4; k++) pw[2 + k] = h[k*64 +: 64];
    pw[6] = e; pw[7] = 64'(clen);
    send_frame(64'(CN_MSG_PROGRAM_READ_RESP), pw, 8);
    nw = (clen + 7) / 8;
    for (int wi = 0; wi < nw; wi++) begin
      w = '0;
      for (int b = 0; b < 8; b++) begin
        bi = off + wi * 8 + b;
        if (bi < off + clen) begin
          w[b*8 +: 8] = img[i*4096+bi];
          if (bi == corrupt) w[b*8 +: 8] = img[i*4096+bi] ^ 8'h01;
        end
      end
      send_word(w);
    end
  endtask

  task automatic send_setup(input int i);
    @(negedge clk);
    setup_valid = 1'b1;
    setup_pid = meta[i][0];
    setup_ver = meta[i][1];
    setup_pkg = meta_pkg(i);
    setup_epoch = meta[i][6];
    setup_slot = meta[i][7];
    setup_base = meta[i][8];
    setup_len = meta[i][9];
    setup_digest = meta_dig(i);
    @(negedge clk);
    require(setup_ready, "setup accepted");
    setup_valid = 1'b0;
  endtask

  // Serve a whole install stream in `sizes` byte chunks (must sum to len).
  task automatic serve_stream(input int i, input int sizes [$]);
    int off;
    logic [63:0] pw [16];
    logic [63:0] skip [16];
    int nsk;
    off = 0;
    foreach (sizes[s]) begin
      wait_frame(64'(CN_MSG_PROGRAM_READ_REQ), pw, nsk, skip);
      require(pw[0] == meta[i][0] && pw[1] == meta[i][1] && pw[6] == meta[i][6],
              "READ_REQ carries the pending identity");
      for (int k = 0; k < 4; k++)
        require(pw[2 + k] == meta_pkg(i)[k*64 +: 64], "READ_REQ package_hash echo");
      send_resp(meta[i][0], meta[i][1], meta[i][6], meta_pkg(i), i, off, sizes[s], -1);
      off += sizes[s];
    end
  endtask

  // Wait for the INSTALL_ACK frame and check identity + len + measured hash.
  task automatic wait_ack(input int i, input logic [255:0] eh,
                          input logic [63:0] elen, output logic [63:0] pw [16]);
    int nsk;
    logic [63:0] skip [16];
    wait_frame(64'(CN_MSG_PROGRAM_INSTALL_ACK), pw, nsk, skip);
    require(pw[0] == meta[i][0] && pw[1] == meta[i][1] && pw[6] == meta[i][6],
            "ACK carries the installed identity");
    for (int k = 0; k < 4; k++)
      require(pw[2 + k] == meta_pkg(i)[k*64 +: 64], "ACK package_hash echo");
    require(pw[7] == elen, "ACK installed_len");
    for (int k = 0; k < 4; k++)
      require(pw[8 + k] == eh[k*64 +: 64], "ACK measured_hash = real text digest");
  endtask

  // The handle sideband must hold a value for the whole ACK frame.
  logic [63:0] handle_seen;
  always @(posedge clk) begin
    if (ack_handle_valid) handle_seen <= ack_handle;
  end

  // Fetch probe through the real C030 port: one aligned line read.
  task automatic fetch_line(input [ADDR_W-1:0] addr, input logic [3:0] wantcode,
                            output logic [FB*8-1:0] data);
    fet_key[0] = {CTX_W'(0), 16'h0001, 32'(fet_seq), 32'h1};
    fet_seq++;
    fet_addr[0] = addr;
    fet_len[0] = 8'd32;
    @(negedge clk);
    fet_valid[0] = 1'b1;
    do @(negedge clk); while (!fet_ready[0]);
    fet_valid[0] = 1'b0;
    while (!rsp_valid) @(negedge clk);
    require(rsp_code == wantcode, "probe response code");
    data = rsp_data;
    rsp_ready = 1'b1;
    @(negedge clk);
    rsp_ready = 1'b0;
  endtask

  int unsigned fet_seq;

  function automatic logic [FB*8-1:0] exp_line(input int i, input int base);
    logic [FB*8-1:0] r;
    int g, len;
    len = int'(meta[i][9]);
    for (int b = 0; b < FB; b++) begin
      g = base + b;
      if (g < len) r[b*8 +: 8] = img[i*4096+g];
      else r[b*8 +: 8] = 8'h00;
    end
    return r;
  endfunction

  // Full probe of resident image i: every line inside its slot range reads
  // the real bytes (text + zero tail) through the actual port.
  task automatic probe_image(input int i);
    logic [FB*8-1:0] data;
    int base, len, padded;
    base = int'(meta[i][8]);
    len = int'(meta[i][9]);
    padded = ((len + FB - 1) / FB) * FB;
    gate_addr = ADDR_W'(base);
    #1;
    require(fetch_allow, "gate open on resident text");
    for (int l = 0; l < padded / FB; l++) begin
      fetch_line(ADDR_W'(base + l * FB), F_OK, data);
      require(data == exp_line(i, l * FB), "fetch bytes == real image text (+zero tail)");
    end
    gate_addr = ADDR_W'(base + padded);
    #1;
    require(!fetch_allow, "gate closed beyond the installed range");
    gate_addr = ADDR_W'(base);
    #1;
  endtask

  task automatic ref_op(input bit inc, input int i);
    @(negedge clk);
    ref_inc = inc;
    ref_dec = !inc;
    ref_pid = PROG_ID_W'(meta[i][0]);
    ref_ep = EPOCH_W'(meta[i][6]);
    @(negedge clk);
    ref_inc = 1'b0;
    ref_dec = 1'b0;
  endtask

  task automatic fault_expect(input logic [63:0] want, input string label);
    int guard;
    guard = 0;
    while (!fault_o) begin
      guard++;
      if (guard > 50000) $fatal(1, "C039_ASSERT tb_install_service %s: no fault", label);
      @(negedge clk);
    end
    require(fault_code == want, {label, ": fault_code"});
  endtask

  task automatic no_t2g_for(input int cycles, input string label);
    for (int c = 0; c < cycles; c++) begin
      @(negedge clk);
      if (t2g_valid && t2g_ready)
        $fatal(1, "C039_ASSERT tb_install_service %s: unexpected T2G frame", label);
    end
  endtask

  // ---------------------------------------------------------------------
  initial begin
    logic [63:0] pw [16];
    logic [63:0] skip [16];
    int nsk;
    logic [FB*8-1:0] data;
    logic [63:0] beats_before, req_before, resp_before, inv_before;

    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      fet_valid[c] = 1'b0; fet_key[c] = '0; fet_addr[c] = '0; fet_len[c] = '0;
    end
    fet_seq = 1;
    t2g_n = 0; scan = 0; handle_seen = '0;
    load_inputs();

    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    require(state_o == ST_IDLE && !fault_seen, "reset: idle, no fault");
    require(resident_o == '0 && ready_o == '0 && slot_valid_o == '0, "reset: no records");
    require(cnt_req == 0 && cnt_resp == 0 && cnt_beats == 0 && cnt_inst == 0, "reset: counters");
    gate_addr = 0;
    #1;
    require(!fetch_allow, "gate closed with no residency");

    // ------------------------------------------------------------------
    // 1) COLD INSTALL A (slot 0): trusted setup -> INSTALL -> READ_REQ x3
    //    -> RESP blobs (64+64+16) -> measured sha256(text) -> INSTALL_ACK
    //    -> PROGRAM_READY with the sideband handle -> byte-exact probe.
    send_setup(0);
    send_install(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), meta[0][7]);
    serve_stream(0, '{64, 64, 16});
    wait_ack(0, meta_dig(0), meta[0][9], pw);
    require(cnt_req == 64'd3, "cold install: one READ_REQ per chunk");
    require(cnt_resp == 64'(meta[0][9]), "cold install: real bytes consumed");
    require(cnt_beats == 64'(160 / 4), "commit wrote ceil(144/32)*32 bytes");
    require(resident_o == 2'b01, "slot 0 resident after commit");
    send_ready(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), handle_seen);
    repeat (4) @(negedge clk);
    require(state_o == ST_IDLE, "READY completes the install");
    require(ready_o == 2'b01, "slot 0 ready");
    require(!fault_seen, "no fault through cold install");
    probe_image(0);

    // ------------------------------------------------------------------
    // 2) WARM HIT: identical INSTALL for A -> immediate ACK, ZERO READ_REQ,
    //    same handle, same measurement (one shared completion point).
    req_before = cnt_req;
    beats_before = cnt_beats;
    send_install(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), meta[0][7]);
    wait_ack(0, meta_dig(0), meta[0][9], pw);
    require(cnt_req == req_before, "warm hit: ZERO READ_REQ");
    require(cnt_beats == beats_before, "warm hit: ZERO install beats");
    require(cnt_warm == 64'd1, "warm hit counted");
    send_ready(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), handle_seen);
    repeat (4) @(negedge clk);
    require(state_o == ST_IDLE && ready_o == 2'b01, "warm re-READY");

    // ------------------------------------------------------------------
    // 3) DIGEST MISMATCH: corrupt one streamed byte of C -> the measured
    //    sha256 disagrees with the trusted digest -> fault before a single
    //    port write; slot 1 stays free; the gate stays closed.
    beats_before = cnt_beats;
    send_setup(2);
    send_install(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), meta[2][8]);
    begin
      logic [63:0] lpw [16];
      wait_frame(64'(CN_MSG_PROGRAM_READ_REQ), lpw, nsk, skip);
      send_resp(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), 2, 0, 64, -1);
      wait_frame(64'(CN_MSG_PROGRAM_READ_REQ), lpw, nsk, skip);
      send_resp(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), 2, 64, 64, -1);
      wait_frame(64'(CN_MSG_PROGRAM_READ_REQ), lpw, nsk, skip);
      send_resp(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), 2, 128, 16, 130);
    end
    fault_expect(FC_INVALID_DESCRIPTOR, "digest mismatch faults");
    require(cnt_beats == beats_before, "verify-first: nothing written on mismatch");
    require(slot_valid_o == 2'b01, "slot 1 still free after mismatch");
    gate_addr = ADDR_W'(meta[2][8]);
    #1;
    require(!fetch_allow, "gate closed on faulted install");
    fetch_line(ADDR_W'(meta[2][8]), F_UNWRITTEN, data);

    // ------------------------------------------------------------------
    // 4) OVERSHOOT: a chunk_len beyond the trusted remaining length faults
    //    at the header (fail-stop, no unbounded drain, no ACK).
    send_install(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), meta[2][8]);
    begin
      logic [63:0] lpw [16];
      wait_frame(64'(CN_MSG_PROGRAM_READ_REQ), lpw, nsk, skip);
      lpw[0] = meta[2][0]; lpw[1] = meta[2][1];
      for (int k = 0; k < 4; k++) lpw[2 + k] = meta_pkg(2)[k*64 +: 64];
      lpw[6] = meta[2][6]; lpw[7] = 64'd160;
      send_frame(64'(CN_MSG_PROGRAM_READ_RESP), lpw, 8);  // header only, no blob
      fault_expect(FC_STREAM_PROTO, "overshoot chunk faults");
    end
    require(slot_valid_o == 2'b01, "overshoot leaves slot 1 free");

    // ------------------------------------------------------------------
    // 5) READY with a wrong handle faults (ACK-verified handle required).
    send_install(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), meta[2][8]);
    serve_stream(2, '{144});
    wait_ack(2, meta_dig(2), meta[2][9], pw);
    require(resident_o == 2'b11, "both slots resident");
    send_ready(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), handle_seen + 64'd1);
    repeat (4) @(negedge clk);
    require(ready_o == 2'b01, "wrong handle leaves slot 1 un-ready");
    gate_addr = ADDR_W'(meta[2][8]);                         // inside C's range
    #1;
    require(!fetch_allow, "committed-but-unready text is not fetchable");
    gate_addr = 0;
    send_ready(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2), handle_seen);
    repeat (4) @(negedge clk);
    require(ready_o == 2'b11, "both slots ready");
    probe_image(2);

    // ------------------------------------------------------------------
    // 6) INVALIDATE is a serial point: an exec ref holds the ACK, the gate
    //    closes immediately, a queued INSTALL for the SAME identity cannot
    //    warm-hit (payload deferred until the drain completes).
    ref_op(1'b1, 0);                                   // exec ref on A
    send_invalidate(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0));
    repeat (6) @(negedge clk);
    inv_before = cnt_inv;
    require(resident_o == 2'b10, "gate closed at the serial point");
    gate_addr = 0;
    #1;
    require(!fetch_allow, "no new warm hit while invalidating");
    no_t2g_for(8, "drain holds the INV_ACK until refs release");
    // Queue an INSTALL for A: while the drain is open the header itself
    // cannot be admitted (rx_admit gates INSTALL on !inv_active), so
    // g2t_ready stays low - that held request IS the bounded-stream
    // backpressure evidence.
    @(negedge clk);
    g2t_valid = 1'b1;
    g2t_word = 64'(CN_MSG_PROGRAM_INSTALL);
    repeat (4) @(negedge clk);
    require(!g2t_ready, "INSTALL header defers while the drain is open");
    // Release the ref inline; INV_ACK can now emit, r_inv clears, and the
    // same held frame is admitted word-by-word without being re-driven.
    ref_dec = 1'b1;
    ref_pid = PROG_ID_W'(meta[0][0]);
    ref_ep = EPOCH_W'(meta[0][6]);
    @(negedge clk);
    ref_dec = 1'b0;
    // The same frame completes word-by-word once the drain frees it; the
    // receiver kept the pending payload slot open through the stall.
    do @(negedge clk); while (!g2t_ready);             // word0 consumed
    g2t_word = 64'd8;                                  // payload_words
    do @(negedge clk); while (!g2t_ready);             // word1 consumed
    g2t_word = 64'd0;
    do @(negedge clk); while (!g2t_ready);             // message_seq
    g2t_word = 64'd0;
    do @(negedge clk); while (!g2t_ready);             // reserved
    g2t_word = meta[0][0];
    do @(negedge clk); while (!g2t_ready);
    g2t_word = meta[0][1];
    do @(negedge clk); while (!g2t_ready);
    for (int k = 0; k < 4; k++) begin
      g2t_word = meta_pkg(0)[k*64 +: 64];
      do @(negedge clk); while (!g2t_ready);
    end
    g2t_word = meta[0][6];
    do @(negedge clk); while (!g2t_ready);
    g2t_word = meta[0][7];
    do @(negedge clk); while (!g2t_ready);
    g2t_valid = 1'b0;
    begin
      logic [63:0] lpw [16];
      wait_frame(64'(CN_MSG_INVALIDATE_ACK), lpw, nsk, skip);
      require(lpw[0] == meta[0][0] && lpw[6] == meta[0][6], "INV_ACK identity");
    end
    require(cnt_inv == inv_before + 64'd1, "invalidate counted");
    require(slot_valid_o == 2'b10, "slot 0 dropped at ACK emit");
    // The deferred INSTALL now streams its remaining words and proceeds as
    // a COLD install (a "miss": real READ_REQs, not a hit).
    req_before = cnt_req;
    serve_stream(0, '{64, 64, 16});
    require(cnt_req == req_before + 64'd3, "missed install pulls the stream");
    wait_ack(0, meta_dig(0), meta[0][9], pw);
    send_ready(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), handle_seen);
    repeat (4) @(negedge clk);
    require(resident_o == 2'b11 && ready_o == 2'b11, "re-install completes cold");
    probe_image(0);

    // ------------------------------------------------------------------
    // 7) CMP-18: a conflicting re-write of C's trusted setup faults; an
    //    INSTALL of B's identity onto A's occupied slot faults before a
    //    single port write; A's bytes stay untouched.
    @(negedge clk);
    setup_valid = 1'b1;
    setup_pid = meta[2][0];
    setup_ver = meta[2][1];
    setup_pkg = meta_pkg(2);
    setup_epoch = meta[2][6];
    setup_slot = 64'd0;
    setup_base = 64'd0;
    setup_len = meta[2][9];
    setup_digest = meta_dig(2);
    @(negedge clk);
    setup_valid = 1'b0;
    fault_expect(FC_STREAM_PROTO, "conflicting trusted setup faults");
    // pid 9 has no setup at all: INSTALL faults closed at the trusted check.
    send_install(64'd9, 64'd1, 64'd1, meta_pkg(1), 64'd0);
    fault_expect(FC_INVALID_DESCRIPTOR, "unregistered identity faults closed");
    // Register B's metadata; its INSTALL onto slot 0 (held by A) is CMP-18.
    send_install(meta[1][0], meta[1][1], meta[1][6], meta_pkg(1), meta[0][7]);
    // B has no setup yet: also closed at the trusted check.
    fault_expect(FC_INVALID_DESCRIPTOR, "no setup for B yet");
    // A registered, fitting identity (fresh pid, A's own slot geometry)
    // reaches the slot-owner check: INSTALL onto A's occupied slot is
    // CMP-18 with no READ_REQ and no port write.
    @(negedge clk);
    setup_valid = 1'b1;
    setup_pid = 64'd11;
    setup_ver = 64'd1;
    setup_pkg = meta_pkg(2);
    setup_epoch = 64'd11;
    setup_slot = 64'd0;
    setup_base = 64'd0;
    setup_len = meta[0][9];
    setup_digest = meta_dig(0);
    @(negedge clk);
    setup_valid = 1'b0;
    req_before = cnt_req;
    beats_before = cnt_beats;
    send_install(64'd11, 64'd1, 64'd11, meta_pkg(2), 64'd0);
    fault_expect(FC_INVALID_DESCRIPTOR, "CMP-18: occupied slot rejects a conflicting install");
    require(cnt_req == req_before, "CMP-18 issues no READ_REQ");
    require(cnt_beats == beats_before, "CMP-18 writes no port bytes");
    probe_image(0);   // A's bytes untouched

    // ------------------------------------------------------------------
    // 8) CAPACITY: a trusted length beyond the slot faults closed on a
    //    free slot.
    send_invalidate(meta[2][0], meta[2][1], meta[2][6], meta_pkg(2));
    begin
      logic [63:0] lpw [16];
      wait_frame(64'(CN_MSG_INVALIDATE_ACK), lpw, nsk, skip);
    end
    require(slot_valid_o == 2'b01, "only slot 0 resident");
    @(negedge clk);
    setup_valid = 1'b1;
    setup_pid = 64'd10; setup_ver = 64'd1;
    setup_pkg = meta_pkg(0); setup_epoch = 64'd10;
    setup_slot = 64'd4096; setup_base = 64'd4096;
    setup_len = 64'd4097; setup_digest = meta_dig(0);
    @(negedge clk);
    setup_valid = 1'b0;
    send_install(64'd10, 64'd1, 64'd10, meta_pkg(0), 64'd4096);
    fault_expect(FC_L2_CAPACITY, "oversized text faults");

    // ------------------------------------------------------------------
    // 9) IDENTITY CHANGE: after slot 1 is invalidated, a DIFFERENT
    //    program (B) installs cold onto C's stale bytes and takes over.
    send_setup(1);
    send_install(meta[1][0], meta[1][1], meta[1][6], meta_pkg(1), meta[1][7]);
    serve_stream(1, '{72, 8, 64});
    wait_ack(1, meta_dig(1), meta[1][9], pw);
    send_ready(meta[1][0], meta[1][1], meta[1][6], meta_pkg(1), handle_seen);
    repeat (4) @(negedge clk);
    require(resident_o == 2'b11 && ready_o == 2'b11, "identity change installs");
    probe_image(1);

    // ------------------------------------------------------------------
    // 10) INVALIDATE of an absent identity is answered idempotently.
    send_invalidate(64'd77, 64'd1, 64'd1, meta_pkg(0));
    begin
      logic [63:0] lpw [16];
      wait_frame(64'(CN_MSG_INVALIDATE_ACK), lpw, nsk, skip);
      require(lpw[0] == 64'd77, "noop invalidate still ACKs");
    end

    // ------------------------------------------------------------------
    // 11) PENDING-INSTALL ABORT: an INVALIDATE naming the in-flight
    //    install kills it (staged bytes die unpublished); the deferred
    //    re-install then misses and pulls the stream cold.
    send_invalidate(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0));
    begin
      logic [63:0] lpw [16];
      wait_frame(64'(CN_MSG_INVALIDATE_ACK), lpw, nsk, skip);
    end
    send_install(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), meta[0][7]);
    begin
      logic [63:0] lpw [16];
      wait_frame(64'(CN_MSG_PROGRAM_READ_REQ), lpw, nsk, skip);
      send_resp(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), 0, 0, 64, -1);
      resp_before = cnt_resp;
      // INVALIDATE the pending install mid-stream: the stream dies and an
      // idempotent ACK (no record was ever published) carries the identity.
      send_invalidate(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0));
      wait_frame(64'(CN_MSG_INVALIDATE_ACK), lpw, nsk, skip);
      require(lpw[0] == meta[0][0], "abort ACK carries the identity");
      require(slot_valid_o == 2'b10, "aborted install never publishes");
      require(cnt_resp == resp_before, "abort stops the stream mid-flight");
    end
    // Re-install A cold; the stream restarts from byte zero.
    send_install(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), meta[0][7]);
    serve_stream(0, '{144});
    wait_ack(0, meta_dig(0), meta[0][9], pw);
    send_ready(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), handle_seen);
    repeat (4) @(negedge clk);
    require(ready_o == 2'b11, "re-installed after abort");

    // ------------------------------------------------------------------
    // 12) T2G backpressure: an INSTALL_ACK is held whole under ready=0.
    t2g_ready = 1'b0;
    send_invalidate(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0));
    begin
      logic [63:0] lpw [16];
      // The INV_ACK cannot emit while ready is low; release and parse it.
      repeat (4) @(negedge clk);
      require(!t2g_valid || t2g_word == 64'(CN_MSG_INVALIDATE_ACK),
              "INV_ACK stalls whole under backpressure");
      t2g_ready = 1'b1;
      wait_frame(64'(CN_MSG_INVALIDATE_ACK), lpw, nsk, skip);
    end
    // The deferred INSTALL (none queued) — re-install then stall its ACK.
    send_install(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), meta[0][7]);
    serve_stream(0, '{144});
    t2g_ready = 1'b0;
    repeat (2) @(negedge clk);
    while (!t2g_valid) @(negedge clk);
    require(t2g_word == 64'(CN_MSG_PROGRAM_INSTALL_ACK), "stalls before the ACK frame");
    begin
      logic [63:0] held;
      held = t2g_word;
      repeat (4) begin
        @(negedge clk);
        require(t2g_valid && t2g_word == held, "ACK held stable under backpressure");
      end
    end
    t2g_ready = 1'b1;
    wait_ack(0, meta_dig(0), meta[0][9], pw);
    send_ready(meta[0][0], meta[0][1], meta[0][6], meta_pkg(0), handle_seen);
    repeat (4) @(negedge clk);

    // ------------------------------------------------------------------
    // 13) Terminal state: bitmaps and counters reflect the whole run.
    require(state_o == ST_IDLE, "ends idle");
    require(resident_o == 2'b11 && ready_o == 2'b11, "A and B resident+ready");
    require(cnt_inst >= 64'd4, "cold installs counted");
    require(cnt_warm >= 64'd1, "warm hit counted");
    require(cnt_inv >= 64'd4, "invalidates counted");
    require(!p_inv_err, "port invariants hold throughout");
    probe_image(0);
    probe_image(1);

    $display("C039_PASS tb_install_service contexts=%0d checks=%0d installs=%0d warm=%0d inv=%0d",
             CONTEXT_COUNT, checks, cnt_inst, cnt_warm, cnt_inv);
    $finish;
  end
endmodule
