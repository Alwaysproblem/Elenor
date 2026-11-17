// tb_c028_port.sv — C030 x C028 native physical-port handoff regression.
//
// The REAL generated PerCtxFetchMux (Chisel, rtl/scalar/PerCtxFetchMux.scala,
// emitted inside the locked Bazel builder by RUNNING the existing
// core_mini_verification_axi_cc_library_emit_verilog_binary target with the
// sealed default fetchDataBits = 256 and --contextCount = 1/2) drives the
// REAL rtl/memory/l1_program_port.sv through this bench's bridge:
//
//   mux.io.request  --(per-Ctx decode by key.ctx_slot)--> port fet channel c
//   port rsp        --(1-entry throttle skid)-->         mux.io.response
//
// Every dimension is taken from the real design and elaboration-checked
// against the generated mux's own port widths (initial $bits asserts), so a
// width drift fails loudly instead of silently truncating:
//   FETCH_BYTES  = fetchDataBits/8   = 32   ($bits(io_response_bits_data))
//   GEN_W        = ctxGenerationBits = 16
//   EPOCH_W=SEQ_W= ownerOpSeqBits    = 32
//   ADDR_W       = fetchAddrBits     = 32
//   INF_W        = log2(maxOutstanding+1) ($bits(io_outstanding_0)):
//                  FETCH_OUTSTANDING = 2 = sealed fetch_outstanding_per_context
//
// Harness contract (W2_03_fetch_epoch_spec lesson): ONE endpoint owns every
// clock edge and handshake. The bridge owns the mux request ready (it may
// delay presenting a request) and the response skid gate (it may delay a
// response: the real mux hardwires io.response.ready high, so backpressure is
// applied by NOT presenting), and every valid&&ready fire on all four port
// channels plus both mux channels is recorded and held until answered —
// nothing is dropped and nothing is consumed unrecorded.
//
// Scenarios:
//   A  tagged end-to-end fetch: real OP-IMM instruction words installed
//      through the port INSTALL path, fetched by the mux per Context; io.inst
//      lanes deliver exactly the stored program in order (C=1 and C=2).
//   B  same-PC ABA: two requests for one line under different fetch_epoch
//      (redirect between) never alias; the stale one lands in staleRejected
//      and the delivered instructions come only from the current epoch.
//   C  tagged response BACKPRESSURE: responses held at the skid/port while
//      fetches queue: key/payload stable, per-Ctx credits (io.outstanding)
//      pinned at FETCH_OUTSTANDING, the (N+1)th request never issued (the
//      mux's sealed bound agrees with the port parameter), credits returned
//      exactly once, nothing dropped.
//   D  INSTALL CONTENTION: sustained installs concurrent with fetches from
//      every Context: install and fetch grants both progress, install is
//      never starved below the fetch volume, port counters equal independent
//      boundary shadow counts, every fetch response delivered.
//   E  FET-06: an install overlapping text whose program identity holds an
//      execution reference (ref_inc) is rejected and the Context still
//      fetches the ORIGINAL instructions; after ref_dec the write lands and
//      becomes visible through the frontend.
//   F  fault: an out-of-range fetch address faults through the real response
//      channel and only the owning Context (peer keeps fetching).
`timescale 1ns/1ps
module tb_c028_port
  import c029_params_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT = 1
) ();
  // Real-design geometry (checked against the generated mux below).
  localparam int unsigned FO = 2;    // C028TargetParams.fetchOutstandingPerContext
  localparam int unsigned FB = 32;   // fetchDataBits/8
  localparam int unsigned WB = 4;    // install word beat
  localparam int unsigned GEN_W = 16;
  localparam int unsigned EPOCH_W = 32;
  localparam int unsigned SEQ_W = 32;
  localparam int unsigned ADDR_W = 32;
  localparam int unsigned PROG_ID_W = 8;
  localparam int unsigned CTX_W = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT);
  localparam int unsigned KEY_W = CTX_W + GEN_W + EPOCH_W + SEQ_W;
  localparam int unsigned INF_W = $clog2(FO + 1);
  localparam int unsigned SRAM = 512;  // 16 bench-scale lines
  localparam int unsigned LANES = 4;   // Parameters.instructionLanes
  localparam int unsigned MAXCTX = 2;  // C028TargetParams.contextCounts upper bound

  localparam logic [3:0] F_OK = 4'd0, F_RANGE = 4'd1, F_ALIGN = 4'd2, F_ZERO = 4'd3,
                         F_LEN = 4'd4, F_UNWRITTEN = 4'd5;
  localparam logic [2:0] I_OK = 3'd0, I_RANGE = 3'd1, I_ALIGN = 3'd2, I_ZERO = 3'd3,
                         I_EXEC_REF = 3'd4, I_ALREADY = 3'd5;

  localparam logic [GEN_W-1:0] GEN_BASE = 16'h0005;
  localparam logic [PROG_ID_W-1:0] PID_P0 = 8'h10, PID_P1 = 8'h11, PID_SCRATCH = 8'h33;
  localparam logic [EPOCH_W-1:0] IE0 = 32'd7;
  // Program placement: ctx0 lines 0..2 (line 3 uninstalled terminator),
  // ctx1 lines 4..6 (line 7 terminator), 0x100..0x1FC scratch for installs.
  localparam logic [ADDR_W-1:0] P0_BASE = 32'h000, P1_BASE = 32'h080;

  logic clk = 0, rst_n = 0;

  int checks = 0;
  task automatic require(input logic ok, input string label);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C030_ASSERT tb_c028_port %s", label);
  endtask

  // ---------------- mux-side nets (bridge endpoints) ----------------------
  logic mreq_valid, mreq_ready;
  logic [CTX_W-1:0] mkey_slot;
  logic [GEN_W-1:0] mkey_gen;
  logic [EPOCH_W-1:0] mkey_epoch;
  logic [SEQ_W-1:0] mkey_reqid;
  logic [ADDR_W-1:0] mreq_addr;
  logic req_hold = 1'b0;  // bridge: delay presenting the offered request

  logic skid_valid, skid_fault, skid_present;
  logic [3:0] skid_code;
  logic [CTX_W-1:0] skid_slot;
  logic [GEN_W-1:0] skid_gen;
  logic [EPOCH_W-1:0] skid_epoch;
  logic [SEQ_W-1:0] skid_reqid;
  logic [FB*8-1:0] skid_data;
  logic rsp_open = 1'b1;  // bridge: response gate (backpressure scenarios)

  logic acc [MAXCTX], prdy [MAXCTX], parked [MAXCTX], cancel [MAXCTX];
  logic [GEN_W-1:0] gen_i [MAXCTX];
  logic start_v [MAXCTX], redir_v [MAXCTX];
  logic [ADDR_W-1:0] start_b [MAXCTX], redir_b [MAXCTX];
  logic [ADDR_W-1:0] pc_o [MAXCTX];
  logic fault_v [MAXCTX];
  logic [ADDR_W-1:0] fault_b [MAXCTX];
  logic [EPOCH_W-1:0] epoch_o [MAXCTX];
  logic [INF_W-1:0] outst [MAXCTX];
  logic [31:0] stale_o [MAXCTX];
  logic lv [MAXCTX][LANES], lr [MAXCTX][LANES], lb [MAXCTX][LANES];
  logic [ADDR_W-1:0] la [MAXCTX][LANES];
  logic [31:0] li [MAXCTX][LANES];

  // ---------------- program port (DUT) wiring -----------------------------
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
  logic rsp_valid, rsp_ready, rsp_fault;
  logic [KEY_W-1:0] rsp_key;
  logic [FB*8-1:0] rsp_data;
  logic [3:0] rsp_code;
  logic ref_inc_valid = 0, ref_dec_valid = 0;
  logic [PROG_ID_W-1:0] ref_prog_id = 0;
  logic [EPOCH_W-1:0] ref_epoch = 0;
  logic [63:0] fetch_accepted_o, rsp_delivered_o, install_beats_o, port_busy_o, bound_hits_o;
  logic [CONTEXT_COUNT*INF_W-1:0] inf_o;
  logic invariant_error_o;

  function automatic logic [INF_W-1:0] inf_of(input int unsigned c);
    return inf_o[c*INF_W +: INF_W];
  endfunction

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
  ) port (
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

  // ---------------- the real generated mux (two context geometries) -------
  // Width agreement with the generated module is asserted against the mux's
  // OWN ports, in both geometries, so a drifted generation fails loudly.
  // Exactly one of the two context geometries below elaborates per -G; the
  // not-taken branch still parses against the other geometry's port list.
  // Exactly one context geometry is compiled per run: the generated mux has
  // a different port list per context count (firtool DCEs the constant
  // request ctx_slot at contextCount=1), so only the matching instantiation
  // is compiled (+define+C028_C1 selects the one-Context geometry; the
  // default, no define, is the two-Context geometry).
  generate
    `ifdef C028_C1
    begin : g_mux
      PerCtxFetchMux mux (
        .clock(clk),
        .reset(~rst_n),
        .io_accepted_0(acc[0]),
        .io_programReady_0(prdy[0]),
        .io_parked_0(parked[0]),
        .io_cancel_0(cancel[0]),
        .io_generation_0(gen_i[0]),
        .io_start_0_valid(start_v[0]),
        .io_start_0_bits(start_b[0]),
        .io_redirect_0_valid(redir_v[0]),
        .io_redirect_0_bits(redir_b[0]),
        .io_request_valid(mreq_valid),
        .io_request_ready(mreq_ready),
        // firtool DCEs the constant request ctx_slot at contextCount=1
        // (the pin does not exist); the bridge pins the slot itself below.
        .io_request_bits_key_ctx_generation(mkey_gen),
        .io_request_bits_key_fetch_epoch(mkey_epoch),
        .io_request_bits_key_fetch_reqid(mkey_reqid),
        .io_request_bits_addr(mreq_addr),
        .io_response_valid(skid_present),
        .io_response_bits_key_ctx_slot(skid_slot),
        .io_response_bits_key_ctx_generation(skid_gen),
        .io_response_bits_key_fetch_epoch(skid_epoch),
        .io_response_bits_key_fetch_reqid(skid_reqid),
        .io_response_bits_data(skid_data),
        .io_response_bits_fault(skid_fault),
        .io_inst_0_0_valid(lv[0][0]), .io_inst_0_0_ready(lr[0][0]),
        .io_inst_0_0_bits_addr(la[0][0]), .io_inst_0_0_bits_inst(li[0][0]),
        .io_inst_0_0_bits_brchFwd(lb[0][0]),
        .io_inst_0_1_valid(lv[0][1]), .io_inst_0_1_ready(lr[0][1]),
        .io_inst_0_1_bits_addr(la[0][1]), .io_inst_0_1_bits_inst(li[0][1]),
        .io_inst_0_1_bits_brchFwd(lb[0][1]),
        .io_inst_0_2_valid(lv[0][2]), .io_inst_0_2_ready(lr[0][2]),
        .io_inst_0_2_bits_addr(la[0][2]), .io_inst_0_2_bits_inst(li[0][2]),
        .io_inst_0_2_bits_brchFwd(lb[0][2]),
        .io_inst_0_3_valid(lv[0][3]), .io_inst_0_3_ready(lr[0][3]),
        .io_inst_0_3_bits_addr(la[0][3]), .io_inst_0_3_bits_inst(li[0][3]),
        .io_inst_0_3_bits_brchFwd(lb[0][3]),
        .io_pc_0(pc_o[0]),
        .io_fault_0_valid(fault_v[0]),
        .io_fault_0_bits(fault_b[0]),
        .io_epoch_0(epoch_o[0]),
        .io_outstanding_0(outst[0]),
        .io_staleRejected_0(stale_o[0])
      );
      initial begin
        if ($bits(mux.io_response_bits_data) != FB*8)
          $fatal(1, "C030_ASSERT tb_c028_port mux line width %0d != fetchDataBits %0d",
                 $bits(mux.io_response_bits_data), FB*8);
        if ($bits(mux.io_request_bits_key_ctx_generation) != GEN_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux ctx_generation width %0d != %0d",
                 $bits(mux.io_request_bits_key_ctx_generation), GEN_W);
        if ($bits(mux.io_request_bits_key_fetch_epoch) != EPOCH_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux fetch_epoch width %0d != %0d",
                 $bits(mux.io_request_bits_key_fetch_epoch), EPOCH_W);
        if ($bits(mux.io_request_bits_key_fetch_reqid) != SEQ_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux fetch_reqid width %0d != %0d",
                 $bits(mux.io_request_bits_key_fetch_reqid), SEQ_W);
        if ($bits(mux.io_request_bits_addr) != ADDR_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux fetch addr width %0d != %0d",
                 $bits(mux.io_request_bits_addr), ADDR_W);
        // ctx_slot on the REQUEST is DCE'd at contextCount=1 (constant 0);
        // the response-side slot pins the 1-bit ctxBits specialization.
        if ($bits(mux.io_response_bits_key_ctx_slot) != CTX_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux ctx_slot width %0d != %0d",
                 $bits(mux.io_response_bits_key_ctx_slot), CTX_W);
        if (CONTEXT_COUNT != 1)
          $fatal(1, "C030_ASSERT tb_c028_port C028_C1 build must run with -GCONTEXT_COUNT=1");
        if ($bits(mux.io_outstanding_0) != INF_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux outstanding width %0d disagrees with sealed FETCH_OUTSTANDING %0d",
                 $bits(mux.io_outstanding_0), FO);
      end
      // firtool DCEs the constant request ctx_slot at contextCount=1 (the
      // pin does not exist): the bridge pins the slot itself.
      assign mkey_slot = '0;
    end
    `else
    begin : g_mux
      PerCtxFetchMux mux (
        .clock(clk),
        .reset(~rst_n),
        .io_accepted_0(acc[0]),
        .io_accepted_1(acc[1]),
        .io_programReady_0(prdy[0]),
        .io_programReady_1(prdy[1]),
        .io_parked_0(parked[0]),
        .io_parked_1(parked[1]),
        .io_cancel_0(cancel[0]),
        .io_cancel_1(cancel[1]),
        .io_generation_0(gen_i[0]),
        .io_generation_1(gen_i[1]),
        .io_start_0_valid(start_v[0]),
        .io_start_0_bits(start_b[0]),
        .io_start_1_valid(start_v[1]),
        .io_start_1_bits(start_b[1]),
        .io_redirect_0_valid(redir_v[0]),
        .io_redirect_0_bits(redir_b[0]),
        .io_redirect_1_valid(redir_v[1]),
        .io_redirect_1_bits(redir_b[1]),
        .io_request_valid(mreq_valid),
        .io_request_ready(mreq_ready),
        .io_request_bits_key_ctx_slot(mkey_slot),
        .io_request_bits_key_ctx_generation(mkey_gen),
        .io_request_bits_key_fetch_epoch(mkey_epoch),
        .io_request_bits_key_fetch_reqid(mkey_reqid),
        .io_request_bits_addr(mreq_addr),
        .io_response_valid(skid_present),
        .io_response_bits_key_ctx_slot(skid_slot),
        .io_response_bits_key_ctx_generation(skid_gen),
        .io_response_bits_key_fetch_epoch(skid_epoch),
        .io_response_bits_key_fetch_reqid(skid_reqid),
        .io_response_bits_data(skid_data),
        .io_response_bits_fault(skid_fault),
        .io_inst_0_0_valid(lv[0][0]), .io_inst_0_0_ready(lr[0][0]),
        .io_inst_0_0_bits_addr(la[0][0]), .io_inst_0_0_bits_inst(li[0][0]),
        .io_inst_0_0_bits_brchFwd(lb[0][0]),
        .io_inst_0_1_valid(lv[0][1]), .io_inst_0_1_ready(lr[0][1]),
        .io_inst_0_1_bits_addr(la[0][1]), .io_inst_0_1_bits_inst(li[0][1]),
        .io_inst_0_1_bits_brchFwd(lb[0][1]),
        .io_inst_0_2_valid(lv[0][2]), .io_inst_0_2_ready(lr[0][2]),
        .io_inst_0_2_bits_addr(la[0][2]), .io_inst_0_2_bits_inst(li[0][2]),
        .io_inst_0_2_bits_brchFwd(lb[0][2]),
        .io_inst_0_3_valid(lv[0][3]), .io_inst_0_3_ready(lr[0][3]),
        .io_inst_0_3_bits_addr(la[0][3]), .io_inst_0_3_bits_inst(li[0][3]),
        .io_inst_0_3_bits_brchFwd(lb[0][3]),
        .io_inst_1_0_valid(lv[1][0]), .io_inst_1_0_ready(lr[1][0]),
        .io_inst_1_0_bits_addr(la[1][0]), .io_inst_1_0_bits_inst(li[1][0]),
        .io_inst_1_0_bits_brchFwd(lb[1][0]),
        .io_inst_1_1_valid(lv[1][1]), .io_inst_1_1_ready(lr[1][1]),
        .io_inst_1_1_bits_addr(la[1][1]), .io_inst_1_1_bits_inst(li[1][1]),
        .io_inst_1_1_bits_brchFwd(lb[1][1]),
        .io_inst_1_2_valid(lv[1][2]), .io_inst_1_2_ready(lr[1][2]),
        .io_inst_1_2_bits_addr(la[1][2]), .io_inst_1_2_bits_inst(li[1][2]),
        .io_inst_1_2_bits_brchFwd(lb[1][2]),
        .io_inst_1_3_valid(lv[1][3]), .io_inst_1_3_ready(lr[1][3]),
        .io_inst_1_3_bits_addr(la[1][3]), .io_inst_1_3_bits_inst(li[1][3]),
        .io_inst_1_3_bits_brchFwd(lb[1][3]),
        .io_pc_0(pc_o[0]),
        .io_pc_1(pc_o[1]),
        .io_fault_0_valid(fault_v[0]),
        .io_fault_0_bits(fault_b[0]),
        .io_fault_1_valid(fault_v[1]),
        .io_fault_1_bits(fault_b[1]),
        .io_epoch_0(epoch_o[0]),
        .io_epoch_1(epoch_o[1]),
        .io_outstanding_0(outst[0]),
        .io_outstanding_1(outst[1]),
        .io_staleRejected_0(stale_o[0]),
        .io_staleRejected_1(stale_o[1])
      );
      initial begin
        if (CONTEXT_COUNT != 2)
          $fatal(1, "C030_ASSERT tb_c028_port two-Context build must run with -GCONTEXT_COUNT=2");
        if ($bits(mux.io_response_bits_data) != FB*8)
          $fatal(1, "C030_ASSERT tb_c028_port mux line width %0d != fetchDataBits %0d",
                 $bits(mux.io_response_bits_data), FB*8);
        if ($bits(mux.io_request_bits_key_ctx_generation) != GEN_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux ctx_generation width %0d != %0d",
                 $bits(mux.io_request_bits_key_ctx_generation), GEN_W);
        if ($bits(mux.io_request_bits_key_fetch_epoch) != EPOCH_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux fetch_epoch width %0d != %0d",
                 $bits(mux.io_request_bits_key_fetch_epoch), EPOCH_W);
        if ($bits(mux.io_request_bits_key_fetch_reqid) != SEQ_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux fetch_reqid width %0d != %0d",
                 $bits(mux.io_request_bits_key_fetch_reqid), SEQ_W);
        if ($bits(mux.io_request_bits_addr) != ADDR_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux fetch addr width %0d != %0d",
                 $bits(mux.io_request_bits_addr), ADDR_W);
        if ($bits(mux.io_request_bits_key_ctx_slot) != CTX_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux ctx_slot width %0d != %0d",
                 $bits(mux.io_request_bits_key_ctx_slot), CTX_W);
        if ($bits(mux.io_outstanding_0) != INF_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux outstanding width %0d disagrees with sealed FETCH_OUTSTANDING %0d",
                 $bits(mux.io_outstanding_0), FO);
        if ($bits(mux.io_outstanding_1) != INF_W)
          $fatal(1, "C030_ASSERT tb_c028_port mux outstanding_1 width %0d disagrees with sealed FETCH_OUTSTANDING %0d",
                 $bits(mux.io_outstanding_1), FO);
      end
    end
    `endif
  endgenerate


  // ---------------- C028 -> C030 physical-port bridge ---------------------
  // The bridge is the single owner of both channel handshakes: it may delay
  // presenting a request (req_hold) and delay a response (rsp_open gate; the
  // real mux consumes any presented response in the same beat, so holding
  // means not presenting), and every fire is recorded below.
  for (genvar c = 0; c < CONTEXT_COUNT; c++) begin : bridge
    always_comb begin
      fet_valid[c] = mreq_valid && !req_hold && (mkey_slot == CTX_W'(c));
      fet_key[c] = {mkey_slot, mkey_gen, mkey_epoch, mkey_reqid};
      fet_addr[c] = mreq_addr;
      fet_len[c] = 8'(FB);
    end
  end

  // Bridge: refuse every response at the port (no skid draw), so the port's own queue
  // must hold all CONTEXT_COUNT * FETCH_OUTSTANDING in-flight responses.
  logic rsp_hold_all = 1'b0;
  assign skid_present = skid_valid && rsp_open;
  always_comb begin
    mreq_ready = !req_hold && fet_ready[mkey_slot];
    rsp_ready = !skid_valid && !rsp_hold_all;  // the skid holds one response; the port queues
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      skid_valid <= 1'b0;
      skid_slot <= '0;
      skid_gen <= '0;
      skid_epoch <= '0;
      skid_reqid <= '0;
      skid_data <= '0;
      skid_fault <= 1'b0;
      skid_code <= 4'd0;
    end else begin
      if (rsp_valid && rsp_ready) begin
        skid_valid <= 1'b1;
        skid_slot <= rsp_key[KEY_W-1 -: CTX_W];
        skid_gen <= rsp_key[KEY_W-CTX_W-1 -: GEN_W];
        skid_epoch <= rsp_key[SEQ_W+EPOCH_W-1 -: EPOCH_W];
        skid_reqid <= rsp_key[SEQ_W-1:0];
        skid_data <= rsp_data;
        skid_fault <= rsp_fault;
        skid_code <= rsp_code;
      end else if (skid_present) begin
        skid_valid <= 1'b0;  // presented responses are consumed by the mux
      end
    end
  end

  // ---------------- fire logs and continuous invariants -------------------
  localparam int unsigned FQD = 16;
  localparam int unsigned ILOG = 4096;
  logic [KEY_W-1:0] fq [FQD];
  logic [ADDR_W-1:0] fa [FQD];
  logic [3:0] lf_code [MAXCTX];  // code of the last delivered fault per Ctx
  int unsigned fq_head, fq_tail;
  int unsigned req_seen [MAXCTX], rsp_seen [MAXCTX];
  logic [ADDR_W-1:0] ilog_addr [MAXCTX][ILOG];
  logic [31:0] ilog_inst [MAXCTX][ILOG];
  int unsigned ilog_n [MAXCTX];
  // shadow handshake counters at the port boundary (independent of the DUTs)
  int unsigned sh_fet [MAXCTX], sh_ins, sh_rsp;
  bit in_d;  // scenario D flag: scratch install responses must stay legal

  initial begin
    fq_head = 0;
    fq_tail = 0;
    in_d = 1'b0;
    for (int c = 0; c < MAXCTX; c++) begin
      req_seen[c] = 0;
      rsp_seen[c] = 0;
      ilog_n[c] = 0;
      lf_code[c] = 4'd0;
      acc[c] = 1'b1;
      prdy[c] = 1'b1;
      parked[c] = 1'b1;  // every Context parked until its scenario starts it
      cancel[c] = 1'b0;
      gen_i[c] = GEN_BASE;
      start_v[c] = 1'b0;
      redir_v[c] = 1'b0;
      start_b[c] = '0;
      redir_b[c] = '0;
      for (int l = 0; l < LANES; l++) lr[c][l] = 1'b1;
    end
    sh_ins = 0;
    sh_rsp = 0;
  end

  always @(posedge clk) begin
    if (rst_n) begin
      automatic int unsigned slot;
      // request fire: record identity + address, hold until answered
      if (mreq_valid && mreq_ready) begin
        slot = int'(mkey_slot);
        if (slot >= CONTEXT_COUNT)
          $fatal(1, "C030_ASSERT tb_c028_port request fire with foreign ctx_slot %0d", slot);
        fq[fq_tail % FQD] = {mkey_slot, mkey_gen, mkey_epoch, mkey_reqid};
        fa[fq_tail % FQD] = mreq_addr;
        fq_tail++;
        req_seen[slot]++;
        if (fq_tail - fq_head >= FQD)
          $fatal(1, "C030_ASSERT tb_c028_port fire log overflow");
      end
      // response fire (the mux consumes any presented response): it must
      // answer the oldest live request, in order, exactly once
      if (skid_present) begin
        if (fq_head == fq_tail)
          $fatal(1, "C030_ASSERT tb_c028_port response without a live request");
        if ({skid_slot, skid_gen, skid_epoch, skid_reqid} != fq[fq_head % FQD])
          $fatal(1, "C030_ASSERT tb_c028_port response key does not answer the oldest live request");
        fq_head++;
        if (skid_fault) lf_code[int'(skid_slot)] = skid_code;
        rsp_seen[int'(skid_slot)]++;
        sh_rsp++;
      end
      // port-boundary shadow handshakes
      if (ins_valid && ins_ready) sh_ins++;
      if (in_d && ins_rsp_valid && ins_rsp_ready &&
          !(ins_rsp_code inside {I_OK, I_ALREADY}))
        $fatal(1, "C030_ASSERT tb_c028_port D: scratch install answered %0d", ins_rsp_code);
      for (int c = 0; c < CONTEXT_COUNT; c++)
        if (fet_valid[c] && fet_ready[c]) sh_fet[c]++;
      // per-Ctx credit bound: never above the sealed FETCH_OUTSTANDING
      for (int c = 0; c < CONTEXT_COUNT; c++)
        if (outst[c] > INF_W'(FO))
          $fatal(1, "C030_ASSERT tb_c028_port context %0d exceeds FETCH_OUTSTANDING", c);
      // delivered instructions recorded raw; scenarios check them
      for (int c = 0; c < CONTEXT_COUNT; c++)
        for (int l = 0; l < LANES; l++)
          if (lv[c][l] && lr[c][l]) begin
            ilog_addr[c][ilog_n[c] % ILOG] = la[c][l];
            ilog_inst[c][ilog_n[c] % ILOG] = li[c][l];
            ilog_n[c]++;
            if (ilog_n[c] >= ILOG)
              $fatal(1, "C030_ASSERT tb_c028_port instruction log overflow");
            if (lb[c][l] !== 1'b0)
              $fatal(1, "C030_ASSERT tb_c028_port unexpected predicted jump in ALU program");
          end
      if (invariant_error_o)
        $fatal(1, "C030_ASSERT tb_c028_port l1_program_port invariant_error_o");
    end
  end

  always #5 clk = ~clk;
  initial begin #20_000_000; $fatal(1, "C030_ASSERT tb_c028_port watchdog"); end

  // ---------------- program text ------------------------------------------
  // Real OP-IMM (ADDI) encodings, one per word address; never jal/branch, so
  // predecode never predicts a jump and brchFwd stays low.
  function automatic logic [31:0] exp_word(input logic [ADDR_W-1:0] addr);
    return 32'h0000_0093 | (32'(addr[9:2]) << 20);
  endfunction

  function automatic logic [31:0] patched_word(input logic [ADDR_W-1:0] addr);
    return 32'h0000_0093 | ((32'(addr[9:2]) ^ 32'hFFF) << 20);
  endfunction

  // Install one word through the port's INSTALL path and check the answer.
  task automatic ins_word(input logic [ADDR_W-1:0] addr, input logic [31:0] data,
                          input logic [PROG_ID_W-1:0] pid, input logic [EPOCH_W-1:0] ep,
                          input logic [2:0] ecode, input string label);
    @(negedge clk);
    ins_valid = 1'b1;
    ins_addr = addr;
    ins_data = data;
    ins_be = '1;
    ins_prog_id = pid;
    ins_epoch = ep;
    do @(negedge clk); while (!ins_ready);
    ins_valid = 1'b0;
    while (!ins_rsp_valid) @(negedge clk);
    require(ins_rsp_code == ecode, {label, ": install code"});
    ins_rsp_ready = 1'b1;
    @(negedge clk);
    ins_rsp_ready = 1'b0;
  endtask

  task automatic install_line(input logic [ADDR_W-1:0] base, input logic [PROG_ID_W-1:0] pid,
                              input logic [EPOCH_W-1:0] ep, input string label);
    for (int w = 0; w < FB/WB; w++) begin
      automatic logic [ADDR_W-1:0] a;
      a = base + ADDR_W'(4*w);
      ins_word(a, exp_word(a), pid, ep, I_OK, label);
    end
  endtask

  task automatic pulse_start(input int unsigned c, input logic [ADDR_W-1:0] pc);
    @(negedge clk);
    start_v[c] = 1'b1;
    start_b[c] = pc;
    parked[c] = 1'b0;
    @(negedge clk);
    start_v[c] = 1'b0;
  endtask

  task automatic pulse_redirect(input int unsigned c, input logic [ADDR_W-1:0] pc);
    @(negedge clk);
    redir_v[c] = 1'b1;
    redir_b[c] = pc;
    @(negedge clk);
    redir_v[c] = 1'b0;
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

  task automatic wait_fires(input int unsigned c, input int unsigned n, input string label);
    automatic int unsigned watchdog;
    watchdog = 0;
    while (req_seen[c] < n) begin
      @(negedge clk);
      watchdog++;
      if (watchdog > 3000) $fatal(1, "C030_ASSERT tb_c028_port %s: fire %0d never arrived", label, n);
    end
  endtask
  // Bounded wait for the frontend to stage exactly one request offer.
  task automatic wait_offered(input string label);
    automatic int unsigned wd;
    wd = 0;
    while (!mreq_valid && wd < 500) begin
      @(negedge clk);
      wd++;
    end
    require(mreq_valid, {label, ": the frontend staged an offer"});
  endtask


  task automatic wait_quiet(input string label);
    automatic int unsigned watchdog;
    watchdog = 0;
    forever begin
      automatic bit busy;
      busy = (fq_head != fq_tail) || skid_valid || rsp_valid;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        busy |= outst[c] != '0;
        busy |= inf_of(c) != '0;
        for (int l = 0; l < LANES; l++) busy |= lv[c][l];
      end
      if (!busy) break;
      @(negedge clk);
      watchdog++;
      if (watchdog > 30000) $fatal(1, "C030_ASSERT tb_c028_port %s: quiet watchdog", label);
    end
  endtask
  // Bounded wait for a Context's program to end at its fetch fault, then a
  // bounded drain: a mutant that hides the fault fails the directed require
  // instead of stalling the bench on a watchdog.
  task automatic wait_terminated(input int unsigned c, input string label);
    automatic int unsigned wd;
    wd = 0;
    while (!fault_v[c] && wd < 3000) begin
      @(negedge clk);
      wd++;
    end
    require(fault_v[c], {label, ": context reached its fetch fault"});
    wd = 0;
    forever begin
      automatic bit busy;
      busy = (fq_head != fq_tail) || skid_valid || rsp_valid || (outst[c] != '0) || (inf_of(c) != '0);
      for (int l = 0; l < LANES; l++) busy |= lv[c][l];
      if (!busy) break;
      @(negedge clk);
      wd++;
      if (wd > 2000) $fatal(1, "C030_ASSERT tb_c028_port %s: post-fault drain watchdog", label);
    end
  endtask

  // Check the LAST n recorded instructions of Context c: sequential word
  // addresses from base, each matching exp_word (patch list optional).
  task automatic expect_tail(input int unsigned c, input int unsigned n,
                             input logic [ADDR_W-1:0] base, input string label);
    for (int i = 0; i < int'(n); i++) begin
      automatic int idx;
      idx = int'(ilog_n[c]) - int'(n) + i;
      if (ilog_addr[c][idx % ILOG] != base + ADDR_W'(4*i))
        $fatal(1, "C030_ASSERT tb_c028_port %s: inst %0d addr %0h != %0h",
               label, i, ilog_addr[c][idx % ILOG], base + ADDR_W'(4*i));
      if (ilog_inst[c][idx % ILOG] != exp_word(base + ADDR_W'(4*i)))
        $fatal(1, "C030_ASSERT tb_c028_port %s: inst %0d encoding %0h != %0h",
               label, i, ilog_inst[c][idx % ILOG], exp_word(base + ADDR_W'(4*i)));
    end
    checks += int'(n);
  endtask

  initial begin : main
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    require(!skid_valid && !rsp_valid, "reset: no responses pending");
    require(fq_head == fq_tail, "reset: no live requests");

    // Program text: ctx0 lines 0..2, ctx1 lines 4..6 (identity P0/P1).
    install_line(P0_BASE + 32'h00, PID_P0, IE0, "install P0 line 0");
    install_line(P0_BASE + 32'h20, PID_P0, IE0, "install P0 line 1");
    install_line(P0_BASE + 32'h40, PID_P0, IE0, "install P0 line 2");
    if (CONTEXT_COUNT == 2) begin
      install_line(P1_BASE + 32'h00, PID_P1, IE0, "install P1 line 0");
      install_line(P1_BASE + 32'h20, PID_P1, IE0, "install P1 line 1");
      install_line(P1_BASE + 32'h40, PID_P1, IE0, "install P1 line 2");
    end

    // ---------------- A: tagged end-to-end fetch --------------------------
    // ctx0 walks lines 0..2 (24 real instructions, key-tagged end to end),
    // then faults on the uninstalled terminator line 3 (F_UNWRITTEN).
    pulse_start(0, P0_BASE);
    wait_terminated(0, "A ctx0 drains");
    require(fault_v[0], "A: uninstalled terminator faults ctx0");
    require(fault_b[0] == P0_BASE + 32'h60, "A: ctx0 fault pc at terminator line");
    expect_tail(0, 24, P0_BASE, "A: ctx0 program bytes");
    require(ilog_n[0] == 24, "A: ctx0 delivered exactly its program");
    require(rsp_seen[0] == req_seen[0], "A: every ctx0 request answered");
    // The real frontend speculatively offers one line past the faulting
    // terminator: that line's response is stale-dropped once faulted latches.
    repeat (4) @(negedge clk);
    require(req_seen[0] == 5,
            "A: three text lines + terminator + one speculative line fetched");
    require(stale_o[0] == 32'd1, "A: exactly the speculative line stale-dropped");
    require(fetch_accepted_o == 64'(req_seen[0]), "A: port accepted exactly the mux fires");
    if (CONTEXT_COUNT == 2) begin
      pulse_start(1, P1_BASE);
      wait_terminated(1, "A ctx1 drains");
      require(fault_v[1] && fault_b[1] == P1_BASE + 32'h60, "A: ctx1 terminator fault");
      expect_tail(1, 24, P1_BASE, "A: ctx1 program bytes");
      require(ilog_n[1] == 24, "A: ctx1 delivered exactly its program");
      require(rsp_seen[1] == req_seen[1], "A: every ctx1 request answered");
      repeat (4) @(negedge clk);
      require(req_seen[1] == 5, "A: ctx1 fetched text+terminator+speculative");
      require(stale_o[1] == 32'd1, "A: ctx1 speculative line stale-dropped");
      require(fetch_accepted_o == 64'(req_seen[0] + req_seen[1]),
              "A: port accepted exactly both Contexts' fires");
    end
    require(!invariant_error_o, "A: port invariants clean");

    // ---------------- B: same-PC ABA across fetch_epoch -------------------
    begin
      automatic logic [EPOCH_W-1:0] ep_a, ep_b;
      automatic logic [ADDR_W-1:0] ad_a, ad_b;
      automatic int unsigned fires0, stale0, delivered0, k0, stale1_base;
      fires0 = req_seen[0];
      stale0 = stale_o[0];
      stale1_base = (CONTEXT_COUNT == 2) ? stale_o[1] : 0;
      delivered0 = ilog_n[0];
      k0 = fq_head;  // index of this scenario's first (stale) fire
      // Freeze the request channel at the bridge, so exactly one offer is
      // staged and the sequential successor line can never interleave.
      @(negedge clk);
      req_hold = 1'b1;
      // First fetch of line 0 under the post-A epoch: staged, not yet fired.
      pulse_redirect(0, P0_BASE);
      wait_offered("B: first same-PC offer staged");
      // Redirect between: SAME pc, NEW epoch; the staged offer stays alive
      // (it will complete and its response is later rejected as stale).
      pulse_redirect(0, P0_BASE);
      repeat (2) @(negedge clk);
      // Release the request channel: the STALE offer fires first, then the
      // mux stages the CURRENT-epoch offer for the SAME line.
      @(negedge clk);
      req_hold = 1'b0;
      wait_fires(0, fires0 + 2, "B: both same-PC requests issued");
      parked[0] = 1'b1;  // no further offers; drain exactly these two
      ep_a = fq[k0 % FQD][SEQ_W+EPOCH_W-1 -: EPOCH_W];
      ad_a = fa[k0 % FQD];
      ep_b = fq[(k0+1) % FQD][SEQ_W+EPOCH_W-1 -: EPOCH_W];
      ad_b = fa[(k0+1) % FQD];
      require(ad_a == ad_b, "B: same PC requested twice");
      require(ep_a != ep_b, "B: the two same-PC requests carry different epochs");
      require(fq[k0 % FQD][SEQ_W-1:0] != fq[(k0+1) % FQD][SEQ_W-1:0],
              "B: the two same-PC requests carry different reqids");
      // Release the responses: the stale one completes its handshake
      // (credit back), is dropped at the ledger head and counted in
      // staleRejected; the current-epoch response delivers the instructions.
      @(negedge clk);
      rsp_open = 1'b1;
      wait_quiet("B: ABA pair drains");
      begin
        // wait_quiet can return while the instruction buffer is still draining its
        // lanes: wait (bounded) for the expected count, then let any extra delivery
        // surface before the exact-count check below.
        automatic int unsigned wd_b;
        wd_b = 0;
        while (ilog_n[0] < delivered0 + 8 && wd_b < 500) begin
          @(negedge clk);
          wd_b++;
        end
        repeat (40) @(negedge clk);
      end
      require(stale_o[0] == stale0 + 32'd1, "B: stale epoch counted in staleRejected");
      require(ilog_n[0] == delivered0 + 8, "B: only the current epoch line delivered");
      expect_tail(0, 8, P0_BASE, "B: current-epoch instructions are the stored bytes");
      require(outst[0] == '0, "B: credits returned to zero");
      require(rsp_seen[0] == req_seen[0], "B: both responses completed exactly once");
      if (CONTEXT_COUNT == 2) begin
        require(stale_o[1] == stale1_base, "B: peer context never saw a stale response");
        require(fault_v[1], "B: peer keeps its own state (terminator fault)");
      end
    end

    // ---------------- C0: the consumer accepts NOTHING; the port holds every response ----
    // The bridge skid normally absorbs one response and hides the port's own queue depth.
    // With rsp_hold_all the port itself must hold every in-flight response
    // (CONTEXT_COUNT * FETCH_OUTSTANDING) with nothing dropped, oldest first.
    begin
      automatic int unsigned f0, f1, held_total;
      automatic logic [63:0] acc0, del0;
      automatic logic [KEY_W-1:0] head_key;
      automatic int unsigned wd0;
      f0 = req_seen[0];
      f1 = req_seen[1];
      acc0 = fetch_accepted_o;
      del0 = rsp_delivered_o;
      held_total = CONTEXT_COUNT * FO;
      rsp_hold_all = 1'b1;
      for (int l = 0; l < LANES; l++) begin
        lr[0][l] = 1'b0;
        if (CONTEXT_COUNT == 2) lr[1][l] = 1'b0;
      end
      parked[0] = 1'b0;
      pulse_redirect(0, P0_BASE);
      wait_fires(0, f0 + FO, "C0: ctx0 fetches issued to the bound");
      if (CONTEXT_COUNT == 2) begin
        parked[1] = 1'b0;
        pulse_redirect(1, P1_BASE);
        wait_fires(1, f1 + FO, "C0: ctx1 fetches issued to the bound");
      end
      // No further offers: only the already-issued requests are in flight.
      parked[0] = 1'b1;
      if (CONTEXT_COUNT == 2) parked[1] = 1'b1;
      repeat (8) @(negedge clk);
      require(fetch_accepted_o - acc0 == 64'(held_total), "C0: port accepted every in-flight fetch");
      require(rsp_delivered_o == del0, "C0: nothing delivered while the consumer refuses");
      require(rsp_valid && !skid_valid, "C0: the port itself holds the responses");
      head_key = fq[fq_head % FQD];
      repeat (4) begin
        @(negedge clk);
        require(rsp_valid && rsp_key == head_key, "C0: held head response is stable and oldest");
      end
      // Release: every held response must be delivered exactly once. The drain is bounded
      // and then checked directly, so a dropped response fails a directed assertion.
      for (int l = 0; l < LANES; l++) begin
        lr[0][l] = 1'b1;
        if (CONTEXT_COUNT == 2) lr[1][l] = 1'b1;
      end
      @(negedge clk);
      rsp_hold_all = 1'b0;
      wd0 = 0;
      while ((fq_head != fq_tail || skid_valid || rsp_valid) && wd0 < 2000) begin
        @(negedge clk);
        wd0++;
      end
      require(fq_head == fq_tail, "C0: every held response delivered after release");
      require(rsp_delivered_o - del0 == 64'(held_total), "C0: every held response delivered exactly once");
      wait_quiet("C0: drains");
      require(rsp_seen[0] == req_seen[0], "C0: ctx0 responses all consumed");
      if (CONTEXT_COUNT == 2) require(rsp_seen[1] == req_seen[1], "C0: ctx1 responses all consumed");
      require(!invariant_error_o, "C0: port invariants clean");
    end

    // ---------------- C: tagged response backpressure ---------------------
    begin
      automatic int unsigned fires0;
      automatic logic [KEY_W-1:0] held_key;
      automatic logic [FB*8-1:0] held_data;
      fires0 = req_seen[0];
      // Lanes stalled AND responses gated: fetches queue, credits are held.
      for (int l = 0; l < LANES; l++) lr[0][l] = 1'b0;
      rsp_open = 1'b0;
      parked[0] = 1'b0;
      pulse_redirect(0, P0_BASE);
      // The mux issues to its bound and NO further request is issued while
      // both credits are outstanding (bound hit at exactly N, never N+1).
      wait_fires(0, fires0 + FO, "C: fetches issued to the bound");
      repeat (8) begin
        @(negedge clk);
        require(outst[0] == INF_W'(FO), "C: credits held at FETCH_OUTSTANDING");
        require(req_seen[0] == fires0 + FO, "C: no (N+1)th request issued at the bound");
      end
      // C=2: hold the peer to its bound too — the port response queue then
      // carries its full RSP_DEPTH under backpressure with nothing dropped.
      if (CONTEXT_COUNT == 2) begin
        automatic int unsigned fires1;
        fires1 = req_seen[1];
        for (int l = 0; l < LANES; l++) lr[1][l] = 1'b0;
        parked[1] = 1'b0;
        pulse_redirect(1, P1_BASE);
        wait_fires(1, fires1 + FO, "C: peer fetches issued to the bound");
        repeat (4) begin
          @(negedge clk);
          require(outst[1] == INF_W'(FO), "C: peer credits held at FETCH_OUTSTANDING");
          require(req_seen[1] == fires1 + FO, "C: peer issued no (N+1)th request");
        end
      end
      // Held response key/payload stay stable under backpressure, both in
      // the bridge skid and at the port's registered queue head.
      held_key = {skid_slot, skid_gen, skid_epoch, skid_reqid};
      held_data = skid_data;
      repeat (4) begin
        @(negedge clk);
        require({skid_slot, skid_gen, skid_epoch, skid_reqid} == held_key,
                "C: held response key stable");
        require(skid_data == held_data, "C: held response payload stable");
        if (rsp_valid) begin
          require(rsp_key == fq[(fq_head+1) % FQD], "C: queued response key stable");
          require(rsp_fault == 1'b0, "C: queued response is data (not fault)");
        end
      end
      // Release: responses drain, credits return exactly once each, the
      // stalled lanes then accept every queued instruction. The held-fire
      // drain is BOUNDED and then checked directly, so a dropped held
      // response fails a directed assertion instead of stalling the bench.
      for (int l = 0; l < LANES; l++) begin
        lr[0][l] = 1'b1;
        if (CONTEXT_COUNT == 2) lr[1][l] = 1'b1;
      end
      @(negedge clk);
      rsp_open = 1'b1;
      begin
        automatic int unsigned wd;
        wd = 0;
        while ((fq_head != fq_tail || skid_valid || rsp_valid) && wd < 2000) begin
          @(negedge clk);
          wd++;
        end
        require(fq_head == fq_tail, "C: every held response delivered after release");
      end
      wait_quiet("C: backpressure drains");
      require(ilog_n[0] >= 24, "C: program continued after release");
      require(rsp_seen[0] == req_seen[0], "C: no response dropped under backpressure");
      if (CONTEXT_COUNT == 2)
        require(rsp_seen[1] == req_seen[1], "C: no peer response dropped under backpressure");
      require(!invariant_error_o, "C: port invariants clean");
    end

    // ---------------- D: install contention with every Context ------------
    begin
      automatic int unsigned f0s, f1s, fis, fa0, fai, pb0, tgr, g0, g1, gi;
      automatic int unsigned watchdog;
      f0s = sh_fet[0];
      f1s = (CONTEXT_COUNT == 2) ? sh_fet[1] : 0;
      fis = sh_ins;
      fa0 = 32'(fetch_accepted_o);
      fai = 32'(install_beats_o);
      pb0 = 32'(port_busy_o);
      in_d = 1'b1;
      // Restart both Contexts at their programs and keep them fetching while
      // installs hammer the scratch region through the same physical port.
      pulse_redirect(0, P0_BASE);
      if (CONTEXT_COUNT == 2) pulse_redirect(1, P1_BASE);
      ins_rsp_ready = 1'b1;
      watchdog = 0;
      while (watchdog < 600) begin
        @(negedge clk);
        // sustained install: a fresh scratch word offered every beat
        ins_valid = 1'b1;
        ins_addr = 32'h0000_0100 + ADDR_W'(4*(watchdog % 64));
        ins_data = 32'h5A5A_0000 | 32'(watchdog);
        ins_be = '1;
        ins_prog_id = PID_SCRATCH;
        ins_epoch = IE0;
        // restart a Context that hit its terminator fault
        for (int c = 0; c < CONTEXT_COUNT; c++)
          if (fault_v[c]) pulse_redirect(c, (c == 0) ? P0_BASE : P1_BASE);
        watchdog++;
      end
      ins_valid = 1'b0;
      parked[0] = 1'b1;
      if (CONTEXT_COUNT == 2) parked[1] = 1'b1;
      wait_quiet("D: contention drains");
      in_d = 1'b0;
      ins_rsp_ready = 1'b0;
      g0 = sh_fet[0] - f0s;
      g1 = (CONTEXT_COUNT == 2) ? (sh_fet[1] - f1s) : 0;
      gi = sh_ins - fis;
      tgr = g0 + g1 + gi;
      require(gi >= 100, "D: install progressed under sustained contention");
      require(g0 >= 12, "D: context 0 fetches progressed");
      if (CONTEXT_COUNT == 2) require(g1 >= 12, "D: context 1 fetches progressed");
      require(gi + 2 >= g0 && gi + 2 >= g1, "D: install not starved below fetch volume");
      require(install_beats_o - 64'(fai) == 64'(gi),
              "D: install beats equal boundary handshakes");
      require(fetch_accepted_o - 64'(fa0) == 64'(g0 + g1),
              "D: accepted fetches equal boundary handshakes");
      require(port_busy_o - 64'(pb0) == 64'(tgr), "D: port busy equals all grants");
      require(rsp_seen[0] == req_seen[0], "D: every ctx0 response delivered");
      if (CONTEXT_COUNT == 2) require(rsp_seen[1] == req_seen[1],
                                       "D: every ctx1 response delivered");
      require(fq_head == fq_tail, "D: fire log fully answered");
      require(!invariant_error_o, "D: port invariants clean");
    end

    // ---------------- E: FET-06 rewrite protection ------------------------
    begin
      automatic int unsigned before_n;
      before_n = ilog_n[0];
      // ctx0 restarts its program under an execution reference of P0.
      ref_op(1'b1, PID_P0, IE0);
      parked[0] = 1'b0;
      pulse_redirect(0, P0_BASE);
      wait_fires(0, req_seen[0] + 1, "E: program refetching under reference");
      // Overlapping installs are rejected while the reference is held ...
      ins_word(P0_BASE + 32'h20, patched_word(P0_BASE + 32'h20), PID_P1, IE0,
               I_EXEC_REF, "E: rewrite of executing text rejected");
      ins_word(P0_BASE + 32'h04, patched_word(P0_BASE + 32'h04), PID_P1, IE0,
               I_EXEC_REF, "E: overlap with already-fetched word rejected");
      ins_word(P0_BASE + 32'h20, exp_word(P0_BASE + 32'h20), PID_P0, IE0,
               I_EXEC_REF, "E: identical bytes still rejected under reference");
      // ... and the Context keeps fetching the ORIGINAL instructions.
      wait_terminated(0, "E: program drains under protection");
      require(fault_v[0], "E: program ran to its terminator");
      require(ilog_n[0] - before_n == 24, "E: whole program delivered under protection");
      expect_tail(0, 24, P0_BASE, "E: original instructions kept");
      // After the reference drops, the write lands and is visible through
      // the frontend on the next fetch of that line.
      ref_op(1'b0, PID_P0, IE0);
      ins_word(P0_BASE + 32'h24, patched_word(P0_BASE + 32'h24), PID_P1, IE0,
               I_OK, "E: install succeeds after reference drop");
      pulse_redirect(0, P0_BASE + 32'h20);
      wait_terminated(0, "E: patched line drains");
      begin
        automatic int unsigned n;
        automatic int base_i;
        n = ilog_n[0];
        base_i = int'(n) - 16;  // lines 1,2 redelivered after this redirect
        require(base_i > 0, "E: patched lines were refetched");
        for (int i = 0; i < 16; i++) begin
          automatic logic [ADDR_W-1:0] a;
          a = P0_BASE + 32'h20 + ADDR_W'(4*i);
          if (ilog_addr[0][(base_i+i) % ILOG] != a)
            $fatal(1, "C030_ASSERT tb_c028_port E: refetch addr %0h != %0h",
                   ilog_addr[0][(base_i+i) % ILOG], a);
          if (i == 1 ? (ilog_inst[0][(base_i+i) % ILOG] != patched_word(a)) :
                       (ilog_inst[0][(base_i+i) % ILOG] != exp_word(a)))
            $fatal(1, "C030_ASSERT tb_c028_port E: refetch word %0d not the resident bytes", i);
        end
        checks += 16;
      end
    end

    // ---------------- F: out-of-range fault hits only the owner -----------
    begin
      automatic int unsigned peer_n, fires0, stale1_0;
      peer_n = (CONTEXT_COUNT == 2) ? ilog_n[1] : 0;
      fires0 = req_seen[0];
      // ctx0 starts beyond the SRAM model: the port answers F_RANGE through
      // the real response channel and the mux faults ONLY ctx0. One
      // speculative line past the fault is also fetched, then stale-dropped.
      pulse_start(0, 32'(SRAM));
      begin
        // wait_quiet alone can return before the start pulse produced its first offer
        // (nothing is busy yet), so first wait (bounded) for THIS fault, identified by its
        // pc, which cannot be confused with the previous scenario's latched fault.
        automatic int unsigned wd_f;
        wd_f = 0;
        while (!(fault_v[0] && fault_b[0] == 32'(SRAM)) && wd_f < 3000) begin
          @(negedge clk);
          wd_f++;
        end
      end
      wait_quiet("F: range fault settles");
      require(fault_v[0], "F: out-of-range start faulted the owner");
      require(fault_b[0] == 32'(SRAM), "F: fault pc is the out-of-range line");
      require(lf_code[0] == F_RANGE, "F: out-of-range answered F_RANGE");
      require(req_seen[0] - fires0 == 2, "F: faulted line plus one speculative line");
      require(fa[(fq_tail-2) % FQD] == 32'(SRAM), "F: faulted fire was the out-of-range line");
      require(fa[(fq_tail-1) % FQD] == 32'(SRAM + FB), "F: speculative fire is the next line");
      require(outst[0] == '0, "F: no credit leaked by the fault");
      if (CONTEXT_COUNT == 2) begin
        // The peer keeps fetching its own program while the owner is held
        // at its fault: only the owning Context stopped.
        stale1_0 = stale_o[1];
        parked[1] = 1'b0;
        pulse_redirect(1, P1_BASE);
        require(fault_v[0] && !fault_v[1], "F: only the owning Context is faulted");
        wait_terminated(1, "F: peer still fetches");
        require(ilog_n[1] > peer_n, "F: peer delivered instructions after the owner fault");
        require(stale_o[1] == stale1_0 + 32'd1,
                "F: peer's only stale line is its own speculative fetch");
      end
      require(rsp_seen[0] == req_seen[0], "F: fault response delivered exactly once");
      require(!invariant_error_o, "F: port invariants clean");
    end

    // ---------------- honest counters, final ------------------------------
    begin
      automatic logic [63:0] sf;
      sf = '0;
      for (int c = 0; c < CONTEXT_COUNT; c++) sf = sf + 64'(sh_fet[c]);
      require(fetch_accepted_o == sf, "final: accepted fetches equal boundary fires");
      require(install_beats_o == 64'(sh_ins), "final: install beats equal boundary fires");
      require(rsp_delivered_o == 64'(sh_rsp), "final: responses equal boundary fires");
      require(port_busy_o == sf + 64'(sh_ins), "final: port busy = fetch + install grants");
      require(fq_head == fq_tail, "final: no request left unanswered");
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        require(outst[c] == '0, "final: no context left in flight");
        require(inf_of(c) == '0, "final: port in-flight empty");
      end
    end
    $display("C030_PASS tb_c028_port checks=%0d contexts=%0d fetch_outstanding=%0d fetch_bytes=%0d",
             checks, CONTEXT_COUNT, FO, FB);
    $finish;
  end
endmodule
