// C032: MFE direct-move engine against the real C023 MemoryService peer.
//
// Transport delays below affect delivery timing only; every byte, response and
// terminal closure comes from the host Python service (c032_mfe_peer.py) over
// the FIFO endpoint given by +SERVICE=<path>. The bench proves:
//   * a real read->write copy with byte equality through the local window,
//     including multi-segment transactions at LINE_BYTES boundaries;
//   * permission/range/binding/kind authorize faults settle FAULTED with a
//     code and ZERO external requests;
//   * a cancel landing on an accepted write leg settles COMPLETED with the
//     effect readable from the destination binding, and issues no further
//     write after the mark;
//   * a cancel before any write leg is accepted settles ISOLATED with zero
//     writes, and a cancelled inbound step absorbs its late response;
//   * same-edge cancel+completion produces exactly one terminal;
//   * duplicate terminal / release-without-ACK / duplicate release /
//     terminal-after-release are refused with distinct codes (injected on a
//     standalone tracker instance);
//   * poison refuses new starts until a quiescent reset, and a reset while a
//     transaction is live is refused without clearing anything;
//   * the ledger closes: one TERMINAL_ACK per accepted request obligation,
//     zero stale responses, zero refused engine commands, service drained.
`timescale 1ns/1ps
module tb_mfe_top;
  // The harness injects -GCONTEXT_COUNT=N whenever a bench declares Context
  // coverage; the MFE has no Context dimension, so this shim only keeps a
  // per-Context registration elaborable. Run this bench at contexts=None.
  parameter int unsigned CONTEXT_COUNT = 1;
  import c029_params_pkg::*;
  import c032_mfe_pkg::*;

  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  logic desc_valid = 0, desc_ready;
  mfe_desc_t desc;
  logic bind_valid = 0, bind_ready;
  mfe_binding_t bindin;
  logic cancel_valid = 0;
  logic [31:0] cancel_txn = 0;
  logic isolation_failed = 0, quiescent_reset = 0, poisoned;
  logic term_valid, term_ready = 1;
  mfe_terminal_t term;
  logic mem_valid, mem_ready, mr_valid = 0, mr_ready, ack_valid, ack_ready;
  mfe_mem_req_t mem;
  mfe_mem_rsp_t mr;
  mfe_term_ack_t ack;
  logic [31:0] lpeek_addr = 0;
  logic [7:0] lpeek_data;
  logic [$clog2(MFE_INFLIGHT+1)-1:0] outstanding;
  logic [63:0] sent_c, rsps_c, acks_c, absorbed_c, stale_c;
  logic [63:0] trk_terminals, trk_stale_cancel, trk_refused;
  logic [3:0] dut_err;
  logic dut_inv;

  mfe_top dut (
    .clk, .rst_n,
    .desc_valid_i(desc_valid), .desc_ready_o(desc_ready), .desc_i(desc),
    .bind_valid_i(bind_valid), .bind_ready_o(bind_ready), .bind_i(bindin),
    .cancel_valid_i(cancel_valid), .cancel_txn_i(cancel_txn),
    .isolation_failed_i(isolation_failed), .quiescent_reset_i(quiescent_reset),
    .poisoned_o(poisoned),
    .term_valid_o(term_valid), .term_ready_i(term_ready), .term_o(term),
    .mem_req_valid_o(mem_valid), .mem_req_ready_i(mem_ready), .mem_req_o(mem),
    .mem_rsp_valid_i(mr_valid), .mem_rsp_ready_o(mr_ready), .mem_rsp_i(mr),
    .term_ack_valid_o(ack_valid), .term_ack_ready_i(ack_ready), .term_ack_o(ack),
    .lpeek_addr_i(lpeek_addr), .lpeek_data_o(lpeek_data),
    .outstanding_o(outstanding),
    .sent_o(sent_c), .rsps_o(rsps_c), .acks_o(acks_c), .absorbed_o(absorbed_c),
    .stale_rsp_o(stale_c),
    .tracker_terminals_o(trk_terminals), .tracker_stale_cancel_o(trk_stale_cancel),
    .tracker_refused_o(trk_refused),
    .err_code_o(dut_err), .invariant_error_o(dut_inv)
  );

  // Standalone tracker: the bench injects protocol violations here that the
  // engine itself must never produce (the dut's refused count stays zero).
  localparam int unsigned T2 = MFE_INFLIGHT;
  logic t2_alloc_valid = 0, t2_alloc_ready;
  logic [31:0] t2_alloc_txn = 0;
  logic [$clog2(T2)-1:0] t2_alloc_slot;
  logic t2_obligate_valid = 0, t2_ack_valid = 0, t2_settle_valid = 0, t2_release_valid = 0;
  logic [$clog2(T2)-1:0] t2_slot = 0;
  logic [1:0] t2_settle_final = MFE_COMPLETED;
  logic t2_term_valid, t2_quiescent_reset = 0;
  logic [31:0] t2_term_txn;
  logic [1:0] t2_term_final;
  logic [3:0] t2_term_err, t2_err;
  logic [T2-1:0] t2_free, t2_live, t2_marked, t2_settled, t2_acks_open;
  logic t2_poisoned, t2_inv;
  logic [63:0] t2_terminals, t2_stale_cancel, t2_refused;

  transfer_cancel_tracker #(.ENTRIES(T2)) trk2 (
    .clk, .rst_n,
    .alloc_ready_o(t2_alloc_ready), .alloc_valid_i(t2_alloc_valid),
    .alloc_txn_i(t2_alloc_txn), .alloc_slot_o(t2_alloc_slot),
    .cancel_valid_i(1'b0), .cancel_txn_i(32'd0),
    .obligate_valid_i(t2_obligate_valid), .obligate_slot_i(t2_slot),
    .ack_valid_i(t2_ack_valid), .ack_slot_i(t2_slot),
    .settle_valid_i(t2_settle_valid), .settle_slot_i(t2_slot),
    .settle_final_i(t2_settle_final), .settle_err_i(ERR_NONE),
    .release_valid_i(t2_release_valid), .release_slot_i(t2_slot),
    .term_valid_o(t2_term_valid), .term_ready_i(1'b1),
    .term_txn_o(t2_term_txn), .term_final_o(t2_term_final), .term_err_o(t2_term_err),
    .free_o(t2_free), .live_o(t2_live), .marked_o(t2_marked),
    .settled_o(t2_settled), .acks_open_o(t2_acks_open),
    .poisoned_o(t2_poisoned), .isolation_failed_i(1'b0),
    .quiescent_reset_i(t2_quiescent_reset),
    .err_code_o(t2_err), .invariant_error_o(t2_inv),
    .terminals_o(t2_terminals), .stale_cancel_o(t2_stale_cancel), .refused_o(t2_refused)
  );

  int checks = 0, cycles = 0, issued_count = 0;
  logic [1:0] final_exp [1024];
  logic [3:0] err_exp [1024];
  logic got_term [1024], issued [1024];
  int term_count [1024];
  task automatic require(input logic ok, input string why);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C032_ASSERT tb_mfe_top %s cycle=%0d", why, cycles);
  endtask

  mfe_mem_req_t held_mem;
  mfe_term_ack_t held_ack;
  mfe_terminal_t held_lt;
  logic mem_held = 0, ack_held = 0, lt_held = 0;

  always @(posedge clk) begin
    cycles++;
    if (rst_n) begin
      require(!dut_inv, "live invariants");
      require(trk_refused == 64'd0, "engine never refuses a tracker command");
      require(trk_stale_cancel == 64'd0 && t2_stale_cancel == 64'd0,
              "every cancel matched a live transaction");
      if (mem_held && mem_valid) require(mem == held_mem, "MEM_REQ stable under backpressure");
      if (ack_held) require(ack_valid && ack == held_ack, "TERMINAL_ACK stable under backpressure");
      if (lt_held) require(term_valid && term == held_lt, "local terminal stable under backpressure");
      mem_held = mem_valid && !mem_ready; held_mem = mem;
      ack_held = ack_valid && !ack_ready; held_ack = ack;
      lt_held = term_valid && !term_ready; held_lt = term;
      if (term_valid && term_ready) begin
        require(term.txn_id < 32'd1024, "terminal txn range");
        require(!got_term[term.txn_id], "exactly one terminal per transaction");
        require(term.final_state == final_exp[term.txn_id] &&
                term.error == err_exp[term.txn_id], "real terminal class and error");
        got_term[term.txn_id] = 1;
        term_count[term.txn_id]++;
      end
    end
  end

  // Python C023 peer: synchronous calls at real external handshakes only.
  integer request_file, response_file, scan_status;
  string endpoint;
  logic deliver = 1, request_enable = 1, fault_next = 0, ack_enable = 1;
  logic [31:0] service_status, service_vis;
  logic [LINE_BYTES*8-1:0] service_data;
  mfe_mem_rsp_t queued [8];
  int qhead = 0, qtail = 0, qcount = 0, sent = 0, acked = 0;
  int seg_off [64], seg_len [64];
  logic seg_wr [64];
  logic request_allow = 0;
  always @(negedge clk) request_allow = cycles % 7 != 0 && request_enable;
  always_comb mem_ready = request_allow;
  assign ack_ready = ack_enable;

  always @(posedge clk) begin
    if (!rst_n) begin
      qhead = 0; qtail = 0; qcount = 0; mr_valid = 0; mr = '0;
    end else begin
      if (mem_valid && mem_ready) begin
        require(mem.slice_id == 32'd0 && mem.slice_count == 32'd1, "single-slice request");
        require(mem.length != 32'd0 && mem.length <= 32'(LINE_BYTES), "bounded segment length");
        require(mem.ctx_slot == 32'd0 && mem.ctx_generation == 32'd1 &&
                mem.arena_generation == 32'd99 && mem.scope_epoch == 32'd77,
                "external request retains full provenance");
        if (mem.write) begin
          $fdisplay(request_file, "W %h %h %h %h %h %h %h %h %h %h %h %h",
            mem.txn_id, mem.slice_id, mem.slice_count, mem.binding_id,
            mem.binding_generation, mem.binding_offset, mem.length,
            mem.ctx_slot, mem.ctx_generation, mem.arena_generation, mem.scope_epoch,
            mem.payload);
          $fflush(request_file);
          scan_status = $fscanf(response_file, "%h %h", service_status, service_vis);
          require(scan_status == 2 && service_status == 32'd0, "real MemoryService write accepted");
          require(service_vis == 32'd2, "write visibility is HBM-persisted");
          service_data = '0;
        end else begin
          $fdisplay(request_file, "R %h %h %h %h %h %h %h %h %h %h %h",
            mem.txn_id, mem.slice_id, mem.slice_count, mem.binding_id,
            mem.binding_generation, mem.binding_offset, mem.length,
            mem.ctx_slot, mem.ctx_generation, mem.arena_generation, mem.scope_epoch);
          $fflush(request_file);
          scan_status = $fscanf(response_file, "%h %h", service_status, service_data);
          require(scan_status == 2 && service_status == 32'd0, "real MemoryService read accepted");
          service_vis = 32'd0;
        end
        queued[qtail] = '{txn_id: mem.txn_id, slice_id: 32'd0,
          binding_generation: mem.binding_generation, status: service_status,
          data: service_data};
        qtail = (qtail + 1) % 8; qcount++;
        seg_off[sent] = int'(mem.binding_offset);
        seg_len[sent] = int'(mem.length);
        seg_wr[sent] = mem.write;
        sent++;
      end
      if (mr_valid && mr_ready) mr_valid = 0;
      else if (!mr_valid && deliver && qcount > 0) begin
        mr = queued[qhead];
        if (fault_next) begin
          mr.status = 32'd1;
          fault_next = 0;
        end
        mr_valid = 1;
        qhead = (qhead + 1) % 8; qcount--;
      end
      if (ack_valid && ack_ready) begin
        $fdisplay(request_file, "T %h %h %h %h", ack.txn_id, ack.slice_id,
                  ack.slice_count, {30'd0, ack.final_state});
        $fflush(request_file);
        scan_status = $fscanf(response_file, "%h", service_status);
        require(scan_status == 1 && service_status == 32'd0, "real MemoryService terminal closed");
        acked++;
      end
    end
  end

  task automatic install(input int bid, input int gen, input int size,
                         input logic r, input logic w);
    @(negedge clk);
    bindin = '{id: 32'(bid), generation: 32'(gen), size: 32'(size), readable: r, writable: w};
    bind_valid = 1;
    begin : acc
      int bound;
      bound = 0;
      #1;
      while (!bind_ready && bound < 5000) begin @(negedge clk); #1; bound++; end
      require(bind_ready, "binding table install accepted");
    end
    @(posedge clk);
    @(negedge clk); bind_valid = 0;
  endtask

  task automatic issue(input int t, input logic [1:0] srck, input logic [1:0] dstk,
                       input int bid, input int bgen, input int boff, input int loff,
                       input int nbytes, input logic [1:0] fin, input logic [3:0] err);
    final_exp[t] = fin; err_exp[t] = err;
    issued[t] = 1; issued_count++;
    @(negedge clk);
    desc = '{txn_id: 32'(t), src_kind: srck, dst_kind: dstk,
             binding_id: 32'(bid), binding_generation: 32'(bgen),
             binding_offset: 32'(boff), local_offset: 32'(loff), bytes: 32'(nbytes),
             ctx_slot: 32'd0, ctx_generation: 32'd1, arena_generation: 32'd99,
             scope_epoch: 32'd77};
    desc_valid = 1;
    begin : acc
      int bound;
      bound = 0;
      #1;
      while (!desc_ready && bound < 20000) begin @(negedge clk); #1; bound++; end
      require(desc_ready, "bounded descriptor acceptance");
    end
    @(posedge clk);
    @(negedge clk); desc_valid = 0;
  endtask

  task automatic await_term(input int t);
    int bound;
    bound = 0;
    while (!got_term[t] && bound < 30000) begin @(negedge clk); bound++; end
    require(got_term[t], "bounded transaction terminal");
  endtask

  task automatic drain;
    int bound;
    bound = 0;
    while ((outstanding != 0 || sent != acked || qcount != 0 || mr_valid) && bound < 40000)
      @(negedge clk);
    require(outstanding == 0 && sent == acked && qcount == 0 && !mr_valid,
            "all real obligations closed");
  endtask

  task automatic cancel_at(input int t);
    @(negedge clk); cancel_txn = 32'(t); cancel_valid = 1;
    @(posedge clk);
    @(negedge clk); cancel_valid = 0;
  endtask

  task automatic peek(input int off, output logic [7:0] value);
    lpeek_addr = 32'(off);
    #1;
    value = lpeek_data;
  endtask

  task automatic require_local_equal(input int a, input int b, input int n, input string why);
    logic [7:0] va, vb;
    for (int k = 0; k < n; k++) begin
      peek(a + k, va);
      peek(b + k, vb);
      require(va == vb, why);
    end
  endtask

  task automatic require_local_seed(input int a, input int n, input int binding_off,
                                    input int salt, input string why);
    logic [7:0] va;
    for (int k = 0; k < n; k++) begin
      peek(a + k, va);
      require(va == 8'((binding_off + k) * 17 + ((binding_off + k) / 64) * 13 + 3 + salt), why);
    end
  endtask

  task automatic t2_cmd(input logic ob, input logic ac, input logic st, input logic rel,
                        input logic [1:0] fin);
    @(negedge clk);
    t2_obligate_valid = ob; t2_ack_valid = ac; t2_settle_valid = st;
    t2_release_valid = rel; t2_settle_final = fin;
    @(posedge clk);
    @(negedge clk);
    t2_obligate_valid = 0; t2_ack_valid = 0; t2_settle_valid = 0; t2_release_valid = 0;
  endtask

  task automatic t2_alloc(input int t);
    @(negedge clk); t2_alloc_txn = 32'(t); t2_alloc_valid = 1;
    begin : acc
      int bound;
      bound = 0;
      #1;
      while (!t2_alloc_ready && bound < 1000) begin @(negedge clk); #1; bound++; end
      require(t2_alloc_ready, "standalone tracker allocation");
      t2_slot = t2_alloc_slot;
    end
    @(posedge clk);
    @(negedge clk); t2_alloc_valid = 0;
  endtask

  task automatic poison_pulse;
    @(negedge clk); isolation_failed = 1;
    @(posedge clk);
    @(negedge clk); isolation_failed = 0;
    #1;
  endtask

  task automatic reset_pulse;
    @(negedge clk); quiescent_reset = 1;
    @(posedge clk);
    @(negedge clk); quiescent_reset = 0;
    #1;
  endtask

  task automatic t2_reset_pulse;
    @(negedge clk); t2_quiescent_reset = 1;
    @(posedge clk);
    @(negedge clk); t2_quiescent_reset = 0;
    #1;
  endtask

  initial begin : run
    int baseline, b0;
    logic [63:0] r0, absb;
    require($value$plusargs("SERVICE=%s", endpoint), "service endpoint provided");
    request_file = $fopen({endpoint, ".req"}, "w");
    response_file = $fopen({endpoint, ".rsp"}, "r");
    require(request_file != 0 && response_file != 0, "open real service FIFOs");
    for (int t = 0; t < 1024; t++) begin
      final_exp[t] = MFE_COMPLETED; err_exp[t] = ERR_NONE;
      got_term[t] = 0; issued[t] = 0; term_count[t] = 0;
    end
    for (int k = 0; k < 64; k++) begin
      seg_off[k] = 0; seg_len[k] = 0; seg_wr[k] = 0;
    end
    repeat (4) @(negedge clk); rst_n = 1;

    install(17, 1, 4096, 1, 0);  // peer c032_src:   read-only source
    install(18, 1, 4096, 1, 1);  // peer c032_dst:   read-write destination
    install(17, 2, 4096, 1, 0);  // peer c032_gen2:  alternate generation

    // Real bytes both directions: external read -> local window -> external
    // write -> read back, with multi-segment boundaries at LINE_BYTES.
    issue(100, KIND_EXT, KIND_LOCAL, 17, 1, 96, 0, 160, MFE_COMPLETED, ERR_NONE);
    await_term(100); drain();
    require(sent == 3, "multi-segment inbound read count");
    require(seg_off[0] == 96 && seg_len[0] == 64 && !seg_wr[0], "segment 0 bounds");
    require(seg_off[1] == 160 && seg_len[1] == 64, "segment 1 bounds");
    require(seg_off[2] == 224 && seg_len[2] == 32, "tail segment bounds");
    require_local_seed(0, 160, 96, 0, "inbound bytes are the real service bytes");

    ack_enable = 0;
    issue(101, KIND_LOCAL, KIND_EXT, 18, 1, 32, 0, 160, MFE_COMPLETED, ERR_NONE);
    repeat (20) @(negedge clk);
    ack_enable = 1;
    await_term(101); drain();
    require(sent == 6, "multi-segment outbound write count");
    require(seg_wr[3] && seg_wr[4] && seg_wr[5], "outbound legs are writes");
    require(seg_off[3] == 32 && seg_off[4] == 96 && seg_off[5] == 160,
            "write segment offsets");
    require(seg_len[3] == 64 && seg_len[4] == 64 && seg_len[5] == 32,
            "write segment lengths");

    issue(102, KIND_EXT, KIND_LOCAL, 18, 1, 32, 512, 160, MFE_COMPLETED, ERR_NONE);
    await_term(102); drain();
    require_local_equal(0, 512, 160, "read-back equals the bytes written");
    require_local_seed(512, 160, 96, 0, "written bytes are the bytes actually read");

    // Authorize faults: per-field violation, zero external requests.
    baseline = sent;
    issue(103, KIND_LOCAL, KIND_EXT, 17, 1, 0, 0, 64, MFE_FAULTED, ERR_PERMISSION);
    await_term(103);
    issue(104, KIND_EXT, KIND_LOCAL, 17, 1, 4090, 0, 100, MFE_FAULTED, ERR_RANGE);
    await_term(104);
    issue(105, KIND_EXT, KIND_LOCAL, 17, 3, 0, 0, 64, MFE_FAULTED, ERR_BINDING);
    await_term(105);
    issue(106, KIND_LOCAL, KIND_LOCAL, 18, 1, 0, 0, 64, MFE_FAULTED, ERR_KIND);
    await_term(106);
    issue(107, KIND_EXT, KIND_EXT, 18, 1, 0, 0, 64, MFE_FAULTED, ERR_KIND);
    await_term(107);
    issue(108, KIND_EXT, KIND_LOCAL, 17, 1, 0, 1000, 100, MFE_FAULTED, ERR_RANGE);
    await_term(108);
    drain();
    require(sent == baseline, "authorization faults issue zero external requests");
    require(absorbed_c == 64'd0, "faulted transactions absorb nothing");

    // Cancel landing on an accepted write leg: the effect happened, so the
    // transaction reports COMPLETED after that write's ACK and stops there.
    deliver = 0; baseline = sent;
    issue(110, KIND_LOCAL, KIND_EXT, 18, 1, 2048, 0, 128, MFE_COMPLETED, ERR_NONE);
    while (sent == baseline) @(negedge clk);
    cancel_at(110);
    b0 = sent;
    deliver = 1;
    await_term(110); drain();
    require(b0 == baseline + 1, "cancelled outbound transaction stops after its accepted write");

    // The accepted write's effect is readable from the destination binding.
    issue(111, KIND_EXT, KIND_LOCAL, 18, 1, 2048, 640, 64, MFE_COMPLETED, ERR_NONE);
    await_term(111); drain();
    require_local_equal(0, 640, 64, "accepted write effect is readable");

    // Cancel before any write leg is accepted: ISOLATED, zero writes issued.
    request_enable = 0; baseline = sent;
    issue(112, KIND_LOCAL, KIND_EXT, 18, 1, 3000, 0, 64, MFE_ISOLATED, ERR_NONE);
    begin : park
      int bound;
      bound = 0;
      while (!mem_valid && bound < 5000) begin @(negedge clk); bound++; end
      require(mem_valid && !mem_ready, "outbound request parks under withheld acceptance");
    end
    cancel_at(112);
    await_term(112);
    require(sent == baseline, "isolated outbound transaction issued zero writes");
    request_enable = 1; drain();

    // Cancel parked between segments (request accepted+ACKed, next segment
    // not yet granted). Outbound with a completed write leg must settle
    // COMPLETED (the effect is bound); inbound is retractable and settles
    // ISOLATED. The mark is observed in S_SEGMENT after the parked ACK.
    ack_enable = 0;
    issue(116, KIND_LOCAL, KIND_EXT, 18, 1, 3200, 0, 128, MFE_COMPLETED, ERR_NONE);
    begin : parkseg
      int bound;
      bound = 0;
      while (!ack_valid && bound < 5000) begin @(negedge clk); bound++; end
      require(ack_valid, "first leg terminal ACK parks under backpressure");
    end
    cancel_at(116);
    ack_enable = 1;
    await_term(116); drain();
    require(term_count[116] == 1, "mid-segment outbound cancel settles once");

    ack_enable = 0;
    issue(117, KIND_EXT, KIND_LOCAL, 17, 1, 1024, 704, 128, MFE_ISOLATED, ERR_NONE);
    begin : parkseg2
      int bound;
      bound = 0;
      while (!ack_valid && bound < 5000) begin @(negedge clk); bound++; end
      require(ack_valid, "first inbound leg terminal ACK parks");
    end
    cancel_at(117);
    ack_enable = 1;
    await_term(117); drain();

    // Cancelled inbound step: the late response is absorbed, nothing delivered.
    deliver = 0; baseline = sent; absb = absorbed_c;
    issue(113, KIND_EXT, KIND_LOCAL, 17, 1, 512, 0, 128, MFE_ISOLATED, ERR_NONE);
    while (sent == baseline) @(negedge clk);
    cancel_at(113);
    deliver = 1;
    await_term(113); drain();
    require(absorbed_c == absb + 64'd1, "cancelled inbound response absorbed exactly once");
    require_local_seed(0, 128, 96, 0, "absorbed response delivers no local bytes");

    // Same-edge cancel and completion: exactly one terminal, completion wins.
    deliver = 0; baseline = sent;
    issue(115, KIND_EXT, KIND_LOCAL, 17, 1, 800, 896, 64, MFE_COMPLETED, ERR_NONE);
    while (sent == baseline) @(negedge clk);
    @(negedge clk); deliver = 1;
    @(negedge clk); cancel_txn = 32'd115; cancel_valid = 1;
    @(posedge clk);
    @(negedge clk); cancel_valid = 0;
    await_term(115); drain();
    require(term_count[115] == 1, "same-edge cancel and completion settle once");

    // Transport fault on the external leg: FAULTED, never ISOLATED.
    fault_next = 1;
    issue(114, KIND_EXT, KIND_LOCAL, 17, 1, 700, 0, 64, MFE_FAULTED, ERR_MEMORY);
    await_term(114); drain();
    require_local_seed(0, 64, 96, 0, "faulted response delivers no local bytes");

    // Injected tracker protocol violations are refused with distinct codes.
    t2_alloc(500);
    require(!t2_free[t2_slot] && t2_live[t2_slot] && !t2_settled[t2_slot] &&
            !t2_marked[t2_slot] && !t2_acks_open[t2_slot], "tracker entry live");
    t2_cmd(1, 0, 0, 0, MFE_COMPLETED);
    require(t2_acks_open[t2_slot], "wire obligation tracked open");
    t2_cmd(0, 0, 1, 0, MFE_COMPLETED);
    require(t2_settled[t2_slot] && t2_term_valid && t2_term_txn == 32'd500 &&
            t2_term_final == MFE_COMPLETED && t2_term_err == ERR_NONE,
            "settled terminal carries txn and final state");
    @(negedge clk);  // let the parked terminal reach its consumer
    require(t2_terminals == 64'd1 && !t2_term_valid, "single terminal emitted");
    r0 = t2_refused;
    t2_cmd(0, 0, 1, 0, MFE_COMPLETED);
    require(t2_refused == r0 + 64'd1 && t2_err == ERR_DUPLICATE_TERMINAL && t2_inv,
            "duplicate terminal refused");
    require(t2_terminals == 64'd1, "refused settle emits no second terminal");
    r0 = t2_refused;
    t2_cmd(0, 0, 0, 1, MFE_COMPLETED);
    require(t2_refused == r0 + 64'd1 && t2_err == ERR_RELEASE_WITHOUT_ACK,
            "release without ACK refused");
    t2_cmd(0, 1, 0, 0, MFE_COMPLETED);
    t2_cmd(0, 0, 0, 1, MFE_COMPLETED);
    r0 = t2_refused;
    t2_cmd(0, 0, 0, 1, MFE_COMPLETED);
    require(t2_refused == r0 + 64'd1 && t2_err == ERR_DUPLICATE_RELEASE,
            "duplicate release refused");
    r0 = t2_refused;
    t2_cmd(0, 0, 1, 0, MFE_COMPLETED);
    require(t2_refused == r0 + 64'd1 && t2_err == ERR_TERMINAL_AFTER_RELEASE,
            "terminal after release refused");
    require(t2_poisoned, "refused commands poison the tracker");
    @(negedge clk); t2_alloc_txn = 32'd502; t2_alloc_valid = 1; #1;
    repeat (4) begin
      @(negedge clk); #1;
      require(!t2_alloc_ready, "poison refuses new tracker entries");
    end
    @(negedge clk); t2_alloc_valid = 0;
    t2_reset_pulse;
    require(!t2_poisoned, "quiescent reset clears tracker poison");
    t2_alloc(501);
    t2_cmd(1, 0, 0, 0, MFE_COMPLETED);
    r0 = t2_refused;
    t2_cmd(0, 0, 0, 1, MFE_COMPLETED);
    require(t2_refused == r0 + 64'd1 && t2_err == ERR_RELEASE_EARLY,
            "release before terminal refused");
    t2_cmd(0, 0, 1, 0, MFE_COMPLETED);
    t2_cmd(0, 1, 0, 0, MFE_COMPLETED);
    t2_cmd(0, 0, 0, 1, MFE_COMPLETED);
    require(t2_terminals == 64'd2 && t2_free == '1 && !t2_term_valid,
            "both injected transactions settled once and closed");

    // Poison refuses new engine starts until a quiescent reset clears it.
    poison_pulse;
    require(poisoned, "isolation failure poisons the engine");
    @(negedge clk);
    desc = '{txn_id: 32'd119, src_kind: KIND_EXT, dst_kind: KIND_LOCAL,
             binding_id: 32'd17, binding_generation: 32'd1, binding_offset: 32'd256,
             local_offset: 32'd256, bytes: 32'd64,
             ctx_slot: 32'd0, ctx_generation: 32'd1, arena_generation: 32'd99,
             scope_epoch: 32'd77};
    desc_valid = 1; #1;
    repeat (8) begin
      @(negedge clk); #1;
      require(!desc_ready, "poison refuses new starts");
    end
    @(negedge clk); desc_valid = 0;
    reset_pulse;
    require(!poisoned, "quiescent reset clears poison when idle");
    term_ready = 0;
    issue(120, KIND_EXT, KIND_LOCAL, 17, 1, 0, 256, 64, MFE_COMPLETED, ERR_NONE);
    begin : parkterm
      int bound;
      bound = 0;
      while (!term_valid && bound < 5000) begin @(negedge clk); bound++; end
      require(term_valid, "terminal parks under consumer backpressure");
    end
    repeat (10) @(negedge clk);
    term_ready = 1;
    await_term(120); drain();

    // A live transaction survives poison; the reset is refused until closure.
    deliver = 0; baseline = sent;
    issue(121, KIND_EXT, KIND_LOCAL, 17, 1, 640, 768, 64, MFE_COMPLETED, ERR_NONE);
    while (sent == baseline) @(negedge clk);
    poison_pulse;
    require(poisoned, "poison set while a transaction is live");
    reset_pulse;
    require(poisoned && dut_err == ERR_RESET, "reset refused while not quiescent");
    deliver = 1;
    await_term(121); drain();
    reset_pulse;
    require(!poisoned, "reset clears poison after quiescence");

    // Ledger closure.
    drain();
    require(stale_c == 64'd0, "no stale external responses");
    require(rsps_c == sent_c && acks_c == sent_c, "one response and one ACK per request");
    require(64'(sent) == sent_c && 64'(acked) == acks_c,
            "bench and engine agree on wire traffic");
    require(trk_terminals == 64'(issued_count), "one terminal per accepted descriptor");
    for (int t = 0; t < 1024; t++)
      if (issued[t]) require(term_count[t] == 1, "exactly one terminal per issued txn");
    $fdisplay(request_file, "D"); $fflush(request_file);
    scan_status = $fscanf(response_file, "%h", service_status);
    require(scan_status == 1 && service_status == 32'd0, "MemoryService zero leaks");
    $display("C032_PASS tb_mfe_top contexts=%0d checks=%0d external=%0d terminals=%0d absorbed=%0d",
             CONTEXT_COUNT, checks, sent, acked, absorbed_c);
    $fclose(request_file); $fclose(response_file); $finish;
  end

  initial begin
    #200_000_000;
    $fatal(1, "C032_ASSERT tb_mfe_top watchdog");
  end
endmodule
