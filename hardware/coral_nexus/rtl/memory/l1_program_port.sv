// l1_program_port.sv — C030 program-SRAM physical port: one shared install/
// fetch port with fair arbitration (RES-12), honest counting and FET-06
// rewrite protection (design/coral_nexus_tile/execution/G2_scalar.md :: C030;
// W4_Memory_Subsystem.md: the program SRAM physical array is owned by this
// package and installs use the reserved path without consuming data
// Arena/MSHR/credit).
//
//   * ONE physical port: at most one grant per cycle. Requesters are the
//     install channel plus one fetch channel per Context; a rotating pointer
//     arbitrates round-robin over all requesters, so nobody starves under
//     sustained contention from the others (RES-12).
//   * Honest counting: fetch_accepted_o / rsp_delivered_o / install_beats_o /
//     port_busy_o / bound_hits_o move only on real port handshakes, and live
//     invariants keep them equal to the in-flight bookkeeping.
//   * Fetch identity is the frozen C028 CtxFetchKey
//     (rtl/scalar/PerCtxFetchMux.scala): {ctx_slot, ctx_generation,
//     fetch_epoch, fetch_reqid} packed MSB-first; a request also carries an
//     address. Responses carry {key, data, fault} and NO address, and the
//     delivered key always equals the accepted key, so a same-PC ABA across
//     epochs can never alias.
//   * Per-Context in-flight bound: at most FETCH_OUTSTANDING accepted but
//     undelivered fetches per Context; the (N+1)th request is simply not
//     granted (ready low) and is never dropped.
//   * FET-06: an install overlapping text whose program identity
//     (program_id+epoch) holds execution references is rejected with
//     I_EXEC_REF and the bytes stay unchanged. A warm re-install of IDENTICAL
//     resident text of the same identity is a no-op answered I_ALREADY (the
//     bytes are not rewritten); any other unreferenced install writes and
//     answers I_OK.
//   * Zero-length, misaligned and out-of-range installs and fetches are
//     answered with a fault/reject code and have no side effects; fetching
//     never-installed bytes faults with F_UNWRITTEN instead of inventing data.
//
// Parameters: dimensions come from the REAL C028 frontend design
// (rtl/scalar/PerCtxFetchMux.scala over the sealed upstream Parameters):
//   * FETCH_BYTES = fetchDataBits/8 = 32 (Parameters.scala leaves fetchDataBits
//     at its sealed 256 "do not change"): the fetch line / response width,
//     decoupled from WORD_BYTES, the install write-beat width.
//   * GEN_W = ctxGenerationBits = 16, EPOCH_W = SEQ_W = ownerOpSeqBits = 32,
//     ADDR_W = fetchAddrBits = programCounterBits = 32 (the real CtxFetchKey
//     and fetch-address widths, not the old 8/8/8/16 transport containers).
//   * CONTEXT_COUNT defaults to the sealed C2 target count (c029_params_pkg ::
//     C2_CONTEXTS); C028TargetParams seals contextCounts = Seq(1, 2).
//   * FETCH_OUTSTANDING defaults to the sealed per-Context fetch capacity
//     fetch_outstanding_per_context = 2 shared by both cs1 targets (symbol
//     Fetcher.maxConcurrentTx,
//     upstream/coralnpu/hdl/chisel/src/coralnpu/scalar/UncachedFetch.scala:335);
//     it is NOT derivable through gen_memory_params.py's names map (that map
//     only sources w4_memory_constraints.yaml bounds_env, which carries no
//     fetch entry) and the generator/package are outside this slice's edit
//     set, so the bound stays a module parameter carrying the sealed value.
//     tb_c028_port.sv elaboration-checks the generated PerCtxFetchMux really
//     stops offering at exactly FETCH_OUTSTANDING.
//   * SRAM_BYTES is a bench-scale model of the program SRAM (validator
//     hardware_config tile_program_sram.capacity_bytes = 65536 is a baseline,
//     not a sealed C019 constant); WORD_BYTES, PROG_ID_W and the ref-slot
//     count are transport containers, not frozen silicon widths.
`timescale 1ns/1ps
module l1_program_port
  import c029_params_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT = C2_CONTEXTS,
  parameter int unsigned FETCH_OUTSTANDING = 2,
  parameter int unsigned SRAM_BYTES = 256,
  parameter int unsigned FETCH_BYTES = 32,
  parameter int unsigned WORD_BYTES = 4,
  parameter int unsigned INS_RSP_DEPTH = 4,
  parameter int unsigned REF_SLOTS = 8,
  parameter int unsigned PROG_ID_W = 8,
  parameter int unsigned EPOCH_W = 32,
  parameter int unsigned GEN_W = 16,
  parameter int unsigned SEQ_W = 32,
  parameter int unsigned ADDR_W = 32,
  // Derived geometry (transport containers, not sealed widths).
  localparam int unsigned CTX_W = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT),
  localparam int unsigned KEY_W = CTX_W + GEN_W + EPOCH_W + SEQ_W,
  localparam int unsigned INF_W = $clog2(FETCH_OUTSTANDING + 1),
  localparam int unsigned RSP_DEPTH = CONTEXT_COUNT * FETCH_OUTSTANDING,
  localparam int unsigned REQS = CONTEXT_COUNT + 1
) (
  input  logic clk,
  input  logic rst_n,

  // Install channel (requester 0): one beat per handshake.
  input  logic ins_valid,
  output logic ins_ready,
  input  logic [ADDR_W-1:0] ins_addr,
  input  logic [WORD_BYTES*8-1:0] ins_data,
  input  logic [WORD_BYTES-1:0] ins_be,
  input  logic [PROG_ID_W-1:0] ins_prog_id,
  input  logic [EPOCH_W-1:0] ins_epoch,
  output logic ins_rsp_valid,
  input  logic ins_rsp_ready,
  output logic [2:0] ins_rsp_code,
  output logic [PROG_ID_W-1:0] ins_rsp_prog_id,
  output logic [EPOCH_W-1:0] ins_rsp_epoch,

  // Per-Context fetch channels (requesters 1..CONTEXT_COUNT).
  input  logic fet_valid [CONTEXT_COUNT],
  output logic fet_ready [CONTEXT_COUNT],
  input  logic [KEY_W-1:0] fet_key [CONTEXT_COUNT],
  input  logic [ADDR_W-1:0] fet_addr [CONTEXT_COUNT],
  input  logic [7:0] fet_len [CONTEXT_COUNT],

  // Fetch response channel: {key, data, fault}, NO address. data is the full
  // fetch LINE (FETCH_BYTES wide, = fetchDataBits), not an install word.
  output logic rsp_valid,
  input  logic rsp_ready,
  output logic [KEY_W-1:0] rsp_key,
  output logic [FETCH_BYTES*8-1:0] rsp_data,
  output logic rsp_fault,
  output logic [3:0] rsp_code,

  // Execution reference tracking (FET-06), identity = program_id + epoch.
  input  logic ref_inc_valid,
  input  logic ref_dec_valid,
  input  logic [PROG_ID_W-1:0] ref_prog_id,
  input  logic [EPOCH_W-1:0] ref_epoch,

  // Honest counters and observability.
  output logic [63:0] fetch_accepted_o,
  output logic [63:0] rsp_delivered_o,
  output logic [63:0] install_beats_o,
  output logic [63:0] port_busy_o,
  output logic [63:0] bound_hits_o,
  output logic [CONTEXT_COUNT*INF_W-1:0] inf_o,
  output logic invariant_error_o
);
  localparam int unsigned PTR_W = $clog2(REQS);

  // fetch response codes
  localparam logic [3:0] F_OK = 4'd0, F_RANGE = 4'd1, F_ALIGN = 4'd2, F_ZERO = 4'd3,
                         F_LEN = 4'd4, F_UNWRITTEN = 4'd5;
  // install response codes
  localparam logic [2:0] I_OK = 3'd0, I_RANGE = 3'd1, I_ALIGN = 3'd2, I_ZERO = 3'd3,
                         I_EXEC_REF = 3'd4, I_ALREADY = 3'd5;

  // Program SRAM model with bytewise validity and owner identity.
  logic [7:0] mem [SRAM_BYTES];
  logic byte_valid [SRAM_BYTES];
  logic [PROG_ID_W-1:0] byte_pid [SRAM_BYTES];
  logic [EPOCH_W-1:0] byte_ep [SRAM_BYTES];

  // Execution reference counts per program identity (bounded model).
  logic [15:0] ref_cnt [REF_SLOTS];
  logic [PROG_ID_W-1:0] ref_pid [REF_SLOTS];
  logic [EPOCH_W-1:0] ref_ep [REF_SLOTS];

  // Registered response queues (held under backpressure).
  logic [KEY_W-1:0] rsp_q_key [RSP_DEPTH];
  logic [FETCH_BYTES*8-1:0] rsp_q_data [RSP_DEPTH];
  logic [3:0] rsp_q_code [RSP_DEPTH];
  int unsigned rsp_count;
  logic [2:0] ins_q_code [INS_RSP_DEPTH];
  logic [PROG_ID_W-1:0] ins_q_pid [INS_RSP_DEPTH];
  logic [EPOCH_W-1:0] ins_q_ep [INS_RSP_DEPTH];
  int unsigned ins_count;

  // Response outputs come straight from the registered queue heads: they are
  // registered values and are held stable until the consumer accepts them.
  assign rsp_valid = rsp_count != 0;
  assign rsp_key = rsp_q_key[0];
  assign rsp_data = rsp_q_data[0];
  assign rsp_fault = rsp_q_code[0] != F_OK;
  assign rsp_code = rsp_q_code[0];
  assign ins_rsp_valid = ins_count != 0;
  assign ins_rsp_code = ins_q_code[0];
  assign ins_rsp_prog_id = ins_q_pid[0];
  assign ins_rsp_epoch = ins_q_ep[0];

  logic [INF_W-1:0] inf [CONTEXT_COUNT];
  logic [PTR_W-1:0] rr_ptr;

  logic [63:0] fetch_accepted, rsp_delivered, install_beats, port_busy, bound_hits;
  logic inv_err;

  assign fetch_accepted_o = fetch_accepted;
  assign rsp_delivered_o = rsp_delivered;
  assign install_beats_o = install_beats;
  assign port_busy_o = port_busy;
  assign bound_hits_o = bound_hits;
  assign invariant_error_o = inv_err;

  always_comb begin
    for (int c = 0; c < CONTEXT_COUNT; c++) inf_o[c*INF_W +: INF_W] = inf[c];
  end

  function automatic int unsigned ctxslot(input logic [KEY_W-1:0] k);
    return int'(k[KEY_W-1 -: CTX_W]);
  endfunction

  // Total live execution references for one program identity.
  function automatic logic [15:0] ref_of(input logic [PROG_ID_W-1:0] pid,
                                         input logic [EPOCH_W-1:0] ep);
    logic [15:0] r;
    r = '0;
    for (int i = 0; i < REF_SLOTS; i++)
      if (ref_cnt[i] != 16'd0 && ref_pid[i] == pid && ref_ep[i] == ep)
        r = r + ref_cnt[i];
    return r;
  endfunction

  // ---------------- shared-port arbitration (RES-12) ----------------
  logic req [REQS];
  logic grant [REQS];
  logic g_found;
  int unsigned g_idx;

  always_comb begin
    req[0] = ins_valid && (ins_count < INS_RSP_DEPTH);
    for (int c = 0; c < CONTEXT_COUNT; c++)
      req[1+c] = fet_valid[c] && (inf[c] < INF_W'(FETCH_OUTSTANDING));
    for (int q = 0; q < REQS; q++) grant[q] = 1'b0;
    g_found = 1'b0;
    g_idx = 0;
    for (int i = 0; i < REQS; i++) begin
      automatic int unsigned idx = (32'(rr_ptr) + i) % REQS;
      if (!g_found && req[idx]) begin
        g_found = 1'b1;
        g_idx = idx;
      end
    end
    if (g_found) grant[g_idx] = 1'b1;
  end

  assign ins_ready = grant[0];

  always_comb begin
    for (int c = 0; c < CONTEXT_COUNT; c++) fet_ready[c] = grant[1+c];
  end

  logic ins_fire;
  logic a_fire;
  int unsigned actx;
  logic delivering;
  int unsigned dctx;
  logic ins_delivering;

  always_comb begin
    ins_fire = ins_valid && grant[0];
    a_fire = 1'b0;
    actx = 0;
    for (int c = 0; c < CONTEXT_COUNT; c++)
      if (fet_valid[c] && grant[1+c]) begin
        a_fire = 1'b1;
        actx = c;
      end
    delivering = (rsp_count != 0) && rsp_ready;
    dctx = (rsp_count != 0) ? ctxslot(rsp_q_key[0]) : 0;
    ins_delivering = (ins_count != 0) && ins_rsp_ready;
  end

  // Push slots account for a same-cycle pop (shift first, push after).
  int unsigned rsp_push_idx;
  int unsigned ins_push_idx;

  always_comb begin
    rsp_push_idx = rsp_count - (delivering ? 1 : 0);
    ins_push_idx = ins_count - (ins_delivering ? 1 : 0);
  end

  // Requester-cycles blocked at the per-Context in-flight bound.
  int unsigned bound_now;

  always_comb begin
    bound_now = 0;
    for (int c = 0; c < CONTEXT_COUNT; c++)
      if (fet_valid[c] && inf[c] == INF_W'(FETCH_OUTSTANDING)) bound_now++;
  end

  // ---------------- fetch decision ----------------
  // A fetch moves one LINE (FETCH_BYTES = fetchDataBits/8), line-aligned like
  // the real frontend's alignLine requests; WORD_BYTES only paces installs.
  logic [3:0] fet_code;
  logic [FETCH_BYTES*8-1:0] fet_rdata;

  always_comb begin
    fet_code = F_OK;
    fet_rdata = '0;
    if (a_fire) begin
      automatic logic [31:0] aw;
      aw = 32'(fet_addr[actx]);
      if (fet_len[actx] == 8'd0) fet_code = F_ZERO;
      else if (32'(fet_len[actx]) != FETCH_BYTES) fet_code = F_LEN;
      else if ((aw & (32'(FETCH_BYTES) - 32'd1)) != 32'd0) fet_code = F_ALIGN;
      else if (aw + 32'(FETCH_BYTES) > 32'(SRAM_BYTES)) fet_code = F_RANGE;
      else if (aw + 32'(FETCH_BYTES) <= 32'(SRAM_BYTES)) begin
        // The guard keeps a broken range check from reading out of bounds.
        for (int b = 0; b < FETCH_BYTES; b++)
          if (!byte_valid[aw + b]) fet_code = F_UNWRITTEN;
        if (fet_code == F_OK)
          for (int b = 0; b < FETCH_BYTES; b++)
            fet_rdata[b*8 +: 8] = mem[aw + b];
      end
    end
  end


  // ---------------- install decision (FET-06 order: protection first) ----
  logic [2:0] ins_code;
  logic ins_exec_hit;
  logic ins_same;

  always_comb begin
    ins_code = I_OK;
    ins_exec_hit = 1'b0;
    ins_same = 1'b1;
    if (ins_fire) begin
      automatic logic [31:0] aw;
      aw = 32'(ins_addr);
      if (ins_be == '0) ins_code = I_ZERO;
      else if ((aw & (32'(WORD_BYTES) - 32'd1)) != 32'd0) ins_code = I_ALIGN;
      else if (aw + 32'(WORD_BYTES) > 32'(SRAM_BYTES)) ins_code = I_RANGE;
      else if (aw + 32'(WORD_BYTES) <= 32'(SRAM_BYTES)) begin
        for (int b = 0; b < WORD_BYTES; b++) begin
          if (ins_be[b]) begin
            automatic int unsigned off;
            off = aw + b;
            if (byte_valid[off]) begin
              if (ref_of(byte_pid[off], byte_ep[off]) != 16'd0)
                ins_exec_hit = 1'b1;
              if (mem[off] != ins_data[b*8 +: 8] || byte_pid[off] != ins_prog_id ||
                  byte_ep[off] != ins_epoch)
                ins_same = 1'b0;
              // A same-edge acquisition of this identity takes effect with
              // this write; include it so the FIRST reference can never race
              // an overlapping install (FET-06 same-edge race). A same-edge
              // release is conservatively still protecting (old count read).
              if (ref_inc_valid && ref_prog_id == byte_pid[off] && ref_epoch == byte_ep[off])
                ins_exec_hit = 1'b1;
            end else ins_same = 1'b0;
          end
        end
        if (ins_exec_hit) ins_code = I_EXEC_REF;
        else if (ins_same) ins_code = I_ALREADY;
        else ins_code = I_OK;
      end
    end
  end

  // ---------------- port grant, queues, memory, references ----------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rsp_count <= 0;
      ins_count <= 0;
      rr_ptr <= '0;
      fetch_accepted <= 64'd0;
      rsp_delivered <= 64'd0;
      install_beats <= 64'd0;
      port_busy <= 64'd0;
      bound_hits <= 64'd0;
      inv_err <= 1'b0;
      for (int i = 0; i < REF_SLOTS; i++) begin
        ref_cnt[i] <= 16'd0;
        ref_pid[i] <= '0;
        ref_ep[i] <= '0;
      end
      for (int a = 0; a < SRAM_BYTES; a++) byte_valid[a] <= 1'b0;
    end else begin
      // Queue pops: shift the registered entries down.
      if (delivering) begin
        for (int i = 0; i < RSP_DEPTH-1; i++) begin
          rsp_q_key[i] <= rsp_q_key[i+1];
          rsp_q_data[i] <= rsp_q_data[i+1];
          rsp_q_code[i] <= rsp_q_code[i+1];
        end
      end
      if (ins_delivering) begin
        for (int i = 0; i < INS_RSP_DEPTH-1; i++) begin
          ins_q_code[i] <= ins_q_code[i+1];
          ins_q_pid[i] <= ins_q_pid[i+1];
          ins_q_ep[i] <= ins_q_ep[i+1];
        end
      end
      // Queue pushes land after the shifts (a same-edge pop+push orders right).
      if (ins_fire) begin
        ins_q_code[ins_push_idx] <= ins_code;
        ins_q_pid[ins_push_idx] <= ins_prog_id;
        ins_q_ep[ins_push_idx] <= ins_epoch;
        if (ins_code == I_OK) begin
          for (int b = 0; b < WORD_BYTES; b++)
            if (ins_be[b] && (32'(ins_addr) + b) < SRAM_BYTES) begin
              mem[32'(ins_addr) + b] <= ins_data[b*8 +: 8];
              byte_valid[32'(ins_addr) + b] <= 1'b1;
              byte_pid[32'(ins_addr) + b] <= ins_prog_id;
              byte_ep[32'(ins_addr) + b] <= ins_epoch;
            end
        end
      end
      if (a_fire) begin
        rsp_q_key[rsp_push_idx] <= fet_key[actx];
        rsp_q_data[rsp_push_idx] <= fet_rdata;
        rsp_q_code[rsp_push_idx] <= fet_code;
      end
      // Combined occupancies (a push and a pop may coincide).
      if (delivering || a_fire)
        rsp_count <= rsp_count - (delivering ? 1 : 0) + (a_fire ? 1 : 0);
      if (ins_delivering || ins_fire)
        ins_count <= ins_count - (ins_delivering ? 1 : 0) + (ins_fire ? 1 : 0);
      // Honest counters: only real port handshakes move them.
      if (ins_fire) install_beats <= install_beats + 64'd1;
      if (a_fire) fetch_accepted <= fetch_accepted + 64'd1;
      if (delivering) rsp_delivered <= rsp_delivered + 64'd1;
      if (g_found) port_busy <= port_busy + 64'd1;
      if (bound_now != 0) bound_hits <= bound_hits + 64'(bound_now);
      if (g_found) rr_ptr <= PTR_W'((g_idx + 1) % REQS);
      // Execution reference bookkeeping (identity = program_id + epoch).
      if (ref_inc_valid && ref_dec_valid) begin
        inv_err <= 1'b1;
      end else if (ref_inc_valid) begin
        automatic int unsigned hit;
        automatic int unsigned free_i;
        hit = REF_SLOTS;
        free_i = REF_SLOTS;
        for (int i = 0; i < REF_SLOTS; i++) begin
          if (ref_cnt[i] != 16'd0 && ref_pid[i] == ref_prog_id && ref_ep[i] == ref_epoch &&
              hit == REF_SLOTS)
            hit = i;
          if (ref_cnt[i] == 16'd0 && free_i == REF_SLOTS) free_i = i;
        end
        if (hit != REF_SLOTS) ref_cnt[hit] <= ref_cnt[hit] + 16'd1;
        else if (free_i != REF_SLOTS) begin
          ref_cnt[free_i] <= 16'd1;
          ref_pid[free_i] <= ref_prog_id;
          ref_ep[free_i] <= ref_epoch;
        end else inv_err <= 1'b1;
      end else if (ref_dec_valid) begin
        automatic int unsigned hit;
        hit = REF_SLOTS;
        for (int i = 0; i < REF_SLOTS; i++)
          if (ref_cnt[i] != 16'd0 && ref_pid[i] == ref_prog_id && ref_ep[i] == ref_epoch &&
              hit == REF_SLOTS)
            hit = i;
        if (hit != REF_SLOTS) ref_cnt[hit] <= ref_cnt[hit] - 16'd1;
        else inv_err <= 1'b1;
      end
    end
  end

  // In-flight bookkeeping is updated separately so an accept and a delivery
  // of the same Context in one cycle can never lose an update.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int c = 0; c < CONTEXT_COUNT; c++) inf[c] <= '0;
    end else begin
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        automatic int unsigned d;
        d = 32'(inf[c]);
        if (delivering && dctx == c) d--;
        if (a_fire && actx == c) d++;
        inf[c] <= INF_W'(d);
      end
    end
  end

  // Live honesty invariants, checked in the clock domain separately from the
  // grant FSM so a mutation cannot mask itself.
  always @(posedge clk) begin
    if (rst_n) begin
      automatic logic [63:0] s;
      s = '0;
      for (int c = 0; c < CONTEXT_COUNT; c++) s = s + 64'(inf[c]);
      if (s != fetch_accepted - rsp_delivered)
        $error("C030_ASSERT l1_program_port accepted/delivered counters drift from in-flight");
      if (64'(rsp_count) != s)
        $error("C030_ASSERT l1_program_port response queue disagrees with in-flight");
      if (port_busy != fetch_accepted + install_beats)
        $error("C030_ASSERT l1_program_port port-busy counter is not handshake-honest");
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (inf[c] > INF_W'(FETCH_OUTSTANDING))
          $error("C030_ASSERT l1_program_port context %0d exceeds the fetch in-flight bound", c);
        if (fet_valid[c] && ctxslot(fet_key[c]) != c)
          $error("C030_ASSERT l1_program_port channel %0d carries a foreign ctx_slot key", c);
      end
      if (delivering && inf[dctx] == '0)
        $error("C030_ASSERT l1_program_port delivery without an accepted request");
    end
  end
endmodule
