// tb_c039_cosim.sv — C039 co-simulation endpoint: the real `w5/install_service`
// driving the real C030 `memory/l1_program_port` program SRAM.
//
// This bench owns no stimulus and asserts no scenario. The Group side (live
// Python: `ProgramService` + the Group residency manager) is the sole time
// authority: every command below is a bounded number of clock edges with fixed
// inputs, and the reply reports, in order, everything the Tile boundary did at
// those edges. The sealed C017 layout is read through the generated
// `cn_offsets.svh` only (message ids and payload word counts).
//
// Geometry (C038 slot_freeze.json; the Python driver reads the same file):
//   SRAM = instruction_storage.itcm_bytes (8192), SLOTS = program_slots (2),
//   one 32 B fetch line, fetch_outstanding_per_context = 2 (C019 target).
//
// The bench is the Tile-side link layer around the service: it re-frames the
// T2G word stream and checks what a link must guarantee on its own — a sealed
// message id, the sealed payload word count, a contiguous message_seq, a zero
// reserved word, and a word/handle that stay stable while the Group holds
// ready low. The C030 port's fetch channels carry the bridge's probes, and the
// execution-reference pulse reaches the service and the port on the same edge.
//
// Commands: ASCII hex words, opcode first (`%h` scanning only).
//   01 n w[n]                          deliver n G2T words (one sealed frame + its blob words)
//   02 max                             idle ticks until the next boundary event (bounded by max)
//   03 ctx gen epoch reqid addr len    arm a fetch probe on Context ctx (real C030 fetch port)
//   04 inc pid epoch                   one tick with an execution-reference inc/dec pulse
//   05 v                               T2G ready level (0 = Group backpressure)
//   06 v                               fetch-response ready level (0 = probe backpressure)
//   07 pid ver h[4] epoch slot base len dig[4]
//                                      trusted per-slot setup from the verified sealed image
//   08                                 status only (no tick)
//   09                                 Tile reset pulse
//   0a code                            Group verdict: the live Python side rejected the Tile behavior
//   0b                                 quit: print the PASS record and finish
// Reply (one line): status ticks nev g2t_acc g2t_pending,
//   nev x (type nwords words...), then the fixed observation tail of reply().
// Events: 01 T2G frame [id seq frame-words... handle] · 02 fault [code] ·
//   03 fetch probe accepted [ctx gate] · 04 fetch response [ctx fault code key2 data4] ·
//   05 C030 install response that is neither I_OK nor I_ALREADY [code pid epoch] ·
//   06 setup accepted [].
`timescale 1ns/1ps
module tb_c039_cosim
  import c029_params_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT = C2_CONTEXTS
) ();
  // The generated header carries 12-bit CSR literals in 32-bit int localparams (allocator
  // output format); only that known width expansion is silenced, as in c038_scalar_pkg.
  /* verilator lint_off WIDTHEXPAND */
  `include "cn_offsets.svh"
  /* verilator lint_on WIDTHEXPAND */

  localparam int unsigned FO = 2;        // fetch_outstanding_per_context (sealed C019 target)
  localparam int unsigned FB = 32;       // fetchDataBits/8: one fetch line
  localparam int unsigned WB = 4;        // install word beat
  localparam int unsigned SRAM = 8192;   // slot_freeze.instruction_storage.itcm_bytes
  localparam int unsigned SLOTS = 2;     // slot_freeze.slot_geometry.program_slots
  localparam int unsigned SETUP_SLOTS = 4;
  localparam int unsigned GEN_W = 16, EPOCH_W = 32, SEQ_W = 32, PROG_ID_W = 8, ADDR_W = 32;
  localparam int unsigned CTX_W = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT);
  localparam int unsigned KEY_W = CTX_W + GEN_W + EPOCH_W + SEQ_W;
  localparam int unsigned INF_W = $clog2(FO + 1);
  localparam int unsigned KQ_DEPTH = FO + 2;
  localparam int unsigned FRAME_MAX = CN_HEADER_WORDS + CN_WORDS_PROGRAM_INSTALL_ACK;
  localparam int unsigned EV_WORDS = FRAME_MAX + 4, MAX_EV = 64, MAX_ARG = 64, MAX_TICKS = 8192;
  localparam int unsigned TAIL_WORDS = 24;
  // reply statuses
  localparam int unsigned ST_OK = 0, ST_EVENT = 1, ST_NOT_ACCEPTED = 2, ST_NO_EVENT = 4;
  // event kinds
  localparam int unsigned EV_FRAME = 1, EV_FAULT = 2, EV_FETCH_ACCEPT = 3, EV_FETCH_RESPONSE = 4,
                         EV_INSTALL_REJECT = 5, EV_SETUP = 6;
  // C030 install response codes (l1_program_port.sv); the bench only classifies.
  localparam logic [2:0] I_OK = 3'd0, I_ALREADY = 3'd5;

  logic clk = 1'b0, rst_n = 1'b0;
  always #5 clk = ~clk;
  initial begin
    #50_000_000;
    $fatal(1, "C039_ASSERT tb_c039_cosim watchdog: the Group stopped ticking");
  end

  // ---------------- Tile-side PROGRAM endpoint: w5/install_service ----------------
  logic        g2t_valid = 1'b0, g2t_ready;
  logic [63:0] g2t_word = '0;
  logic        t2g_valid, t2g_ready = 1'b1;
  logic [63:0] t2g_word;
  logic [63:0] ack_handle;
  logic        ack_handle_valid;
  logic        setup_valid = 1'b0, setup_ready;
  logic [63:0] setup_program_id = '0, setup_version = '0, setup_program_epoch = '0,
               setup_slot = '0, setup_text_base = '0, setup_text_len = '0;
  logic [255:0] setup_package_hash = '0, setup_text_digest = '0;
  logic        exec_ref_inc = 1'b0, exec_ref_dec = 1'b0;
  logic [PROG_ID_W-1:0] exec_ref_prog_id = '0;
  logic [EPOCH_W-1:0] exec_ref_epoch = '0;
  logic [ADDR_W-1:0] fetch_addr;
  logic             fetch_allow;
  logic [3:0]       state;
  logic             fault, fault_seen, invariant_error;
  logic [63:0]      fault_code;
  logic [SLOTS-1:0] resident_map, ready_map, slot_valid;
  logic [63:0] read_req_count, resp_bytes, ins_beats, install_count, warm_hit_count, inv_count;
  // C030 install port: the service is its only requester; the port answers each beat.
  logic                 ins_valid, ins_ready, ins_rsp_valid, ins_rsp_ready;
  logic [ADDR_W-1:0]    ins_addr, ins_data;
  logic [WB-1:0]        ins_be;
  logic [PROG_ID_W-1:0] ins_prog_id, ins_rsp_prog_id;
  logic [EPOCH_W-1:0]   ins_epoch, ins_rsp_epoch;
  logic [2:0]           ins_rsp_code;
  // Execution references: one pulse, seen by the service counter and by the C030 port.
  logic                 ref_inc_valid, ref_dec_valid;
  logic [PROG_ID_W-1:0] ref_prog_id;
  logic [EPOCH_W-1:0]   ref_epoch;
  assign ref_inc_valid = exec_ref_inc;
  assign ref_dec_valid = exec_ref_dec;
  assign ref_prog_id   = exec_ref_prog_id;
  assign ref_epoch     = exec_ref_epoch;

  // ---------------- real C030 program port + the bridge's fetch probes ----------------
  logic fet_valid [CONTEXT_COUNT];
  logic fet_ready [CONTEXT_COUNT];
  logic [KEY_W-1:0] fet_key [CONTEXT_COUNT];
  logic [ADDR_W-1:0] fet_addr [CONTEXT_COUNT];
  logic [7:0] fet_len [CONTEXT_COUNT];
  logic rsp_valid, rsp_ready = 1'b1, rsp_fault;
  logic [KEY_W-1:0] rsp_key;
  logic [FB*8-1:0] rsp_data;
  logic [3:0] rsp_code;
  logic [63:0] fetch_accepted, rsp_delivered, install_beats, port_busy, bound_hits;
  logic [CONTEXT_COUNT*INF_W-1:0] inf;

  initial begin
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      fet_valid[c] = 1'b0;
      fet_key[c] = '0;
      fet_addr[c] = '0;
      fet_len[c] = 8'd0;
    end
  end

  // The exported residency gate is asked about the probe address being offered
  // (lowest valid Context first).
  always_comb begin
    fetch_addr = '0;
    for (int c = CONTEXT_COUNT; c > 0; c--)
      if (fet_valid[c - 1]) fetch_addr = fet_addr[c - 1];
  end

  install_service #(
    .SLOTS(SLOTS),
    .SLOT_BYTES(SRAM / SLOTS),
    .SRAM_BYTES(SRAM),
    .SETUP_SLOTS(SETUP_SLOTS),
    .WORD_BYTES(WB),
    .LINE_BYTES(FB),
    .ADDR_W(ADDR_W),
    .PROG_ID_W(PROG_ID_W),
    .EPOCH_W(EPOCH_W)
  ) service (
    .clk(clk),
    .rst_n(rst_n),
    .setup_valid_i(setup_valid),
    .setup_ready_o(setup_ready),
    .setup_program_id_i(setup_program_id),
    .setup_version_i(setup_version),
    .setup_package_hash_i(setup_package_hash),
    .setup_program_epoch_i(setup_program_epoch),
    .setup_slot_i(setup_slot),
    .setup_text_base_i(setup_text_base),
    .setup_text_len_i(setup_text_len),
    .setup_text_digest_i(setup_text_digest),
    .g2t_valid_i(g2t_valid),
    .g2t_ready_o(g2t_ready),
    .g2t_word_i(g2t_word),
    .t2g_valid_o(t2g_valid),
    .t2g_ready_i(t2g_ready),
    .t2g_word_o(t2g_word),
    .ack_handle_valid_o(ack_handle_valid),
    .ack_handle_o(ack_handle),
    .ins_valid_o(ins_valid),
    .ins_ready_i(ins_ready),
    .ins_addr_o(ins_addr),
    .ins_data_o(ins_data),
    .ins_be_o(ins_be),
    .ins_prog_id_o(ins_prog_id),
    .ins_epoch_o(ins_epoch),
    .ins_rsp_valid_i(ins_rsp_valid),
    .ins_rsp_ready_o(ins_rsp_ready),
    .ins_rsp_code_i(ins_rsp_code),
    .ins_rsp_prog_id_i(ins_rsp_prog_id),
    .ins_rsp_epoch_i(ins_rsp_epoch),
    .exec_ref_inc_i(exec_ref_inc),
    .exec_ref_dec_i(exec_ref_dec),
    .exec_ref_prog_id_i(exec_ref_prog_id),
    .exec_ref_epoch_i(exec_ref_epoch),
    .fetch_addr_i(fetch_addr),
    .fetch_allow_o(fetch_allow),
    .state_o(state),
    .resident_o(resident_map),
    .ready_o(ready_map),
    .slot_valid_o(slot_valid),
    .fault_o(fault),
    .fault_seen_o(fault_seen),
    .fault_code_o(fault_code),
    .read_req_count_o(read_req_count),
    .resp_bytes_o(resp_bytes),
    .ins_beats_o(ins_beats),
    .install_count_o(install_count),
    .warm_hit_count_o(warm_hit_count),
    .inv_count_o(inv_count)
  );

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
  ) program_port (
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
    .fetch_accepted_o(fetch_accepted),
    .rsp_delivered_o(rsp_delivered),
    .install_beats_o(install_beats),
    .port_busy_o(port_busy),
    .bound_hits_o(bound_hits),
    .inf_o(inf),
    .invariant_error_o(invariant_error)
  );

  // ---------------- bookkeeping ----------------
  integer req_fd, rsp_fd;
  int unsigned ticks_total = 0, checks = 0, reset_edges = 0;
  int unsigned ev_count = 0, ev_kind [MAX_EV], ev_n [MAX_EV];
  logic [63:0] ev_w [MAX_EV][EV_WORDS], stage[EV_WORDS], arg_w [MAX_ARG];
  int unsigned g2t_accepted = 0;
  bit stop, install_rejected;

  logic [KEY_W-1:0] kq [CONTEXT_COUNT][KQ_DEPTH];
  int unsigned kq_n [CONTEXT_COUNT];

  logic [63:0] t2g_buf[FRAME_MAX];
  int unsigned t2g_n = 0, t2g_type = 0, t2g_exp = 0, t2g_seq = 0;
  logic [63:0] t2g_handle = '0;
  bit held;
  logic [63:0] held_word, held_handle;
  bit held_handle_valid;
  bit fault_latched;
  logic [63:0] fault_latched_code;

  initial begin
    for (int c = 0; c < CONTEXT_COUNT; c++) kq_n[c] = 0;
    held               = 1'b0;
    held_word          = '0;
    held_handle        = '0;
    held_handle_valid  = 1'b0;
    fault_latched      = 1'b0;
    fault_latched_code = '0;
  end

  function automatic int unsigned ctxslot(input logic [KEY_W-1:0] k);
    return int'(k[KEY_W-1 -: CTX_W]);
  endfunction

  task automatic require(input bit ok, input string label);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C039_ASSERT tb_c039_cosim %s (tick %0d)", label, ticks_total);
  endtask

  task automatic push_ev(input int unsigned kind, input int unsigned n);
    if (ev_count == MAX_EV)
      $fatal(1, "C039_ASSERT tb_c039_cosim event overflow (tick %0d)", ticks_total);
    ev_kind[ev_count] = kind;
    ev_n[ev_count]   = n;
    for (int i = 0; i < n; i++) ev_w[ev_count][i] = stage[i];
    ev_count++;
  endtask

  task automatic begin_command();
    ev_count         = 0;
    stop             = 1'b0;
    g2t_accepted     = 0;
    install_rejected = 1'b0;
  endtask

  // Values the boundary showed just before the edge under observation.
  bit s_g2t_acc, s_setup_acc, s_t2g_valid, s_t2g_fire, s_handle_valid, s_rsp_fire, s_fault;
  bit s_ins_rsp_fire, s_gate;
  bit s_fet_acc [CONTEXT_COUNT];
  logic [63:0] s_t2g_word, s_handle, s_fault_code;
  logic [KEY_W-1:0] s_rsp_key;
  logic [FB*8-1:0] s_rsp_data;
  logic s_rsp_fault;
  logic [3:0] s_rsp_code;
  logic [2:0] s_ins_rsp_code;
  logic [PROG_ID_W-1:0] s_ins_rsp_pid;
  logic [EPOCH_W-1:0] s_ins_rsp_epoch;

  task automatic observe();
    int unsigned c;
    // T2G word stream: re-frame the sealed layout and keep the link invariants.
    if (held)
      require(s_t2g_valid && s_t2g_word == held_word && s_handle_valid == held_handle_valid &&
              (!held_handle_valid || s_handle == held_handle),
              "T2G word and ACK handle stay stable while the Group holds ready low");
    held              = !t2g_ready && s_t2g_valid;
    held_word         = s_t2g_word;
    held_handle       = s_handle;
    held_handle_valid = s_handle_valid;
    if (s_t2g_fire) begin
      t2g_buf[t2g_n] = s_t2g_word;
      if (t2g_n == 0) begin
        require(s_t2g_word == 64'(CN_MSG_PROGRAM_READ_REQ) ||
                s_t2g_word == 64'(CN_MSG_PROGRAM_INSTALL_ACK) ||
                s_t2g_word == 64'(CN_MSG_INVALIDATE_ACK),
                "T2G word0 is a sealed PROGRAM T2G message id");
        t2g_type = s_t2g_word[31:0];
        case (t2g_type)
          CN_MSG_PROGRAM_READ_REQ:    t2g_exp = CN_WORDS_PROGRAM_READ_REQ;
          CN_MSG_PROGRAM_INSTALL_ACK: t2g_exp = CN_WORDS_PROGRAM_INSTALL_ACK;
          default:                    t2g_exp = CN_WORDS_INVALIDATE_ACK;
        endcase
        if (t2g_type == CN_MSG_PROGRAM_INSTALL_ACK) begin
          require(s_handle_valid, "INSTALL_ACK frame carries the local-handle sideband");
          t2g_handle = s_handle;
        end
      end else begin
        if (t2g_n == 1) require(s_t2g_word == 64'(t2g_exp), "T2G payload word count equals the sealed layout");
        if (t2g_n == 2) require(s_t2g_word == 64'(t2g_seq), "T2G message_seq is contiguous");
        if (t2g_n == 3) require(s_t2g_word == 64'd0, "T2G reserved header word is zero");
        if (t2g_type == CN_MSG_PROGRAM_INSTALL_ACK)
          require(s_handle_valid && s_handle == t2g_handle, "ACK handle sideband is stable across its frame");
      end
      t2g_n++;
      if (t2g_n == CN_HEADER_WORDS + t2g_exp) begin
        stage[0] = 64'(t2g_type);
        stage[1] = 64'(t2g_seq);
        for (int i = 0; i < t2g_n; i++) stage[2 + i] = t2g_buf[i];
        stage[2 + t2g_n] = t2g_handle;
        push_ev(EV_FRAME, 3 + t2g_n);  // id, seq, every frame word, handle sideband
        $display("C039_T2G tb_c039_cosim id=%0d seq=%0d tick=%0d", t2g_type, t2g_seq, ticks_total);
        t2g_n      = 0;
        t2g_seq++;
        t2g_handle = '0;
        stop       = 1'b1;
      end
    end
    // Group -> Tile accounting (the Python side holds the matching half).
    if (s_g2t_acc) g2t_accepted++;
    if (s_setup_acc) begin
      push_ev(EV_SETUP, 0);
      stop = 1'b1;
    end
    if (s_fault) begin
      stage[0] = s_fault_code;
      push_ev(EV_FAULT, 1);
      $display("C039_FAULT tb_c039_cosim code=%0d tick=%0d", s_fault_code, ticks_total);
      if (!fault_latched) begin
        fault_latched      = 1'b1;
        fault_latched_code = s_fault_code;
      end
      stop = 1'b1;
    end
    if (s_ins_rsp_fire && s_ins_rsp_code != I_OK && s_ins_rsp_code != I_ALREADY) begin
      install_rejected = 1'b1;
      stage[0] = 64'(s_ins_rsp_code);
      stage[1] = 64'(s_ins_rsp_pid);
      stage[2] = 64'(s_ins_rsp_epoch);
      push_ev(EV_INSTALL_REJECT, 3);
      stop = 1'b1;
    end
    // Fetch probes on the real C030 fetch port.
    for (int i = 0; i < CONTEXT_COUNT; i++) begin
      if (s_fet_acc[i]) begin
        require(kq_n[i] < KQ_DEPTH, "fetch probe in-flight bookkeeping is bounded");
        kq[i][kq_n[i]] = fet_key[i];
        kq_n[i]++;
        fet_valid[i] = 1'b0;  // held until accepted, then dropped
        stage[0] = 64'(i);
        stage[1] = 64'(s_gate);
        push_ev(EV_FETCH_ACCEPT, 2);
        stop = 1'b1;
      end
    end
    if (s_rsp_fire) begin
      c = ctxslot(s_rsp_key);
      require(kq_n[c] != 0 && kq[c][0] == s_rsp_key, "fetch response echoes its request key");
      for (int i = 0; i + 1 < kq_n[c]; i++) kq[c][i] = kq[c][i + 1];
      kq_n[c]--;
      stage[0] = 64'(c);
      stage[1] = 64'(s_rsp_fault);
      stage[2] = 64'(s_rsp_code);
      stage[3] = s_rsp_key[63:0];
      stage[4] = 64'(s_rsp_key >> 64);
      for (int i = 0; i < FB / 8; i++) stage[5 + i] = s_rsp_data[i * 64 +: 64];
      push_ev(EV_FETCH_RESPONSE, 5 + FB / 8);
      $display("C039_PROBE tb_c039_cosim ctx=%0d fault=%0d code=%0d tick=%0d", c, s_rsp_fault,
               s_rsp_code, ticks_total);
      stop = 1'b1;
    end
    require(!invariant_error, "C030 program-port honesty invariants hold");
  endtask

  // One clock edge with the current inputs.
  task automatic step();
    @(negedge clk);
    #1;
    s_g2t_acc       = g2t_valid && g2t_ready;
    s_setup_acc     = setup_valid && setup_ready;
    s_t2g_valid     = t2g_valid;
    s_t2g_fire      = t2g_valid && t2g_ready;
    s_t2g_word      = t2g_word;
    s_handle        = ack_handle;
    s_handle_valid  = ack_handle_valid;
    s_gate          = fetch_allow;
    s_rsp_fire      = rsp_valid && rsp_ready;
    s_rsp_key       = rsp_key;
    s_rsp_data      = rsp_data;
    s_rsp_fault     = rsp_fault;
    s_rsp_code      = rsp_code;
    s_ins_rsp_fire  = ins_rsp_valid && ins_rsp_ready;
    s_ins_rsp_code  = ins_rsp_code;
    s_ins_rsp_pid   = ins_rsp_prog_id;
    s_ins_rsp_epoch = ins_rsp_epoch;
    s_fault         = fault;
    s_fault_code    = fault_code;
    for (int c = 0; c < CONTEXT_COUNT; c++) s_fet_acc[c] = fet_valid[c] && fet_ready[c];
    @(posedge clk);
    #1;
    ticks_total++;
    if (!rst_n) begin
      reset_edges++;
      if (reset_edges > 1)
        require(!s_t2g_fire && !s_fault, "no Tile traffic while in reset");
    end else begin
      observe();
    end
  endtask

  task automatic reply(input int unsigned status, input int unsigned ticks);
    string s;
    s = $sformatf("%0h %0h %0h %0h %0h", status, ticks, ev_count, g2t_accepted, g2t_valid);
    for (int i = 0; i < ev_count; i++) begin
      s = {s, $sformatf(" %0h %0h", ev_kind[i], ev_n[i])};
      for (int j = 0; j < ev_n[i]; j++) s = {s, $sformatf(" %0h", ev_w[i][j])};
    end
    s = {s, $sformatf(
      " %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h %0h",
      64'(state), 64'(resident_map), 64'(ready_map), 64'(slot_valid), fault_latched,
      fault_latched_code, fault_seen, t2g_valid, fetch_allow, read_req_count, resp_bytes, ins_beats,
      install_count, warm_hit_count, inv_count, fetch_accepted, rsp_delivered, install_beats,
      port_busy, bound_hits, 64'(invariant_error), 64'(t2g_seq), 64'(checks), 64'(ticks_total))};
    $fdisplay(rsp_fd, "%s", s);
    $fflush(rsp_fd);
  endtask

  task automatic deliver(input int unsigned count);
    int unsigned sent, ticks;
    begin_command();
    require(!g2t_valid, "deliver: the previous G2T word was accepted first");
    sent  = 0;
    ticks = 0;
    while (sent < count && !stop && ticks < MAX_TICKS) begin
      g2t_valid = 1'b1;
      g2t_word  = arg_w[sent];
      step();
      ticks++;
      if (s_g2t_acc) sent++;
    end
    if (sent == count) g2t_valid = 1'b0;  // consumed: nothing is left to hold
    if (stop) reply(ST_EVENT, ticks);
    else if (sent == count) reply(ST_OK, ticks);
    else reply(ST_NOT_ACCEPTED, ticks);
  endtask

  task automatic wait_event(input int unsigned max_ticks);
    int unsigned ticks;
    begin_command();
    ticks = 0;
    while (!stop && ticks < max_ticks) begin
      step();
      ticks++;
    end
    if (stop) reply(ST_EVENT, ticks);
    else reply(ST_NO_EVENT, ticks);
  endtask

  task automatic probe(input logic [63:0] ctx, gen, epoch, reqid, addr, len);
    int unsigned c;
    begin_command();
    c = int'(ctx);
    require(c < CONTEXT_COUNT, "probe: Context index inside the target geometry");
    require(!fet_valid[c], "probe: one outstanding probe arm per Context");
    fet_valid[c] = 1'b1;
    fet_key[c]   = {CTX_W'(c), gen[GEN_W-1:0], epoch[EPOCH_W-1:0], reqid[SEQ_W-1:0]};
    fet_addr[c]  = ADDR_W'(addr);
    fet_len[c]   = 8'(len);
    reply(ST_OK, 0);
  endtask

  task automatic ref_pulse(input logic [63:0] inc, pid, epoch);
    begin_command();
    if (inc != 64'd0) exec_ref_inc = 1'b1;
    else exec_ref_dec = 1'b1;
    exec_ref_prog_id = PROG_ID_W'(pid);
    exec_ref_epoch   = EPOCH_W'(epoch);
    step();
    exec_ref_inc = 1'b0;
    exec_ref_dec = 1'b0;
    if (stop) reply(ST_EVENT, 1);
    else reply(ST_OK, 1);
  endtask

  task automatic setup_write();
    int unsigned ticks;
    begin_command();
    setup_valid         = 1'b1;  // fields were loaded by the caller
    ticks               = 0;
    while (!stop && !s_setup_acc && ticks < MAX_TICKS) begin
      step();
      ticks++;
    end
    setup_valid = 1'b0;
    if (stop) reply(ST_EVENT, ticks);
    else if (s_setup_acc) reply(ST_OK, ticks);
    else reply(ST_NOT_ACCEPTED, ticks);
  endtask

  task automatic reset_pulse();
    int unsigned ticks;
    begin_command();
    g2t_valid    = 1'b0;
    setup_valid  = 1'b0;
    exec_ref_inc = 1'b0;
    exec_ref_dec = 1'b0;
    for (int c = 0; c < CONTEXT_COUNT; c++) fet_valid[c] = 1'b0;
    t2g_ready = 1'b1;
    rsp_ready = 1'b1;
    rst_n     = 1'b0;
    ticks     = 0;
    repeat (4) begin
      step();
      ticks++;
    end
    rst_n = 1'b1;
    repeat (2) begin
      step();
      ticks++;
    end
    reset_edges        = 0;
    t2g_n              = 0;
    t2g_seq            = 0;
    t2g_handle         = '0;
    held               = 1'b0;
    fault_latched      = 1'b0;
    fault_latched_code = '0;
    for (int c = 0; c < CONTEXT_COUNT; c++) kq_n[c] = 0;
    reply(ST_OK, ticks);
  endtask

  initial begin : main
    int n;
    logic [63:0] op, a, b, c, d, e, f;
    string peer;
    if (!$value$plusargs("PEER=%s", peer)) $fatal(1, "C039_ASSERT tb_c039_cosim requires +PEER");
    req_fd = $fopen({peer, ".cmd"}, "r");   // commands: Python -> bench
    rsp_fd = $fopen({peer, ".evt"}, "w");   // replies: bench -> Python
    if (req_fd == 0 || rsp_fd == 0) $fatal(1, "C039_ASSERT tb_c039_cosim cannot open %s", peer);
    $display("C039_READY tb_c039_cosim contexts=%0d", CONTEXT_COUNT);
    repeat (3) @(negedge clk);
    rst_n = 1'b1;
    repeat (2) step();
    forever begin
      n = $fscanf(req_fd, "%h", op);
      if (n != 1) $fatal(1, "C039_ASSERT tb_c039_cosim command stream closed");
      case (op)
        64'h01: begin
          n = $fscanf(req_fd, "%h", a);
          if (n != 1 || a < 64'd1 || a > 64'(MAX_ARG))
            $fatal(1, "C039_ASSERT tb_c039_cosim bad deliver count");
          for (int i = 0; i < int'(a); i++) begin
            n = $fscanf(req_fd, "%h", arg_w[i]);
            if (n != 1) $fatal(1, "C039_ASSERT tb_c039_cosim truncated deliver");
          end
          deliver(int'(a));
        end
        64'h02: begin
          n = $fscanf(req_fd, "%h", a);
          if (n != 1 || a < 64'd1 || a > 64'(MAX_TICKS))
            $fatal(1, "C039_ASSERT tb_c039_cosim bad wait bound");
          wait_event(int'(a));
        end
        64'h03: begin
          n = $fscanf(req_fd, "%h %h %h %h %h %h", a, b, c, d, e, f);
          if (n != 6) $fatal(1, "C039_ASSERT tb_c039_cosim truncated probe");
          probe(a, b, c, d, e, f);
        end
        64'h04: begin
          n = $fscanf(req_fd, "%h %h %h", a, b, c);
          if (n != 3) $fatal(1, "C039_ASSERT tb_c039_cosim truncated ref");
          ref_pulse(a, b, c);
        end
        64'h05: begin
          n = $fscanf(req_fd, "%h", a);
          if (n != 1 || a > 64'd1) $fatal(1, "C039_ASSERT tb_c039_cosim bad t2g ready level");
          t2g_ready = a[0];
          begin_command();
          reply(ST_OK, 0);
        end
        64'h06: begin
          n = $fscanf(req_fd, "%h", a);
          if (n != 1 || a > 64'd1) $fatal(1, "C039_ASSERT tb_c039_cosim bad fetch ready level");
          rsp_ready = a[0];
          begin_command();
          reply(ST_OK, 0);
        end
        64'h07: begin
          n = $fscanf(
            req_fd, "%h %h %h %h %h %h %h %h %h %h %h %h %h %h",
            arg_w[0], arg_w[1], arg_w[2], arg_w[3], arg_w[4], arg_w[5], arg_w[6], arg_w[7],
            arg_w[8], arg_w[9], arg_w[10], arg_w[11], arg_w[12], arg_w[13]);
          if (n != 14) $fatal(1, "C039_ASSERT tb_c039_cosim truncated setup");
          setup_program_id    = arg_w[0];
          setup_version       = arg_w[1];
          setup_package_hash  = {arg_w[5], arg_w[4], arg_w[3], arg_w[2]};
          setup_program_epoch = arg_w[6];
          setup_slot          = arg_w[7];
          setup_text_base     = arg_w[8];
          setup_text_len      = arg_w[9];
          setup_text_digest   = {arg_w[13], arg_w[12], arg_w[11], arg_w[10]};
          setup_write();
        end
        64'h08: begin
          begin_command();
          reply(ST_OK, 0);
        end
        64'h09: reset_pulse();
        64'h0a: begin
          n = $fscanf(req_fd, "%h", a);
          $fatal(1, "C039_ASSERT tb_c039_cosim peer_reject code=%0d tick=%0d", a, ticks_total);
        end
        64'h0b: begin
          $fclose(rsp_fd);
          $fclose(req_fd);
          $display("C039_PASS tb_c039_cosim contexts=%0d checks=%0d ticks=%0d", CONTEXT_COUNT, checks,
                   ticks_total);
          $finish;
        end
        default: $fatal(1, "C039_ASSERT tb_c039_cosim unknown command %h", op);
      endcase
    end
  end
endmodule
