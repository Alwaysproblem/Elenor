// tb_scalar_helper_issue.sv — C038 directed behavior bench.
//
// One scalar_helper_issue adapter drives the REAL W4/W6 owners end to end:
// mfe_top (with an SV memory peer), event_table (with its wait_record) and
// terminal_store. No model, no fabricated owner.
//
// What the bench proves:
//   * generated-constant decode: CSRRW/CSRRWI to each of the four helper
//     CSRs reaches only its owner; non-helper CSRs and non-Zicsr encodings
//     take RSP_PASS untouched;
//   * operand extraction: CSRRW uses the rs1 register value, CSRRWI uses
//     the uimm5 field; the adapter selects launch_descriptors[operand]
//     immutably for the MFE (the real mfe_desc_t fields are checked);
//   * accepted/stalled control per owner: an MFE submit stalls while the
//     engine's single record is busy (fields stable under the stall), a
//     phase instruction stalls while the slot owes an unfired byte, a
//     task_ret stalls while drain evidence is absent — and in every stall
//     the instruction stays presented, never re-formed;
//   * WAIT routing: an unresolvable event is an owner reject -> fault; a
//     parked wait arms the real W6 record and wakes on the real
//     completion; a direct-DONE verdict retires the instruction;
//   * invalid forms fault with ZERO owner side effects (descriptor
//     accepts, phase messages, DRAIN commands and lifecycle state are
//     counted across the whole run);
//   * task_ret drives the real C033 chain as far as the accepted
//     CMD_DRAIN: the bench finishes DRAIN->RR->RETIRE->FINISH on the real
//     store, proving an accepted task_ret actually enters the chain, and
//     a task_ret outside a live task is a consumed reject -> fault.
//
// Contexts: the bench runs at the C1/C2 sealed counts; geometry is one
// live task owner on Context slot 0. All geometry here is a verification
// fixture, not silicon sizing.
`timescale 1ns/1ps
module tb_scalar_helper_issue
  import c029_params_pkg::*;
  import c032_mfe_pkg::*;
  import c033_event_pkg::*;
  import c034_reset_pkg::*;
  import c038_scalar_pkg::*;
#(
  parameter int unsigned CONTEXT_COUNT = 1,  // harness injects -GCONTEXT_COUNT=N
  parameter string BENCH_NAME = "tb_scalar_helper_issue"
);
  localparam int unsigned DESC_COUNT = 2, EVENTS = 2, OPERANDS = 4;
  localparam int unsigned SOURCES = 1, FRONTIER = 4, COUNTER_BITS = 4;
  localparam int unsigned INFLIGHT = 2, BINDINGS = 2, LOCAL_BYTES = 512;

  logic clk = 0, rst_n = 0;
  always #5 clk = ~clk;

  // ---- launch environment (expanded sealed tables) ----
  // Desc 0: l2_to_frame — read binding BIDA@96, 32B into local offset 0.
  // Desc 1: frame_to_l2 — write local offset 0, 32B to binding BIDB@208.
  localparam logic [31:0] BIDA = 32'd17, BIDB = 32'd18;
  localparam logic [31:0] ARENA = 32'd99, SCOPE = 32'd77;
  scalar_desc_t launch_desc [DESC_COUNT];
  event_key_t wait_events [EVENTS];

  localparam word_t CTX_SLOT = 64'd0, CTX_GEN = 64'd9;
  localparam word_t PROD = 64'h1_0000_0007;
  owner_t owner;
  event_key_t K0, K1;

  initial begin
    K0 = '{ctx: '{tile_id: 64'd0, ctx_slot: CTX_SLOT, ctx_generation: CTX_GEN},
           local_event_id: 64'h1000, event_generation: 64'd7, seq: 64'd1,
           scope: EV_SCOPE_TASK_PRIVATE};
    K1 = '{ctx: '{tile_id: 64'd0, ctx_slot: CTX_SLOT, ctx_generation: CTX_GEN},
           local_event_id: 64'h1001, event_generation: 64'd7, seq: 64'd2,
           scope: EV_SCOPE_TASK_PRIVATE};
    owner = '{ctx: '{tile_id: 64'd0, ctx_slot: CTX_SLOT, ctx_generation: CTX_GEN},
              task_key: '{launch_generation: 64'd5, root_launch_generation: 64'd5,
                         dispatch_ordinal: 64'd1, task_id: 64'hBEEF,
                         context_handle: 64'hCAFE}};
    launch_desc[0] = '{dir_in: 1'b1, binding_id: BIDA, binding_generation: 32'd1,
                      binding_offset: 32'd96, local_off: 32'd0, bytes: 32'd32};
    launch_desc[1] = '{dir_in: 1'b0, binding_id: BIDB, binding_generation: 32'd1,
                      binding_offset: 32'd208, local_off: 32'd0, bytes: 32'd32};
    wait_events[0] = K0;
    wait_events[1] = K1;
  end

  // ---- instruction front ----
  logic        instr_valid;
  logic        instr_ready;
  logic [31:0] instr, rs1_op;
  logic [2:0]  rsp;
  word_t       op_seq;
  logic        fault;
  word_t       fault_code, fault_op_seq;
  logic [3:0]  fault_abi;
  logic [11:0] fault_csr;
  logic [31:0] mfe_txn_seq;

  // ---- owner-facing wires ----
  logic          mfe_desc_valid, mfe_desc_ready;
  mfe_desc_t     mfe_desc;
  logic          wait_valid, wait_all, wait_ready, wait_reject;
  context_key_t  wait_ctx;
  word_t         wait_count;
  event_key_t [OPERANDS-1:0] wait_keys;
  logic          direct_ok, direct_fail, park;
  logic          ph_valid, ph_ready;
  owner_t        ph_owner;
  logic          ph_dir;
  word_t         ph_seq;
  logic          ph_valid_e, ph_ready_i_b;
  phase_msg_t    ph_msg;
  owner_t        ret_owner;
  task_status_t  ret_fstatus;
  word_t         ret_evseq, ret_ffref;
  cmd_kind_t     ret_kind;
  logic          ret_valid, ret_ready, ret_reject;
  logic          ret_exec, ret_noold, ret_region, ret_reclaimed, ret_w4closed;


  // ---- terminal-store cmd mux: adapter DRAIN bundle vs bench lifecycle ----
  logic          b_cmd_valid;
  cmd_kind_t     b_cmd_kind;
  owner_t        b_cmd_owner;
  logic          b_exec, b_noold, b_region, b_reclaimed, b_w4closed;
  task_status_t  b_fstatus;
  word_t         b_evseq, b_ffref;

  logic          cmd_valid, cmd_ready, cmd_reject;
  cmd_kind_t     cmd_kind;
  owner_t        cmd_owner;
  logic          cmd_exec, cmd_noold, cmd_region, cmd_reclaimed, cmd_w4closed;
  task_status_t  cmd_fstatus;
  word_t         cmd_evseq, cmd_ffref;

  always_comb begin
    cmd_valid    = ret_valid | b_cmd_valid;
    cmd_kind     = ret_valid ? ret_kind     : b_cmd_kind;
    cmd_owner    = ret_valid ? ret_owner    : b_cmd_owner;
    cmd_exec     = ret_valid ? ret_exec     : b_exec;
    cmd_noold    = ret_valid ? ret_noold    : b_noold;
    cmd_region   = ret_valid ? ret_region   : b_region;
    cmd_reclaimed= ret_valid ? ret_reclaimed: b_reclaimed;
    cmd_w4closed = ret_valid ? ret_w4closed : b_w4closed;
    cmd_fstatus  = ret_valid ? ret_fstatus  : b_fstatus;
    cmd_evseq    = ret_valid ? ret_evseq    : b_evseq;
    cmd_ffref    = ret_valid ? ret_ffref    : b_ffref;
  end

  // ---- memory-service stub (real handshakes, deterministic bytes) ----
  logic          mem_valid, mem_ready, mr_valid, mr_ready, ack_valid, ack_ready;
  mfe_mem_req_t  mem;
  mfe_mem_rsp_t  mr;
  mfe_term_ack_t t_ack;
  logic          term_valid, term_ready;
  mfe_terminal_t term;
  logic          term_hold;
  logic [31:0]   lpeek_addr;
  logic [7:0]    lpeek_data;
  logic [$clog2(INFLIGHT+1)-1:0] outstanding;
  logic [63:0]   sent_c, rsps_c, acks_c, absorbed_c, stale_c;
  logic [63:0]   trk_terminals, trk_stale, trk_refused;
  logic [3:0]    dut_err;
  logic          dut_inv, poisoned;
  logic          bind_valid, bind_ready;
  mfe_binding_t  bindin;

  // ---- event table ports (C034 segment idle) ----
  logic          alloc_valid, alloc_ready, alloc_reject;
  event_key_t    alloc_key, query_key;
  logic [1:0]    alloc_kind;
  word_t         alloc_future, alloc_producer;
  logic [SOURCES-1:0] comp_valid, comp_ready;
  completion_t [SOURCES-1:0] comp_i;
  logic [CONTEXT_COUNT-1:0] armed, wake_ok, wake_fail;
  event_view_t   query_view;
  counts_t [CONTEXT_COUNT-1:0] counts;
  counter_flags_t [CONTEXT_COUNT-1:0] counts_seen;
  word_t         mark_count, live_events;
  logic          abandon_ready;
  logic          fault_valid;
  context_key_t  fault_ctx;

  // ---- terminal-store ports ----
  logic          abort_valid;
  owner_t        abort_owner;
  task_status_t  abort_status;
  word_t         abort_ref;
  logic          msg_valid, msg_ready;
  terminal_msg_t msg;
  logic [CONTEXT_COUNT-1:0] active, ready_pend, retire_sent, comp_pend;
  owner_t [CONTEXT_COUNT-1:0] live_owner;
  logic [CONTEXT_COUNT-1:0] aborted, phase_pend;

  // ============================================================ DUT chain
  scalar_helper_issue #(
    .DESC_COUNT(DESC_COUNT), .EVENTS(EVENTS), .OPERANDS(OPERANDS)
  ) adapter (
    .clk_i(clk), .rst_ni(rst_n),
    .launch_desc_i(launch_desc), .wait_events_i(wait_events),
    .instr_valid_i(instr_valid), .instr_ready_o(instr_ready),
    .instr_i(instr), .rs1_operand_i(rs1_op), .owner_i(owner),
    .op_seq_i(op_seq), .rsp_o(rsp),
    .arena_generation_i(ARENA), .scope_epoch_i(SCOPE),
    .mfe_desc_valid_o(mfe_desc_valid), .mfe_desc_ready_i(mfe_desc_ready),
    .mfe_desc_o(mfe_desc),
    .wait_valid_o(wait_valid), .wait_all_o(wait_all), .wait_ctx_o(wait_ctx),
    .wait_count_o(wait_count), .wait_keys_o(wait_keys),
    .wait_ready_i(wait_ready), .wait_reject_i(wait_reject),
    .wait_direct_ok_i(direct_ok), .wait_direct_fail_i(direct_fail),
    .wait_park_i(park),
    .phase_valid_o(ph_valid), .phase_owner_o(ph_owner),
    .phase_direction_o(ph_dir), .phase_seq_o(ph_seq),
    .phase_seq_i(64'h5EED_0001), .phase_ready_i(ph_ready),
    .ret_cmd_valid_o(ret_valid), .ret_cmd_kind_o(ret_kind),
    .ret_cmd_owner_o(ret_owner), .ret_exec_drained_o(ret_exec),
    .ret_no_old_inflight_o(ret_noold), .ret_region_ended_o(ret_region),
    .ret_local_reclaimed_o(ret_reclaimed), .ret_w4_closed_o(ret_w4closed),
    .ret_finish_status_o(ret_fstatus), .ret_drain_evidence_seq_o(ret_evseq),
    .ret_first_fault_ref_o(ret_ffref),
    .ret_cmd_ready_i(ret_ready), .ret_cmd_reject_i(ret_reject),
    .exec_drained_i(b_exec), .no_old_inflight_i(b_noold),
    .region_ended_i(b_region), .local_reclaimed_i(b_reclaimed),
    .w4_closed_i(b_w4closed), .drain_evidence_seq_i(b_evseq),
    .first_fault_ref_i(b_ffref),
    .fault_o(fault), .fault_code_o(fault_code), .fault_abi_o(fault_abi),
    .fault_csr_o(fault_csr), .fault_op_seq_o(fault_op_seq),
    .mfe_txn_seq_o(mfe_txn_seq)
  );

  assign ret_ready  = cmd_ready;
  assign ret_reject = cmd_reject & ret_valid;

  mfe_top #(
    .INFLIGHT(INFLIGHT), .LOCAL_BYTES(LOCAL_BYTES), .BINDINGS(BINDINGS)
  ) mfe (
    .clk, .rst_n,
    .desc_valid_i(mfe_desc_valid), .desc_ready_o(mfe_desc_ready),
    .desc_i(mfe_desc),
    .bind_valid_i(bind_valid), .bind_ready_o(bind_ready), .bind_i(bindin),
    .cancel_valid_i(1'b0), .cancel_txn_i(32'd0),
    .isolation_failed_i(1'b0), .quiescent_reset_i(1'b0), .poisoned_o(poisoned),
    .term_valid_o(term_valid), .term_ready_i(term_ready), .term_o(term),
    .mem_req_valid_o(mem_valid), .mem_req_ready_i(mem_ready), .mem_req_o(mem),
    .mem_rsp_valid_i(mr_valid), .mem_rsp_ready_o(mr_ready), .mem_rsp_i(mr),
    .term_ack_valid_o(ack_valid), .term_ack_ready_i(ack_ready),
    .term_ack_o(t_ack),
    .lpeek_addr_i(lpeek_addr), .lpeek_data_o(lpeek_data),
    .outstanding_o(outstanding),
    .sent_o(sent_c), .rsps_o(rsps_c), .acks_o(acks_c),
    .absorbed_o(absorbed_c), .stale_rsp_o(stale_c),
    .tracker_terminals_o(trk_terminals), .tracker_stale_cancel_o(trk_stale),
    .tracker_refused_o(trk_refused),
    .err_code_o(dut_err), .invariant_error_o(dut_inv)
  );

  event_table #(
    .CONTEXT_COUNT(CONTEXT_COUNT), .FRONTIER(FRONTIER), .OPERANDS(OPERANDS),
    .SOURCES(SOURCES), .COUNTER_BITS(COUNTER_BITS), .TILE_ID(0)
  ) events (
    .clk_i(clk), .rst_ni(rst_n),
    .alloc_valid_i(alloc_valid), .alloc_key_i(alloc_key),
    .alloc_kind_i(alloc_kind), .alloc_producer_i(alloc_producer),
    .alloc_future_i(alloc_future), .alloc_ready_o(alloc_ready),
    .alloc_reject_o(alloc_reject),
    .comp_valid_i(comp_valid), .comp_i(comp_i), .comp_ready_o(comp_ready),
    .wait_valid_i(wait_valid), .wait_all_i(wait_all), .wait_ctx_i(wait_ctx),
    .wait_count_i(wait_count), .wait_keys_i(wait_keys),
    .wait_ready_o(wait_ready), .wait_reject_o(wait_reject),
    .direct_ok_o(direct_ok), .direct_fail_o(direct_fail), .park_o(park),
    .armed_o(armed), .wake_ok_o(wake_ok), .wake_fail_o(wake_fail),
    .fault_valid_i(fault_valid), .fault_ctx_i(fault_ctx),
    .query_key_i(query_key), .query_o(query_view),
    .counts_o(counts), .counts_seen_o(counts_seen),
    .mark_events_i(1'b0), .mark_count_o(mark_count),
    .abandon_events_i(1'b0), .abandon_ready_o(abandon_ready),
    .live_events_o(live_events)
  );

  terminal_store #(
    .CONTEXT_COUNT(CONTEXT_COUNT), .TILE_ID(0)
  ) terms (
    .clk_i(clk), .rst_ni(rst_n),
    .issue_frozen_i(1'b0), .reset_failure_i(1'b0), .reset_ref_i('0),
    .cmd_valid_i(cmd_valid), .cmd_kind_i(cmd_kind), .cmd_owner_i(cmd_owner),
    .exec_drained_i(cmd_exec), .no_old_inflight_i(cmd_noold),
    .region_ended_i(cmd_region), .local_reclaimed_i(cmd_reclaimed),
    .w4_closed_i(cmd_w4closed), .finish_status_i(cmd_fstatus),
    .drain_evidence_seq_i(cmd_evseq), .first_fault_ref_i(cmd_ffref),
    .cmd_ready_o(cmd_ready), .cmd_reject_o(cmd_reject),
    .abort_valid_i(abort_valid), .abort_owner_i(abort_owner),
    .abort_status_i(abort_status), .abort_ref_i(abort_ref),
    .phase_valid_i(ph_valid), .phase_owner_i(ph_owner),
    .phase_direction_i(ph_dir), .phase_seq_i(ph_seq), .phase_ready_o(ph_ready),
    .phase_valid_o(ph_valid_e), .phase_o(ph_msg), .phase_ready_i(ph_ready_i_b),
    .msg_valid_o(msg_valid), .msg_o(msg), .msg_ready_i(msg_ready),
    .active_o(active), .ready_pending_o(ready_pend), .retire_sent_o(retire_sent),
    .completion_pending_o(comp_pend), .live_owner_o(live_owner),
    .aborted_o(aborted), .phase_pending_o(phase_pend)
  );

  // ================================================================ check
  int checks = 0, cycles = 0;
  task automatic require(input logic ok, input string why);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C038_ASSERT %s %s cycle=%0d", BENCH_NAME, why, cycles);
  endtask

  // ---- global no-side-effect ledgers and offer-stability monitors ----
  int desc_accepts, phase_fires, drain_accepts, rr_fires, comp_fires;
  int fault_pulses, wakes_ok, wakes_fail, term_count;
  logic [3:0]  last_fabi;
  word_t       last_fcode;
  logic [11:0] last_fcsr;
  logic [1:0]  term_final [64];
  logic [3:0]  term_err [64];
  mfe_desc_t   held_desc;
  logic        desc_held;
  logic        ph_held; logic ph_dir_held; owner_t ph_owner_held;
  word_t       ph_seq_held;
  logic        ret_held;
  terminal_msg_t msg_held_v; logic msg_held;
  phase_msg_t    ph_held_v;  logic ph_e_held;
  int            ph_dir_n;
  logic          ph_dir_seen [8];

  always @(posedge clk) begin
    cycles++;
    if (rst_n) begin
      require(!dut_inv, "mfe invariants");
      require(trk_refused == 64'd0 && trk_stale == 64'd0, "tracker clean");
      // Offered payload stability under any stall.
      if (desc_held) require(mfe_desc_valid && mfe_desc == held_desc,
                             "descriptor payload stable under MFE stall");
      if (ph_held) require(ph_valid && ph_dir == ph_dir_held &&
                           ph_owner == ph_owner_held && ph_seq == ph_seq_held,
                           "phase ingress payload stable under stall");
      if (ret_held) require(ret_valid && ret_owner == owner && ret_kind == CMD_DRAIN,
                            "retire-entry payload stable under stall");
      if (ph_e_held) require(ph_valid_e && ph_msg == ph_held_v,
                             "PHASE_SIGNAL egress stable under stall");
      if (msg_held) require(msg_valid && msg == msg_held_v,
                            "terminal egress stable under stall");
      desc_held = mfe_desc_valid && !mfe_desc_ready; held_desc = mfe_desc;
      ph_held = ph_valid && !ph_ready;
      ph_dir_held = ph_dir; ph_owner_held = ph_owner; ph_seq_held = ph_seq;
      ret_held = ret_valid && !ret_ready;
      ph_e_held = ph_valid_e && !ph_ready_i_b; ph_held_v = ph_msg;
      msg_held = msg_valid && !msg_ready; msg_held_v = msg;
      if (mfe_desc_valid && mfe_desc_ready) desc_accepts++;
      if (ph_valid_e && ph_ready_i_b) begin
        require(ph_msg.task_key == owner.task_key &&
                ph_msg.message_seq == 64'h5EED_0001, "phase egress identity");
        ph_dir_seen[ph_dir_n] = ph_msg.direction; ph_dir_n++;
        phase_fires++;
      end
      if (cmd_valid && cmd_ready && !cmd_reject && cmd_kind == CMD_DRAIN) drain_accepts++;
      if (msg_valid && msg_ready && !msg.is_completion) rr_fires++;
      if (msg_valid && msg_ready && msg.is_completion) comp_fires++;
      if (wake_ok[0]) wakes_ok++;
      if (wake_fail[0]) wakes_fail++;
      if (term_valid && term_ready && term.txn_id < 32'd64) begin
        term_final[term.txn_id] = term.final_state;
        term_err[term.txn_id] = term.error;
        term_count++;
      end
      if (fault) begin
        fault_pulses++;
        last_fabi = fault_abi; last_fcode = fault_code; last_fcsr = fault_csr;
        // Fault class decides what may reach an owner: domain/form faults
        // present NOTHING; owner-reject faults present exactly the request
        // the owner refused (it is counted, never a side effect).
        if (fault_abi == FAB_ACCESS_FORM || fault_abi == FAB_OPERAND_DOMAIN)
          require(!mfe_desc_valid && !wait_valid && !ph_valid && !ret_valid,
                  "form/domain fault drove an owner request");
        else if (fault_abi == FAB_WAIT_REJECTED || fault_abi == FAB_WAIT_FAILED)
          require(wait_valid && !mfe_desc_valid && !ph_valid && !ret_valid,
                  "wait fault presented more than the refused wait");
        else if (fault_abi == FAB_RETIRE_REJECTED)
          require(ret_valid && !mfe_desc_valid && !wait_valid && !ph_valid,
                  "retire fault presented more than the refused drain");
      end
    end
  end

  // ---- memory stub: one response FIFO, patterned reads, recorded writes ----
  logic        rsp_pend;
  int          rsp_wait;
  mfe_mem_rsp_t rsp_next;
  logic [31:0] wr_off [8], wr_len [8];
  logic [LINE_BYTES*8-1:0] wr_payload [8];
  int wr_count;
  logic [31:0] rd_off [8], rd_len [8];
  int rd_count;

  function automatic logic [LINE_BYTES*8-1:0] rd_pattern(input logic [31:0] off);
    logic [LINE_BYTES*8-1:0] r;
    for (int j = 0; j < LINE_BYTES; j++) r[j*8 +: 8] = 8'((off + 32'(j)) ^ 32'h5A);
    return r;
  endfunction

  always @(posedge clk) begin
    if (!rst_n) begin
      mr_valid <= 1'b0; mr <= '0; rsp_pend <= 1'b0; rsp_wait <= 0;
      rsp_next <= '0; wr_count <= 0; rd_count <= 0;
    end else begin
      if (mem_valid && mem_ready) begin
        require(mem.slice_id == 32'd0 && mem.slice_count == 32'd1, "single slice");
        require(mem.ctx_slot == 32'd0 && mem.ctx_generation == 32'd9 &&
                mem.arena_generation == ARENA && mem.scope_epoch == SCOPE,
                "external request keeps adapter provenance");
        if (mem.write) begin
          wr_off[wr_count] <= mem.binding_offset; wr_len[wr_count] <= mem.length;
          wr_payload[wr_count] <= mem.payload; wr_count <= wr_count + 1;
        end else begin
          rd_off[rd_count] <= mem.binding_offset; rd_len[rd_count] <= mem.length;
          rd_count <= rd_count + 1;
        end
        rsp_pend <= 1'b1; rsp_wait <= 2;
        rsp_next <= '{txn_id: mem.txn_id, slice_id: 32'd0,
                      binding_generation: mem.binding_generation, status: 32'd0,
                      data: rd_pattern(mem.binding_offset)};
      end
      if (mr_valid && mr_ready) mr_valid <= 1'b0;
      else if (rsp_pend && !mr_valid) begin
        if (rsp_wait > 0) rsp_wait <= rsp_wait - 1;
        else begin mr <= rsp_next; mr_valid <= 1'b1; rsp_pend <= 1'b0; end
      end
    end
  end
  assign mem_ready  = rst_n && !rsp_pend && !mr_valid;
  assign ack_ready  = 1'b1;
  assign term_ready = rst_n && !term_hold;

  // =============================================================== drive
  function automatic logic [31:0] enc(input logic [2:0] f3, input logic [4:0] r,
                                      input logic [4:0] s, input logic [11:0] c);
    return {c, s, f3, r, 7'b1110011};
  endfunction

  // Present one instruction until the adapter consumes it; return rsp_o.
  task automatic issue_instr(input logic [31:0] i, input logic [31:0] rs1v,
                             input int bound, output logic [2:0] got);
    int lim;
    lim = 0;
    @(negedge clk);
    instr = i; rs1_op = rs1v; instr_valid = 1;
    #1;
    while (!instr_ready && lim < bound) begin @(negedge clk); #1; lim++; end
    require(instr_ready, "instruction consumed within bound");
    got = rsp;
    @(posedge clk);
    @(negedge clk); instr_valid = 0; #1;
  endtask

  task automatic issue_expect(input logic [31:0] i, input logic [31:0] rs1v,
                              input logic [2:0] want, input string why);
    logic [2:0] got;
    issue_instr(i, rs1v, 20000, got);
    require(got == want, why);
  endtask

  // Consume a fault: rsp FAULT, exactly one fault pulse, right ABI class.
  task automatic issue_fault(input logic [31:0] i, input logic [31:0] rs1v,
                             input logic [3:0] want_abi, input word_t want_code,
                             input logic [11:0] want_csr, input string why);
    logic [2:0] got;
    int before_f;
    before_f = fault_pulses;
    issue_instr(i, rs1v, 20000, got);
    require(got == RSP_FAULT, why);
    require(fault_pulses == before_f + 1 && last_fabi == want_abi &&
            last_fcode == want_code && last_fcsr == want_csr, why);
  endtask

  task automatic wait_cycles(input int n);
    repeat (n) @(negedge clk);
  endtask

  task automatic install_binding(input int id, input int size,
                                 input logic r, input logic w);
    @(negedge clk);
    bindin = '{id: 32'(id), generation: 32'd1, size: 32'(size),
               readable: r, writable: w};
    bind_valid = 1; #1;
    require(bind_ready, "binding install accepted");
    @(posedge clk);
    @(negedge clk); bind_valid = 0; #1;
  endtask

  task automatic alloc_event(input event_key_t key);
    @(negedge clk);
    alloc_key = key; alloc_kind = EV_KIND_TRANSFER_DONE;
    alloc_producer = PROD; alloc_future = 64'd4; alloc_valid = 1; #1;
    require(alloc_ready && !alloc_reject, "event registered");
    @(posedge clk);
    @(negedge clk); alloc_valid = 0; #1;
  endtask

  task automatic complete(input event_key_t key, input event_status_t st);
    completion_t c;
    @(negedge clk);
    c = '{key: key, status: st, producer_id: PROD, error_code: 64'd0};
    comp_i[0] = c; comp_valid[0] = 1; #1;
    require(comp_ready[0], "completion accepted");
    @(posedge clk);
    @(negedge clk); comp_valid[0] = 0; #1;
  endtask

  task automatic bench_cmd(input cmd_kind_t k);
    @(negedge clk);
    b_cmd_kind = k; b_cmd_owner = owner; b_cmd_valid = 1; #1;
    begin : acc
      int bound;
      bound = 0;
      while (!cmd_ready && bound < 20000) begin @(negedge clk); #1; bound++; end
      require(cmd_ready && !cmd_reject, "bench lifecycle command consumed");
    end
    @(posedge clk);
    @(negedge clk); b_cmd_valid = 0; #1;
  endtask

  task automatic peek(input int off, output logic [7:0] value);
    lpeek_addr = 32'(off); #1; value = lpeek_data;
  endtask

  task automatic await_mfe_idle;
    int bound;
    bound = 0;
    while ((outstanding != 0 || sent_c != acks_c || mr_valid || rsp_pend) &&
           bound < 40000) @(negedge clk);
    require(outstanding == 0 && sent_c == acks_c && !mr_valid && !rsp_pend,
            "MFE drained every real obligation");
  endtask

  // =============================================================== main
  initial begin : run
    logic [7:0] v;
    instr_valid = 0; instr = '0; rs1_op = '0; op_seq = 64'hA1;
    alloc_valid = 0; alloc_key = '0; alloc_kind = '0; alloc_producer = '0;
    alloc_future = '0; comp_valid = '0; comp_i[0] = '0;
    fault_valid = 0; fault_ctx = '0; query_key = '0;
    b_cmd_valid = 0; b_cmd_kind = CMD_START; b_cmd_owner = '0;
    b_exec = 0; b_noold = 0; b_region = 0; b_reclaimed = 0; b_w4closed = 0;
    b_fstatus = TS_DONE; b_evseq = '0; b_ffref = '0;
    abort_valid = 0; abort_owner = '0; abort_status = TS_DONE; abort_ref = '0;
    ph_ready_i_b = 1; msg_ready = 1; term_hold = 0; bind_valid = 0; bindin = '0;
    lpeek_addr = '0;
    ph_dir_n = 0;
    repeat (4) @(negedge clk); rst_n = 1; #1;

    // ---------- task admission on the real store ----------
    b_exec = 1; b_noold = 1; b_region = 1;
    bench_cmd(CMD_START);
    require(active[0], "task admitted on Context 0");

    // ---------- invalid carrier / operand forms: fault, no side effect --
    issue_fault(enc(3'b010, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_MFE_SUBMIT)), 32'd0,
                FAB_ACCESS_FORM, FC_UNSUPPORTED_ABI, 12'(CN_CSR_NEXUS_MFE_SUBMIT),
                "CSRRS to a helper CSR faults");
    issue_fault(enc(3'b011, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_EVENT_WAIT)), 32'd0,
                FAB_ACCESS_FORM, FC_UNSUPPORTED_ABI, 12'(CN_CSR_NEXUS_EVENT_WAIT),
                "CSRRC to a helper CSR faults");
    issue_fault(enc(3'b001, 5'd3, 5'd0, 12'(CN_CSR_NEXUS_EVENT_WAIT)), 32'd0,
                FAB_ACCESS_FORM, FC_UNSUPPORTED_ABI, 12'(CN_CSR_NEXUS_EVENT_WAIT),
                "CSRRW with rd != x0 faults");
    issue_fault(enc(3'b101, 5'd1, 5'd0, 12'(CN_CSR_NEXUS_PHASE_SIGNAL)), 32'd0,
                FAB_ACCESS_FORM, FC_UNSUPPORTED_ABI, 12'(CN_CSR_NEXUS_PHASE_SIGNAL),
                "CSRRWI with rd != x0 faults");
    issue_fault(enc(3'b001, 5'd0, 5'd6, 12'(CN_CSR_NEXUS_MFE_SUBMIT)), 32'd77,
                FAB_OPERAND_DOMAIN, FC_INVALID_DESCRIPTOR, 12'(CN_CSR_NEXUS_MFE_SUBMIT),
                "MFE_SUBMIT operand >= DESC_COUNT faults");
    issue_fault(enc(3'b101, 5'd0, 5'd2, 12'(CN_CSR_NEXUS_EVENT_WAIT)), 32'd0,
                FAB_OPERAND_DOMAIN, FC_INVALID_DESCRIPTOR, 12'(CN_CSR_NEXUS_EVENT_WAIT),
                "WAIT operand >= EVENTS faults");
    issue_fault(enc(3'b101, 5'd0, 5'd3, 12'(CN_CSR_NEXUS_PHASE_SIGNAL)), 32'd0,
                FAB_OPERAND_DOMAIN, FC_INVALID_DESCRIPTOR, 12'(CN_CSR_NEXUS_PHASE_SIGNAL),
                "SIGNAL_PHASE operand > 1 faults");
    issue_fault(enc(3'b001, 5'd0, 5'd2, 12'(CN_CSR_NEXUS_TASK_RET)), 32'd1,
                FAB_OPERAND_DOMAIN, FC_INVALID_DESCRIPTOR, 12'(CN_CSR_NEXUS_TASK_RET),
                "TASK_RET operand != 0 faults");
    issue_expect(enc(3'b001, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_MREGION_START)), 32'd5,
                 RSP_PASS, "non-scalar helper CSR passes to the front-end");
    issue_expect(enc(3'b010, 5'd0, 5'd1, 12'h001), 32'd0, RSP_PASS,
                 "non-helper CSRRS passes untouched");
    issue_expect(32'h00208133, 32'd0, RSP_PASS, "non-Zicsr encoding passes");
    require(fault_pulses == 8 && desc_accepts == 0 && phase_fires == 0 &&
            drain_accepts == 0 && counts[0].protocol_error == word_t'(0),
            "invalid forms caused no owner side effect");

    // ---------- WAIT on an unresolvable event: owner reject = fault ----
    install_binding(BIDA, 4096, 1, 0);
    install_binding(BIDB, 4096, 1, 1);
    issue_fault(enc(3'b101, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_EVENT_WAIT)), 32'd0,
                FAB_WAIT_REJECTED, FC_STREAM_PROTOCOL_ERROR, 12'(CN_CSR_NEXUS_EVENT_WAIT),
                "WAIT on an unregistered event faults via owner reject");
    require(counts[0].protocol_error == word_t'(1), "reject counted once");

    // ---------- register the slot-local events ----------
    alloc_event(K0);
    alloc_event(K1);

    // ---------- MFE submits (CSRRW + CSRRWI) + stall proof on desc 1 ----
    // INFLIGHT=2: park both records in S_TERM behind a held terminal, then
    // desc 1's offer has no free record -> real backpressure stall.
    term_hold = 1;
    issue_expect(enc(3'b001, 5'd0, 5'd5, 12'(CN_CSR_NEXUS_MFE_SUBMIT)), 32'd0,
                 RSP_OK, "MFE_SUBMIT desc 0 accepted (CSRRW)");
    issue_expect(enc(3'b101, 5'd0, 5'd0, 12'(CN_CSR_NEXUS_MFE_SUBMIT)), 32'd0,
                 RSP_OK, "MFE_SUBMIT desc 0 accepted again (CSRRWI)");
    require(desc_accepts == 2 && mfe_txn_seq == 32'd2,
            "both records occupied behind the held terminal");
    // desc 1 stalls while the engine record is busy; fields stay stable.
    begin : stall_check
      logic [2:0] g;
      @(negedge clk);
      instr = enc(3'b101, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_MFE_SUBMIT));
      rs1_op = '0; instr_valid = 1; #1;
      require(!instr_ready && mfe_desc_valid, "desc 1 stalls behind busy records");
      require(mfe_desc.binding_id == BIDB && mfe_desc.bytes == 32'd32 &&
              mfe_desc.src_kind == KIND_LOCAL && mfe_desc.dst_kind == KIND_EXT &&
              mfe_desc.binding_offset == 32'd208 && mfe_desc.local_offset == 32'd0 &&
              mfe_desc.txn_id == {CTX_SLOT[7:0], 24'd2},
              "stalled desc 1 already carries the right record");
      repeat (3) begin
        @(negedge clk); #1;
        require(mfe_desc_valid && mfe_desc.binding_id == BIDB,
                "stalled offer stays stable");
      end
      term_hold = 0;   // terminals drain, records settle, desc 1 gets a slot
      #1;  // read instr_ready only after the comb settle
      begin : acc
        int bound;
        bound = 0;
        while (!instr_ready && bound < 30000) begin @(negedge clk); #1; bound++; end
        require(instr_ready, "desc 1 accepted once a record frees");
        g = rsp;
      end
      require(g == RSP_OK, "desc 1 retires OK");
      @(posedge clk);
      @(negedge clk); instr_valid = 0; #1;
    end
    await_mfe_idle();
    require(desc_accepts == 3, "exactly three descriptors accepted");
    require(rd_count == 2 && rd_off[0] == 32'd96 && rd_len[0] == 32'd32 &&
            rd_off[1] == 32'd96 && rd_len[1] == 32'd32,
            "two inbound reads went to binding offset 96 for 32B");
    require(wr_count == 1 && wr_off[0] == 32'd208 && wr_len[0] == 32'd32,
            "outbound write went to binding offset 208 for 32B");
    // Inbound pattern landed in the local window and the outbound write
    // carried exactly those delivered bytes.
    for (int j = 0; j < 32; j++) begin
      peek(j, v);
      require(v == 8'((32'(96 + j)) ^ 32'h5A), "local window holds real inbound bytes");
      require(wr_payload[0][j*8 +: 8] == 8'((32'(96 + j)) ^ 32'h5A),
              "outbound payload is the delivered window bytes");
    end
    require(term_count == 3 && term_final[0] == MFE_COMPLETED &&
            term_err[0] == ERR_NONE && term_final[1] == MFE_COMPLETED &&
            term_err[1] == ERR_NONE && term_final[2] == MFE_COMPLETED &&
            term_err[2] == ERR_NONE, "all three descriptors settled COMPLETED");

    // ---------- WAIT: direct-fail, park, real wake, direct-DONE ---------
    complete(K0, EV_ERROR);
    wait_cycles(2);
    issue_fault(enc(3'b101, 5'd0, 5'd0, 12'(CN_CSR_NEXUS_EVENT_WAIT)), 32'd0,
                FAB_WAIT_FAILED, FC_EVENT_DEPENDENCY_TIMEOUT, 12'(CN_CSR_NEXUS_EVENT_WAIT),
                "WAIT on a FAILED event faults via direct_fail");
    issue_expect(enc(3'b101, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_EVENT_WAIT)), 32'd0,
                 RSP_PARK, "WAIT desc-1 event parks the next issue");
    require(armed[0], "the real W6 wait record armed");
    complete(K1, EV_DONE);
    wait_cycles(2);
    require(!armed[0], "wake released the record");
    issue_expect(enc(3'b101, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_EVENT_WAIT)), 32'd0,
                 RSP_OK, "WAIT on a DONE event retires directly");
    require(wakes_ok == 1 && wakes_fail == 0, "exactly one real wake");

    // ---------- PHASE_SIGNAL: accepted, stalled, egress-fired ----------
    ph_ready_i_b = 0;   // hold egress: an owed byte stalls the next phase
    issue_expect(enc(3'b001, 5'd0, 5'd0, 12'(CN_CSR_NEXUS_PHASE_SIGNAL)), 32'd0,
                 RSP_OK, "phase input_released accepted");
    begin : ph_stall
      logic [2:0] g;
      @(negedge clk);
      instr = enc(3'b101, 5'd0, 5'd1, 12'(CN_CSR_NEXUS_PHASE_SIGNAL));
      instr_valid = 1; #1;
      require(!instr_ready && ph_valid && ph_dir == PHASE_OUTPUT_READY,
              "second phase stalls behind the owed byte");
      repeat (2) begin
        @(negedge clk); #1;
        require(ph_valid && ph_dir == PHASE_OUTPUT_READY, "phase offer stable");
      end
      ph_ready_i_b = 1;
      #1;  // read instr_ready only after the comb settle
      begin : acc
        int bound;
        bound = 0;
        while (!instr_ready && bound < 20000) begin @(negedge clk); #1; bound++; end
        require(instr_ready, "phase accepted after the owed byte fired");
        g = rsp;
      end
      require(g == RSP_OK, "phase output_ready retires OK");
      @(posedge clk);
      @(negedge clk); instr_valid = 0; #1;
    end
    wait_cycles(3);
    require(phase_fires == 2 && ph_dir_seen[0] == PHASE_INPUT_RELEASED &&
            ph_dir_seen[1] == PHASE_OUTPUT_READY,
            "exactly two PHASE_SIGNAL messages in order");

    // ---------- task_ret: stall on absent evidence, then the chain ------
    b_exec = 0; b_noold = 0; b_region = 0; b_evseq = 64'hAA;
    begin : ret_stall
      logic [2:0] g;
      @(negedge clk);
      instr = enc(3'b101, 5'd0, 5'd0, 12'(CN_CSR_NEXUS_TASK_RET));
      instr_valid = 1; #1;
      require(!instr_ready && ret_valid && ret_kind == CMD_DRAIN &&
              ret_owner == owner, "task_ret stalls on absent drain evidence");
      repeat (3) begin
        @(negedge clk); #1;
        require(ret_valid && ret_kind == CMD_DRAIN, "drain intent stable");
      end
      b_exec = 1; b_noold = 1; b_region = 1;
      #1;  // let comb settle: the consume edge is the NEXT posedge
      begin : acc
        int bound;
        bound = 0;
        while (!instr_ready && bound < 20000) begin @(negedge clk); #1; bound++; end
        require(instr_ready, "task_ret consumed on accepted DRAIN");
        g = rsp;
      end
      require(g == RSP_RET, $sformatf(
              "task_ret retires at the DRAIN edge (rsp=%0d cmd_ready=%b cmd_reject=%b ret_reject=%b)",
              g, ret_ready, cmd_reject, ret_reject));
      @(posedge clk);
      @(negedge clk); instr_valid = 0; #1;
    end
    require(drain_accepts == 1, "exactly one DRAIN accepted");
    // The real chain: RR offer -> bench RETIRE -> FINISH -> COMPLETION DONE.
    msg_ready = 0;
    wait_cycles(2);
    require(msg_valid && !msg.is_completion, "RETIRE_READY offered");
    require(msg.drain_evidence_seq == 64'hAA, "drain evidence rode the RR");
    msg_ready = 1;
    wait_cycles(1);
    require(rr_fires == 1, "RETIRE_READY fired once");
    bench_cmd(CMD_RETIRE);
    b_reclaimed = 1; b_w4closed = 1; b_ffref = 64'd0;
    bench_cmd(CMD_FINISH);
    wait_cycles(1);
    require(msg_valid && msg.is_completion && msg.status == TS_DONE,
            "TASK_COMPLETION DONE offered");
    wait_cycles(2);
    require(comp_fires == 1 && !active[0], "completion fired and slot freed");

    // ---------- task_ret outside a live task: consumed reject = fault ---
    issue_fault(enc(3'b101, 5'd0, 5'd0, 12'(CN_CSR_NEXUS_TASK_RET)), 32'd0,
                FAB_RETIRE_REJECTED, FC_STREAM_PROTOCOL_ERROR, 12'(CN_CSR_NEXUS_TASK_RET),
                "task_ret with no live task is a protocol fault");

    // ---------- close-out ----------
    require(fault_pulses == 11 && counts[0].protocol_error == word_t'(1),
            "fault and owner-reject ledgers exact");
    require(stale_c == 64'd0 && absorbed_c == 64'd0, "no wire anomalies");
    $display("C038_PASS %s contexts=%0d checks=%0d desc=%0d phases=%0d rr=%0d comp=%0d",
             BENCH_NAME, CONTEXT_COUNT, checks, desc_accepts, phase_fires,
             rr_fires, comp_fires);
    $finish;
  end

  initial begin : watchdog
    #5000000;
    $fatal(1, "C038_ASSERT %s watchdog", BENCH_NAME);
  end
endmodule
