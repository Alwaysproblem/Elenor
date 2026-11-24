// scalar_helper_issue.sv — C038 synthesizable scalar helper issue adapter.
//
// Issue-side decode for the four scalar control helpers of the sealed G1 v5
// ABI (protocol/scalar_helper_abi_v1.json, cn_scalar_helper_abi/v1). It
// accepts a Context's presented Zicsr instruction, extracts the operand from
// the only two legal carrier forms (CSRRW x0,csr,rs1 / CSRRWI x0,csr,imm),
// and qualifies the control effect against the real owner interfaces:
//
//   nexus.mfe.submit   (W4): selects launch_descriptors[operand] — an
//     immutable, environment-expanded record — and presents it as one
//     mfe_desc_t on the MFE descriptor ingress. A stall is real
//     backpressure: the instruction is held (instr_ready_o=0), never
//     re-formed. One accepted descriptor is exactly one transaction; the
//     local transaction id is {ctx_slot[7:0], per-adapter sequence} so two
//     adapters can never collide and a cancel request can name it.
//   nexus.event.wait   (W6): presents the slot-local wait_events[operand]
//     key on the event_table WAIT issue head. The owner verdict is
//     consumed the same cycle: direct DONE -> instruction OK; park ->
//     instruction retires and the Context's NEXT issue is held by C040
//     (RSP_PARK; the park record lives in W6, EVT-04); a failed operand
//     event (direct_fail) or an owner reject is a protocol fault.
//   nexus.phase.signal (W6): one accepted instruction emits exactly one
//     PHASE_SIGNAL (duplicates included, EVT-14) — a stall (phase byte
//     owed / slot not LS_RUN / cancelled / frozen) is backpressure, never
//     a dropped or doubled message.
//   nexus.task.ret     (W2/W6): the retire-chain ENTRY. The instruction
//     retires exactly when its CMD_DRAIN intent is consumed by the
//     terminal store (backpressured while drain evidence is absent,
//     rejected -> protocol fault). The DRAIN->RR->RETIRE->FINISH chain
//     itself remains C033/C035: this unit presents no success semantics
//     beyond the accepted DRAIN.
//
// Carrier faults (any access outside the two legal forms — CSRRS/CSRRC*
// variants, rd != x0, or any other encoding) and operand faults
// (index >= table size, phase enum > 1, task_ret code != 0) are consumed
// with exactly one fault pulse and produce ZERO owner side effects.
// Instructions outside the four helper CSR addresses take RSP_PASS for
// the front-end's normal path. There is no "wait stall": a WAIT is
// accepted-with-verdict or rejected in the presentation cycle (the event
// table supplies no WAIT backpressure, only reject). rsp_o is meaningful
// only in cycles where instr_ready_o is 1. rst_ni is the synchronous
// active-low cold reset; it never substitutes for a protocol state.
`timescale 1ns/1ps
module scalar_helper_issue
  import c033_event_pkg::*;
  import c034_reset_pkg::*;
  import c032_mfe_pkg::*;
  import c038_scalar_pkg::*;
#(
  parameter int unsigned DESC_COUNT = 2,  // launch_descriptors entries; K-SCALAR-ADD7 fixes 2
  parameter int unsigned EVENTS = 2,      // wait_events entries (ABI: parallel to launch_descriptors)
  parameter int unsigned OPERANDS = 4     // event_table issue-head operand array (C040 wiring)
) (
  input  logic           clk_i,
  input  logic           rst_ni,

  // Sealed launch tables, expanded by the launch environment (immutable for
  // the Task's lifetime; the kernel names only an index into them).
  input  scalar_desc_t   launch_desc_i [DESC_COUNT],
  input  event_key_t     wait_events_i [EVENTS],

  // Instruction issue front (Context-private Zicsr stream).
  input  logic           instr_valid_i,
  output logic           instr_ready_o,
  input  logic [31:0]    instr_i,
  input  logic [31:0]    rs1_operand_i,   // rs1 read data (CSRRW operand)
  input  owner_t         owner_i,         // full task/context identity at dispatch
  input  word_t          op_seq_i,        // owner_op_seq for fault attribution
  output logic [2:0]     rsp_o,

  // Frame/descriptor-owner provenance riding every external request
  // (c032_mfe_pkg mfe_desc_t fields; supplied at submit time, never by the
  // kernel — scalar_helper_abi_v1.json descriptor_record.no_full_addresses).
  input  logic [31:0]    arena_generation_i,
  input  logic [31:0]    scope_epoch_i,

  // W4 MFE descriptor ingress (one transaction per accepted descriptor).
  output logic           mfe_desc_valid_o,
  input  logic           mfe_desc_ready_i,
  output mfe_desc_t      mfe_desc_o,

  // W6 event table WAIT issue head (single global issue point).
  output logic           wait_valid_o,
  output logic           wait_all_o,
  output context_key_t   wait_ctx_o,
  output word_t          wait_count_o,
  output event_key_t [OPERANDS-1:0] wait_keys_o,   // packed: event_table issue-head shape
  input  logic           wait_ready_i,
  input  logic           wait_reject_i,
  input  logic           wait_direct_ok_i,
  input  logic           wait_direct_fail_i,
  input  logic           wait_park_i,

  // W6 terminal-store PHASE_SIGNAL ingress (one accepted = one message).
  output logic           phase_valid_o,
  output owner_t         phase_owner_o,
  output logic           phase_direction_o,
  output word_t          phase_seq_o,
  input  word_t          phase_seq_i,      // link-side sequencer value, carried only
  input  logic           phase_ready_i,

  // Retire-chain entry: a complete terminal-store cmd bundle (drop-in wired
  // by C040; kind is always CMD_DRAIN). Evidence fields are pass-throughs
  // of the drain evidence owner, not adapter state.
  output logic           ret_cmd_valid_o,
  output cmd_kind_t      ret_cmd_kind_o,
  output owner_t         ret_cmd_owner_o,
  output logic           ret_exec_drained_o,
  output logic           ret_no_old_inflight_o,
  output logic           ret_region_ended_o,
  output logic           ret_local_reclaimed_o,
  output logic           ret_w4_closed_o,
  output task_status_t   ret_finish_status_o,
  output word_t          ret_drain_evidence_seq_o,
  output word_t          ret_first_fault_ref_o,
  input  logic           ret_cmd_ready_i,
  input  logic           ret_cmd_reject_i,

  // Drain-evidence pass-throughs (wired to the terminal_store cmd bundle).
  input  logic           exec_drained_i,
  input  logic           no_old_inflight_i,
  input  logic           region_ended_i,
  input  logic           local_reclaimed_i,
  input  logic           w4_closed_i,
  input  word_t          drain_evidence_seq_i,
  input  word_t          first_fault_ref_i,

  // Protocol-fault observation: exactly one pulse per consumed fault, with
  // the C017 wire fault_code plus the ABI fault class for observability.
  output logic           fault_o,
  output word_t          fault_code_o,
  output logic [3:0]     fault_abi_o,
  output logic [11:0]    fault_csr_o,
  output word_t          fault_op_seq_o,

  // Local MFE transaction sequence observation (the txn_id[23:0] the NEXT
  // accepted submit will carry; 0 until the first submit).
  output logic [31:0]    mfe_txn_seq_o
);

  // ---- instruction fields ----
  logic        is_zicsr, is_csrw, is_csrwi, form_ok;
  logic [11:0] csr;
  logic [2:0]  funct3;
  logic [4:0]  rd, rs1f;
  logic [31:0] operand;
  logic        hit_mfe, hit_wait, hit_phase, hit_ret, helper_hit;

  assign is_zicsr = (instr_i[6:0] == 7'b1110011);
  assign funct3   = instr_i[14:12];
  assign csr      = instr_i[31:20];
  assign rd       = instr_i[11:7];
  assign rs1f     = instr_i[19:15];
  assign is_csrw  = is_zicsr && (funct3 == 3'b001);
  assign is_csrwi = is_zicsr && (funct3 == 3'b101);
  assign form_ok  = (is_csrw || is_csrwi) && (rd == 5'd0);
  // CSRRWI carries the uimm5 literal; CSRRW carries the rs1 register value.
  assign operand  = is_csrwi ? {27'd0, rs1f} : rs1_operand_i;

  assign hit_mfe   = (csr == 12'(CN_CSR_NEXUS_MFE_SUBMIT));
  assign hit_wait  = (csr == 12'(CN_CSR_NEXUS_EVENT_WAIT));
  assign hit_phase = (csr == 12'(CN_CSR_NEXUS_PHASE_SIGNAL));
  assign hit_ret   = (csr == 12'(CN_CSR_NEXUS_TASK_RET));
  assign helper_hit = hit_mfe || hit_wait || hit_phase || hit_ret;

  logic [31:0] desc_index;
  assign desc_index = operand;

  // ---- local MFE transaction sequence ----
  // {ctx_slot[7:0], seq[23:0]}: unique across adapters via the Context slot,
  // bounded by MFE_INFLIGHT live records and 2^24-1 issues per slot.
  logic [23:0] txn_seq_q;
  wire   mfe_fire = mfe_desc_valid_o && mfe_desc_ready_i;
  assign mfe_txn_seq_o = {8'd0, txn_seq_q};

  always_ff @(posedge clk_i) begin
    if (!rst_ni) txn_seq_q <= 24'd0;
    else if (mfe_fire) txn_seq_q <= txn_seq_q + 24'd1;
  end

  // ---- descriptor select (immutable launch table by index) ----
  scalar_desc_t sel_desc;
  logic         desc_ok;
  always_comb begin
    sel_desc = launch_desc_i[0];
    for (int i = 0; i < DESC_COUNT; i++)
      if (desc_index == 32'(i)) sel_desc = launch_desc_i[i];
    desc_ok = (desc_index < 32'(DESC_COUNT));
  end

  assign mfe_desc_o = '{
    txn_id:             {owner_i.ctx.ctx_slot[7:0], txn_seq_q},
    src_kind:           sel_desc.dir_in ? KIND_EXT : KIND_LOCAL,
    dst_kind:           sel_desc.dir_in ? KIND_LOCAL : KIND_EXT,
    binding_id:         sel_desc.binding_id,
    binding_generation: sel_desc.binding_generation,
    binding_offset:     sel_desc.binding_offset,
    local_offset:       sel_desc.local_off,
    bytes:              sel_desc.bytes,
    ctx_slot:           owner_i.ctx.ctx_slot[31:0],
    ctx_generation:     owner_i.ctx.ctx_generation[31:0],
    arena_generation:   arena_generation_i,
    scope_epoch:        scope_epoch_i
  };

  // ---- WAIT key select (one operand; wait_all stays 0) ----
  always_comb begin
    for (int i = 0; i < OPERANDS; i++) wait_keys_o[i] = '0;
    for (int i = 0; i < EVENTS; i++)
      if (desc_index == 32'(i)) wait_keys_o[0] = wait_events_i[i];
  end

  // ---- retire-chain entry bundle (drop-in CMD_DRAIN; finish fields are
  // inert on a DRAIN — see terminal_store.sv cmd decode) ----
  assign ret_cmd_kind_o           = CMD_DRAIN;
  assign ret_cmd_owner_o          = owner_i;
  assign ret_exec_drained_o       = exec_drained_i;
  assign ret_no_old_inflight_o    = no_old_inflight_i;
  assign ret_region_ended_o       = region_ended_i;
  assign ret_local_reclaimed_o    = local_reclaimed_i;
  assign ret_w4_closed_o          = w4_closed_i;
  assign ret_finish_status_o      = TS_DONE;          // inert on DRAIN
  assign ret_drain_evidence_seq_o = drain_evidence_seq_i;
  assign ret_first_fault_ref_o    = first_fault_ref_i;

  // ---- accept / stall / fault qualification ----
  logic        consumed, is_fault;
  logic [2:0]  rsp_c;
  logic [3:0]  fault_abi_c;
  word_t       fault_code_c;

  always_comb begin
    mfe_desc_valid_o   = 1'b0;
    wait_valid_o       = 1'b0;
    wait_all_o         = 1'b0;
    wait_ctx_o         = owner_i.ctx;
    wait_count_o       = word_t'(1);
    phase_valid_o      = 1'b0;
    phase_owner_o      = owner_i;
    phase_direction_o  = PHASE_INPUT_RELEASED;
    phase_seq_o        = phase_seq_i;
    ret_cmd_valid_o    = 1'b0;

    consumed    = 1'b0;
    is_fault    = 1'b0;
    rsp_c       = RSP_PASS;
    fault_abi_c = FAB_NONE;
    fault_code_c = FC_INVALID_DESCRIPTOR;

    if (instr_valid_i) begin
      if (!is_zicsr || !helper_hit) begin
        consumed = 1'b1;                       // not ours: PASS through
      end else if (!form_ok) begin
        consumed = 1'b1;                       // HELPER_ACCESS_FORM
        is_fault = 1'b1;
        rsp_c = RSP_FAULT;
        fault_abi_c = FAB_ACCESS_FORM;
        fault_code_c = FC_UNSUPPORTED_ABI;
      end else if (hit_mfe) begin
        if (!desc_ok) begin
          consumed = 1'b1;                     // OPERAND_OUT_OF_DOMAIN
          is_fault = 1'b1;
          rsp_c = RSP_FAULT;
          fault_abi_c = FAB_OPERAND_DOMAIN;
          fault_code_c = FC_INVALID_DESCRIPTOR;
        end else begin
          mfe_desc_valid_o = 1'b1;
          consumed = mfe_desc_ready_i;         // real MFE backpressure stalls
          rsp_c = RSP_OK;
        end
      end else if (hit_wait) begin
        if (desc_index >= 32'(EVENTS)) begin
          consumed = 1'b1;                     // OPERAND_OUT_OF_DOMAIN
          is_fault = 1'b1;
          rsp_c = RSP_FAULT;
          fault_abi_c = FAB_OPERAND_DOMAIN;
          fault_code_c = FC_INVALID_DESCRIPTOR;
        end else begin
          wait_valid_o = 1'b1;
          consumed = wait_ready_i | wait_reject_i;
          if (wait_reject_i) begin             // owner reject = protocol fault
            is_fault = 1'b1;
            rsp_c = RSP_FAULT;
            fault_abi_c = FAB_WAIT_REJECTED;
            fault_code_c = FC_STREAM_PROTOCOL_ERROR;
          end else if (wait_direct_fail_i) begin
            is_fault = 1'b1;                   // waited event already failed
            rsp_c = RSP_FAULT;
            fault_abi_c = FAB_WAIT_FAILED;
            fault_code_c = FC_EVENT_DEPENDENCY_TIMEOUT;
          end else if (wait_park_i) begin
            rsp_c = RSP_PARK;                  // record armed: park NEXT issue
          end else begin
            rsp_c = RSP_OK;                    // direct DONE
          end
        end
      end else if (hit_phase) begin
        if (operand > 32'd1) begin
          consumed = 1'b1;                     // OPERAND_OUT_OF_DOMAIN
          is_fault = 1'b1;
          rsp_c = RSP_FAULT;
          fault_abi_c = FAB_OPERAND_DOMAIN;
          fault_code_c = FC_INVALID_DESCRIPTOR;
        end else begin
          phase_valid_o = 1'b1;
          phase_direction_o = operand[0] ? PHASE_OUTPUT_READY : PHASE_INPUT_RELEASED;
          consumed = phase_ready_i;            // owed byte / not LS_RUN stalls
          rsp_c = RSP_OK;
        end
      end else begin  // hit_ret
        if (operand != 32'd0) begin
          consumed = 1'b1;                     // OPERAND_OUT_OF_DOMAIN
          is_fault = 1'b1;
          rsp_c = RSP_FAULT;
          fault_abi_c = FAB_OPERAND_DOMAIN;
          fault_code_c = FC_INVALID_DESCRIPTOR;
        end else begin
          ret_cmd_valid_o = 1'b1;
          consumed = ret_cmd_ready_i;          // absent drain evidence stalls
          if (ret_cmd_reject_i) begin          // consume-with-reject = fault
            is_fault = 1'b1;
            rsp_c = RSP_FAULT;
            fault_abi_c = FAB_RETIRE_REJECTED;
            fault_code_c = FC_STREAM_PROTOCOL_ERROR;
          end else begin
            rsp_c = RSP_RET;
          end
        end
      end
    end
  end

  assign instr_ready_o = rst_ni && consumed;
  assign rsp_o         = rsp_c;

  assign fault_o        = rst_ni && consumed && is_fault;
  assign fault_code_o   = fault_code_c;
  assign fault_abi_o    = fault_abi_c;
  assign fault_csr_o    = csr;
  assign fault_op_seq_o = op_seq_i;

  // Internal never-happens guards (verifiable by the directed bench and by
  // the mutant family): a consumed FORM/DOMAIN-fault edge drives no owner
  // request (owner-reject faults DO present — the refused request is what
  // the owner saw), and a descriptor offer never exists without a live
  // instruction.
  always @(posedge clk_i) begin
    if (rst_ni && consumed && is_fault &&
        (fault_abi_c == FAB_ACCESS_FORM || fault_abi_c == FAB_OPERAND_DOMAIN) &&
        (mfe_desc_valid_o || wait_valid_o || phase_valid_o || ret_cmd_valid_o)) begin
      $error("C038_ASSERT scalar_helper_issue form/domain fault carried an owner request");
    end
    if (rst_ni && mfe_desc_valid_o && !instr_valid_i) begin
      $error("C038_ASSERT scalar_helper_issue descriptor offer without an instruction");
    end
  end

endmodule
