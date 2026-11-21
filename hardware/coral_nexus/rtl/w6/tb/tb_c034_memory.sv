// C034 unit peer: control an actual C032 MFE descriptor and its C023 service
// response independently. A service write takes effect when its MEM_REQ fires;
// while the response or terminal ACK is held, the W6 drain still owes work.
`timescale 1ns/1ps
module tb_c034_memory (
  input logic clk_i, rst_ni, issue_frozen_i, abort_valid_i,
  input c033_event_pkg::owner_t abort_owner_i,
  input integer request_fd, response_fd,
  output logic quiescent_o, poison_o
);
  import c032_mfe_pkg::*;
  logic dv, dr, bv, br, local_cancel, isolation_failed, recover;
  logic req_valid, req_ready, rsp_valid, rsp_ready, ack_valid, ack_ready, term_valid, term_ready;
  mfe_desc_t desc;
  mfe_binding_t binding;
  mfe_mem_req_t req, accepted;
  mfe_mem_rsp_t rsp;
  mfe_term_ack_t ack, fired_ack;
  mfe_terminal_t term_bus;
  logic [31:0] peek_addr;
  logic [7:0] peek_byte;
  logic [1:0] outstanding;
  logic [63:0] sent, responses, acked, service_data;
  logic invariant_error;
  wire owner_cancel = abort_valid_i && outstanding != 0 && abort_owner_i.task_key.task_id == 64'(desc.txn_id);
  assign quiescent_o = outstanding == 0 && !term_valid && sent == acked;
  // Forward only the record sampled on an actual handshake, never a preview.
  always @(posedge clk_i) if (rst_ni) begin
    if (req_valid && req_ready) accepted=req;
    if (ack_valid && ack_ready) fired_ack=ack;
  end
  mfe_top #(.INFLIGHT(2),.LOCAL_BYTES(64)) mfe (
    .clk(clk_i),.rst_n(rst_ni),.desc_valid_i(dv && !issue_frozen_i),.desc_ready_o(dr),.desc_i(desc),
    .bind_valid_i(bv),.bind_ready_o(br),.bind_i(binding),
    .cancel_valid_i(local_cancel || owner_cancel),.cancel_txn_i(desc.txn_id),
    .isolation_failed_i(isolation_failed),.quiescent_reset_i(recover),.poisoned_o(poison_o),
    .term_valid_o(term_valid),.term_ready_i(term_ready),.term_o(term_bus),
    .mem_req_valid_o(req_valid),.mem_req_ready_i(req_ready),.mem_req_o(req),
    .mem_rsp_valid_i(rsp_valid),.mem_rsp_ready_o(rsp_ready),.mem_rsp_i(rsp),
    .term_ack_valid_o(ack_valid),.term_ack_ready_i(ack_ready),.term_ack_o(ack),
    .lpeek_addr_i(peek_addr),.lpeek_data_o(peek_byte),.outstanding_o(outstanding),
    .sent_o(sent),.rsps_o(responses),.acks_o(acked),.absorbed_o(),.stale_rsp_o(),
    .tracker_terminals_o(),.tracker_stale_cancel_o(),.tracker_refused_o(),.err_code_o(),
    .invariant_error_o(invariant_error)
  );
  task automatic step(); @(negedge clk_i); #1; endtask
  task automatic check(input logic ok, input string detail);
    if (ok !== 1'b1) $fatal(1,"C034_ASSERT memory %s",detail);
  endtask
  task automatic prepare();
    dv=0; bv=0; desc='0; binding='0; local_cancel=0; isolation_failed=0; recover=0;
    req_ready=0; rsp_valid=0; rsp='0; ack_ready=0; term_ready=0; peek_addr=0;
  endtask
  task automatic peer(input string cmd, output logic[63:0] a,b);
    logic [63:0] status;
    int n;
    $fdisplay(request_fd,"%s",cmd); $fflush(request_fd);
    n=$fscanf(response_fd,"%h %h %h",status,a,b);
    if (n!=3 || status!=0) $fatal(1,"C034_PEER memory command %s failed: status=%h",cmd,status);
  endtask
  task automatic open_service();
    logic[63:0] a,b;
    peer("M",a,b);
    for (int i=0; i<2; i++) begin
      binding='{id:32'(17+i),generation:1,size:4096,readable:1,writable:i==1}; bv=1; #1;
      check(br,"install real binding"); step(); bv=0;
    end
  endtask
  task automatic request(input int unsigned id, input bit write_op,
                         input int unsigned ext_offset, local_offset, cgen);
    logic [63:0] a,b;
    string cmd;
    desc='{txn_id:id,src_kind:write_op?KIND_LOCAL:KIND_EXT,dst_kind:write_op?KIND_EXT:KIND_LOCAL,
           binding_id:write_op?18:17,binding_generation:1,binding_offset:ext_offset,local_offset:local_offset,
           bytes:8,ctx_slot:0,ctx_generation:cgen,arena_generation:99,scope_epoch:77};
    dv=1; #1; check(dr && !issue_frozen_i,"MFE accepted descriptor before cancel"); step(); dv=0;
    for(int n=0;n<16 && !req_valid;n++) step();
    check(req_valid,"watchdog: MFE did not reach MEM_REQ");
    check(req.length==8 && req.write==write_op && req.binding_offset==ext_offset &&
          req.ctx_generation==cgen,"real request provenance, direction and extent");
    req_ready=1; step(); req_ready=0;
    cmd=$sformatf("%s %h %h %h %h %h %h %h %h %h %h %h",write_op?"W":"R",accepted.txn_id,
      accepted.slice_id,accepted.slice_count,accepted.binding_id,accepted.binding_generation,
      accepted.binding_offset,accepted.length,accepted.ctx_slot,accepted.ctx_generation,
      accepted.arena_generation,accepted.scope_epoch);
    if (write_op) cmd={cmd,$sformatf(" %h",accepted.payload[63:0])};
    peer(cmd,a,b); service_data=a;
    if (write_op) check(a==2,"real HBM write is persisted, not just acknowledged locally");
    check(!quiescent_o && sent==acked+1,"accepted request stays outstanding without its response");
  endtask
  task automatic cancel(); local_cancel=1; step(); local_cancel=0; endtask
  task automatic refuse_isolation();
    isolation_failed=1; step(); isolation_failed=0;
    check(poison_o && !quiescent_o && !dr,"isolation refusal retains transfer and rejects new MFE work");
    recover=1; step(); recover=0;
    check(poison_o && !quiescent_o,"premature MFE quiescent reset cannot erase write obligations");
  endtask
  task automatic clear_poison();
    check(quiescent_o,"physical closure precedes explicit recovery"); recover=1; step(); recover=0;
    check(!poison_o,"explicit MFE recovery after closure");
  endtask
  task automatic respond(input logic[31:0] status, input logic[1:0] outcome);
    logic[63:0] a,b;
    rsp='0; rsp.txn_id=accepted.txn_id; rsp.slice_id=accepted.slice_id;
    rsp.binding_generation=accepted.binding_generation; rsp.status=status; rsp.data[63:0]=service_data;
    rsp_valid=1; #1; check(rsp_ready,"MFE ready for its actual response"); step(); rsp_valid=0;
    for(int n=0;n<10 && !ack_valid;n++) step();
    check(ack_valid,"watchdog: MFE wire ACK");
    check(ack.txn_id==accepted.txn_id && ack.final_state==outcome && !quiescent_o,
          "W4 wire terminal class and identity while owed");
    ack_ready=1; step(); ack_ready=0;
    check(sent==acked && fired_ack.txn_id==accepted.txn_id && fired_ack.final_state==outcome,
          "service reference releases only on the observed wire ACK fire");
    peer($sformatf("T %h %h %h %h",fired_ack.txn_id,fired_ack.slice_id,
                   fired_ack.slice_count,fired_ack.final_state),a,b);
    for(int n=0;n<16 && !term_valid;n++) step();
    check(term_valid,"watchdog: local MFE terminal");
    check(term_bus.txn_id==desc.txn_id && term_bus.final_state==outcome && !quiescent_o,
          "local MFE terminal survives cancellation until consumed");
    term_ready=1; step(); term_ready=0;
    for(int n=0;n<4 && !quiescent_o;n++) step();
    check(quiescent_o && sent==responses && sent==acked && !invariant_error,"MFE obligation ledger closes");
  endtask
  task automatic peek_local(input int unsigned off,input logic[63:0] expected);
    for(int i=0;i<8;i++) begin peek_addr=off+32'(i); #1;
      check(peek_byte==expected[i*8+:8],"only delivered reads may change Tile-local bytes"); end
  endtask
  task automatic check_persisted(input logic[63:0] expected);
    logic [63:0] value, unused;
    peer("B 40",value,unused); check(value==expected,"external bytes survive cancel/drain");
  endtask
  task automatic close_service(input int unsigned expected);
    logic [63:0] reqs,acks;
    peer("E",reqs,acks); check(reqs==64'(expected) && acks==reqs && reqs==sent,
                                "C023 service and C032 agree on every released wire request");
  endtask
endmodule
