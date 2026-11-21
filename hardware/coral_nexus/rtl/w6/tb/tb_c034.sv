// C034: observed handshakes from the real Tile table, terminal store, MFE
// and first-fault record. U3-f/h/i take each phase from a live Python Group.
`timescale 1ns/1ps
module tb_c034 #(parameter int unsigned CONTEXT_COUNT=2);
  import c033_event_pkg::*;
  import c034_reset_pkg::*;
  localparam word_t HIGH=64'h1_0000_0000, PROD=HIGH+7;
  localparam int unsigned FS=4;
  logic clk_i=0, rst_ni=0;
  always #20 clk_i=~clk_i;
  word_t gen,tag,hold_cycles,wanted[6];
  int checks=0,cycles=0,rr_fires,task_fires,phase_fires,marks,clears,invalidates,rebuilds;
  word_t marked_sum,observed_src;
  string case_name,vector_file,peer_path;
  integer request_fd,response_fd;

  logic run_begin_i,run_ready_o,group_poisoned_i,issue_frozen_o,reset_active_o,reset_done_o;
  word_t run_generation_i,req_generation_i,ack_generation_o,ack_marked_o,reset_fault_ref_i;
  logic reset_failure_o;
  word_t reset_ref_o;
  reset_phase_t req_phase_i,ack_phase_o;
  logic req_valid_i,req_ready_o,req_reject_o,ack_valid_o,ack_ready_i;
  logic [5:0] ack_mask_o,drain_evidence_i,is_quiescent_snapshot_o,injected_evidence;
  logic clear_ingress_valid_o,clear_ingress_ready_i,invalidate_valid_o,invalidate_ready_i,refs_zero_i;
  logic mark_events_o,abandon_events_o,abandon_ready_i,rebuild_valid_o,rebuild_ready_i;
  word_t mark_count_i,live_events_o;
  logic [CONTEXT_COUNT-1:0] live_i,aborted_i,terminal_committed_i,phase_owed;
  owner_t [CONTEXT_COUNT-1:0] live_owner_i;
  logic cancel_valid_i,cancel_ready_o,cancel_fault_o,abort_valid_o;
  cancel_class_t cancel_class_o;
  owner_t cancel_owner_i,abort_owner_o;
  word_t cancel_ref_i,stale_cancels_o,abort_ref_o;
  task_status_t abort_status_o;
  reset_endpoint #(.CONTEXT_COUNT(CONTEXT_COUNT),.TILE_ID(0)) endpoint (.*);

  logic [FS-1:0] lanes,fault_valid;
  fault_t [FS-1:0] faults,latch_input;
  fault_t record_o;
  logic record_valid_o,report_valid_o,report_ready_i,ack_valid_i,priv_clear_i,clear_ready_o,suppressed_seen_o;
  word_t ack_tile_i,suppressed_o;
  always_comb begin
    fault_valid=lanes; latch_input=faults;
    if(cancel_fault_o) begin
      fault_valid[FS-1]=1; latch_input[FS-1].owner=cancel_owner_i;
    end
  end
  first_fault_latch #(.SOURCES(FS),.COUNTER_BITS(3),.TILE_ID(0)) first_fault (
    .fault_valid_i(fault_valid),.fault_i(latch_input),.reset_active_i(reset_active_o),
    .reset_done_i(reset_done_o),.*);

  // A burst may be wider than the counter itself; losing the carry wraps
  // eight suppressed faults to zero for this legal verification geometry.
  logic [7:0] burst_valid;
  word_t burst_count;
  fault_t [7:0] burst_faults;
  for(genvar lane=0;lane<8;lane++) assign burst_faults[lane]=faults[lane%FS];
  first_fault_latch #(.SOURCES(8),.COUNTER_BITS(1),.TILE_ID(0)) narrow_counter (
    .clk_i,.rst_ni,.reset_active_i(1'b1),.reset_done_i(1'b0),.fault_valid_i(burst_valid),.fault_i(burst_faults),
    .record_valid_o(),.record_o(),.report_valid_o(),.report_ready_i(1'b0),.ack_valid_i(1'b0),
    .ack_tile_i('0),.priv_clear_i(1'b0),.clear_ready_o(),.suppressed_o(burst_count),.suppressed_seen_o());

  logic alloc_valid_i,alloc_ready_o,alloc_reject_o,wait_valid_i,wait_ready_o,wait_reject_o;
  event_key_t alloc_key_i,query_key_i;
  word_t alloc_future_i;
  logic [1:0] alloc_kind_i;
  event_view_t query_o;
  completion_t [0:0] comp_i;
  logic [0:0] comp_valid_i,comp_ready_o;
  context_key_t wait_ctx_i;
  event_key_t [1:0] wait_keys_i;
  logic direct_ok_o,direct_fail_o,park_o;
  logic [CONTEXT_COUNT-1:0] armed_o,wake_ok_o,wake_fail_o;
  event_table #(.CONTEXT_COUNT(CONTEXT_COUNT),.FRONTIER(4),.OPERANDS(2),.SOURCES(1),.COUNTER_BITS(4)) events (
    .alloc_valid_i(alloc_valid_i && !issue_frozen_o),.alloc_producer_i(PROD),.wait_all_i(1'b0),
    .wait_valid_i(wait_valid_i && !issue_frozen_o),.wait_count_i(64'd1),.fault_valid_i(1'b0),.fault_ctx_i('0),
    .mark_events_i(mark_events_o),.mark_count_o(mark_count_i),.abandon_events_i(abandon_events_o),
    .abandon_ready_o(abandon_ready_i),.counts_o(),.counts_seen_o(),.*);

  logic cmd_valid_i,cmd_ready_o,cmd_reject_o,msg_ready_i,msg_valid_o,phase_valid_i,phase_ready_o,phase_valid_o;
  logic region_failure,w4_quiet,w4_poison,ingress_owed;
  logic phase_sink_ready;
  cmd_kind_t cmd_kind_i;
  owner_t cmd_owner_i,phase_owner_i;
  terminal_msg_t msg_o;
  task_status_t finish_status_i;
  word_t terminal_ref_i;
  wire region_abort=region_failure && (|wake_fail_o) && record_valid_o;
  terminal_store #(.CONTEXT_COUNT(CONTEXT_COUNT),.TILE_ID(0)) terminals (
    .issue_frozen_i(issue_frozen_o),.live_owner_o(live_owner_i),.active_o(live_i),.aborted_o(aborted_i),
    .reset_failure_i(reset_failure_o),.reset_ref_i(reset_ref_o),
    .phase_pending_o(phase_owed),.completion_pending_o(terminal_committed_i),.ready_pending_o(),.retire_sent_o(),
    .abort_valid_i(abort_valid_o || region_abort),.abort_owner_i(region_abort?record_o.owner:abort_owner_o),
    .abort_status_i(region_abort?TS_ERROR:abort_status_o),.abort_ref_i(region_abort?reset_fault_ref_i:abort_ref_o),
    .exec_drained_i(1'b1),.no_old_inflight_i(w4_quiet),.region_ended_i(1'b1),.local_reclaimed_i(1'b1),
    .w4_closed_i(w4_quiet),.drain_evidence_seq_i(gen),.first_fault_ref_i(terminal_ref_i),
    .phase_ready_i(phase_sink_ready),
    .phase_direction_i(1'b1),.phase_seq_i(gen),.phase_o(),.*);
  tb_c034_memory memory(.clk_i,.rst_ni,.issue_frozen_i(issue_frozen_o),.abort_valid_i(abort_valid_o),
                        .abort_owner_i(abort_owner_o),.request_fd,.response_fd,.quiescent_o(w4_quiet),.poison_o(w4_poison));
  // The other (future W2/W5/debug) producers are controlled inputs. W4
  // closure and the C033 owed phase are observations, not scenario booleans.
  always_comb begin
    drain_evidence_i=injected_evidence;
    drain_evidence_i[1]&=!ingress_owed;
    drain_evidence_i[3]&=w4_quiet;
    drain_evidence_i[5]&=!(|phase_owed);
  end

  task automatic check(input logic ok,input string detail);
    checks++; if(ok !== 1'b1) $fatal(1,"C034_ASSERT %s cycle=%0d %s",case_name,cycles,detail);
  endtask
  task automatic tick(); @(negedge clk_i); #1; endtask
  logic held_ack,held_report,marked_last_edge;
  logic [136:0] old_ack; // phase=3 + generation=64 + count=64 + mask=6
  fault_t old_report;
  always @(posedge clk_i) begin
    cycles++;
    if(!rst_ni) begin
      rr_fires=0; task_fires=0; phase_fires=0; marks=0; clears=0; invalidates=0; rebuilds=0;
      marked_sum=0; observed_src=0; held_ack=0; held_report=0; marked_last_edge=0;
    end else begin
      if(held_ack) check(ack_valid_o && {ack_phase_o,ack_generation_o,ack_marked_o,ack_mask_o}==old_ack,
                         "stable phase ACK including release edge");
      if(held_report) check(report_valid_o && record_o==old_report,"stable full first-fault report through fire");
      held_ack=ack_valid_o && !ack_ready_i; old_ack={ack_phase_o,ack_generation_o,ack_marked_o,ack_mask_o};
      held_report=report_valid_o && !report_ready_i; old_report=record_o;
      if(record_valid_o) observed_src=record_o.source_id;
      if(marked_last_edge) marked_sum+=mark_count_i; // committed writer result, not a request-side estimate
      if(mark_events_o) marks++;
      marked_last_edge=mark_events_o;
      if(clear_ingress_valid_o && clear_ingress_ready_i) clears++;
      if(invalidate_valid_o && invalidate_ready_i) invalidates++;
      if(rebuild_valid_o && rebuild_ready_i) rebuilds++;
      if(msg_valid_o && msg_ready_i) begin if(msg_o.is_completion) task_fires++; else rr_fires++; end
      if(phase_valid_o && phase_sink_ready) phase_fires++;
    end
  end
  // One queued, NOT accepted, operation is cleared by the real phase command.
  always_ff @(posedge clk_i) begin
    if(!rst_ni) ingress_owed<=0;
    else if(run_begin_i && run_ready_o) ingress_owed<=1;
    else if(clear_ingress_valid_o && clear_ingress_ready_i) ingress_owed<=0;
  end
  task automatic peer(input string text,output word_t a,b);
    word_t status;
    int n;
    $fdisplay(request_fd,"%s",text); $fflush(request_fd);
    n=$fscanf(response_fd,"%h %h %h",status,a,b);
    if(n!=3 || status!=0) $fatal(1,"C034_PEER %s status=%h",text,status);
  endtask
  function automatic owner_t owner(input word_t id,input int slot=0);
    owner_t o;
    o.ctx='{tile_id:0,ctx_slot:word_t'(slot),ctx_generation:gen};
    o.task_key='{launch_generation:gen,root_launch_generation:gen+HIGH,dispatch_ordinal:HIGH+7,
                 task_id:id,context_handle:HIGH+word_t'(slot)};
    return o;
  endfunction
  function automatic event_key_t key(input word_t id);
    owner_t o;
    event_key_t k;
    o=owner(tag); k='{ctx:o.ctx,local_event_id:id,event_generation:gen+2,seq:gen+3,scope:EV_SCOPE_TASK_PRIVATE};
    return k;
  endfunction
  function automatic fault_t diagnosis(input word_t src,engine,op,input int slot=0);
    fault_t f;
    f='0; f.owner=owner(tag,slot); f.op_seq=op; f.code=FC_ENGINE_INTERNAL_FAULT;
    f.source_id=src; f.severity=3; f.command_id=HIGH+11; f.event_id=HIGH+12;
    f.program_id=HIGH+13; f.desc_id=HIGH+14; f.group_id=5; f.queue_id=6; f.slot_id=7; f.patch_id=8;
    f.engine_id=engine; f.offending_addr=HIGH+16; f.aux0=gen; f.aux1=~gen; f.cycle=HIGH+22; return f;
  endfunction
  task automatic start_case();
    rst_ni=0; run_begin_i=0; run_generation_i=gen; group_poisoned_i=0;
    req_valid_i=0; req_phase_i=RP_FREEZE; req_generation_i=gen; ack_ready_i=0;
    clear_ingress_ready_i=1; invalidate_ready_i=1; refs_zero_i=1; rebuild_ready_i=1;
    injected_evidence='1; reset_fault_ref_i=HIGH+tag; cancel_valid_i=0; cancel_owner_i='0; cancel_ref_i=HIGH+23;
    lanes='0; faults='0; report_ready_i=0; ack_valid_i=0; ack_tile_i=0; priv_clear_i=0;
    burst_valid='0;
    alloc_valid_i=0; alloc_kind_i=0; alloc_key_i='0; alloc_future_i=0; comp_valid_i=0; comp_i='0;
    wait_valid_i=0; wait_ctx_i='0; wait_keys_i='0; query_key_i='0;
    cmd_valid_i=0; cmd_kind_i=CMD_START; cmd_owner_i='0; msg_ready_i=0; finish_status_i=TS_DONE;
    terminal_ref_i=0; phase_valid_i=0; phase_owner_i='0; region_failure=0; memory.prepare();
    phase_sink_ready=1;
    // No queued ingress work is necessary to load a fresh epoch. The owner
    // injects a queued request after it has admitted the run (below).
    tick(); rst_ni=1; run_begin_i=1; #1; check(run_ready_o,"Group RUN_BEGIN admitted"); tick(); run_begin_i=0;
    check(!reset_active_o && !reset_done_o && !issue_frozen_o,"cold epoch admits work");
  endtask

  task automatic alloc(input event_key_t k,input word_t uses,input logic[1:0] kind=EV_KIND_COMPUTE_DONE);
    alloc_valid_i=1; alloc_key_i=k; alloc_kind_i=kind; alloc_future_i=uses; #1;
    check(alloc_ready_o && !alloc_reject_o,"event admission"); tick(); alloc_valid_i=0;
  endtask
  task automatic complete(input event_key_t k,input event_status_t st);
    comp_valid_i=1; comp_i[0]='{key:k,status:st,producer_id:PROD,error_code:st==EV_ERROR?HIGH+1:0};
    #1; check(comp_ready_o[0],"accepted completion in real event table"); tick(); comp_valid_i=0;
  endtask
  task automatic park(input event_key_t k);
    wait_valid_i=1; wait_ctx_i=k.ctx; wait_keys_i[0]=k; #1;
    check(wait_ready_o && park_o && !wait_reject_o,"real wait registered"); tick(); wait_valid_i=0;
  endtask
  task automatic view(input event_key_t k,input event_status_t st,input word_t refs,future);
    query_key_i=k; #1; check(query_o.live && query_o.status==st && query_o.consumer_refs==refs &&
                            query_o.future_uses==future,"retained event state + both reference counts");
  endtask
  task automatic command(input cmd_kind_t c,input owner_t o);
    cmd_kind_i=c; cmd_owner_i=o; cmd_valid_i=1; #1;
    check(cmd_ready_o && !cmd_reject_o,$sformatf("eligible terminal-store command %0d",c)); tick(); cmd_valid_i=0;
  endtask
  task automatic refuse_start(input owner_t o);
    cmd_kind_i=CMD_START; cmd_owner_i=o; cmd_valid_i=1; #1;
    check(cmd_ready_o && cmd_reject_o,"START visibly rejected while frozen/poisoned"); tick(); cmd_valid_i=0;
  endtask
  task automatic terminal(input bit last,input owner_t o,input task_status_t st,input word_t refword);
    #1; for(int i=0;i<12 && !msg_valid_o;i++) tick();
    check(msg_valid_o,"watchdog: missing terminal offer");
    check(msg_o.owner==o && msg_o.is_completion==last,"stored terminal type and full owner");
    if(last) check(msg_o.status==st && msg_o.first_fault_ref==refword,"exact terminal class and fault reference");
    else check(msg_o.drain_evidence_seq==gen,"RETIRE_READY is an evidence report, not Task success");
    msg_ready_i=1; tick(); msg_ready_i=0;
  endtask
  task automatic retire(input owner_t o,input task_status_t st,input word_t refword);
    command(CMD_DRAIN,o); terminal(0,o,TS_DONE,0); command(CMD_RETIRE,o);
    command(CMD_FINISH,o); terminal(1,o,st,refword);
  endtask
  task automatic capture(input fault_t f);
    lanes=1; faults[0]=f; tick(); lanes=0;
    check(record_valid_o && record_o==f,"entire first fault is queryable before its error event");
  endtask
  task automatic deliver_report(input bit live_group);
    fault_t f;
    word_t response,b;
    #1; check(report_valid_o && record_valid_o,"pending reliable fault report"); f=record_o;
    report_ready_i=1; tick(); report_ready_i=0; check(!report_valid_o && record_valid_o,"delivery does not clear latch");
    if(live_group) begin
      peer($sformatf("F %h %h %h %h %h %h %h %h %h",gen,f.owner.ctx.tile_id,f.owner.ctx.ctx_slot,
           f.owner.ctx.ctx_generation,f.code,f.severity,f.source_id,f.engine_id,f.op_seq),response,b);
      reset_fault_ref_i=response; check(response!=64'hffffffffffffffff,"Group assigned a fault record index");
    end
  endtask
  task automatic request_phase(input reset_phase_t p,input word_t epoch,input bit reject=0);
    req_phase_i=p; req_generation_i=epoch; req_valid_i=1; #1;
    for(int i=0;i<10 && !req_ready_o;i++) tick();
    check(req_ready_o,"watchdog: reset request ingress"); check(req_reject_o==reject,"reject only stale/invalid phase");
    tick(); req_valid_i=0;
  endtask
  task automatic begin_phase(input reset_phase_t expected,input bit live_group=0);
    word_t cmd,epoch;
    if(live_group) begin
      peer("N",cmd,epoch); check(cmd==word_t'(expected) && epoch==gen,"actual Group phase and epoch");
      request_phase(reset_phase_t'(cmd),epoch);
    end else request_phase(expected,gen);
  endtask
  task automatic take_ack(input reset_phase_t p,input word_t count,input bit live_group=0);
    word_t a,b,saved_gen,saved_count;
    logic[5:0] mask;
    reset_phase_t got;
    #1; for(int i=0;i<24+int'(CONTEXT_COUNT) && !ack_valid_o;i++) tick();
    check(ack_valid_o,"watchdog: phase owner failed to acknowledge");
    got=ack_phase_o; saved_gen=ack_generation_o; saved_count=ack_marked_o; mask=ack_mask_o;
    check(got==p && saved_gen==gen && saved_count==count && mask==(p==RP_DRAIN?6'h3f:6'h00),
          "ACK reports actual phase, generation, MARK count and drain evidence");
    tick(); ack_ready_i=1; tick(); ack_ready_i=0;
    if(live_group) peer($sformatf("A %h %h %h %h",got,saved_gen,saved_count,mask),a,b);
  endtask
  task automatic phase_step(input reset_phase_t p,input word_t count=0,input bit live_group=0);
    begin_phase(p,live_group); take_ack(p,count,live_group);
  endtask
  task automatic finish_reset(input bit live_group=0);
    word_t a,b;
    phase_step(RP_INVALIDATE,0,live_group); phase_step(RP_REBUILD,0,live_group); phase_step(RP_RESUME,0,live_group);
    check(reset_done_o && !reset_active_o && clears==1 && invalidates==1 && rebuilds==1,
          "one actual owner action per phase, reset reached DONE");
    check(!(|armed_o) && live_events_o==0 && live_i=='0 && w4_quiet,"no live identity, pinned event, transfer or terminal");
    if(live_group) begin
      peer("N",a,b); check(a==255 && b==gen,"Group reached DONE only after accepted RESUME ACK");
    end
  endtask
  task automatic cancel_req(input owner_t o,input cancel_class_t expected);
    cancel_valid_i=1; cancel_owner_i=o; #1;
    check(cancel_ready_o && cancel_class_o==expected && cancel_fault_o==(expected==CC_CONFLICT),
          "classify cancel by complete live task identity, including generation");
    check(abort_valid_o==(expected==CC_APPLY),"only first exact live cancel applies an abort");
    if(expected==CC_APPLY) check(abort_status_o==TS_RESET && abort_owner_o==o,"scoped failure only");
    tick(); cancel_valid_i=0;
  endtask

  task automatic multi_fault();
    fault_t f[4];
    f[0]=diagnosis(tag+1,0,0); f[1]=diagnosis(tag,2,0);
    f[2]=diagnosis(tag,1,CONTEXT_COUNT>1?0:9,CONTEXT_COUNT>1?1:0); f[3]=diagnosis(tag,1,5);
    for(int i=0;i<4;i++) faults[i]=f[i]; lanes='1; tick(); lanes=0;
    check(record_valid_o && record_o==f[3] && suppressed_o==3,"lexicographic winner, count all losers");
    ack_valid_i=1; tick(); ack_valid_i=0; check(report_valid_o && record_valid_o,"unsent report survives early ACK");
    repeat(int'(hold_cycles)) tick(); faults[0]=diagnosis(tag-1,0,0); lanes=1; tick(); lanes=0;
    check(record_o==f[3] && suppressed_o==4,"later lower source cannot overwrite a captured first");
    deliver_report(0); ack_valid_i=1; ack_tile_i=1; tick(); ack_valid_i=0;
    check(record_valid_o,"foreign ACK cannot clear this Tile");
    for(int i=0;i<4;i++) faults[i]=f[3-i]; lanes='1; ack_tile_i=0; ack_valid_i=1;
    #1; check(clear_ready_o,"authorized ACK after reliable report"); tick(); ack_valid_i=0; lanes=0;
    check(report_valid_o && record_o==f[3],"clear + new reports rearm deterministically under lane reversal");
    deliver_report(0); priv_clear_i=1; tick(); priv_clear_i=0; check(!record_valid_o,"privileged clear");
  endtask
  task automatic ordered_reset();
    owner_t o;
    o=owner(tag); command(CMD_START,o); alloc(key(1),1); park(key(1)); alloc(key(2),2);
    alloc(key(3),1); complete(key(3),EV_ERROR); capture(diagnosis(tag,2,4)); deliver_report(0);
    phase_sink_ready=0; phase_owner_i=o; phase_valid_i=1; #1;
    check(phase_ready_o,"old phase instruction accepted before freeze"); tick(); phase_valid_i=0;
    request_phase(RP_DRAIN,gen,1); request_phase(RP_FREEZE,gen-1,1); request_phase(reset_phase_t'(7),gen,1);
    check(!reset_active_o && !ack_valid_o && live_events_o==3,"rejected phase has no effects");
    begin_phase(RP_FREEZE); repeat(int'(hold_cycles)) tick(); take_ack(RP_FREEZE,0);
    refuse_start(owner(tag+1)); check(reset_active_o && issue_frozen_o,"Group freeze persists while idle");
    lanes=1; faults[0]=diagnosis(0,0,0); tick(); lanes=0;
    clear_ingress_ready_i=0; begin_phase(RP_CANCEL); repeat(int'(hold_cycles)) begin tick();
      check(!ack_valid_o,"unaccepted work owner has not cleared its queue"); end
    clear_ingress_ready_i=1; take_ack(RP_CANCEL,0);
    check(aborted_i[0] && clears==1 && reset_active_o,"reset canceled, not retired, the real Task");
    view(key(1),EV_PENDING,1,0); begin_phase(RP_DRAIN);
    repeat(int'(hold_cycles)) begin tick(); check(!is_quiescent_snapshot_o[5] && !ack_valid_o && phase_fires==0,
      "real owed PHASE_SIGNAL blocks DRAIN while local PENDING events do not"); end
    phase_sink_ready=1; take_ack(RP_DRAIN,0); check(phase_fires==1,"older phase drains exactly once");
    view(key(2),EV_PENDING,0,2);
    phase_step(RP_MARK,2); check(armed_o=='0,"MARK releases the armed waiter via failed event");
    query_key_i=key(1); #1; check(!query_o.live,"marked zero-future event reclaimed after pin release");
    view(key(2),EV_RESET,0,2); view(key(3),EV_ERROR,0,1);
    check(query_o.error_code==HIGH+1,"stored error payload unchanged by functional reset");
    phase_step(RP_MARK,2); check(marks==1,"replay returns original MARK count without executing twice");
    retire(o,TS_RESET,reset_fault_ref_i);
    refs_zero_i=0; begin_phase(RP_INVALIDATE);
    repeat(int'(hold_cycles)) begin tick(); check(!ack_valid_o && invalidates==0 && live_events_o==2,
      "no invalidation/reclaim before last residency reference"); end
    refs_zero_i=1; invalidate_ready_i=0; repeat(3) tick();
    check(!ack_valid_o && live_events_o==2,"invalidate owner has not accepted action");
    invalidate_ready_i=1; take_ack(RP_INVALIDATE,0);
    rebuild_ready_i=0; begin_phase(RP_REBUILD); repeat(3) tick(); check(!ack_valid_o,"no early credit epoch");
    rebuild_ready_i=1; take_ack(RP_REBUILD,0); phase_step(RP_RESUME);
    check(reset_done_o && !reset_active_o && marks==1 && invalidates==1 && rebuilds==1 && live_events_o==0,
          "reset completes exactly once, zero residual events");
    ack_valid_i=1; tick(); ack_valid_i=0;
    lanes=1; tick(); lanes=0; check(!record_valid_o && suppressed_seen_o,"DONE does not silently rearm fault capture");
  endtask
  task automatic drain_masks();
    alloc(key(1),1); alloc(key(2),1);
    cmd_owner_i=owner(tag); cmd_kind_i=CMD_START; cmd_valid_i=1;
    req_phase_i=RP_FREEZE; req_valid_i=1; #1;
    check(req_ready_o && !req_reject_o && cmd_ready_o && cmd_reject_o,
          "Group FREEZE prevents a simultaneous START on a free slot");
    tick(); req_valid_i=0; cmd_valid_i=0; take_ack(RP_FREEZE,0); phase_step(RP_CANCEL);
    injected_evidence=6'b111110; begin_phase(RP_DRAIN);
    for(int i=0;i<int'(hold_cycles);i++) begin
      injected_evidence=~(6'b000001 << (i%6)); #1;
      check(is_quiescent_snapshot_o==injected_evidence,"named quiescence mask observes live evidence"); tick();
      check(!ack_valid_o && !reset_done_o,"DRAIN cannot OR evidence across cycles or time out to success");
    end
    injected_evidence='1; take_ack(RP_DRAIN,0); view(key(1),EV_PENDING,0,1);
    // Align an already-accepted completion to the physical edge that marks the table.
    req_valid_i=1; req_phase_i=RP_MARK; req_generation_i=gen; #1;
    check(req_ready_o && !req_reject_o,"MARK accepted");
    if(!mark_events_o) begin tick(); req_valid_i=0;
      for(int i=0;i<5 && !mark_events_o;i++) tick(); end
    check(mark_events_o,"watchdog: MARK action"); comp_valid_i=1;
    comp_i[0]='{key:key(1),producer_id:PROD,status:EV_DONE,error_code:0}; #1;
    check(comp_ready_o[0],"completion actually accepted on the physical MARK edge");
    tick(); req_valid_i=0; comp_valid_i=0;
    check(mark_count_i==1,"writer excluded the real completion from its committed MARK count");
    take_ack(RP_MARK,1);
    view(key(1),EV_DONE,0,1); view(key(2),EV_RESET,0,1); finish_reset();
  endtask
  task automatic scoped_cancel();
    owner_t o,other,bad;
    o=owner(tag); other=owner(tag+1,CONTEXT_COUNT>1?1:0); command(CMD_START,o);
    if(CONTEXT_COUNT>1) command(CMD_START,other);
    bad=o; bad.task_key.dispatch_ordinal^=HIGH; cancel_req(bad,CC_STALE);
    bad=o; bad.ctx.ctx_generation^=HIGH; faults[FS-1]=diagnosis(tag,3,3); cancel_req(bad,CC_CONFLICT);
    check(record_valid_o && record_o.owner==bad && aborted_i=='0,"conflict reports fault without aborting either Task");
    bad=o; bad.ctx.ctx_slot=1; cancel_req(bad,CC_CONFLICT);
    cancel_req(o,CC_APPLY); repeat(int'(hold_cycles)) tick();
    check(aborted_i[0] && task_fires==0,"cancel alone never publishes Task success or releases its identity");
    cancel_req(o,CC_DUPLICATE); retire(o,TS_RESET,cancel_ref_i);
    cancel_req(o,CC_STALE); if(CONTEXT_COUNT==1) command(CMD_START,other);
    retire(other,TS_DONE,0); check(live_i=='0,"unrelated or reused Context retains healthy outcome");
  endtask
  task automatic external_write();
    owner_t o;
    word_t bytes_read;
    o=owner(tag); command(CMD_START,o); memory.open_service();
    memory.request(1,0,0,0,32'(gen)); bytes_read=memory.service_data;
    check(bytes_read==64'h7a69584736251403,"independent seeded bytes from the C023 input binding");
    memory.respond(0,c032_mfe_pkg::MFE_COMPLETED); memory.peek_local(0,bytes_read);
    memory.request(2,0,8,8,32'(gen)); memory.cancel(); memory.respond(0,c032_mfe_pkg::MFE_ISOLATED);
    memory.peek_local(8,0); memory.request(3,0,8,8,32'(gen)); memory.cancel();
    memory.respond(1,c032_mfe_pkg::MFE_FAULTED); memory.peek_local(8,0);
    memory.request(32'(tag),1,64,0,32'(gen)); memory.check_persisted(bytes_read);
    phase_step(RP_FREEZE); phase_step(RP_CANCEL); begin_phase(RP_DRAIN);
    repeat(int'(hold_cycles)) begin tick(); check(!ack_valid_o && !is_quiescent_snapshot_o[3] && live_i[0],
                   "accepted external write cannot be discarded by reset or watchdog"); end
    memory.respond(0,c032_mfe_pkg::MFE_COMPLETED); take_ack(RP_DRAIN,0);
    phase_step(RP_MARK,0); retire(o,TS_RESET,reset_fault_ref_i); finish_reset();
    memory.check_persisted(bytes_read); memory.close_service(4);
  endtask
  task automatic group_failure(input bit isolate,input bit region);
    owner_t o; fault_t f;
    word_t a,b,expected_mark;
    o=owner(tag); command(CMD_START,o); expected_mark=region?0:1;
    if(region) begin alloc(key(1),2,EV_KIND_REGION_DONE); park(key(1)); end
    else alloc(key(1),1);
    if(isolate) begin
      memory.open_service(); memory.request(32'(tag),0,8,8,32'(gen)); memory.refuse_isolation();
    end
    f=diagnosis(tag,region?2:1,HIGH+9); capture(f);
    repeat(int'(hold_cycles)) tick(); deliver_report(1);
    if(isolate) begin
      peer($sformatf("P %h %h",gen,o.ctx.tile_id),a,b); check(a==1,"observed W4 refusal poisons the Group");
      group_poisoned_i=a[0]; refuse_start(owner(tag+1,CONTEXT_COUNT>1?1:0));
      peer("C",a,b); check(a==0 && b==1,"explicit recovery rejected with real reset outstanding");
    end
    if(region) begin
      region_failure=1; complete(key(1),EV_ERROR); region_failure=0;
      view(key(1),EV_ERROR,0,1); check(query_o.kind==EV_KIND_REGION_DONE && aborted_i[0],
          "failed region exits WAIT through failure and records a failed Task, never success");
    end
    phase_step(RP_FREEZE,0,1); phase_step(RP_CANCEL,0,1); begin_phase(RP_DRAIN,1);
    if(isolate) begin
      repeat(int'(hold_cycles)) begin tick(); check(!ack_valid_o && w4_poison && !w4_quiet,
             "poison is not closure or a successful drain"); end
      memory.respond(0,c032_mfe_pkg::MFE_ISOLATED); memory.peek_local(8,0);
    end
    take_ack(RP_DRAIN,0,1); phase_step(RP_MARK,expected_mark,1);
    if(region) view(key(1),EV_ERROR,0,1); else view(key(1),EV_RESET,0,1);
    retire(o,region?TS_ERROR:TS_RESET,reset_fault_ref_i); finish_reset(1);
    check(record_valid_o && record_o==f && task_fires==1 && phase_fires==0,
          "expected safe failure: retained fault, exactly one failed terminal, no OUTPUT_READY");
    if(isolate) begin
      refuse_start(owner(tag+1)); peer("C",a,b); check(a==1 && b==0,"only explicit Group recovery authorizes reuse");
      group_poisoned_i=b[0]; memory.clear_poison(); memory.close_service(1);
      run_begin_i=1; run_generation_i=gen; #1; check(!run_ready_o,"replay cannot rearm the same epoch");
      run_generation_i=gen+1; #1; check(run_ready_o,"new authenticated run accepted after explicit recovery");
      tick(); run_begin_i=0; o=owner(tag+2); o.ctx.ctx_generation++; command(CMD_START,o); retire(o,TS_DONE,0);
      check(live_i=='0 && w4_quiet,"healthy task in recovered epoch retires without stale failure");
    end
    peer("Z",a,b);
  endtask
  task automatic ring_contrast();
    word_t a,b;
    fault_t f;
    burst_valid='1; tick(); burst_valid='0;
    check(burst_count==1,"eight concurrent suppressed faults saturate a one-bit counter");
    for(int i=0;i<int'(hold_cycles);i++) begin
      f=diagnosis(tag+word_t'(i),1,word_t'(i)); f.code=word_t'(i)%64'd11; faults[0]=f; lanes=1; tick();
    end
    lanes=0; peer($sformatf("G 2 %h",hold_cycles),a,b);
    check(a==hold_cycles && b==hold_cycles-1,"real depth-two Group ring overwrote its oldest slot");
    check(record_valid_o && record_o.source_id==tag && record_o.code==0 && suppressed_seen_o,
          "Tile retained first fault; ring overwrite is not the Tile model"); deliver_report(0);
  endtask

  // Same authorized FINISH before, on, or after the FREEZE admission edge.
  // A previously stored success is immutable. An uncommitted FINISH cannot
  // become success just because the per-Context CANCEL sweep has not run yet.
  task automatic terminal_cutline();
    owner_t o;
    word_t first_ref;
    for(int when_finish=-1; when_finish<=1; when_finish++) begin
      start_case(); o=owner(tag); first_ref=reset_fault_ref_i;
      command(CMD_START,o); command(CMD_DRAIN,o); terminal(0,o,TS_DONE,0); command(CMD_RETIRE,o);
      if(when_finish<0) begin
        command(CMD_FINISH,o); repeat(3) tick();
        check(msg_valid_o && msg_o.status==TS_DONE,"old success already offered before FREEZE");
      end
      if(when_finish==0) begin
        cmd_kind_i=CMD_FINISH; cmd_owner_i=o; cmd_valid_i=1; req_valid_i=1; req_phase_i=RP_FREEZE; #1;
        check(cmd_ready_o && !cmd_reject_o && req_ready_o && !req_reject_o,
              "real FINISH and FREEZE accepted together");
        tick(); req_valid_i=0; cmd_valid_i=0; take_ack(RP_FREEZE,0);
      end else phase_step(RP_FREEZE);
      reset_fault_ref_i=~first_ref;
      if(when_finish>0) command(CMD_FINISH,o);
      terminal(1,o,when_finish<0?TS_DONE:TS_RESET,when_finish<0?0:first_ref);
      phase_step(RP_CANCEL); phase_step(RP_DRAIN); phase_step(RP_MARK); finish_reset();
      check(task_fires==1 && rr_fires==1,"one terminal regardless of the FREEZE/FINISH cutline");
    end
  endtask

  initial begin
    int vector_fd,n;
    if(!$value$plusargs("VECTORS=%s",vector_file) || !$value$plusargs("PEER=%s",peer_path))
      $fatal(1,"C034 bench requires vectors and a live peer");
    vector_fd=$fopen(vector_file,"r"); request_fd=$fopen({peer_path,".req"},"w");
    response_fd=$fopen({peer_path,".rsp"},"r");
    if(vector_fd==0 || request_fd==0 || response_fd==0) $fatal(1,"C034 cannot open its inputs");
    for(int i=0;i<9;i++) begin
      case_name=$sformatf("U3-%c",8'(97+i));
      n=$fscanf(vector_fd,"%h %h %h %h %h %h %h %h %h",gen,tag,hold_cycles,wanted[0],wanted[1],
                wanted[2],wanted[3],wanted[4],wanted[5]);
      if(n!=9) $fatal(1,"C034 truncated vector %s",case_name);
      start_case();
      case(i)
        0:multi_fault(); 1:ordered_reset(); 2:drain_masks(); 3:scoped_cancel(); 4:external_write();
        5:group_failure(1,0); 6:ring_contrast(); 7:group_failure(0,1); 8:group_failure(0,0);
      endcase
      tick(); check(observed_src==wanted[0] && suppressed_o==wanted[1],"scenario first-fault summary");
      check(marked_sum==wanted[2] && word_t'(task_fires)==wanted[3] && task_fires==rr_fires,
            "scenario MARK and exactly-once retirement ledgers");
      check(stale_cancels_o==wanted[4] && word_t'(group_poisoned_i)==wanted[5],"scenario cancel/poison closure");
      $display("C034_CASE %s",case_name);
    end
    case_name="committed-terminal-cutline"; terminal_cutline();
    $fclose(request_fd); $fclose(response_fd); $fclose(vector_fd);
    $display("C034_PASS tb_c034 contexts=%0d checks=%0d",CONTEXT_COUNT,checks); $finish;
  end
endmodule
