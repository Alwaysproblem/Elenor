// C033: drive real producer, UCE and reliable-port handshakes. YAML expectations
// are read by the compiled bench, not compared with another copy of the YAML.
`timescale 1ns/1ps
module tb_c033 #(parameter int unsigned CONTEXT_COUNT=2, OPERANDS=3,
                 parameter string BENCH_NAME="tb_c033");
  import c033_event_pkg::*;
  localparam int unsigned SOURCES=4, FRONTIER=2;
  localparam word_t PROD = 64'h100000007, HIGH = 64'h100000000;
  logic clk_i = 0, rst_ni = 0;
  logic alloc_valid_i, alloc_ready_o, alloc_reject_o;
  event_key_t alloc_key_i, query_key_i;
  logic [1:0] alloc_kind_i;
  word_t alloc_producer_i, alloc_future_i;
  logic [SOURCES-1:0] comp_valid_i, comp_ready_o;
  completion_t [SOURCES-1:0] comp_i;
  logic wait_valid_i, wait_all_i, wait_ready_o, wait_reject_o, direct_ok_o, direct_fail_o, park_o;
  context_key_t wait_ctx_i, fault_ctx_i;
  word_t wait_count_i;
  event_key_t [OPERANDS-1:0] wait_keys_i;
  logic [CONTEXT_COUNT-1:0] armed_o, wake_ok_o, wake_fail_o;
  logic fault_valid_i;
  event_view_t query_o;
  counts_t [CONTEXT_COUNT-1:0] counts_o;
  counter_flags_t [CONTEXT_COUNT-1:0] counts_seen_o;
  // C033 exercises normal issue, not the cold or functional reset sequence.
  wire mark_events_i = 1'b0, abandon_events_i = 1'b0, issue_frozen_i = 1'b0;
  wire reset_failure_i = 1'b0;
  wire word_t reset_ref_i = '0;
  word_t mark_count_o, live_events_o;
  logic abandon_ready_o;
  event_table #(.CONTEXT_COUNT(CONTEXT_COUNT), .FRONTIER(FRONTIER), .OPERANDS(OPERANDS),
                .SOURCES(SOURCES), .COUNTER_BITS(4), .TILE_ID(0)) events (.*);

  logic cmd_valid_i, cmd_ready_o, cmd_reject_o, exec_drained_i, no_old_inflight_i, region_ended_i;
  logic local_reclaimed_i, w4_closed_i, abort_valid_i;
  cmd_kind_t cmd_kind_i;
  owner_t cmd_owner_i, abort_owner_i, phase_owner_i;
  task_status_t finish_status_i, abort_status_i;
  word_t drain_evidence_seq_i, first_fault_ref_i, abort_ref_i, phase_seq_i;
  logic phase_valid_i, phase_direction_i, phase_ready_o, phase_valid_o, phase_ready_i;
  phase_msg_t phase_o;
  logic msg_valid_o, msg_ready_i;
  terminal_msg_t msg_o;
  logic [CONTEXT_COUNT-1:0] active_o, ready_pending_o, retire_sent_o, completion_pending_o;
  owner_t [CONTEXT_COUNT-1:0] live_owner_o;
  logic [CONTEXT_COUNT-1:0] aborted_o, phase_pending_o;
  terminal_store #(.CONTEXT_COUNT(CONTEXT_COUNT), .TILE_ID(0)) terminals (.*);

  // Only these fields cross into a new scenario: four seed words, eight observations.
  word_t seed_id, seed_gen, seed_seq, seed_uses, wanted [8];
  string case_name, fixture;
  int cycles = 0, checks = 0, successes, failures, phases, retire_ready, completions;
  logic phase_held, msg_held;
  phase_msg_t held_phase;
  terminal_msg_t held_msg;

  task automatic require(input logic ok, input string detail);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C033_ASSERT %s cycle=%0d %s", case_name, cycles, detail);
  endtask

  always @(posedge clk_i) begin
    cycles++;
    if (!rst_ni) begin
      successes = 0; failures = 0; phases = 0; retire_ready = 0; completions = 0;
      phase_held = 0; msg_held = 0;
    end else begin
      require(!(direct_ok_o && direct_fail_o) && !(|(wake_ok_o & wake_fail_o)), "one WAIT verdict");
      if (phase_held) require(phase_valid_o && phase_o == held_phase, "phase payload stable through fire");
      if (msg_held) require(msg_valid_o && msg_o == held_msg, "terminal payload stable through fire");
      phase_held = phase_valid_o && !phase_ready_i; held_phase = phase_o;
      msg_held = msg_valid_o && !msg_ready_i; held_msg = msg_o;
      successes += int'(direct_ok_o) + $countones(wake_ok_o);
      failures += int'(direct_fail_o) + $countones(wake_fail_o);
      if (phase_valid_o && phase_ready_i) begin
        phases++;
        if (case_name == "U2-f" || case_name == "U2-g") begin
          $display("C033_PHASE %s %h %h %h %h %h %h %h", case_name,
                   phase_o.task_key.launch_generation, phase_o.task_key.root_launch_generation,
                   phase_o.task_key.dispatch_ordinal, phase_o.task_key.task_id,
                   phase_o.task_key.context_handle, phase_o.direction, phase_o.message_seq);
        end
      end
      if (msg_valid_o && msg_ready_i) begin
        if (msg_o.is_completion) completions++; else retire_ready++;
      end
    end
  end

  // Manual clock: inputs are stable through the edge; probes never race NBA updates.
  task automatic tick();
    #2; clk_i = 1; #2; clk_i = 0; #2;
  endtask

  task automatic cold_reset();
    alloc_valid_i = 0; alloc_key_i = '0; alloc_kind_i = 0; alloc_producer_i = PROD; alloc_future_i = 0;
    comp_valid_i = '0; comp_i = '0; wait_valid_i = 0; wait_all_i = 0; wait_keys_i = '0;
    wait_count_i = 0; wait_ctx_i = '0; query_key_i = '0; fault_valid_i = 0; fault_ctx_i = '0;
    cmd_valid_i = 0; cmd_kind_i = CMD_START; cmd_owner_i = '0; abort_owner_i = '0; abort_valid_i = 0;
    abort_status_i = TS_RESET; abort_ref_i = 0; finish_status_i = TS_DONE; first_fault_ref_i = 0;
    exec_drained_i = 1; no_old_inflight_i = 1; region_ended_i = 1; w4_closed_i = 1;
    local_reclaimed_i = 1; drain_evidence_seq_i = seed_seq;
    phase_valid_i = 0; phase_owner_i = '0; phase_direction_i = 0; phase_seq_i = 0;
    phase_ready_i = 0; msg_ready_i = 0; rst_ni = 0; tick(); rst_ni = 1;
    require(counts_seen_o=='0,"sticky reject flags clear on cold reset");
  endtask

  function automatic event_key_t key(input word_t delta = 0);
    event_key_t k;
    k.ctx = '{tile_id:0, ctx_slot:0, ctx_generation:seed_gen};
    k.local_event_id = seed_id + delta; k.event_generation = HIGH + seed_gen + 1;
    k.seq = seed_seq; k.scope = 0; return k;
  endfunction

  task automatic allocate(input event_key_t k, input word_t uses);
    alloc_valid_i = 1; alloc_key_i = k; alloc_future_i = uses; alloc_kind_i = k.local_event_id[1:0];
    #1; require(alloc_ready_o && !alloc_reject_o, "allocate live event"); tick(); alloc_valid_i = 0;
  endtask

  task automatic entry(input event_key_t k, input event_status_t st, input word_t refs, fut);
    query_key_i = k; #1;
    require(query_o.live && query_o.status == st && query_o.consumer_refs == refs &&
            query_o.future_uses == fut && query_o.kind == k.local_event_id[1:0] &&
            query_o.producer_id == PROD, "persistent status, producer, kind and both reference counts");
  endtask

  task automatic absent(input event_key_t k);
    query_key_i = k; #1; require(!query_o.live, "no matching event slot");
  endtask

  task automatic drive_c(input event_key_t k, input event_status_t st, input word_t producer = PROD);
    comp_i[0] = '{key:k, status:st, producer_id:producer, error_code:st == EV_ERROR ? HIGH+3 : 0};
    comp_valid_i[0] = 1; #1; require(comp_ready_o[0], "completion not blocked by frontier/WAIT");
  endtask

  task automatic complete(input event_key_t k, input event_status_t st, input word_t producer = PROD);
    drive_c(k, st, producer); tick(); comp_valid_i = '0;
  endtask

  task automatic drive_wait(input context_key_t who, input event_key_t a, b,
                            input bit all_mode, input word_t n);
    wait_valid_i = 1; wait_ctx_i = who; wait_all_i = all_mode; wait_count_i = n;
    wait_keys_i = '0; wait_keys_i[0] = a; wait_keys_i[1] = b; #1;
  endtask

  task automatic park(input context_key_t who, input event_key_t a, b, input bit all_mode = 0);
    drive_wait(who, a, b, all_mode, all_mode ? 2 : 1);
    require(wait_ready_o && park_o && !direct_ok_o && !direct_fail_o && !wait_reject_o, "WAIT may park only on a valid pending set");
    tick(); wait_valid_i = 0;
    require(armed_o[int'(who.ctx_slot)], "accepted wait persisted in its Context");
  endtask

  task automatic bad_wait(input event_key_t a, b, input bit all_mode, input word_t n);
    drive_wait(a.ctx, a, b, all_mode, n);
    require(wait_reject_o && !park_o && !direct_ok_o && !direct_fail_o, "malformed wait rejected atomically");
    tick(); wait_valid_i = 0;
  endtask

  task automatic event_first();
    event_key_t a, newer;
    a = key(); allocate(a, seed_uses); complete(a, EV_DONE); allocate(key(1), 1);
    entry(a, EV_DONE, 0, seed_uses);
    alloc_key_i = key(2); alloc_valid_i = 1; #1;
    require(!alloc_ready_o && !alloc_reject_o, "future consumer protects a full frontier from replacement");
    tick(); alloc_valid_i = 0;
    newer = a; newer.event_generation++;
    alloc_valid_i = 1; alloc_key_i = newer; #1;
    require(!alloc_ready_o, "same local id cannot reuse retained terminal under a newer generation");
    tick(); alloc_valid_i = 0; absent(newer); entry(a, EV_DONE, 0, seed_uses);
    drive_wait(a.ctx, a, '0, 0, 1);
    require(wait_ready_o && direct_ok_o && !park_o && armed_o == '0, "event first: direct hit");
    tick(); wait_valid_i = 0; tick(); entry(a, EV_DONE, 0, seed_uses-1);
  endtask

  task automatic same_cycle();
    event_key_t a;
    a = key(); allocate(a, seed_uses); drive_c(a, EV_DONE); drive_wait(a.ctx, a, '0, 0, 1);
    require(comp_ready_o[0] && wait_ready_o && direct_ok_o && !park_o && wake_ok_o == '0,
            "same-cycle accepted completion bypasses pending without a park");
    tick(); wait_valid_i = 0; comp_valid_i = 0; #1;
    require(armed_o == '0 && wake_ok_o == '0, "no second wake for direct decision"); absent(a);
    tick();
  endtask

  task automatic after_park();
    event_key_t a, b;
    a = key(); b = key(1); allocate(a, seed_uses); allocate(b, 1); park(a.ctx, a, '0);
    entry(a, EV_PENDING, 1, seed_uses-1); complete(b, EV_DONE);
    require(armed_o[0] && successes == 0, "unrelated event cannot wake a parked WAIT");
    drive_c(a, EV_DONE); require(wake_ok_o[0] && !wake_fail_o[0], "accepted completion wakes parked owner");
    tick(); comp_valid_i = 0; complete(a, EV_RESET);
    entry(a, EV_DONE, 0, seed_uses-1);
    require(counts_o[0].duplicate_signal == 1 && !armed_o[0], "sticky terminal rejects duplicate exactly once");
  endtask

  task automatic waitall_cases();
    event_key_t a, b;
    event_status_t st;
    a = key(); b = key(1); allocate(a, seed_uses); allocate(b, 1);
    park(a.ctx, a, b, 1); complete(a, EV_DONE);
    require(armed_o[0] && successes == 0, "WAITALL must hold the full record after first DONE");
    entry(a, EV_DONE, 1, 0); entry(b, EV_PENDING, 1, 0);
    complete(b, EV_DONE); require(!armed_o[0] && successes == 1, "last DONE wakes WAITALL once");
    absent(a); absent(b);
    for (int f=0; f<3; f++) begin
      st = event_status_t'(int'(EV_ERROR) + f);
      a.event_generation++; b.event_generation++;
      allocate(a, seed_uses); allocate(b, 1); park(a.ctx, a, b, 1); drive_c(a, st);
      require(wake_fail_o[0] && !wake_ok_o[0], "any failure exits before the pending operand");
      tick(); comp_valid_i = 0; absent(a); entry(b, EV_PENDING, 0, 0);
      complete(b, EV_DONE); absent(b);
    end
    a.event_generation++; allocate(a, seed_uses); complete(a, EV_ERROR); entry(a, EV_ERROR, 0, 1);
    require(query_o.error_code == HIGH+3, "error payload retained until matched");
    drive_wait(a.ctx, a, '0, 0, 1);
    require(direct_fail_o && !direct_ok_o && !park_o, "earlier fault is not an immediate success");
    tick(); wait_valid_i = 0;
  endtask

  task automatic full_key();
    event_key_t a, b, bad;
    counts_t ct;
    a = key(); b = key(1); allocate(a, seed_uses); allocate(b, 0);
    bad_wait(a, a, 1, 2); bad_wait(a, b, 0, 2); bad_wait(a, b, 1, 1);
    bad_wait(a, b, 1, word_t'(OPERANDS)+1);
    bad_wait(a, key(22), 1, 2); bad_wait(a, b, 1, 2); entry(a, EV_PENDING, 0, seed_uses);
    bad = a; bad.seq ^= HIGH; bad_wait(bad, '0, 0, 1);
    require(armed_o == '0 && counts_o[0].protocol_error == 7, "malformed waits neither park nor partially pin A");
    // Three independently rejected operations, not one busy cycle.
    alloc_valid_i=1; alloc_key_i=a; drive_c(a,EV_PENDING); drive_wait(a.ctx,a,b,1,2);
    require(alloc_reject_o && wait_reject_o && comp_ready_o[0], "three bad inputs observed on the same edge");
    tick(); alloc_valid_i=0; comp_valid_i='0; wait_valid_i=0;
    require(counts_o[0].protocol_error==10 && counts_seen_o[0].protocol_error &&
            counts_o[0].unknown_event_notify==0 && counts_o[0].stale_sequence==0,
            "count each rejected request without charging bad WAIT operands as notifications");
    entry(a,EV_PENDING,0,seed_uses);
    park(a.ctx, a, '0);
    for (int f=0; f<6; f++) begin
      bad = a; ct = counts_o[0];
      case (f)
        0: bad.ctx.ctx_generation ^= HIGH;
        1: bad.scope ^= 2'b01;
        2: bad.event_generation ^= HIGH;
        3: bad.seq ^= HIGH;
        4: bad.local_event_id ^= HIGH;
        default: ; // producer mismatch on a full-key match
      endcase
      complete(bad, EV_RESET, f == 5 ? PROD ^ HIGH : PROD);
      entry(a, EV_PENDING, 1, seed_uses-1);
      require(armed_o[0] && successes == 0 && failures == 0, "mismatched notification cannot settle record");
      case (f)
        0: begin ct.wrong_owner++; ct.wrong_context++; end
        1: begin ct.wrong_owner++; ct.wrong_scope++; end
        2: begin ct.stale_sequence++; ct.wrong_event_generation++; end
        3: begin ct.stale_sequence++; ct.wrong_sequence++; end
        4: ct.unknown_event_notify++;
        5: ct.wrong_owner++;
      endcase
      require(counts_o[0] == ct, "wrong identity increments only the correct detail and aggregate");
    end
    bad = a; bad.seq ^= HIGH;
    repeat (18) complete(bad, EV_ERROR);
    require(counts_o[0].wrong_sequence == 15 && counts_o[0].stale_sequence == 15, "reject counters saturate, never wrap");
    require(counts_seen_o[0].wrong_scope && counts_seen_o[0].stale_sequence &&
            !counts_seen_o[0].duplicate_signal,"only observed reject classes set sticky bits");
    complete(a, EV_DONE); entry(a, EV_DONE, 0, seed_uses-1);
    complete(b, EV_DONE); b.event_generation++; allocate(b, 1); park(b.ctx,b,'0);
    fault_valid_i = 1; fault_ctx_i = b.ctx; fault_ctx_i.ctx_generation ^= HIGH;
    tick(); require(armed_o[0], "wrong Context generation cannot void WAIT");
    fault_ctx_i = b.ctx; tick(); fault_valid_i = 0;
    require(!armed_o[0], "fault voids its own wait"); entry(b, EV_PENDING, 0, 0);
    complete(b, EV_DONE); absent(b);
  endtask

  function automatic owner_t owner_for(input word_t tid, input bit grid = 0, input int slot = 0);
    owner_t o;
    o.ctx = '{tile_id:0, ctx_slot:word_t'(slot), ctx_generation:seed_gen};
    o.task_key = '{launch_generation:seed_id, root_launch_generation:grid ? seed_id : seed_id+HIGH,
                   dispatch_ordinal:seed_gen, task_id:tid, context_handle:grid ? 0 : HIGH+9};
    return o;
  endfunction

  task automatic command(input cmd_kind_t op, input owner_t owner);
    cmd_owner_i = owner; cmd_kind_i = op; cmd_valid_i = 1; #1;
    require(cmd_ready_o && !cmd_reject_o, $sformatf("eligible command accepted kind=%0d", op));
    tick(); cmd_valid_i = 0;
  endtask

  task automatic blocked_cmd(input cmd_kind_t op, input owner_t owner, input bit awaiting_evidence=0);
    cmd_owner_i = owner; cmd_kind_i = op; cmd_valid_i = 1; #1;
    if (awaiting_evidence)
      require(!cmd_ready_o && !cmd_reject_o, "missing evidence backpressures the producer without fault");
    else
      require(cmd_ready_o && cmd_reject_o, "invalid command is consumed as a rejection, not committed");
    tick(); cmd_valid_i = 0;
  endtask

  task automatic offered(input logic final_msg, input owner_t owner, input task_status_t st, input word_t refword);
    int n;
    #1;
    for (n=0; n<2*int'(CONTEXT_COUNT)+3 && !msg_valid_o; n++) tick();
    require(msg_valid_o, "watchdog: expected reliable port offer");
    require(msg_o.is_completion == final_msg && msg_o.owner == owner, "message type and exact full owner");
    if (final_msg) require(msg_o.status == st && msg_o.first_fault_ref == refword,
                           "terminal status and first fault tied to stored owner");
    else require(msg_o.drain_evidence_seq == refword, "RETIRE_READY retains drain evidence identity");
  endtask

  task automatic take_msg();
    msg_ready_i = 1; tick(); msg_ready_i = 0;
  endtask

  task automatic send_phase(input owner_t o, input logic d, input word_t seqword);
    int n;
    phase_valid_i=1; phase_owner_i=o; phase_direction_i=d; phase_seq_i=seqword; #1;
    for (n=0; n<4 && !phase_ready_o; n++) tick();
    require(phase_ready_o, "watchdog: phase issue ready"); tick(); phase_valid_i=0;
  endtask

  task automatic check_phase(input owner_t o, input logic d, input word_t seqword);
    for (int n=0; n<int'(CONTEXT_COUNT)+2 && !phase_valid_o; n++) tick();
    #1;
    require(phase_valid_o && phase_o.task_key == o.task_key && phase_o.direction == d &&
            phase_o.message_seq == seqword, "one instruction maps to its exact phase message");
  endtask

  task automatic mark_abort(input owner_t o, input task_status_t st, input word_t refword);
    abort_valid_i = 1; abort_owner_i = o; abort_status_i = st; abort_ref_i = refword;
    tick(); abort_valid_i = 0;
  endtask

  task automatic close_task(input owner_t o, input task_status_t expected_status, input word_t expected_ref);
    command(CMD_DRAIN,o); offered(0,o,TS_DONE,drain_evidence_seq_i); take_msg();
    command(CMD_RETIRE,o); finish_status_i = TS_DONE; first_fault_ref_i = 0;
    command(CMD_FINISH,o); offered(1,o,expected_status,expected_ref); take_msg();
    require(!active_o[int'(o.ctx.ctx_slot)], "owner releases only after terminal is consumed");
  endtask

  task automatic duplicate_phase();
    bit buffered_second;
    owner_t o;
    o = owner_for(0,1); command(CMD_START,o); send_phase(o,0,seed_seq);
    check_phase(o,0,seed_seq);
    blocked_cmd(CMD_DRAIN,o,1); // a parked PHASE_SIGNAL is still owed to the Group
    phase_valid_i=1; phase_owner_i=o; phase_direction_i=0; phase_seq_i=seed_seq+1; #1;
    buffered_second = phase_ready_o;
    tick(); phase_valid_i=0;
    phase_ready_i=1; tick(); phase_ready_i=0;
    // A deeper phase buffer could already have accepted the second issue.
    if (!buffered_second) send_phase(o,0,seed_seq+1);
    check_phase(o,0,seed_seq+1); phase_ready_i=1; tick(); phase_ready_i=0;
    require(active_o[0] && retire_ready == 0 && completions == 0, "input release does not retire the Task");
    mark_abort(o,TS_ERROR,seed_seq+3); close_task(o,TS_ERROR,seed_seq+3);
  endtask

  task automatic legal_phases();
    owner_t o;
    word_t s;
    // PHASE_SIGNAL consumes no local-event slot even with a full frontier.
    allocate(key(),seed_uses); allocate(key(1),0);
    for (int t=0; t<4; t++) begin
      o = owner_for(word_t'(t),1,t % int'(CONTEXT_COUNT)); command(CMD_START,o);
      for (int d=0; d<2; d++) begin
        s = seed_seq + 2*word_t'(t) + word_t'(d); send_phase(o,1'(d),s); check_phase(o,1'(d),s);
        phase_ready_i=1; tick(); phase_ready_i=0;
        require(active_o[int'(o.ctx.ctx_slot)], "legal phase does not release Context or advance retire");
      end
      if (t==0) begin
        entry(key(),EV_PENDING,0,seed_uses); entry(key(1),EV_PENDING,0,0);
        complete(key(),EV_DONE); complete(key(1),EV_DONE);
      end
      close_task(o,TS_DONE,0);
    end
  endtask

  task automatic held_terminal();
    owner_t o, wrong;
    event_key_t a, b;
    o=owner_for(seed_id+HIGH); wrong=o; wrong.task_key.context_handle ^= HIGH;
    // Already terminal, but tile-shared future users retain both slots:
    // the full frontier is genuine; there is no invented EXEC_DRAINED.
    a=key(); a.scope=EV_SCOPE_TILE_SHARED; b=key(1); b.scope=EV_SCOPE_TILE_SHARED;
    allocate(a,seed_uses); allocate(b,1); complete(a,EV_DONE); complete(b,EV_DONE);
    alloc_valid_i=1; alloc_key_i=key(3); #1;
    require(!alloc_ready_o, "frontier full while independent terminal reserve is used");
    command(CMD_START,o);
    exec_drained_i=0; blocked_cmd(CMD_DRAIN,o,1); exec_drained_i=1;
    no_old_inflight_i=0; blocked_cmd(CMD_DRAIN,o,1); no_old_inflight_i=1;
    region_ended_i=0; blocked_cmd(CMD_DRAIN,o,1); region_ended_i=1;
    blocked_cmd(CMD_DRAIN,wrong); command(CMD_DRAIN,o);
    offered(0,o,TS_DONE,seed_seq); require(ready_pending_o[0] && !retire_sent_o[0], "RR saved but unsent");
    drain_evidence_seq_i ^= HIGH; blocked_cmd(CMD_RETIRE,o); repeat(3) tick();
    require(active_o[0] && completions == 0, "RR backpressure is not Task completion"); take_msg();
    blocked_cmd(CMD_RETIRE,wrong); blocked_cmd(CMD_FINISH,o); command(CMD_RETIRE,o);
    local_reclaimed_i=0; blocked_cmd(CMD_FINISH,o,1); local_reclaimed_i=1;
    command(CMD_FINISH,o); offered(1,o,TS_DONE,0);
    require(completion_pending_o[0] && active_o[0], "identity retained for stalled completion");
    finish_status_i=TS_ERROR; first_fault_ref_i=seed_seq; blocked_cmd(CMD_FINISH,o);
    wrong=o; wrong.ctx.ctx_generation++; blocked_cmd(CMD_START,wrong); mark_abort(o,TS_RESET,seed_seq);
    offered(1,o,TS_DONE,0); take_msg(); alloc_valid_i=0;
    require(!active_o[0] && !completion_pending_o[0], "exactly one stored completion consumed");
    wrong.ctx=o.ctx; wrong.ctx.ctx_generation++;
    drive_wait(wrong.ctx,a,'0,0,1); require(direct_ok_o,"later consumer sees retained terminal A");
    tick(); wait_valid_i=0;
    drive_wait(wrong.ctx,b,'0,0,1); require(direct_ok_o,"later consumer sees retained terminal B");
    tick(); wait_valid_i=0; absent(a); absent(b);
  endtask

  task automatic retire_vs_cancel();
    owner_t o, wrong;
    task_status_t st;
    word_t fault_ref;
    for (int f=1; f<=3; f++) begin
      o=owner_for(seed_id+word_t'(f)); o.ctx.ctx_generation+=word_t'(f);
      wrong=o; wrong.ctx.ctx_generation ^= HIGH; st=task_status_t'(f); fault_ref=seed_seq+word_t'(f);
      command(CMD_START,o); command(CMD_DRAIN,o); offered(0,o,TS_DONE,seed_seq); take_msg();
      require(retire_sent_o[0] && completions==f-1, "sent RETIRE_READY did not publish success");
      mark_abort(wrong,TS_RESET,~fault_ref);
      if (f!=3) begin mark_abort(o,st,fault_ref); mark_abort(o,TS_RESET,~fault_ref); end
      command(CMD_RETIRE,o); finish_status_i=TS_DONE; first_fault_ref_i=0;
      w4_closed_i=0; blocked_cmd(CMD_FINISH,o,1);
      require(!completion_pending_o[0] && !msg_valid_o, "cancel/timeout is not W4 closure evidence"); w4_closed_i=1;
      if (f==3) begin abort_valid_i=1; abort_owner_i=o; abort_status_i=st; abort_ref_i=fault_ref; end
      command(CMD_FINISH,o); abort_valid_i=0; offered(1,o,st,fault_ref); repeat(3) tick(); take_msg();
      blocked_cmd(CMD_FINISH,o); require(!active_o[0], "old terminal cannot be replayed");
    end
  endtask

  // Drained or cancelled is a boundary for *new issue*, not permission to
  // discard an older phase. Exercise a simultaneous DRAIN and phase issue.
  task automatic phase_cutline();
    owner_t o, peer;
    bit issued;
    o=owner_for(1); command(CMD_START,o);
    cmd_kind_i=CMD_DRAIN; cmd_owner_i=o; cmd_valid_i=1;
    phase_valid_i=1; phase_owner_i=o; phase_direction_i=1; phase_seq_i=seed_seq; #1;
    require(!cmd_reject_o && (phase_ready_o ^ cmd_ready_o),
            "DRAIN and a new unaccepted phase for its owner cannot both commit or deadlock");
    issued=phase_ready_o; tick(); cmd_valid_i=0; phase_valid_i=0;
    if (issued) begin
      check_phase(o,1,seed_seq); blocked_cmd(CMD_DRAIN,o,1);
      if (CONTEXT_COUNT==2) begin
        peer=owner_for(7,0,1); command(CMD_START,peer); close_task(peer,TS_DONE,0);
        check_phase(o,1,seed_seq); // an unrelated terminal made progress during the stall
      end
      phase_ready_i=1; tick(); phase_ready_i=0; command(CMD_DRAIN,o);
    end
    phase_valid_i=1; phase_seq_i++;
    #1; require(!phase_ready_o, "no new PHASE while RETIRE_READY is pending");
    offered(0,o,TS_DONE,seed_seq); take_msg();
    #1; require(!phase_ready_o, "no new PHASE after RETIRE_READY fires");
    command(CMD_RETIRE,o); #1; require(!phase_ready_o, "no new PHASE after local reclaim is authorized");
    command(CMD_FINISH,o); phase_valid_i=0; offered(1,o,TS_DONE,0); take_msg();
    o=owner_for(2); command(CMD_START,o);
    phase_valid_i=1; phase_owner_i=o; abort_valid_i=1; abort_owner_i=o;
    abort_status_i=TS_RESET; abort_ref_i=HIGH+1; #1;
    require(!phase_ready_o, "matching abort takes priority over a new phase issue");
    tick(); abort_valid_i=0; #1; require(!phase_ready_o, "cancelled owner stays closed for new issue");
    phase_valid_i=0; close_task(o,TS_RESET,HIGH+1);
    require(phases==int'(issued) && completions==2+int'(issued && CONTEXT_COUNT==2) &&
            !phase_valid_o, "no message outlives its owner");
    o=owner_for(3); command(CMD_START,o); command(CMD_DRAIN,o);
    offered(0,o,TS_DONE,seed_seq); take_msg(); command(CMD_RETIRE,o);
    finish_status_i=TS_TIMEOUT; first_fault_ref_i=seed_seq+3; w4_closed_i=0;
    blocked_cmd(CMD_FINISH,o,1); w4_closed_i=1; command(CMD_FINISH,o);
    offered(1,o,TS_TIMEOUT,first_fault_ref_i); take_msg();
  endtask

  task automatic arbitration_and_contexts();
    event_key_t a, bad, b, c;
    bit fault_issue_fired;
    context_key_t other;
    owner_t lo, hi;
    logic [SOURCES-1:0] seen;
    a=key(); a.scope=1; allocate(a,2); seen='0;
    for (int s=0; s<int'(SOURCES); s++) comp_i[s]='{key:a,status:EV_DONE,producer_id:PROD,error_code:0};
    comp_valid_i='1;
    repeat(SOURCES) begin
      #1; require($onehot(comp_ready_o & comp_valid_i), "exactly one completion accepted on a busy bus");
      seen |= comp_valid_i & comp_ready_o; tick();
    end
    comp_valid_i=0;
    require(&seen && counts_o[0].duplicate_signal == 3, "RR gives each sustained source a real grant");
    a=key(2); allocate(a,1); bad=a; bad.seq ^= HIGH;
    drive_c(bad,EV_DONE); drive_wait(a.ctx,a,'0,0,1);
    require(wait_ready_o && park_o && !direct_ok_o, "rejected same-edge notification cannot bypass WAIT");
    tick(); comp_valid_i=0; wait_valid_i=0;
    entry(a,EV_PENDING,1,0); complete(a,EV_DONE); absent(a);
    a=key(3); a.scope=EV_SCOPE_TILE_SHARED; allocate(a,1);
    fault_ctx_i=a.ctx; fault_valid_i=1; drive_wait(a.ctx,a,'0,0,1);
    fault_issue_fired=wait_ready_o && park_o;
    tick(); wait_valid_i=0; fault_valid_i=0; entry(a,EV_PENDING,0,1-word_t'(fault_issue_fired));
    require(armed_o=='0, "a new wait cannot survive its own same-edge Context fault");
    complete(a,EV_DONE);
    if (!fault_issue_fired) begin
      other=a.ctx; other.ctx_generation++; drive_wait(other,a,'0,0,1);
      require(direct_ok_o, "unissued shared use retained after the issuing Context fault");
      tick(); wait_valid_i=0;
    end
    absent(a);
    if (CONTEXT_COUNT==2) begin
      a=key(1); a.scope=1; other=a.ctx; other.ctx_slot=1;
      // Same event id/generation in two physical Contexts is not one event.
      bad=a; bad.ctx=other;
      allocate(a,2); park(a.ctx,a,'0); park(other,a,'0); entry(a,EV_PENDING,2,0);
      allocate(bad,1); complete(bad,EV_PENDING);
      require(counts_o[int'(other.ctx_slot)].protocol_error==1 && counts_o[0].protocol_error==0,
              "malformed notification charged only to its routed Context");
      complete(bad,EV_ERROR); entry(bad,EV_ERROR,0,1); entry(a,EV_PENDING,2,0);
      require(&armed_o, "peer's same-name event cannot wake either waiter on the shared operand");
      drive_c(a,EV_DONE); require(&wake_ok_o, "one shared completion wakes both distinct pinned wait records");
      tick(); comp_valid_i=0; absent(a); require(armed_o=='0, "two pins released once each");
      if (OPERANDS==3) begin
        a=key(); a.scope=EV_SCOPE_TILE_SHARED; // retained DONE, two declared uses
        b=key(5); b.scope=EV_SCOPE_TILE_SHARED; c=key(6); c.ctx=other; c.scope=EV_SCOPE_TILE_SHARED;
        allocate(b,1); allocate(c,1); drive_wait(a.ctx,a,b,1,3); wait_keys_i[OPERANDS-1]=c; #1;
        require(park_o && wait_ready_o,"maximum-width WAITALL accepts three resolved operands");
        tick(); wait_valid_i=0; complete(b,EV_DONE);
        require(armed_o[0], "third WAITALL operand must not be truncated");
        complete(c,EV_DONE); require(!armed_o[0], "third DONE releases the full WAITALL");
        entry(a,EV_DONE,0,1); absent(b); absent(c);
      end
      lo=owner_for(2,0,0); hi=owner_for(1,0,1);
      command(CMD_START,hi); command(CMD_DRAIN,hi); offered(0,hi,TS_DONE,seed_seq); tick();
      command(CMD_START,lo); command(CMD_DRAIN,lo); repeat(3) tick();
      offered(0,hi,TS_DONE,seed_seq); take_msg();
      offered(0,lo,TS_DONE,seed_seq); take_msg(); command(CMD_RETIRE,hi); command(CMD_RETIRE,lo);
      command(CMD_FINISH,hi); offered(1,hi,TS_DONE,0); tick(); command(CMD_FINISH,lo);
      repeat(3) tick();
      offered(1,hi,TS_DONE,0); take_msg(); offered(1,lo,TS_DONE,0); take_msg();
      require(active_o=='0 && completions==2, "stalled peer never loses its selection or identity");
    end
    // Counter-width stress above is a unit geometry, not a product counter budget.
    $display("C033_ARBITRATION sources=%0d contexts=%0d",$countones(seen),CONTEXT_COUNT);
  endtask

  initial begin
    int fd, scanned;
    if (!$value$plusargs("VECTORS=%s",fixture)) $fatal(1,"missing scenario vectors");
    fd=$fopen(fixture,"r"); if(fd==0) $fatal(1,"cannot open %s",fixture);
    for (int n=0; n<9; n++) begin
      case_name=$sformatf("U2-%c",8'(97+n));
      scanned=$fscanf(fd,"%h %h %h %h %h %h %h %h %h %h %h %h",seed_id,seed_gen,seed_seq,seed_uses,
                      wanted[0],wanted[1],wanted[2],wanted[3],wanted[4],wanted[5],wanted[6],wanted[7]);
      if(scanned!=12) $fatal(1,"incomplete vector for %s",case_name);
      cold_reset();
      case(n)
        0: event_first(); 1: same_cycle(); 2: after_park(); 3: waitall_cases(); 4: full_key();
        5: duplicate_phase(); 6: legal_phases(); 7: held_terminal(); 8: retire_vs_cancel();
      endcase
      query_key_i=key(); #1;
      require(word_t'($countones(armed_o))==wanted[0] && word_t'(successes)==wanted[1] &&
              word_t'(failures)==wanted[2], "scenario WAIT verdict totals");
      require(query_o.consumer_refs==wanted[3] && query_o.future_uses==wanted[4], "scenario final refcounts");
      require(word_t'(phases)==wanted[5] && word_t'(retire_ready)==wanted[6] &&
              word_t'(completions)==wanted[7], "scenario exact reliable message counts");
      $display("C033_CASE %s",case_name);
    end
    $fclose(fd); case_name="sharing"; cold_reset(); arbitration_and_contexts();
    case_name="phase-cutline"; cold_reset(); phase_cutline();
    $display("C033_PASS %s contexts=%0d checks=%0d",BENCH_NAME,CONTEXT_COUNT,checks); $finish;
  end
endmodule

// Two is a legal WAITALL maximum. Storing a count of two needs two bits.
module tb_c033_min #(parameter int unsigned CONTEXT_COUNT=1);
  tb_c033 #(.CONTEXT_COUNT(CONTEXT_COUNT),.OPERANDS(2),.BENCH_NAME("tb_c033_min")) bench();
endmodule
