// tb_mshr_table.sv — C030 / W4 §4.4 regression for rtl/memory/mshr_table.sv.
//
// Scope: MSHR token lifetime only. The cache is not part of this slice, so
// "cache eviction cannot destroy waiter data" is shown here as the absence of
// any cache dependency (the bytes are served from the entry-owned fill buffer);
// the end-to-end eviction case needs cache.sv and is NOT claimed by this bench.
//
// Scenarios:
//   S1  two waiters merge into one leader; fill (two beats, byte masks) is
//       retained in the entry; the table is then driven to capacity so the
//       only way to make progress would be to steal the live slot; every
//       allocation stalls with FULL and the slot is never reused while
//       (a) a waiter has not consumed, then (b) a delivered waiter has not
//       closed its token reference.
//   S2  after both deliveries AND both reference releases the slot frees; the
//       reused slot carries a fresh entry generation, zeroed fill buffer, and
//       the old (slot, generation) token is rejected without exposing bytes.
//   S3  profile generations: no cross-generation merge, stale allocate/refill
//       rejected and counted, reconfigure only when idle and strictly forward.
//   S4  waiter bound and cancel: merge beyond WAITERS is refused (never
//       dropped); a cancelled entry answers every registered waiter with a
//       terminal no-data response and frees only after all releases.
//   S5  command/response protocol: response held under backpressure, no new
//       command accepted until it is consumed.
`timescale 1ns/1ps
module tb_mshr_table
  import c029_params_pkg::*;
#(
  parameter int CONTEXT_COUNT = C2_CONTEXTS
) ();
  localparam int E = MSHR_ENTRIES;
  localparam int LB = LINE_BYTES;
  localparam int W = 4;
  localparam int SW = $clog2(E);
  localparam int WW = $clog2(W);

  localparam logic [2:0] OP_ALLOC = 3'd0, OP_ISSUE = 3'd1, OP_FILL = 3'd2, OP_CONSUME = 3'd3,
                         OP_RELEASE = 3'd4, OP_CANCEL = 3'd5, OP_RECONFIG = 3'd6, OP_TERMINAL = 3'd7;
  localparam logic [3:0] R_OK = 4'd0, R_LEADER = 4'd1, R_MERGED = 4'd2, R_FULL = 4'd3,
                         R_WAITERS_FULL = 4'd4, R_STALE_PROFILE = 4'd5, R_BAD_TOKEN = 4'd6,
                         R_BAD_STATE = 4'd7, R_NOT_FILLED = 4'd8, R_CANCELLED = 4'd9,
                         R_NOT_DELIVERED = 4'd10, R_ALREADY = 4'd11, R_ACTIVE = 4'd13;

  logic clk = 0, rst_n = 0;
  logic cmd_valid = 0, cmd_ready;
  logic [2:0] cmd_op = 0;
  logic [63:0] cmd_key = 0;
  logic [31:0] cmd_profile_gen = 0;
  logic [SW-1:0] cmd_slot = 0;
  logic [31:0] cmd_entry_gen = 0;
  logic [WW-1:0] cmd_waiter = 0;
  logic [LB*8-1:0] cmd_data = 0;
  logic [LB-1:0] cmd_mask = 0;
  logic cmd_last = 0;
  logic rsp_valid, rsp_ready = 1;
  logic [3:0] rsp_code;
  logic [SW-1:0] rsp_slot;
  logic [31:0] rsp_entry_gen;
  logic [WW-1:0] rsp_waiter;
  logic [LB*8-1:0] rsp_data;
  logic [LB-1:0] rsp_mask;
  logic [$clog2(E+1)-1:0] active_o;
  logic [63:0] merged_o, stalls_o, stale_rejects_o;
  logic [31:0] profile_gen_o;
  logic [E-1:0] entry_busy_o;
  logic invariant_error_o;

  mshr_table #(.ENTRIES(E), .LINE_B(LB), .WAITERS(W)) dut (.*);

  always #5 clk = ~clk;
  initial begin #5_000_000; $fatal(1, "C030_ASSERT mshr watchdog"); end

  int checks = 0;
  task automatic require(input logic ok, input string label);
    checks++;
    if (ok !== 1'b1) $fatal(1, "C030_ASSERT tb_mshr_table %s", label);
  endtask

  // Issue one command and wait for its registered response (rsp_ready high).
  task automatic cmd(input logic [2:0] op, input logic [63:0] key, input logic [31:0] pg,
                     input logic [SW-1:0] slot, input logic [31:0] eg, input logic [WW-1:0] wt,
                     input logic [LB*8-1:0] data, input logic [LB-1:0] mask, input logic last,
                     input logic [3:0] expect_code, input string label);
    @(negedge clk);
    cmd_op = op; cmd_key = key; cmd_profile_gen = pg; cmd_slot = slot; cmd_entry_gen = eg;
    cmd_waiter = wt; cmd_data = data; cmd_mask = mask; cmd_last = last; cmd_valid = 1;
    require(cmd_ready, {label, ": ready"});
    @(negedge clk);
    cmd_valid = 0;
    require(rsp_valid, {label, ": response"});
    require(rsp_code == expect_code, {label, ": code"});
  endtask

  logic [SW-1:0] a_slot;
  logic [31:0] a_gen;
  logic [LB*8-1:0] pat;
  logic [LB-1:0] allm = '1;
  logic [SW-1:0] other_slot [E];
  logic [31:0] other_gen [E];

  function automatic logic [LB*8-1:0] pattern(input logic [7:0] seed);
    logic [LB*8-1:0] p;
    for (int b = 0; b < LB; b++) p[b*8 +: 8] = seed + 8'(b);
    return p;
  endfunction

  initial begin
    pat = pattern(8'h40);
    repeat (3) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    require(active_o == 0 && !invariant_error_o && profile_gen_o == 0, "reset state");

    // ---------------- S1: merge + retained fill + no early reuse -------------
    cmd(OP_ALLOC, 64'hA000, 0, 0, 0, 0, 0, 0, 0, R_LEADER, "leader alloc");
    a_slot = rsp_slot; a_gen = rsp_entry_gen;
    require(rsp_waiter == 0, "leader is waiter 0");
    cmd(OP_ALLOC, 64'hA000, 0, 0, 0, 0, 0, 0, 0, R_MERGED, "second waiter merges");
    require(rsp_slot == a_slot && rsp_entry_gen == a_gen && rsp_waiter == 1, "merge shares the entry token");
    require(merged_o == 1 && active_o == 1, "one entry, one merge");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_NOT_FILLED, "consume before fill refused");
    cmd(OP_FILL, 0, 0, a_slot, a_gen, 0, pat, allm, 0, R_BAD_STATE, "fill before issue refused");
    cmd(OP_ISSUE, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "issue refill");
    cmd(OP_FILL, 0, 0, a_slot, a_gen, 0, pat, {{(LB/2){1'b1}}, {(LB/2){1'b0}}}, 0, R_OK, "first beat (upper half)");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_NOT_FILLED, "partial fill not consumable");
    cmd(OP_FILL, 0, 0, a_slot, a_gen, 0, pat, {{(LB/2){1'b0}}, {(LB/2){1'b1}}}, 1, R_OK, "last beat (lower half)");

    // Table pressure: every other slot allocates; the filled entry is never a
    // reuse candidate, so allocation past capacity stalls.
    for (int i = 0; i < E - 1; i++) begin
      cmd(OP_ALLOC, 64'hB000 + 64'(i), 0, 0, 0, 0, 0, 0, 0, R_LEADER, "pressure alloc");
      require(rsp_slot != a_slot, "pressure never reuses the live slot");
      other_slot[i] = rsp_slot; other_gen[i] = rsp_entry_gen;
    end
    require(int'(active_o) == E, "table full");
    cmd(OP_ALLOC, 64'hC000, 0, 0, 0, 0, 0, 0, 0, R_FULL, "reuse attempt: full while waiters pending");
    require(stalls_o >= 1, "stall counted");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "waiter 0 receives data");
    require(rsp_data == pat && rsp_mask == allm, "retained fill bytes for waiter 0");
    cmd(OP_ALLOC, 64'hC000, 0, 0, 0, 0, 0, 0, 0, R_FULL, "reuse attempt: waiter 1 still undelivered");
    cmd(OP_RELEASE, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "waiter 0 releases");
    cmd(OP_ALLOC, 64'hC000, 0, 0, 0, 0, 0, 0, 0, R_FULL, "reuse attempt: delivery of waiter 1 pending");
    cmd(OP_RELEASE, 0, 0, a_slot, a_gen, 1, 0, 0, 0, R_NOT_DELIVERED, "release before delivery refused");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 1, 0, 0, 0, R_OK, "waiter 1 receives data later");
    require(rsp_data == pat && rsp_mask == allm, "retained fill bytes for the delayed waiter");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 1, 0, 0, 0, R_ALREADY, "double delivery refused");
    cmd(OP_ALLOC, 64'hC000, 0, 0, 0, 0, 0, 0, 0, R_FULL, "reuse attempt: waiter 1 reference still open");
    require(entry_busy_o[a_slot], "slot still busy until both references close");

    // Normal completion: the terminal ACK has NOT arrived, so closing both waiters
    // must not free the token even though every delivery and release is done.
    cmd(OP_RELEASE, 0, 0, a_slot, a_gen, 1, 0, 0, 0, R_OK, "waiter 1 releases");
    require(entry_busy_o[a_slot] && int'(active_o) == E, "all waiters closed but terminal unacked: slot busy");
    cmd(OP_ALLOC, 64'hC000, 0, 0, 0, 0, 0, 0, 0, R_FULL, "allocation pressure with an unacked terminal");
    cmd(OP_TERMINAL, 0, 0, a_slot, a_gen + 1, 0, 0, 0, 0, R_BAD_TOKEN, "terminal with wrong generation refused");
    require(entry_busy_o[a_slot], "wrong-generation terminal does not free a successful refill");
    cmd(OP_TERMINAL, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "terminal ACK for the successful refill");
    cmd(OP_TERMINAL, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_BAD_TOKEN, "duplicate terminal refused");

    // ---------------- S2: free after delivery, references AND terminal ACK ----------
    require(!entry_busy_o[a_slot] && int'(active_o) == E - 1, "slot freed only after the terminal ACK");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_BAD_TOKEN, "old token cannot read after free");
    cmd(OP_ALLOC, 64'hC000, 0, 0, 0, 0, 0, 0, 0, R_LEADER, "slot reused after full closure");
    require(rsp_slot == a_slot && rsp_entry_gen != a_gen, "fresh entry generation on reuse");
    begin
      automatic logic [31:0] new_gen = rsp_entry_gen;
      cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_BAD_TOKEN, "stale token vs reused slot");
      require(rsp_data == '0 && rsp_mask == '0, "stale token exposes no bytes");
      cmd(OP_ISSUE, 0, 0, a_slot, new_gen, 0, 0, 0, 0, R_OK, "new owner issues");
      cmd(OP_FILL, 0, 0, a_slot, new_gen, 0, pattern(8'h90), allm, 1, R_OK, "new owner fills");
      cmd(OP_TERMINAL, 0, 0, a_slot, new_gen, 0, 0, 0, 0, R_OK, "new owner terminal ACK");
      cmd(OP_CONSUME, 0, 0, a_slot, new_gen, 0, 0, 0, 0, R_OK, "new owner consumes");
      require(rsp_data == pattern(8'h90) && rsp_mask == allm, "new owner sees only its own bytes");
      cmd(OP_RELEASE, 0, 0, a_slot, new_gen, 0, 0, 0, 0, R_OK, "new owner releases");
    end
    // Drain the pressure entries (fill, consume, release) back to idle.
    for (int i = 0; i < E - 1; i++) begin
      cmd(OP_ISSUE, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain issue");
      cmd(OP_FILL, 0, 0, other_slot[i], other_gen[i], 0, pattern(8'(i)), allm, 1, R_OK, "drain fill");
      cmd(OP_TERMINAL, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain terminal ACK");
      cmd(OP_CONSUME, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain consume");
      cmd(OP_RELEASE, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain release");
    end
    require(active_o == 0, "idle after drain");

    // ---------------- S3: profile generations --------------------------------
    cmd(OP_ALLOC, 64'hD000, 0, 0, 0, 0, 0, 0, 0, R_LEADER, "gen0 alloc");
    a_slot = rsp_slot; a_gen = rsp_entry_gen;
    cmd(OP_RECONFIG, 0, 1, 0, 0, 0, 0, 0, 0, R_ACTIVE, "reconfigure refused while active");
    cmd(OP_ALLOC, 64'hD000, 1, 0, 0, 0, 0, 0, 0, R_STALE_PROFILE, "alloc carrying future generation refused");
    require(stale_rejects_o == 1, "stale alloc counted");
    cmd(OP_ISSUE, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen0 issue");
    cmd(OP_FILL, 0, 0, a_slot, a_gen, 0, pat, allm, 1, R_OK, "gen0 fill");
    cmd(OP_TERMINAL, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen0 terminal ACK");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen0 consume");
    cmd(OP_RELEASE, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen0 release");
    cmd(OP_RECONFIG, 0, 0, 0, 0, 0, 0, 0, 0, R_STALE_PROFILE, "generation must advance");
    cmd(OP_RECONFIG, 0, 1, 0, 0, 0, 0, 0, 0, R_OK, "reconfigure when idle");
    require(profile_gen_o == 1, "generation advanced");
    cmd(OP_ALLOC, 64'hD000, 0, 0, 0, 0, 0, 0, 0, R_STALE_PROFILE, "old-generation alloc rejected");
    cmd(OP_ALLOC, 64'hD000, 1, 0, 0, 0, 0, 0, 0, R_LEADER, "same key allocates fresh in gen1");
    a_slot = rsp_slot; a_gen = rsp_entry_gen;
    cmd(OP_ISSUE, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen1 issue");
    cmd(OP_FILL, 0, 0, a_slot, a_gen, 0, pat, allm, 1, R_OK, "gen1 fill");
    cmd(OP_TERMINAL, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen1 terminal ACK");
    cmd(OP_CONSUME, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen1 consume");
    cmd(OP_RELEASE, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "gen1 release");

    // ---------------- S4: waiter bound and cancel ----------------------------
    cmd(OP_ALLOC, 64'hE000, 1, 0, 0, 0, 0, 0, 0, R_LEADER, "cancel leader");
    a_slot = rsp_slot; a_gen = rsp_entry_gen;
    for (int w = 1; w < W; w++) cmd(OP_ALLOC, 64'hE000, 1, 0, 0, 0, 0, 0, 0, R_MERGED, "fill waiter slots");
    cmd(OP_ALLOC, 64'hE000, 1, 0, 0, 0, 0, 0, 0, R_WAITERS_FULL, "merge beyond waiter bound refused");
    cmd(OP_ISSUE, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "cancel issue");
    cmd(OP_CANCEL, 0, 1, a_slot, a_gen, 0, 0, 0, 0, R_OK, "cancel");
    cmd(OP_FILL, 0, 0, a_slot, a_gen, 0, pat, allm, 1, R_BAD_STATE, "late fill into cancelled entry refused");
    cmd(OP_ALLOC, 64'hE000, 1, 0, 0, 0, 0, 0, 0, R_LEADER, "cancelled entry no longer merges");
    require(rsp_slot != a_slot, "new leader gets a different slot");
    begin
      automatic logic [SW-1:0] n_slot = rsp_slot; automatic logic [31:0] n_gen = rsp_entry_gen;
      for (int w = 0; w < W; w++) begin
        cmd(OP_CONSUME, 0, 0, a_slot, a_gen, WW'(w), 0, 0, 0, R_CANCELLED, "terminal cancelled response");
        require(rsp_data == '0 && rsp_mask == '0, "cancelled response carries no data");
        require(entry_busy_o[a_slot], "cancelled entry busy until all releases");
        cmd(OP_RELEASE, 0, 0, a_slot, a_gen, WW'(w), 0, 0, 0, R_OK, "cancelled release");
      end
      // ISSUE -> CANCEL -> consume/release has settled every WAITER reference but
      // the accepted refill is still outstanding: the slot must stay busy and
      // must not be reallocated until the delayed terminal ACK arrives.
      require(entry_busy_o[a_slot], "cancelled issued refill stays busy after all releases");
      cmd(OP_TERMINAL, 0, 0, a_slot, a_gen + 1, 0, 0, 0, 0, R_BAD_TOKEN, "terminal with wrong generation refused");
      require(entry_busy_o[a_slot], "wrong-generation terminal does not free");
      cmd(OP_FILL, 0, 0, a_slot, a_gen, 0, pat, allm, 1, R_BAD_STATE, "late refill data after cancel refused");
      require(entry_busy_o[a_slot], "late data does not settle the refill");
      for (int i = 0; i < E - 2; i++) begin
        cmd(OP_ALLOC, 64'hF100 + 64'(i), 1, 0, 0, 0, 0, 0, 0, R_LEADER, "pressure around the unsettled slot");
        require(rsp_slot != a_slot, "unsettled slot never reallocated");
        other_slot[i] = rsp_slot; other_gen[i] = rsp_entry_gen;
      end
      cmd(OP_ALLOC, 64'hF1FF, 1, 0, 0, 0, 0, 0, 0, R_FULL, "table full while one refill is unsettled");
      cmd(OP_TERMINAL, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_OK, "delayed terminal/isolation ACK");
      require(!entry_busy_o[a_slot], "freed only after terminal ACK");
      cmd(OP_TERMINAL, 0, 0, a_slot, a_gen, 0, 0, 0, 0, R_BAD_TOKEN, "duplicate terminal refused");
      cmd(OP_ALLOC, 64'hF1FF, 1, 0, 0, 0, 0, 0, 0, R_LEADER, "slot reusable after terminal ACK");
      require(rsp_slot == a_slot && rsp_entry_gen != a_gen, "reuse carries a fresh generation");
      begin
        automatic logic [SW-1:0] r_slot = rsp_slot; automatic logic [31:0] r_gen = rsp_entry_gen;
        for (int i = 0; i < E - 2; i++) begin
          cmd(OP_ISSUE, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain issue");
          cmd(OP_FILL, 0, 0, other_slot[i], other_gen[i], 0, pat, allm, 1, R_OK, "drain fill");
          cmd(OP_TERMINAL, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain terminal ACK");
          cmd(OP_CONSUME, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain consume");
          cmd(OP_RELEASE, 0, 0, other_slot[i], other_gen[i], 0, 0, 0, 0, R_OK, "drain release");
        end
        cmd(OP_ISSUE, 0, 0, r_slot, r_gen, 0, 0, 0, 0, R_OK, "reused slot issue");
        cmd(OP_FILL, 0, 0, r_slot, r_gen, 0, pat, allm, 1, R_OK, "reused slot fill");
        cmd(OP_TERMINAL, 0, 0, r_slot, r_gen, 0, 0, 0, 0, R_OK, "reused slot terminal ACK");
        cmd(OP_CONSUME, 0, 0, r_slot, r_gen, 0, 0, 0, 0, R_OK, "reused slot consume");
        cmd(OP_RELEASE, 0, 0, r_slot, r_gen, 0, 0, 0, 0, R_OK, "reused slot release");
      end
      cmd(OP_ISSUE, 0, 0, n_slot, n_gen, 0, 0, 0, 0, R_OK, "survivor issue");
      cmd(OP_FILL, 0, 0, n_slot, n_gen, 0, pat, allm, 1, R_OK, "survivor fill");
      cmd(OP_TERMINAL, 0, 0, n_slot, n_gen, 0, 0, 0, 0, R_OK, "survivor terminal ACK");
      cmd(OP_CONSUME, 0, 0, n_slot, n_gen, 0, 0, 0, 0, R_OK, "survivor consume");
      cmd(OP_RELEASE, 0, 0, n_slot, n_gen, 0, 0, 0, 0, R_OK, "survivor release");
    end
    require(active_o == 0, "idle at end");

    // ---------------- S5: response backpressure ------------------------------
    @(negedge clk);
    require(!rsp_valid && cmd_ready, "previous response drained before backpressure case");
    rsp_ready = 0;
    cmd_op = OP_ALLOC; cmd_key = 64'hF000; cmd_profile_gen = 1; cmd_valid = 1;
    @(negedge clk); cmd_valid = 0;
    require(rsp_valid && rsp_code == R_LEADER, "response pending");
    begin
      automatic logic [SW-1:0] held_slot = rsp_slot; automatic logic [31:0] held_gen = rsp_entry_gen;
      repeat (3) begin
        @(negedge clk);
        require(rsp_valid && !cmd_ready && rsp_slot == held_slot && rsp_entry_gen == held_gen,
                "response stable, no command accepted");
      end
      rsp_ready = 1;
      @(negedge clk);
      require(!rsp_valid && cmd_ready, "response consumed once");
    end
    require(!invariant_error_o, "no internal invariant error");
    $display("C030_PASS tb_mshr_table checks=%0d entries=%0d line_bytes=%0d", checks, E, LB);
    $finish;
  end
endmodule
