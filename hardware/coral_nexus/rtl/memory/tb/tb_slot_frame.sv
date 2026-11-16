// C029 Frame behavior: streamed immutable metadata, lifetime reuse, exact
// handle checks, real shadow lookup, input/output backpressure and work bounds.
`timescale 1ns/1ps
module tb_slot_frame #(
  parameter int CONTEXT_COUNT = c029_params_pkg::C2_CONTEXTS
) ();
  localparam int BANKS = c029_params_pkg::BANK_COUNT;
  localparam int SLOTS = c029_params_pkg::SLOT_COUNT;
  localparam int BYTE_AW = $clog2(c029_params_pkg::BANK_BYTES) + 1;
  localparam int ADDR_AW = $clog2(BANKS * c029_params_pkg::BANK_BYTES) + 1;
  localparam int SLOT_AW = (SLOTS <= 1) ? 1 : $clog2(SLOTS);
  localparam int STRIPE = 4096; // authored layout alignment, not a resource capacity
  localparam int START = ((c029_params_pkg::SYSTEM_RESERVED_BYTES + STRIPE - 1) / STRIPE) * STRIPE;
  localparam int ROUND_BYTES = BANKS * STRIPE;
  localparam logic [2:0] PREPARE = 0, COMMIT = 1, BIND = 2, FREE = 3, RELEASE = 4, CHECK = 5;
  localparam logic [4:0] OK = 0, STATE = 1, IDENTITY = 2, SLOT_RANGE = 3,
    SLOT_EMPTY = 4, SLOT_BUSY = 5, LAYOUT = 6, ABI = 7, BAD_RECORD = 8,
    OVERLAP = 9, BANK_POLICY = 10, NO_PREPARE = 12, RECORD_RANGE = 13;

  logic clk = 0, rst_n = 0, cmd_valid = 0, cmd_ready, rec_valid = 0, rec_ready;
  logic [2:0] h_op = 0, h_profile_idx = 0;
  logic [63:0] h_ctx_generation = 0, h_ticket = 0, h_profile_generation = 0;
  logic [63:0] h_owner = 0, h_view_alloc_id = 0, h_arena_alloc_gen = 0, h_layout_count = 0;
  logic [ADDR_AW-1:0] h_base_addr = 0, h_logical = 0, h_arena_offset = 0;
  logic [ADDR_AW-1:0] h_layout_logical = 0, h_layout_offset = 0, h_reserved_bytes = 0;
  logic [SLOT_AW:0] h_slot = 0, rec_slot = 0, lookup_slot_i = 0;
  logic [BYTE_AW-1:0] h_alignment = BYTE_AW'(STRIPE), h_stripe = BYTE_AW'(STRIPE);
  logic [ADDR_AW-1:0] rec_arena_offset = 0, rec_logical = 0;
  logic [BANKS-1:0] res_valid_flat = 0;
  logic [BANKS*BYTE_AW-1:0] res_start_flat = 0, res_size_flat = 0;
  logic rsp_valid, rsp_ready = 0;
  logic [4:0] rsp_code, fault_code_o;
  logic [63:0] rsp_latency, rsp_gen, arena_generation_o;
  logic [3:0] state_o;
  logic [SLOTS-1:0] staged_mask_o, inst_mask_o, live_mask_o, shadow_mask_o;
  logic prepared_o, dbg_invariant_error_o, lookup_valid_o;
  logic [63:0] stat_permission_faults_o, stat_view_binds_o, stat_view_frees_o;
  logic [63:0] stat_prepare_max_lat_o, stat_last_prepare_lat_o;
  logic [63:0] lookup_alloc_o, lookup_owner_o, lookup_arena_gen_o;
  logic [63:0] lookup_context_gen_o, lookup_profile_gen_o;
  logic [ADDR_AW-1:0] lookup_base_o, lookup_size_o;
  logic preserve_lookup = 0;
  longint unsigned cycles = 0, accepted_cycle = 0, checks = 0, last_latency = 0;

  slot_frame dut (
    .clk(clk), .rst_n(rst_n), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .h_op(h_op),
    .h_ctx_generation(h_ctx_generation), .h_ticket(h_ticket),
    .h_profile_generation(h_profile_generation), .h_owner(h_owner),
    .h_view_alloc_id(h_view_alloc_id), .h_arena_alloc_gen(h_arena_alloc_gen),
    .h_base_addr(h_base_addr), .h_logical(h_logical), .h_arena_offset(h_arena_offset),
    .h_slot(h_slot), .h_layout_logical(h_layout_logical), .h_layout_offset(h_layout_offset),
    .h_alignment(h_alignment), .h_stripe(h_stripe), .h_reserved_bytes(h_reserved_bytes),
    .h_layout_count(h_layout_count), .h_profile_idx(h_profile_idx),
    .rec_valid(rec_valid), .rec_ready(rec_ready), .rec_slot(rec_slot),
    .rec_arena_offset(rec_arena_offset), .rec_logical(rec_logical),
    .res_valid_flat(res_valid_flat), .res_start_flat(res_start_flat), .res_size_flat(res_size_flat),
    .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_code(rsp_code),
    .rsp_latency(rsp_latency), .rsp_gen(rsp_gen), .state_o(state_o), .fault_code_o(fault_code_o),
    .staged_mask_o(staged_mask_o), .inst_mask_o(inst_mask_o), .live_mask_o(live_mask_o),
    .shadow_mask_o(shadow_mask_o), .prepared_o(prepared_o),
    .stat_permission_faults_o(stat_permission_faults_o), .stat_view_binds_o(stat_view_binds_o),
    .stat_view_frees_o(stat_view_frees_o), .stat_prepare_max_lat_o(stat_prepare_max_lat_o),
    .stat_last_prepare_lat_o(stat_last_prepare_lat_o), .dbg_invariant_error_o(dbg_invariant_error_o),
    .lookup_slot_i(lookup_slot_i), .lookup_valid_o(lookup_valid_o), .lookup_alloc_o(lookup_alloc_o),
    .lookup_owner_o(lookup_owner_o), .lookup_arena_gen_o(lookup_arena_gen_o),
    .lookup_context_gen_o(lookup_context_gen_o), .lookup_profile_gen_o(lookup_profile_gen_o),
    .lookup_base_o(lookup_base_o), .lookup_size_o(lookup_size_o),
    .arena_generation_o(arena_generation_o)
  );
  always #5 clk = ~clk;
  always @(posedge clk) cycles <= cycles + 1;
  initial begin #5_000_000; $fatal(1, "C029_ASSERT Frame progress watchdog"); end

  task automatic require(input logic condition, input string label);
    checks++;
    if (condition !== 1'b1) $fatal(1, "C029_ASSERT tb_slot_frame %s", label);
  endtask

  task automatic header(input logic [2:0] op);
    @(negedge clk); h_op = op; cmd_valid = 1;
    #1;
    while (!cmd_ready) begin @(negedge clk); #1; end
    @(posedge clk); #1;
    accepted_cycle = cycles;
    cmd_valid = 0;
  endtask

  task automatic record(input int slot, input int offset, input int bytes);
    @(negedge clk);
    rec_slot = (SLOT_AW+1)'(slot);
    rec_arena_offset = ADDR_AW'(offset);
    rec_logical = ADDR_AW'(bytes);
    rec_valid = 1;
    #1;
    while (!rec_ready) begin @(negedge clk); #1; end
    @(posedge clk); #1; rec_valid = 0;
  endtask

  task automatic response(input logic [4:0] expected, input string label);
    logic [63:0] saved_latency, saved_generation;
    #1;
    while (!rsp_valid) begin
      if (preserve_lookup) require(lookup_valid_o, "existing shadow stays usable during bind checks");
      @(posedge clk); #1;
    end
    require(rsp_code == expected, {label, ": response code"});
    saved_latency = rsp_latency; saved_generation = rsp_gen;
    last_latency = rsp_latency;
    require(rsp_latency == cycles - accepted_cycle + 1, {label, ": measured inclusive latency"});
    repeat (3) begin
      @(posedge clk); #1;
      require(rsp_valid && rsp_code == expected && rsp_gen == saved_generation &&
              rsp_latency == saved_latency, {label, ": stable held response"});
      require(!cmd_ready, "unconsumed response blocks overwrite");
    end
    @(negedge clk); rsp_ready = 1;
    @(posedge clk); #1; rsp_ready = 0;
    require(!rsp_valid, "response consumed exactly once");
  endtask

  task automatic view(
    input int slot, input logic [63:0] id, input int bytes, input int offset, input int base
  );
    h_slot = (SLOT_AW+1)'(slot); h_view_alloc_id = id;
    h_logical = ADDR_AW'(bytes); h_arena_offset = ADDR_AW'(offset);
    h_layout_logical = ADDR_AW'(bytes); h_layout_offset = ADDR_AW'(offset);
    h_base_addr = ADDR_AW'(base);
  endtask

  task automatic lookup(
    input int slot, input logic [63:0] id, input int bytes, input int base
  );
    lookup_slot_i = (SLOT_AW+1)'(slot); #1;
    require(lookup_valid_o && lookup_alloc_o == id && lookup_owner_o == h_owner &&
            lookup_arena_gen_o == h_arena_alloc_gen &&
            lookup_context_gen_o == h_ctx_generation &&
            lookup_profile_gen_o == h_profile_generation &&
            lookup_size_o == ADDR_AW'(bytes) && lookup_base_o == ADDR_AW'(base),
            "engine reads the exact committed shadow handle");
  endtask

  task automatic release_frame;
    header(RELEASE); response(OK, "release");
    require(live_mask_o == 0 && shadow_mask_o == 0 && inst_mask_o == 0 &&
            arena_generation_o == 0, "release clears all binding state");
  endtask

  initial begin
    repeat (3) begin @(posedge clk); #1; end
    rst_n = 1;
    for (int context_index = 0; context_index < CONTEXT_COUNT; context_index++) begin
      h_ctx_generation = 64'(context_index + 1);
      h_ticket = 64'(context_index + 20);
      h_profile_generation = 64'd7; h_owner = 64'd40;
      h_arena_alloc_gen = 64'(100 + context_index);
      h_reserved_bytes = ADDR_AW'(BANKS * 4 * STRIPE);
      h_layout_count = 64'(SLOTS + 3);
      res_valid_flat = '1;
      for (int b = 0; b < BANKS; b++) begin
        res_start_flat[b*BYTE_AW +: BYTE_AW] = BYTE_AW'(START);
        res_size_flat[b*BYTE_AW +: BYTE_AW] = BYTE_AW'(4 * STRIPE);
      end
      header(PREPARE);
      // More program buffers than physical live slots; repeated slot IDs and
      // overlapping templates are legal because their lifetimes do not overlap.
      for (int r = 0; r < SLOTS + 3; r++) begin
        if (r == 3) repeat (4) begin @(posedge clk); #1; end
        if (r == SLOTS + 1) record(1, 0, STRIPE);
        else if (r == SLOTS + 2) record(1, ROUND_BYTES, STRIPE);
        else record(0, 0, (r == 0) ? ROUND_BYTES : STRIPE);
      end
      response(OK, "streamed lifetime-reuse metadata");
      require(prepared_o && inst_mask_o == SLOTS'(3), "eligible slots, not buffer-count capacity");
      require(last_latency <= 64'(SLOTS) + 64'(BANKS) + 64'd11,
              "prepare work plus injected stalls bounded");
      h_ticket++;
      header(COMMIT); response(IDENTITY, "wrong commit ticket");
      h_ticket--;
      header(COMMIT); response(OK, "commit");
      require(arena_generation_o == h_arena_alloc_gen, "committed Arena generation");

      view(0, 101, ROUND_BYTES, 0, START);
      h_logical = ADDR_AW'(ROUND_BYTES - 1);
      header(BIND); response(LAYOUT, "handle size differs from immutable layout");
      h_logical = ADDR_AW'(ROUND_BYTES);
      h_arena_offset = ADDR_AW'(ROUND_BYTES);
      header(BIND); response(LAYOUT, "handle offset differs from immutable layout");
      h_arena_offset = 0;
      h_base_addr = ADDR_AW'(START + 1);
      header(BIND); response(BANK_POLICY, "wrong physical base");
      h_base_addr = ADDR_AW'(START);
      header(BIND); response(OK, "whole stripe round binds");
      lookup(0, 101, ROUND_BYTES, START);
      header(BIND); response(SLOT_BUSY, "live slot cannot bind twice");
      view(SLOTS, 999, STRIPE, 0, START);
      header(BIND); response(SLOT_RANGE, "out-of-range slot");
      view(2, 999, STRIPE, 0, START);
      header(BIND); response(LAYOUT, "undeclared slot");

      // Exact stripe-round boundary must not grow a spurious extra chunk.
      view(1, 102, STRIPE, ROUND_BYTES, START + STRIPE);
      lookup_slot_i = 0; preserve_lookup = 1;
      header(BIND); response(OK, "adjacent live view");
      preserve_lookup = 0;
      lookup(1, 102, STRIPE, START + STRIPE);
      header(FREE); response(OK, "free adjacent view");
      view(1, 103, STRIPE, 0, START);
      header(BIND); response(OVERLAP, "live overlap rejected");
      lookup(0, 101, ROUND_BYTES, START);

      view(0, 101, ROUND_BYTES, 0, START);
      h_view_alloc_id = 999;
      header(FREE); response(IDENTITY, "wrong allocation ID");
      h_view_alloc_id = 101; h_owner++;
      header(FREE); response(IDENTITY, "wrong view owner");
      h_owner--; h_profile_generation++;
      header(FREE); response(IDENTITY, "wrong profile generation");
      h_profile_generation--; h_ctx_generation++;
      header(FREE); response(IDENTITY, "wrong Context generation");
      h_ctx_generation--; h_arena_alloc_gen++;
      header(FREE); response(IDENTITY, "wrong Arena generation");
      h_arena_alloc_gen--; h_base_addr++;
      header(FREE); response(IDENTITY, "wrong base");
      h_base_addr--; h_logical--;
      header(FREE); response(IDENTITY, "wrong size");
      h_logical++;
      header(CHECK); response(OK, "read-only binding preflight");
      lookup(0, 101, ROUND_BYTES, START);
      header(FREE); response(OK, "checked free");
      #1; require(!lookup_valid_o && live_mask_o == 0 && shadow_mask_o == 0,
                  "free invalidates both copies without ending Frame");
      require(arena_generation_o == h_arena_alloc_gen, "free does not bump Frame generation");
      header(FREE); response(SLOT_EMPTY, "double free");

      // The same physical slot is reusable with a DIFFERENT buffer shape.
      view(0, 104, STRIPE, 0, START);
      header(BIND); response(OK, "different lifetime shape reuses slot");
      lookup(0, 104, STRIPE, START);
      view(0, 101, ROUND_BYTES, 0, START);
      header(FREE); response(IDENTITY, "old lifetime handle cannot free new view");
      lookup(0, 104, STRIPE, START);
      view(0, 104, STRIPE, 0, START);
      header(FREE); response(OK, "new lifetime free");
      release_frame();
    end

    // Arena geometry errors, malformed records and incomplete bind reject.
    h_layout_count = 0;
    res_start_flat[0 +: BYTE_AW] = BYTE_AW'(c029_params_pkg::SYSTEM_RESERVED_BYTES - 1);
    header(PREPARE); response(BANK_POLICY, "reserved floor violation");
    release_frame();
    res_start_flat[0 +: BYTE_AW] = BYTE_AW'(START);
    h_reserved_bytes--;
    header(PREPARE); response(BANK_POLICY, "reserve total mismatch");
    release_frame(); h_reserved_bytes++;
    h_alignment = 0;
    header(PREPARE); response(ABI, "zero alignment");
    release_frame(); h_alignment = BYTE_AW'(STRIPE);
    header(COMMIT); response(NO_PREPARE, "commit without prepared Frame");
    release_frame();
    h_layout_count = 1;
    header(PREPARE); record(SLOTS, 0, STRIPE);
    response(RECORD_RANGE, "layout slot outside ABI");
    release_frame();
    header(PREPARE); record(0, 1, STRIPE);
    response(BAD_RECORD, "non-round layout offset");
    release_frame();

    // Empty Arena: no live slots, but a real committed allocation identity.
    res_valid_flat = 0; res_start_flat = 0; res_size_flat = 0;
    h_layout_count = 0; h_reserved_bytes = 0; h_arena_alloc_gen = 999;
    header(PREPARE); response(OK, "empty prepare");
    view(0, 200, STRIPE, 0, START);
    header(BIND); response(STATE, "bind before authorization");
    header(COMMIT); response(OK, "empty commit");
    require(arena_generation_o == 999 && live_mask_o == 0, "empty Frame retains identity");
    header(BIND); response(LAYOUT, "empty Frame has no eligible view");
    release_frame();
    require(!dbg_invariant_error_o, "shadow invariants intact");
    $display("C029_PASS tb_slot_frame checks=%0d context_count=%0d max_prepare=%0d",
             checks, CONTEXT_COUNT, stat_prepare_max_lat_o);
    $finish;
  end
endmodule
