// C029 L1 authority: one exact extent per bank/Context and one real SlotFrame
// per Context. Free space is the complement of that bounded reservation table:
// first-fit skips at most CONTEXT_COUNT occupied intervals, without a second
// free-list/counter authority. HOLD publishes all banks atomically.
//
// COMMIT_FINAL is irrevocable. RETIRE requires local reference closure and a
// generation-qualified execution-drain handshake. Reclamation waits for real
// bank validity-clear ACKs. Completion is retained independently of parent R/L2.
// VIEW_BIND consumes immutable compiler layout fields and creates a fresh
// allocation identity itself, as ArenaPool._view_counter does. The live Slot
// count is never used as a limit on streamed program metadata records.
//
// PIN/INFLIGHT consume exactly-once events from their producer; transaction
// de-duplication stays with that producer. IDs/counters are private transport
// storage, not silicon-width claims. Profile sequencing belongs to C058.
`timescale 1ns/1ps
module arena_ledger #(
  parameter int CONTEXT_COUNT = c029_params_pkg::C2_CONTEXTS,
  parameter int ID_W = 64,
  parameter int ACTIVE_PROFILE_IDX = 0,
  localparam int BANKS = c029_params_pkg::BANK_COUNT,
  localparam int SLOTS = c029_params_pkg::SLOT_COUNT,
  localparam int BYTE_AW = $clog2(c029_params_pkg::BANK_BYTES) + 1,
  localparam int ADDR_AW = $clog2(BANKS * c029_params_pkg::BANK_BYTES) + 1,
  localparam int SLOT_AW = (SLOTS <= 1) ? 1 : $clog2(SLOTS),
  localparam int CTX_AW = (CONTEXT_COUNT <= 1) ? 1 : $clog2(CONTEXT_COUNT),
  localparam int BANK_AW = (BANKS <= 1) ? 1 : $clog2(BANKS)
) (
  input logic clk, rst_n,
  input logic cmd_valid,
  output logic cmd_ready,
  input logic [3:0] cmd_op,
  input logic [CTX_AW-1:0] cmd_ctx,
  input logic [ID_W-1:0] cmd_ctx_generation, cmd_ticket, cmd_profile_generation,
  input logic [ID_W-1:0] cmd_owner, cmd_view_alloc_id, cmd_arena_alloc_gen,
  input logic [ADDR_AW-1:0] cmd_reserved_bytes, cmd_logical, cmd_arena_offset, cmd_base_addr,
  input logic [BYTE_AW-1:0] cmd_alignment, cmd_stripe,
  input logic [BANK_AW:0] cmd_seg_count,
  input logic [63:0] cmd_layout_count,
  input logic [SLOTS-1:0] cmd_slot_mask,
  input logic [SLOT_AW:0] cmd_slot,
  input logic rec_valid,
  output logic rec_ready,
  input logic [BANK_AW:0] rec_bank,
  input logic [BYTE_AW-1:0] rec_size,
  input logic [SLOT_AW:0] rec_slot,
  input logic [ADDR_AW-1:0] rec_arena_offset, rec_logical,
  output logic rsp_valid,
  input logic rsp_ready,
  output logic [6:0] rsp_code,
  output logic [63:0] rsp_latency,
  output logic [ID_W-1:0] rsp_arena_gen, rsp_view_alloc_id,
  output logic [CTX_AW-1:0] rsp_ctx,
  input logic [ID_W-1:0] profile_generation_i,
  input logic [CONTEXT_COUNT-1:0] exec_drained,
  input logic [CONTEXT_COUNT*ID_W-1:0] exec_drained_generation_i,
  output logic [CONTEXT_COUNT-1:0] completion_valid, retire_ready,
  output logic [CONTEXT_COUNT*ID_W-1:0] completion_ctx_generation,
  output logic [CONTEXT_COUNT*ID_W-1:0] completion_arena_gen, completion_ticket,
  output logic [CONTEXT_COUNT-1:0] inv_valid_o,
  input logic [CONTEXT_COUNT-1:0] inv_ready_i, inv_ack_i, inv_rej_i,
  output logic [CONTEXT_COUNT*ID_W-1:0] inv_alloc_o, inv_gen_o,
  output logic [CONTEXT_COUNT*(BANK_AW+1)-1:0] inv_bank_o,
  output logic [CONTEXT_COUNT*BYTE_AW-1:0] inv_addr_o, inv_len_o,
  input logic [CONTEXT_COUNT*(SLOT_AW+1)-1:0] query_slot_i,
  output logic [CONTEXT_COUNT-1:0] query_valid_o,
  output logic [CONTEXT_COUNT*ID_W-1:0] query_alloc_o, query_owner_o, query_arena_gen_o,
  output logic [CONTEXT_COUNT*ID_W-1:0] query_context_gen_o, query_profile_gen_o,
  output logic [CONTEXT_COUNT*ADDR_AW-1:0] query_base_o, query_size_o,
  output logic [BANKS*BYTE_AW-1:0] dbg_held_bytes, dbg_bound_bytes, dbg_free_bytes,
  output logic [CONTEXT_COUNT*4-1:0] dbg_ctx_state, dbg_frame_state,
  output logic [CONTEXT_COUNT*SLOTS-1:0] dbg_frame_live_mask, dbg_frame_shadow_mask,
  output logic [CONTEXT_COUNT*BANKS-1:0] dbg_cres_valid,
  output logic [CONTEXT_COUNT*BANKS*BYTE_AW-1:0] dbg_cres_start, dbg_cres_size,
  output logic [CONTEXT_COUNT*SLOTS-1:0] dbg_view_live_mask, dbg_view_invalid_mask,
  output logic [BANKS-1:0] dbg_place_valid,
  output logic [BANKS*BYTE_AW-1:0] dbg_place_start,
  output logic [ID_W-1:0] dbg_arena_gen,
  output logic [4:0] dbg_frame_fault,
  output logic dbg_invariant_error
);
  localparam logic [3:0] HOLD = 0, PREPARE = 1, COMMIT = 2, COMPENSATE = 3,
    BIND = 4, FREE = 5, PIN = 6, UNPIN = 7, BEGIN_ACCESS = 8, END_ACCESS = 9,
    RETIRE = 10, COMPLETION_ACK = 11, FRAME_RELEASE = 12;
  localparam logic [6:0] OK = 0, COMPENSATED = 1, RETIRED = 2,
    BAD_CONTEXT = 3, DUP_TICKET = 4, BAD_STATE = 5, STALE = 6, BAD_PARAM = 7,
    BAD_BANK = 8, DUP_BANK = 9, BAD_SIZE = 10, PERMANENT = 11,
    WAIT_CAPACITY = 12, WAIT_FRAGMENTATION = 13, BAD_TOTAL = 14,
    NOT_HELD = 15, POST_AUTH_COMPENSATE = 16, NOT_PREPARED = 17,
    GENERATION_EXHAUSTED = 18, NOT_BOUND = 19, BAD_SLOT = 20, SLOT_BUSY = 21,
    VIEW_OVERLAP = 22, VIEW_ESCAPE = 23, VIEW_LAYOUT = 24, UNKNOWN_VIEW = 25,
    DOUBLE_FREE = 26, LIVE_REFS = 27, STALE_VIEW = 28, DEAD_VIEW = 29,
    NOT_DRAINED = 30, ALREADY_RETIRED = 31, BAD_COMPLETION = 32,
    PIN_OVERFLOW = 33, ACCESS_OVERFLOW = 34, REF_UNDERFLOW = 35,
    BAD_PROFILE = 36, BAD_COUNT = 37, FRAME_FAULT = 38, INVALIDATION_REJECTED = 39;
  localparam logic [2:0] F_PREPARE = 0, F_COMMIT = 1, F_BIND = 2, F_FREE = 3, F_RELEASE = 4;
  localparam logic [1:0] C_FREE = 0, C_HELD = 1, C_BOUND = 2, C_COMPLETE = 3;
  localparam logic [1:0] V_EMPTY = 0, V_LIVE = 1, V_INVALID = 2, V_CLOSING = 3;
  typedef enum logic [3:0] {
    IDLE, DISPATCH, HOLD_RECORDS, HOLD_SCAN, HOLD_APPLY, FRAME_SEND,
    FRAME_RECORDS, FRAME_WAIT, INVALIDATE_SEND, INVALIDATE_WAIT, RESPOND
  } state_t;
  localparam int SPM = c029_params_pkg::SPM_BYTES_PER_BANK[ACTIVE_PROFILE_IDX];
  localparam int RESERVED = c029_params_pkg::SYSTEM_RESERVED_BYTES;
  localparam int USER_CAPACITY = SPM - RESERVED;
  localparam int CLEAR_WORDS = c029_params_pkg::BANK_BYTES / c029_params_pkg::DATA_BYTES;

  typedef struct packed {
    logic [3:0] op;
    logic [CTX_AW-1:0] ctx;
    logic [ID_W-1:0] context_gen, ticket, profile_gen, owner, allocation_id, arena_gen;
    logic [ADDR_AW-1:0] reserved, logical_bytes, offset, base;
    logic [BYTE_AW-1:0] alignment, stripe;
    logic [BANK_AW:0] segments;
    logic [63:0] layouts;
    logic [SLOTS-1:0] slots;
    logic [SLOT_AW:0] slot;
  } command_t;
  typedef struct packed {
    logic [ID_W-1:0] allocation_id, owner;
    logic [ADDR_AW-1:0] base, bytes, offset;
    logic [63:0] pins, accesses;
  } view_t;
  command_t request;
  state_t state;
  logic [1:0] context_state [CONTEXT_COUNT], view_state [CONTEXT_COUNT][SLOTS];
  logic context_seen [CONTEXT_COUNT];
  logic [ID_W-1:0] context_gen [CONTEXT_COUNT], tickets [CONTEXT_COUNT];
  logic [ID_W-1:0] profiles [CONTEXT_COUNT], arenas [CONTEXT_COUNT];
  logic [BYTE_AW-1:0] alignments [CONTEXT_COUNT], stripes [CONTEXT_COUNT];
  logic [ADDR_AW-1:0] reserved_totals [CONTEXT_COUNT];
  logic [SLOTS-1:0] slot_masks [CONTEXT_COUNT];
  logic [BANKS*BYTE_AW-1:0] reserve_start [CONTEXT_COUNT], reserve_size [CONTEXT_COUNT];
  logic [BANKS-1:0] reserve_valid [CONTEXT_COUNT];
  view_t views [CONTEXT_COUNT][SLOTS];
  logic [ID_W-1:0] arena_serial, view_serial, proposed_view;
  logic [BANKS-1:0] requested_banks;
  logic [BYTE_AW-1:0] requested_sizes [BANKS], placements [BANKS];
  logic [63:0] records_seen, requested_total, elapsed, outside_wait;
  logic [6:0] hold_error;
  int unsigned bank_cursor;
  logic [2:0] frame_op;
  logic frame_record_error, abort_prepare, clear_arena;
  logic [ADDR_AW-1:0] new_view_base;
  logic [CONTEXT_COUNT-1:0] f_cmd_ready, f_rec_ready, f_rsp_valid, f_prepared, f_query_valid;
  logic [CONTEXT_COUNT*5-1:0] f_rsp_code;
  logic [CONTEXT_COUNT-1:0] f_invariant_error;
  logic [63:0] clear_start, clear_length;

  initial begin
    assert (CONTEXT_COUNT == c029_params_pkg::C1_CONTEXTS ||
            CONTEXT_COUNT == c029_params_pkg::C2_CONTEXTS)
      else $fatal(1, "C029_ASSERT ledger Context count is not a sourced target");
    assert (ACTIVE_PROFILE_IDX >= 0 && ACTIVE_PROFILE_IDX < c029_params_pkg::PROFILE_COUNT)
      else $fatal(1, "C029_ASSERT ledger profile index");
  end

  function automatic logic [63:0] align_up(input logic [63:0] value, quantum);
    return ((value + quantum - 64'd1) / quantum) * quantum;
  endfunction
  function automatic logic [63:0] view_span(
    input logic [ADDR_AW-1:0] bytes, input int unsigned bank,
    input logic [BYTE_AW-1:0] stripe
  );
    logic [63:0] chunks, count, tail_bytes;
    chunks = (64'(bytes) + 64'(stripe) - 64'd1) / 64'(stripe);
    if (64'(bank) >= chunks) return 64'd0;
    count = 64'd1 + (chunks - 64'd1 - 64'(bank)) / 64'(BANKS);
    tail_bytes = (64'(bytes) - 64'd1) % 64'(stripe) + 64'd1;
    return count * 64'(stripe) -
      (((chunks - 64'd1) % 64'(BANKS) == 64'(bank)) ? 64'(stripe) - tail_bytes : 64'd0);
  endfunction
  function automatic logic owner_matches;
    return context_gen[request.ctx] == request.context_gen &&
      tickets[request.ctx] == request.ticket && profiles[request.ctx] == request.profile_gen;
  endfunction
  function automatic logic allocation_matches(input view_t value);
    return value.allocation_id == request.allocation_id && value.owner == request.owner &&
      value.base == request.base && value.bytes == request.logical_bytes &&
      arenas[request.ctx] == request.arena_gen;
  endfunction
  function automatic logic context_holds(input int c);
    return context_state[CTX_AW'(c)] == C_HELD || context_state[CTX_AW'(c)] == C_BOUND;
  endfunction

  task automatic reply(
    input logic [6:0] code, input logic [ID_W-1:0] arena, allocation
  );
    logic [64:0] bound;
    case (request.op)
      HOLD: bound = 65'(request.segments) + 65'(BANKS) + 65'd8;
      PREPARE: bound = 65'(request.layouts) + 65'(BANKS) + 65'd16;
      BIND: bound = 65'(BANKS) + 65'd16;
      FREE, RETIRE: bound = 65'(BANKS) * (65'(CLEAR_WORDS) + 65'd6) + 65'd24;
      default: bound = 65'd16;
    endcase
    assert (65'(elapsed) + 65'd1 <= bound + 65'(outside_wait))
      else $error("C029_ASSERT ledger bounded local work excludes external backpressure");
    rsp_valid <= 1'b1; rsp_code <= code; rsp_latency <= elapsed + 64'd1;
    rsp_arena_gen <= arena; rsp_view_alloc_id <= allocation; rsp_ctx <= request.ctx;
    state <= RESPOND;
  endtask

  always_comb begin
    cmd_ready = state == IDLE && !rsp_valid;
    rec_ready = state == HOLD_RECORDS ||
      (state == FRAME_RECORDS && (frame_record_error || f_rsp_valid[request.ctx] ||
                                 f_rec_ready[request.ctx]));
    inv_valid_o = '0; inv_alloc_o = '0; inv_gen_o = '0;
    inv_bank_o = '0; inv_addr_o = '0; inv_len_o = '0;
    clear_start = '0; clear_length = '0;
    if (state == INVALIDATE_SEND || state == INVALIDATE_WAIT) begin
      clear_start = 64'(reserve_start[request.ctx][bank_cursor*BYTE_AW +: BYTE_AW]);
      if (clear_arena) begin
        clear_length = 64'(reserve_size[request.ctx][bank_cursor*BYTE_AW +: BYTE_AW]);
      end else begin
        clear_start += (64'(views[request.ctx][SLOT_AW'(request.slot)].offset) /
          (64'(stripes[request.ctx]) * 64'(BANKS))) * 64'(stripes[request.ctx]);
        clear_length = view_span(views[request.ctx][SLOT_AW'(request.slot)].bytes,
                                 bank_cursor, stripes[request.ctx]);
      end
      inv_valid_o[request.ctx] = state == INVALIDATE_SEND && clear_length != 0;
      inv_alloc_o[request.ctx*ID_W +: ID_W] = clear_arena ? '0 : request.allocation_id;
      inv_gen_o[request.ctx*ID_W +: ID_W] = arenas[request.ctx];
      inv_bank_o[request.ctx*(BANK_AW+1) +: BANK_AW+1] = (BANK_AW+1)'(bank_cursor);
      inv_addr_o[request.ctx*BYTE_AW +: BYTE_AW] = BYTE_AW'(clear_start);
      inv_len_o[request.ctx*BYTE_AW +: BYTE_AW] = BYTE_AW'(clear_length);
    end
  end

  generate
    for (genvar c = 0; c < CONTEXT_COUNT; c++) begin : frames
      slot_frame #(.ID_W(ID_W)) frame (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(state == FRAME_SEND && request.ctx == CTX_AW'(c)),
        .cmd_ready(f_cmd_ready[c]), .h_op(frame_op),
        .h_ctx_generation(request.context_gen), .h_ticket(request.ticket),
        .h_profile_generation(request.profile_gen), .h_owner(request.owner),
        .h_view_alloc_id(request.op == BIND ? proposed_view : request.allocation_id),
        .h_arena_alloc_gen(arenas[c]),
        .h_base_addr(request.op == BIND ? new_view_base : request.base),
        .h_logical(request.logical_bytes), .h_arena_offset(request.offset),
        .h_layout_logical(request.logical_bytes), .h_layout_offset(request.offset),
        .h_slot(request.slot), .h_alignment(alignments[c]), .h_stripe(stripes[c]),
        .h_reserved_bytes(reserved_totals[c]), .h_layout_count(request.layouts),
        .h_profile_idx(3'(ACTIVE_PROFILE_IDX)),
        .rec_valid(state == FRAME_RECORDS && request.ctx == CTX_AW'(c) && rec_valid &&
                   !frame_record_error && !f_rsp_valid[c]),
        .rec_ready(f_rec_ready[c]), .rec_slot(rec_slot),
        .rec_arena_offset(rec_arena_offset), .rec_logical(rec_logical),
        .res_valid_flat(reserve_valid[c]), .res_start_flat(reserve_start[c]),
        .res_size_flat(reserve_size[c]), .rsp_valid(f_rsp_valid[c]),
        .rsp_ready(request.ctx == CTX_AW'(c) &&
                   (state == FRAME_WAIT || (state == FRAME_SEND && frame_op == F_RELEASE))),
        .rsp_code(f_rsp_code[c*5 +: 5]), .rsp_latency(), .rsp_gen(),
        .state_o(dbg_frame_state[c*4 +: 4]), .fault_code_o(),
        .staged_mask_o(), .inst_mask_o(), .live_mask_o(dbg_frame_live_mask[c*SLOTS +: SLOTS]),
        .shadow_mask_o(dbg_frame_shadow_mask[c*SLOTS +: SLOTS]), .prepared_o(f_prepared[c]),
        .stat_permission_faults_o(), .stat_view_binds_o(), .stat_view_frees_o(),
        .stat_prepare_max_lat_o(), .stat_last_prepare_lat_o(),
        .dbg_invariant_error_o(f_invariant_error[c]),
        .lookup_slot_i(query_slot_i[c*(SLOT_AW+1) +: SLOT_AW+1]),
        .lookup_valid_o(f_query_valid[c]), .lookup_alloc_o(query_alloc_o[c*ID_W +: ID_W]),
        .lookup_owner_o(query_owner_o[c*ID_W +: ID_W]),
        .lookup_arena_gen_o(query_arena_gen_o[c*ID_W +: ID_W]),
        .lookup_context_gen_o(query_context_gen_o[c*ID_W +: ID_W]),
        .lookup_profile_gen_o(query_profile_gen_o[c*ID_W +: ID_W]),
        .lookup_base_o(query_base_o[c*ADDR_AW +: ADDR_AW]),
        .lookup_size_o(query_size_o[c*ADDR_AW +: ADDR_AW]), .arena_generation_o()
      );
    end
  endgenerate

  // Accounting is a projection of actual reservations; no duplicate mutable counters.
  always_comb begin : observations
    logic [63:0] held, bound;
    logic local_settled;
    int q;
    dbg_cres_valid = '0; dbg_cres_start = '0; dbg_cres_size = '0;
    dbg_view_live_mask = '0; dbg_view_invalid_mask = '0;
    dbg_ctx_state = '0; completion_valid = '0; retire_ready = '0; query_valid_o = '0;
    completion_ctx_generation = '0; completion_arena_gen = '0; completion_ticket = '0;
    for (int c = 0; c < CONTEXT_COUNT; c++) begin
      dbg_ctx_state[c*4 +: 4] = 4'(context_state[c]);
      dbg_cres_valid[c*BANKS +: BANKS] = reserve_valid[c];
      dbg_cres_start[c*BANKS*BYTE_AW +: BANKS*BYTE_AW] = reserve_start[c];
      dbg_cres_size[c*BANKS*BYTE_AW +: BANKS*BYTE_AW] = reserve_size[c];
      completion_valid[c] = context_state[c] == C_COMPLETE;
      completion_ctx_generation[c*ID_W +: ID_W] = context_gen[c];
      completion_arena_gen[c*ID_W +: ID_W] = arenas[c];
      completion_ticket[c*ID_W +: ID_W] = tickets[c];
      local_settled = context_state[c] == C_BOUND;
      for (int s = 0; s < SLOTS; s++) begin
        dbg_view_live_mask[c*SLOTS+s] = view_state[c][s] == V_LIVE ||
                                       view_state[c][s] == V_CLOSING;
        dbg_view_invalid_mask[c*SLOTS+s] = view_state[c][s] == V_INVALID;
        if (view_state[c][s] == V_LIVE || view_state[c][s] == V_CLOSING ||
            views[c][s].pins != 0 || views[c][s].accesses != 0) local_settled = 0;
      end
      retire_ready[c] = local_settled && exec_drained[c] &&
        exec_drained_generation_i[c*ID_W +: ID_W] == context_gen[c] &&
        !(state != IDLE && state != RESPOND && request.ctx == CTX_AW'(c));
      q = int'(query_slot_i[c*(SLOT_AW+1) +: SLOT_AW+1]);
      if (q < SLOTS) query_valid_o[c] = f_query_valid[c] && context_state[c] == C_BOUND &&
                                      view_state[c][SLOT_AW'(q)] == V_LIVE;
    end
    for (int b = 0; b < BANKS; b++) begin
      held = 0; bound = 0;
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        if (context_state[c] == C_HELD) held += 64'(reserve_size[c][b*BYTE_AW +: BYTE_AW]);
        if (context_state[c] == C_BOUND) bound += 64'(reserve_size[c][b*BYTE_AW +: BYTE_AW]);
      end
      dbg_held_bytes[b*BYTE_AW +: BYTE_AW] = BYTE_AW'(held);
      dbg_bound_bytes[b*BYTE_AW +: BYTE_AW] = BYTE_AW'(bound);
      dbg_free_bytes[b*BYTE_AW +: BYTE_AW] = BYTE_AW'(64'(USER_CAPACITY) - held - bound);
      dbg_place_start[b*BYTE_AW +: BYTE_AW] = placements[b];
    end
    dbg_place_valid = requested_banks;
    dbg_arena_gen = arena_serial;
  end

  always_ff @(posedge clk or negedge rst_n) begin : controller
    logic duplicate, bad_record, settled;
    logic [6:0] error_code;
    logic [63:0] size, candidate, bumped, start_byte, end_byte, used, physical_base;
    if (!rst_n) begin
      state <= IDLE; request <= '0; frame_op <= F_RELEASE;
      frame_record_error <= 0; abort_prepare <= 0; clear_arena <= 0;
      proposed_view <= '0; new_view_base <= '0;
      arena_serial <= '0; view_serial <= '0; elapsed <= 0; outside_wait <= 0;
      records_seen <= 0; requested_total <= 0; requested_banks <= '0;
      bank_cursor <= 0; hold_error <= OK;
      rsp_valid <= 0; rsp_code <= OK; rsp_latency <= 0;
      rsp_arena_gen <= '0; rsp_view_alloc_id <= '0; rsp_ctx <= '0;
      dbg_frame_fault <= 0; dbg_invariant_error <= 0;
      for (int b = 0; b < BANKS; b++) begin requested_sizes[b] <= 0; placements[b] <= 0; end
      for (int c = 0; c < CONTEXT_COUNT; c++) begin
        context_state[c] <= C_FREE; context_seen[c] <= 0;
        context_gen[c] <= '0; tickets[c] <= '0; profiles[c] <= '0; arenas[c] <= '0;
        alignments[c] <= '0; stripes[c] <= '0; reserved_totals[c] <= '0;
        slot_masks[c] <= '0; reserve_valid[c] <= '0;
        reserve_start[c] <= '0; reserve_size[c] <= '0;
        for (int s = 0; s < SLOTS; s++) begin view_state[c][s] <= V_EMPTY; views[c][s] <= '0; end
      end
    end else begin
      if (rsp_valid && rsp_ready) rsp_valid <= 0;
      if (state != IDLE && state != RESPOND) elapsed <= elapsed + 64'd1;
      if ((state == HOLD_RECORDS || state == FRAME_RECORDS) && rec_ready && !rec_valid)
        outside_wait <= outside_wait + 64'd1;
      if (state == INVALIDATE_SEND && clear_length != 0 && !inv_ready_i[request.ctx])
        outside_wait <= outside_wait + 64'd1;

      for (int b = 0; b < BANKS; b++) begin
        used = 0;
        for (int c = 0; c < CONTEXT_COUNT; c++) begin
          if (context_holds(c)) begin
            start_byte = 64'(reserve_start[c][b*BYTE_AW +: BYTE_AW]);
            size = 64'(reserve_size[c][b*BYTE_AW +: BYTE_AW]);
            used += size;
            if (size != 0 && (start_byte < 64'(RESERVED) || start_byte + size > 64'(SPM))) begin
              dbg_invariant_error <= 1;
              $error("C029_ASSERT ledger extent escapes user SPM");
            end
            for (int peer = c + 1; peer < CONTEXT_COUNT; peer++) begin
              if (context_holds(peer) && size != 0 &&
                  reserve_size[peer][b*BYTE_AW +: BYTE_AW] != 0 &&
                  start_byte < 64'(reserve_start[peer][b*BYTE_AW +: BYTE_AW]) +
                               64'(reserve_size[peer][b*BYTE_AW +: BYTE_AW]) &&
                  64'(reserve_start[peer][b*BYTE_AW +: BYTE_AW]) < start_byte + size) begin
                dbg_invariant_error <= 1;
                $error("C029_ASSERT ledger held and bound extents must be disjoint");
              end
            end
          end
        end
        if (used > 64'(USER_CAPACITY)) begin
          dbg_invariant_error <= 1;
          $error("C029_ASSERT ledger no oversell");
        end
      end
      if (|f_invariant_error) begin
        dbg_invariant_error <= 1;
        $error("C029_ASSERT ledger child Frame invariant");
      end

      case (state)
        IDLE: if (cmd_valid && cmd_ready) begin
          request <= '{cmd_op, cmd_ctx, cmd_ctx_generation, cmd_ticket, cmd_profile_generation,
                       cmd_owner, cmd_view_alloc_id, cmd_arena_alloc_gen, cmd_reserved_bytes,
                       cmd_logical, cmd_arena_offset, cmd_base_addr, cmd_alignment, cmd_stripe,
                       cmd_seg_count, cmd_layout_count, cmd_slot_mask, cmd_slot};
          elapsed <= 64'd1; outside_wait <= 0; abort_prepare <= 0;
          state <= DISPATCH;
        end
        DISPATCH: begin
          if (int'(request.ctx) >= CONTEXT_COUNT) reply(BAD_CONTEXT, '0, '0);
          else case (request.op)
            HOLD: begin
              duplicate = 0;
              for (int c = 0; c < CONTEXT_COUNT; c++)
                if (context_state[c] != C_FREE && tickets[c] == request.ticket) duplicate = 1;
              if (duplicate) reply(DUP_TICKET, '0, '0);
              else if (context_state[request.ctx] != C_FREE) reply(BAD_STATE, '0, '0);
              else if (context_seen[request.ctx] && request.context_gen <= context_gen[request.ctx])
                reply(STALE, '0, '0);
              else if (request.profile_gen != profile_generation_i) reply(BAD_PROFILE, '0, '0);
              else if (request.alignment == 0 || request.stripe == 0 ||
                       request.stripe % request.alignment != 0 || int'(request.stripe) > SPM)
                reply(BAD_PARAM, '0, '0);
              else begin
                requested_banks <= '0; records_seen <= 0; requested_total <= 0;
                hold_error <= int'(request.segments) > BANKS ? BAD_COUNT : OK;
                for (int b = 0; b < BANKS; b++) begin requested_sizes[b] <= 0; placements[b] <= 0; end
                bank_cursor <= 0;
                state <= request.segments == 0 ? HOLD_SCAN : HOLD_RECORDS;
              end
            end
            PREPARE, FRAME_RELEASE, COMMIT, COMPENSATE: begin
              if (request.op == COMPENSATE && context_state[request.ctx] == C_BOUND)
                reply(POST_AUTH_COMPENSATE, '0, '0);
              else if (context_state[request.ctx] != C_HELD) reply(NOT_HELD, '0, '0);
              else if (!owner_matches()) reply(STALE, '0, '0);
              else if ((request.op == PREPARE || request.op == COMMIT) &&
                       request.profile_gen != profile_generation_i) reply(BAD_PROFILE, '0, '0);
              else if (request.op == COMMIT && !f_prepared[request.ctx]) reply(NOT_PREPARED, '0, '0);
              else if (request.op == COMMIT && (&arena_serial)) reply(GENERATION_EXHAUSTED, '0, '0);
              else begin
                frame_op <= request.op == PREPARE ? F_PREPARE :
                            request.op == COMMIT ? F_COMMIT : F_RELEASE;
                if (request.op == COMMIT) begin
                  arena_serial <= arena_serial + ID_W'(1);
                  arenas[request.ctx] <= arena_serial + ID_W'(1);
                  context_state[request.ctx] <= C_BOUND;
                end
                records_seen <= 0; frame_record_error <= 0;
                state <= FRAME_SEND;
              end
            end
            BIND: begin
              if (context_state[request.ctx] != C_BOUND) reply(NOT_BOUND, '0, '0);
              else if (!owner_matches() || request.profile_gen != profile_generation_i ||
                       request.arena_gen != arenas[request.ctx]) reply(STALE_VIEW, '0, '0);
              else if (int'(request.slot) >= SLOTS ||
                       !slot_masks[request.ctx][SLOT_AW'(request.slot)]) reply(BAD_SLOT, '0, '0);
              else if (view_state[request.ctx][SLOT_AW'(request.slot)] == V_LIVE ||
                       view_state[request.ctx][SLOT_AW'(request.slot)] == V_CLOSING)
                reply(SLOT_BUSY, '0, '0);
              else if (request.logical_bytes == 0 ||
                       64'(request.offset) % (64'(stripes[request.ctx]) * 64'(BANKS)) != 0)
                reply(VIEW_LAYOUT, '0, '0);
              else if (&view_serial) reply(GENERATION_EXHAUSTED, '0, '0);
              else begin
                physical_base = 64'(reserve_start[request.ctx][0 +: BYTE_AW]) +
                  (64'(request.offset) / (64'(stripes[request.ctx]) * 64'(BANKS))) *
                  64'(stripes[request.ctx]);
                if (physical_base >= 64'(c029_params_pkg::BANK_BYTES)) reply(VIEW_ESCAPE, '0, '0);
                else begin
                  new_view_base <= ADDR_AW'(physical_base);
                  proposed_view <= view_serial + ID_W'(1);
                  frame_op <= F_BIND; state <= FRAME_SEND;
                end
              end
            end
            FREE, PIN, UNPIN, BEGIN_ACCESS, END_ACCESS: begin
              if (context_state[request.ctx] != C_BOUND) reply(NOT_BOUND, '0, '0);
              else if (!owner_matches() || request.arena_gen != arenas[request.ctx])
                reply(STALE_VIEW, '0, '0);
              else if (int'(request.slot) >= SLOTS ||
                       view_state[request.ctx][SLOT_AW'(request.slot)] == V_EMPTY)
                reply(UNKNOWN_VIEW, '0, '0);
              else if (view_state[request.ctx][SLOT_AW'(request.slot)] != V_LIVE)
                reply(request.op == FREE ? DOUBLE_FREE : DEAD_VIEW, '0, '0);
              else if (!allocation_matches(views[request.ctx][SLOT_AW'(request.slot)]))
                reply(STALE_VIEW, '0, '0);
              else if (request.op == FREE) begin
                if (views[request.ctx][SLOT_AW'(request.slot)].pins != 0 ||
                    views[request.ctx][SLOT_AW'(request.slot)].accesses != 0)
                  reply(LIVE_REFS, '0, '0);
                else begin
                  view_state[request.ctx][SLOT_AW'(request.slot)] <= V_CLOSING;
                  clear_arena <= 0; bank_cursor <= 0; state <= INVALIDATE_SEND;
                end
              end else if (request.op == PIN || request.op == BEGIN_ACCESS) begin
                if (request.profile_gen != profile_generation_i) reply(BAD_PROFILE, '0, '0);
                else if (request.op == PIN && (&views[request.ctx][SLOT_AW'(request.slot)].pins))
                  reply(PIN_OVERFLOW, '0, '0);
                else if (request.op == BEGIN_ACCESS &&
                         (&views[request.ctx][SLOT_AW'(request.slot)].accesses))
                  reply(ACCESS_OVERFLOW, '0, '0);
                else begin
                  if (request.op == PIN)
                    views[request.ctx][SLOT_AW'(request.slot)].pins <=
                      views[request.ctx][SLOT_AW'(request.slot)].pins + 64'd1;
                  else views[request.ctx][SLOT_AW'(request.slot)].accesses <=
                    views[request.ctx][SLOT_AW'(request.slot)].accesses + 64'd1;
                  reply(OK, arenas[request.ctx], request.allocation_id);
                end
              end else begin
                if ((request.op == UNPIN && views[request.ctx][SLOT_AW'(request.slot)].pins == 0) ||
                    (request.op == END_ACCESS &&
                     views[request.ctx][SLOT_AW'(request.slot)].accesses == 0))
                  reply(REF_UNDERFLOW, '0, '0);
                else begin
                  if (request.op == UNPIN)
                    views[request.ctx][SLOT_AW'(request.slot)].pins <=
                      views[request.ctx][SLOT_AW'(request.slot)].pins - 64'd1;
                  else views[request.ctx][SLOT_AW'(request.slot)].accesses <=
                    views[request.ctx][SLOT_AW'(request.slot)].accesses - 64'd1;
                  reply(OK, arenas[request.ctx], request.allocation_id);
                end
              end
            end
            RETIRE: begin
              settled = context_state[request.ctx] == C_BOUND && exec_drained[request.ctx] &&
                exec_drained_generation_i[request.ctx*ID_W +: ID_W] == context_gen[request.ctx];
              for (int s = 0; s < SLOTS; s++)
                if (view_state[request.ctx][s] == V_LIVE || view_state[request.ctx][s] == V_CLOSING ||
                    views[request.ctx][s].pins != 0 || views[request.ctx][s].accesses != 0) settled = 0;
              if (context_state[request.ctx] == C_COMPLETE) reply(ALREADY_RETIRED, '0, '0);
              else if (context_state[request.ctx] != C_BOUND) reply(NOT_BOUND, '0, '0);
              else if (!owner_matches() || arenas[request.ctx] != request.arena_gen)
                reply(STALE, '0, '0);
              else if (!settled) reply(NOT_DRAINED, '0, '0);
              else begin clear_arena <= 1; bank_cursor <= 0; state <= INVALIDATE_SEND; end
            end
            COMPLETION_ACK: begin
              if (context_state[request.ctx] != C_COMPLETE) reply(BAD_STATE, '0, '0);
              else if (!owner_matches() || arenas[request.ctx] != request.arena_gen)
                reply(BAD_COMPLETION, '0, '0);
              else begin context_state[request.ctx] <= C_FREE; reply(OK, arenas[request.ctx], '0); end
            end
            default: reply(BAD_PARAM, '0, '0);
          endcase
        end
        HOLD_RECORDS: if (rec_valid && rec_ready) begin
          error_code = hold_error;
          if (int'(rec_bank) >= BANKS) error_code = BAD_BANK;
          else if (requested_banks[BANK_AW'(rec_bank)]) error_code = DUP_BANK;
          else if (rec_size == 0 || rec_size % request.stripe != 0) error_code = BAD_SIZE;
          if (error_code == OK) begin
            requested_banks[BANK_AW'(rec_bank)] <= 1;
            requested_sizes[BANK_AW'(rec_bank)] <= rec_size;
            requested_total <= requested_total + 64'(rec_size);
          end
          hold_error <= error_code;
          records_seen <= records_seen + 64'd1;
          if (records_seen + 64'd1 == 64'(request.segments)) begin
            if (error_code != OK) reply(error_code, '0, '0);
            else state <= HOLD_SCAN;
          end
        end
        HOLD_SCAN: begin
          error_code = OK;
          size = 64'(requested_sizes[bank_cursor]);
          physical_base = 64'(bank_cursor) * 64'(c029_params_pkg::BANK_BYTES);
          candidate = align_up(physical_base + 64'(RESERVED), 64'(request.alignment)) - physical_base;
          used = 0;
          for (int c = 0; c < CONTEXT_COUNT; c++)
            if (context_holds(c)) used += 64'(reserve_size[c][bank_cursor*BYTE_AW +: BYTE_AW]);
          if (size != 0) begin
            if (candidate + size > 64'(SPM)) error_code = PERMANENT;
            else if (size > 64'(USER_CAPACITY) - used) error_code = WAIT_CAPACITY;
            else begin
              for (int pass = 0; pass < CONTEXT_COUNT; pass++) begin
                bumped = candidate;
                for (int c = 0; c < CONTEXT_COUNT; c++) begin
                  start_byte = 64'(reserve_start[c][bank_cursor*BYTE_AW +: BYTE_AW]);
                  end_byte = start_byte + 64'(reserve_size[c][bank_cursor*BYTE_AW +: BYTE_AW]);
                  if (context_holds(c) && end_byte > start_byte &&
                      candidate < end_byte && start_byte < candidate + size && end_byte > bumped)
                    bumped = end_byte;
                end
                candidate = align_up(physical_base + bumped, 64'(request.alignment)) - physical_base;
              end
              if (candidate + size > 64'(SPM)) error_code = WAIT_FRAGMENTATION;
            end
          end
          if (error_code != OK) reply(error_code, '0, '0);
          else begin
            placements[bank_cursor] <= BYTE_AW'(candidate);
            if (bank_cursor == BANKS - 1) begin
              if (requested_total != 64'(request.reserved)) reply(BAD_TOTAL, '0, '0);
              else state <= HOLD_APPLY;
            end else bank_cursor <= bank_cursor + 1;
          end
        end
        HOLD_APPLY: begin
          context_state[request.ctx] <= C_HELD; context_seen[request.ctx] <= 1;
          context_gen[request.ctx] <= request.context_gen; tickets[request.ctx] <= request.ticket;
          profiles[request.ctx] <= request.profile_gen; arenas[request.ctx] <= '0;
          alignments[request.ctx] <= request.alignment; stripes[request.ctx] <= request.stripe;
          reserved_totals[request.ctx] <= request.reserved; slot_masks[request.ctx] <= request.slots;
          reserve_valid[request.ctx] <= requested_banks;
          for (int b = 0; b < BANKS; b++) begin
            reserve_start[request.ctx][b*BYTE_AW +: BYTE_AW] <= placements[b];
            reserve_size[request.ctx][b*BYTE_AW +: BYTE_AW] <= requested_sizes[b];
          end
          reply(OK, '0, '0);
        end
        FRAME_SEND: if (f_cmd_ready[request.ctx]) begin
          if (frame_op == F_PREPARE && request.layouts != 0) begin
            records_seen <= 0; state <= FRAME_RECORDS;
          end else state <= FRAME_WAIT;
        end
        FRAME_RECORDS: if (rec_valid && rec_ready) begin
          bad_record = int'(rec_slot) >= SLOTS ||
                       !slot_masks[request.ctx][SLOT_AW'(rec_slot)];
          if (bad_record) frame_record_error <= 1;
          records_seen <= records_seen + 64'd1;
          if (records_seen + 64'd1 == request.layouts) begin
            if (bad_record || frame_record_error) begin
              abort_prepare <= 1; frame_op <= F_RELEASE; state <= FRAME_SEND;
            end else state <= FRAME_WAIT;
          end
        end
        FRAME_WAIT: if (f_rsp_valid[request.ctx]) begin
          dbg_frame_fault <= f_rsp_code[request.ctx*5 +: 5];
          if (abort_prepare) reply(BAD_SLOT, '0, '0);
          else if (f_rsp_code[request.ctx*5 +: 5] != 0) begin
            if (request.op == BIND) begin
              case (f_rsp_code[request.ctx*5 +: 5])
                5'd9: reply(VIEW_OVERLAP, '0, '0);
                5'd10: reply(VIEW_ESCAPE, '0, '0);
                default: reply(VIEW_LAYOUT, '0, '0);
              endcase
            end else reply(FRAME_FAULT, '0, '0);
          end else case (request.op)
            PREPARE, FRAME_RELEASE: reply(OK, '0, '0);
            COMMIT: reply(OK, arenas[request.ctx], '0);
            BIND: begin
              view_serial <= proposed_view;
              views[request.ctx][SLOT_AW'(request.slot)] <=
                '{proposed_view, request.owner, new_view_base, request.logical_bytes, request.offset,
                  64'd0, 64'd0};
              view_state[request.ctx][SLOT_AW'(request.slot)] <= V_LIVE;
              reply(OK, arenas[request.ctx], proposed_view);
            end
            FREE: begin
              view_state[request.ctx][SLOT_AW'(request.slot)] <= V_INVALID;
              reply(OK, arenas[request.ctx], request.allocation_id);
            end
            COMPENSATE, RETIRE: begin
              reserve_valid[request.ctx] <= '0; reserve_start[request.ctx] <= '0;
              reserve_size[request.ctx] <= '0; reserved_totals[request.ctx] <= '0;
              slot_masks[request.ctx] <= '0;
              for (int s = 0; s < SLOTS; s++) begin
                views[request.ctx][s] <= '0; view_state[request.ctx][s] <= V_EMPTY;
              end
              context_state[request.ctx] <= request.op == RETIRE ? C_COMPLETE : C_FREE;
              reply(request.op == RETIRE ? RETIRED : COMPENSATED, arenas[request.ctx], '0);
            end
            default: reply(FRAME_FAULT, '0, '0);
          endcase
        end
        INVALIDATE_SEND: begin
          if (clear_length == 0) begin
            if (bank_cursor == BANKS - 1) begin
              frame_op <= clear_arena ? F_RELEASE : F_FREE; state <= FRAME_SEND;
            end else bank_cursor <= bank_cursor + 1;
          end else if (inv_ready_i[request.ctx]) state <= INVALIDATE_WAIT;
        end
        INVALIDATE_WAIT: begin
          if (inv_rej_i[request.ctx]) reply(INVALIDATION_REJECTED, arenas[request.ctx], '0);
          else if (inv_ack_i[request.ctx]) begin
            if (bank_cursor == BANKS - 1) begin
              frame_op <= clear_arena ? F_RELEASE : F_FREE; state <= FRAME_SEND;
            end else begin bank_cursor <= bank_cursor + 1; state <= INVALIDATE_SEND; end
          end
        end
        RESPOND: if (rsp_valid && rsp_ready) state <= IDLE;
        default: begin
          dbg_invariant_error <= 1;
          $error("C029_ASSERT ledger illegal controller state");
        end
      endcase
    end
  end
endmodule
