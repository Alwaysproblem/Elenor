// C029 Slot Frame. Physical storage is bounded by live SLOT_COUNT, not by
// the number of buffers in a program. PREPARE validates a stream of immutable
// layout records and retains only their eligible-slot mask. Lifetimes may
// reuse a slot and an address, with different shapes.
//
// VIEW_BIND consumes the immutable per-buffer layout lookup supplied by the
// resident program/descriptor owner, separately from the allocation handle.
// It validates that handle, Arena containment and overlap with CURRENT live
// bindings before publishing an engine-visible shadow record. No mutable
// Python L1 ledger or whole-program layout CAM is introduced.
//
// ArenaPool.bind_view assigns handle.generation = arena.allocation_generation;
// allocation_id distinguishes views within that Arena. Context, profile and
// Arena generation remain separate domains. ID_W and 64-bit work counters are
// experimental transport storage, not frozen silicon widths.
//
// PREPARE work <= layout_count + BANK_COUNT + 4. Input-stream wait cycles are
// measured separately and added to the wall-cycle bound. VIEW_BIND scans one
// bank/cycle against the bounded live-slot table. Responses retain payloads
// under backpressure. Empty Frames still carry their committed Arena identity.
`timescale 1ns/1ps

module slot_frame #(
  parameter int ID_W = 64,
  localparam int BANKS = c029_params_pkg::BANK_COUNT,
  localparam int SLOTS = c029_params_pkg::SLOT_COUNT,
  localparam int BYTE_AW = $clog2(c029_params_pkg::BANK_BYTES) + 1,
  localparam int ADDR_AW = $clog2(BANKS * c029_params_pkg::BANK_BYTES) + 1,
  localparam int SLOT_AW = (SLOTS <= 1) ? 1 : $clog2(SLOTS)
) (
  input logic clk,
  input logic rst_n,
  input logic cmd_valid,
  output logic cmd_ready,
  input logic [2:0] h_op,
  input logic [ID_W-1:0] h_ctx_generation, h_ticket, h_profile_generation,
  input logic [ID_W-1:0] h_owner, h_view_alloc_id, h_arena_alloc_gen,
  input logic [ADDR_AW-1:0] h_base_addr, h_logical, h_arena_offset,
  input logic [SLOT_AW:0] h_slot,
  // Trusted immutable lookup result, not mutable handle fields.
  input logic [ADDR_AW-1:0] h_layout_logical, h_layout_offset,
  input logic [BYTE_AW-1:0] h_alignment, h_stripe,
  input logic [ADDR_AW-1:0] h_reserved_bytes,
  input logic [63:0] h_layout_count,
  input logic [2:0] h_profile_idx,
  input logic rec_valid,
  output logic rec_ready,
  input logic [SLOT_AW:0] rec_slot,
  input logic [ADDR_AW-1:0] rec_arena_offset, rec_logical,
  input logic [BANKS-1:0] res_valid_flat,
  input logic [BANKS*BYTE_AW-1:0] res_start_flat, res_size_flat,
  output logic rsp_valid,
  input logic rsp_ready,
  output logic [4:0] rsp_code,
  output logic [63:0] rsp_latency,
  output logic [ID_W-1:0] rsp_gen,
  output logic [3:0] state_o,
  output logic [4:0] fault_code_o,
  output logic [SLOTS-1:0] staged_mask_o, inst_mask_o, live_mask_o, shadow_mask_o,
  output logic prepared_o,
  output logic [63:0] stat_permission_faults_o, stat_view_binds_o, stat_view_frees_o,
  output logic [63:0] stat_prepare_max_lat_o, stat_last_prepare_lat_o,
  output logic dbg_invariant_error_o,
  // Real engine-facing shadow lookup. Invalid queries return valid=0.
  input logic [SLOT_AW:0] lookup_slot_i,
  output logic lookup_valid_o,
  output logic [ID_W-1:0] lookup_alloc_o, lookup_owner_o, lookup_arena_gen_o,
  output logic [ID_W-1:0] lookup_context_gen_o, lookup_profile_gen_o,
  output logic [ADDR_AW-1:0] lookup_base_o, lookup_size_o,
  output logic [ID_W-1:0] arena_generation_o
);
  localparam logic [2:0] FOP_PREPARE = 0, FOP_COMMIT = 1, FOP_VIEW_BIND = 2,
    FOP_VIEW_FREE = 3, FOP_RELEASE = 4, FOP_VIEW_CHECK = 5;
  localparam logic [3:0] ST_IDLE = 0, ST_FETCH = 1, ST_BANKPOL = 5,
    ST_INSTALL = 6, ST_ACTIVE = 7, ST_FAULTED = 8, ST_PREPARED = 9, ST_VIEW = 10;
  localparam logic [4:0] FR_OK = 0, FR_REJ_STATE = 1, FR_REJ_IDENTITY = 2,
    FR_REJ_SLOT_RANGE = 3, FR_REJ_SLOT_EMPTY = 4, FR_REJ_SLOT_BUSY = 5,
    FR_REJ_LAYOUT = 6, FR_FAULT_ABI = 7, FR_FAULT_SLOT_TABLE = 8,
    FR_FAULT_OVERLAP = 9, FR_FAULT_BANK_POLICY = 10,
    FR_FAULT_NO_PREPARE = 12, FR_REJ_REC_RANGE = 13;
  localparam int PROFILE_AW = (c029_params_pkg::PROFILE_COUNT <= 1)
    ? 1 : $clog2(c029_params_pkg::PROFILE_COUNT);

  typedef struct packed {
    logic [ID_W-1:0] allocation_id, owner, generation;
    logic [ADDR_AW-1:0] base, size, offset;
  } binding_t;
  binding_t live [SLOTS], shadow [SLOTS], candidate;
  logic [SLOT_AW-1:0] candidate_slot;
  logic [3:0] state;
  logic [ID_W-1:0] context_gen, ticket, profile_gen, arena_gen;
  logic [BYTE_AW-1:0] alignment, stripe;
  logic [ADDR_AW-1:0] reserved_bytes;
  logic [PROFILE_AW-1:0] profile_idx;
  logic [63:0] layout_count, records_seen, elapsed, input_stalls, reserve_sum;
  int unsigned bank_cursor;
  logic [SLOTS-1:0] staged_mask, installed_mask, live_mask, shadow_mask;

  function automatic logic [63:0] increment(input logic [63:0] value);
    return (&value) ? value : value + 64'd1;
  endfunction

  // Valid bytes, not padded bytes, on bank b of an all-bank striped view.
  function automatic logic [63:0] span_bytes(
    input logic [ADDR_AW-1:0] bytes, input int unsigned b
  );
    logic [63:0] chunks, count, tail_bytes;
    chunks = (64'(bytes) + 64'(stripe) - 64'd1) / 64'(stripe);
    if (64'(b) >= chunks) return 64'd0;
    count = 64'd1 + (chunks - 64'd1 - 64'(b)) / 64'(BANKS);
    tail_bytes = ((64'(bytes) - 64'd1) % 64'(stripe)) + 64'd1;
    return count * 64'(stripe) -
      (((chunks - 64'd1) % 64'(BANKS) == 64'(b)) ? 64'(stripe) - tail_bytes : 64'd0);
  endfunction

  function automatic logic [63:0] bank_offset(input logic [ADDR_AW-1:0] offset);
    return (64'(offset) / (64'(stripe) * 64'(BANKS))) * 64'(stripe);
  endfunction

  function automatic logic identity_matches;
    return h_ctx_generation == context_gen && h_ticket == ticket &&
      h_profile_generation == profile_gen && h_arena_alloc_gen == arena_gen;
  endfunction

  function automatic logic binding_matches(input binding_t value);
    return value.allocation_id == h_view_alloc_id && value.owner == h_owner &&
      value.generation == h_arena_alloc_gen && value.base == h_base_addr &&
      value.size == h_logical;
  endfunction

  task automatic respond(
    input logic [4:0] code, input logic [63:0] latency, input logic [ID_W-1:0] generation
  );
    rsp_valid <= 1'b1;
    rsp_code <= code;
    rsp_latency <= latency;
    rsp_gen <= generation;
  endtask

  task automatic prepare_fault(input logic [4:0] code, input logic [63:0] latency);
    state <= ST_FAULTED;
    fault_code_o <= code;
    stat_permission_faults_o <= increment(stat_permission_faults_o);
    respond(code, latency, '0);
  endtask

  always_comb begin
    cmd_ready = (!rsp_valid || rsp_ready) &&
      ((state == ST_IDLE || state == ST_ACTIVE || state == ST_PREPARED ||
        state == ST_FAULTED) || h_op == FOP_RELEASE);
    rec_ready = state == ST_FETCH && !(cmd_valid && cmd_ready && h_op == FOP_RELEASE);
    state_o = state;
    staged_mask_o = staged_mask;
    inst_mask_o = installed_mask;
    live_mask_o = live_mask;
    shadow_mask_o = shadow_mask;
    prepared_o = state == ST_PREPARED;
    arena_generation_o = arena_gen;
    lookup_valid_o = 1'b0;
    lookup_alloc_o = '0;
    lookup_owner_o = '0;
    lookup_arena_gen_o = '0;
    lookup_context_gen_o = '0;
    lookup_profile_gen_o = '0;
    lookup_base_o = '0;
    lookup_size_o = '0;
    if ((state == ST_ACTIVE || state == ST_VIEW) && int'(lookup_slot_i) < SLOTS) begin
      if (shadow_mask[SLOT_AW'(lookup_slot_i)]) begin
        lookup_valid_o = 1'b1;
        lookup_alloc_o = shadow[SLOT_AW'(lookup_slot_i)].allocation_id;
        lookup_owner_o = shadow[SLOT_AW'(lookup_slot_i)].owner;
        lookup_arena_gen_o = shadow[SLOT_AW'(lookup_slot_i)].generation;
        lookup_context_gen_o = context_gen;
        lookup_profile_gen_o = profile_gen;
        lookup_base_o = shadow[SLOT_AW'(lookup_slot_i)].base;
        lookup_size_o = shadow[SLOT_AW'(lookup_slot_i)].size;
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin : frame
    logic bad_range, overlap;
    logic [63:0] start_byte, size_bytes, next_sum, view_start, view_bytes;
    logic [63:0] old_start, old_bytes;
    if (!rst_n) begin
      state <= ST_IDLE;
      context_gen <= '0; ticket <= '0; profile_gen <= '0; arena_gen <= '0;
      alignment <= '0; stripe <= '0; reserved_bytes <= '0; profile_idx <= '0;
      layout_count <= '0; records_seen <= '0; elapsed <= '0; input_stalls <= '0;
      reserve_sum <= '0; bank_cursor <= 0;
      candidate <= '0; candidate_slot <= '0;
      staged_mask <= '0; installed_mask <= '0; live_mask <= '0; shadow_mask <= '0;
      rsp_valid <= 1'b0; rsp_code <= FR_OK; rsp_latency <= '0; rsp_gen <= '0;
      fault_code_o <= FR_OK; dbg_invariant_error_o <= 1'b0;
      stat_permission_faults_o <= '0; stat_view_binds_o <= '0; stat_view_frees_o <= '0;
      stat_prepare_max_lat_o <= '0; stat_last_prepare_lat_o <= '0;
      for (int s = 0; s < SLOTS; s++) begin live[s] <= '0; shadow[s] <= '0; end
    end else begin
      if (rsp_valid && rsp_ready) rsp_valid <= 1'b0;
      for (int s = 0; s < SLOTS; s++) begin
        if (live_mask[s] != shadow_mask[s] || (live_mask[s] && live[s] != shadow[s])) begin
          dbg_invariant_error_o <= 1'b1;
          $error("C029_ASSERT frame_shadow_live_records_identical");
        end
      end

      if (cmd_valid && cmd_ready) begin
        case (h_op)
          FOP_PREPARE: begin
            if (state != ST_IDLE) begin
              stat_permission_faults_o <= increment(stat_permission_faults_o);
              respond(FR_REJ_STATE, 64'd1, '0);
            end else if (h_alignment == 0 || h_stripe == 0 ||
                         (h_stripe % h_alignment) != 0 ||
                         int'(h_profile_idx) >= c029_params_pkg::PROFILE_COUNT) begin
              prepare_fault(FR_FAULT_ABI, 64'd1);
            end else begin
              context_gen <= h_ctx_generation;
              ticket <= h_ticket;
              profile_gen <= h_profile_generation;
              alignment <= h_alignment; stripe <= h_stripe;
              reserved_bytes <= h_reserved_bytes; profile_idx <= PROFILE_AW'(h_profile_idx);
              layout_count <= h_layout_count; records_seen <= '0;
              staged_mask <= '0; installed_mask <= '0;
              reserve_sum <= '0; bank_cursor <= 0;
              elapsed <= 64'd1; input_stalls <= '0; fault_code_o <= FR_OK;
              state <= (h_layout_count == 0) ? ST_BANKPOL : ST_FETCH;
            end
          end
          FOP_COMMIT: begin
            if (state != ST_PREPARED) begin
              prepare_fault(FR_FAULT_NO_PREPARE, 64'd1);
            end else if (h_ctx_generation != context_gen || h_ticket != ticket ||
                         h_profile_generation != profile_gen) begin
              respond(FR_REJ_IDENTITY, 64'd1, '0);
            end else begin
              arena_gen <= h_arena_alloc_gen;
              state <= ST_ACTIVE;
              respond(FR_OK, 64'd1, h_arena_alloc_gen);
            end
          end
          FOP_VIEW_BIND: begin
            if (state != ST_ACTIVE) respond(FR_REJ_STATE, 64'd1, '0);
            else if (!identity_matches()) respond(FR_REJ_IDENTITY, 64'd1, '0);
            else if (int'(h_slot) >= SLOTS) respond(FR_REJ_SLOT_RANGE, 64'd1, '0);
            else if (!installed_mask[SLOT_AW'(h_slot)]) respond(FR_REJ_LAYOUT, 64'd1, '0);
            else if (live_mask[SLOT_AW'(h_slot)]) respond(FR_REJ_SLOT_BUSY, 64'd1, '0);
            else if (h_logical == 0 || h_logical != h_layout_logical ||
                     h_arena_offset != h_layout_offset ||
                     64'(h_layout_offset) % (64'(stripe) * 64'(BANKS)) != 0)
              respond(FR_REJ_LAYOUT, 64'd1, '0);
            else begin
              candidate <= '{h_view_alloc_id, h_owner, h_arena_alloc_gen,
                             h_base_addr, h_logical, h_arena_offset};
              candidate_slot <= SLOT_AW'(h_slot);
              bank_cursor <= 0; elapsed <= 64'd1;
              state <= ST_VIEW;
            end
          end
          FOP_VIEW_FREE, FOP_VIEW_CHECK: begin
            if (state != ST_ACTIVE) respond(FR_REJ_STATE, 64'd1, '0);
            else if (!identity_matches()) respond(FR_REJ_IDENTITY, 64'd1, '0);
            else if (int'(h_slot) >= SLOTS) respond(FR_REJ_SLOT_RANGE, 64'd1, '0);
            else if (!live_mask[SLOT_AW'(h_slot)] || !shadow_mask[SLOT_AW'(h_slot)])
              respond(FR_REJ_SLOT_EMPTY, 64'd1, '0);
            else if (!binding_matches(live[SLOT_AW'(h_slot)]) ||
                     !binding_matches(shadow[SLOT_AW'(h_slot)]))
              respond(FR_REJ_IDENTITY, 64'd1, '0);
            else begin
              if (h_op == FOP_VIEW_FREE) begin
                live_mask[SLOT_AW'(h_slot)] <= 1'b0;
                shadow_mask[SLOT_AW'(h_slot)] <= 1'b0;
                live[SLOT_AW'(h_slot)] <= '0; shadow[SLOT_AW'(h_slot)] <= '0;
                stat_view_frees_o <= increment(stat_view_frees_o);
              end
              respond(FR_OK, 64'd1, arena_gen);
            end
          end
          FOP_RELEASE: begin
            state <= ST_IDLE; arena_gen <= '0;
            staged_mask <= '0; installed_mask <= '0; live_mask <= '0; shadow_mask <= '0;
            for (int s = 0; s < SLOTS; s++) begin live[s] <= '0; shadow[s] <= '0; end
            respond(FR_OK, 64'd1, '0);
          end
          default: respond(FR_REJ_STATE, 64'd1, '0);
        endcase
      end else begin
        case (state)
          ST_FETCH: begin
            elapsed <= increment(elapsed);
            if (rec_valid && rec_ready) begin
              if (int'(rec_slot) >= SLOTS) prepare_fault(FR_REJ_REC_RANGE, increment(elapsed));
              else if (rec_logical == 0 ||
                       64'(rec_arena_offset) % (64'(stripe) * 64'(BANKS)) != 0)
                prepare_fault(FR_FAULT_SLOT_TABLE, increment(elapsed));
              else begin
                // Repeated slot IDs and spatial overlap are legal across
                // disjoint lifetimes. Only live bindings participate in overlap.
                staged_mask[SLOT_AW'(rec_slot)] <= 1'b1;
                records_seen <= records_seen + 64'd1;
                if (records_seen + 64'd1 == layout_count) state <= ST_BANKPOL;
              end
            end else input_stalls <= increment(input_stalls);
          end
          ST_BANKPOL: begin
            elapsed <= increment(elapsed);
            start_byte = 64'(res_start_flat[bank_cursor*BYTE_AW +: BYTE_AW]);
            size_bytes = 64'(res_size_flat[bank_cursor*BYTE_AW +: BYTE_AW]);
            next_sum = reserve_sum;
            bad_range = 1'b0;
            if (res_valid_flat[bank_cursor]) begin
              next_sum = reserve_sum + size_bytes;
              bad_range = start_byte < 64'(c029_params_pkg::SYSTEM_RESERVED_BYTES) ||
                size_bytes == 0 || start_byte % 64'(alignment) != 0 ||
                start_byte + size_bytes > 64'(c029_params_pkg::SPM_BYTES_PER_BANK[profile_idx]);
            end
            if (bad_range || (bank_cursor == BANKS - 1 && next_sum != 64'(reserved_bytes)))
              prepare_fault(FR_FAULT_BANK_POLICY, increment(elapsed));
            else if (bank_cursor == BANKS - 1) state <= ST_INSTALL;
            else begin reserve_sum <= next_sum; bank_cursor <= bank_cursor + 1; end
          end
          ST_INSTALL: begin
            installed_mask <= staged_mask;
            state <= ST_PREPARED;
            stat_last_prepare_lat_o <= increment(elapsed);
            if (increment(elapsed) > stat_prepare_max_lat_o)
              stat_prepare_max_lat_o <= increment(elapsed);
            assert (65'(elapsed) + 65'd1 <=
                    65'(layout_count) + 65'(BANKS + 4) + 65'(input_stalls))
              else $error("C029_ASSERT frame_prepare_bounded_stream_work");
            respond(FR_OK, increment(elapsed), '0);
          end
          ST_VIEW: begin
            elapsed <= increment(elapsed);
            view_start = bank_offset(candidate.offset);
            view_bytes = span_bytes(candidate.size, bank_cursor);
            size_bytes = 64'(res_size_flat[bank_cursor*BYTE_AW +: BYTE_AW]);
            bad_range = view_bytes != 0 &&
              (!res_valid_flat[bank_cursor] || view_start + view_bytes > size_bytes);
            if (bank_cursor == 0 && 64'(candidate.base) !=
                64'(res_start_flat[0 +: BYTE_AW]) + view_start) bad_range = 1'b1;
            overlap = 1'b0;
            for (int s = 0; s < SLOTS; s++) begin
              if (live_mask[s]) begin
                old_start = bank_offset(live[s].offset);
                old_bytes = span_bytes(live[s].size, bank_cursor);
                if (live[s].allocation_id == candidate.allocation_id ||
                    (view_bytes != 0 && old_bytes != 0 &&
                     view_start < old_start + old_bytes && old_start < view_start + view_bytes))
                  overlap = 1'b1;
              end
            end
            if (bad_range || overlap) begin
              state <= ST_ACTIVE;
              stat_permission_faults_o <= increment(stat_permission_faults_o);
              respond(overlap ? FR_FAULT_OVERLAP : FR_FAULT_BANK_POLICY,
                      increment(elapsed), '0);
            end else if (bank_cursor == BANKS - 1) begin
              live[candidate_slot] <= candidate;
              shadow[candidate_slot] <= candidate;
              live_mask[candidate_slot] <= 1'b1;
              shadow_mask[candidate_slot] <= 1'b1;
              stat_view_binds_o <= increment(stat_view_binds_o);
              state <= ST_ACTIVE;
              assert (elapsed <= 64'(BANKS + 2))
                else $error("C029_ASSERT frame_view_check_bounded_by_banks");
              respond(FR_OK, increment(elapsed), arena_gen);
            end else bank_cursor <= bank_cursor + 1;
          end
          default: begin end
        endcase
      end
    end
  end
endmodule
