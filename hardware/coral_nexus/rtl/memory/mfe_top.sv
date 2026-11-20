// mfe_top.sv — C032 MFE direct-move engine with per-transaction settlement.
//
// Direct moves only: one accepted descriptor is exactly one transaction; a
// bounded in-flight set of records (INFLIGHT, default 4) each run the FSM
// PARSE -> AUTHORIZE -> SEGMENT -> MOVE -> WAIT_RSP -> TERMINAL -> DONE and
// segment the move into LINE_BYTES requests on the C023-style external channel
// (pipeline_validator/cosim/wire.py MEM_REQ / MEM_RSP / TERMINAL_ACK, served by
// memory_service.py). Indexed/gather/scatter and per-page leases stay
// C055-C057; no empty module is created for them here.
//
// Descriptor sides: one leg is the Tile-local window (KIND_LOCAL + local_offset
// into LOCAL_BYTES), the other is an external L2-view/HBM binding carried as
// (binding_id, binding_generation, binding_offset). Any other kind combination
// is an AUTHORIZE/PARSE fault. The write payload is read out of the local
// window bytes that a real MEM_RSP delivered in this Tile: the write path
// never fabricates bytes.
//
// AUTHORIZE performs the per-field C013-style checks against the installed
// binding table (live id+generation, direction permission, offset+bytes range
// with wrap-free arithmetic, local window range, non-zero length). A violation
// settles FAULTED with the matching error code, moves zero bytes and issues
// zero external requests.
//
// Cancel semantics (all marks settle through transfer_cancel_tracker):
//   * cancel only marks CANCEL_REQUESTED; it never settles anything;
//   * a mark observed before the current step's request is accepted stops
//     issuing: no write leg is accepted after the mark and the transaction
//     settles ISOLATED (an in-flight read response is absorbed, not delivered);
//   * a cancelled transaction still owning an accepted write leg waits for that
//     write's real ACK and settles COMPLETED (the effect happened) — never a
//     timeout shortcut;
//   * same-edge cancel+completion yields exactly one terminal (the settle wins;
//     the late mark is counted stale by the tracker).
//
// Wire obligations: every accepted MEM_REQ gets exactly one TERMINAL_ACK
// (COMPLETED delivered / ISOLATED absorbed / FAULTED transport status) before
// the transaction settles; the ledger must close to zero.
//
// Poison: isolation_failed_i, or any refused tracker command, sets a sticky
// poison that refuses new descriptor starts until quiescent_reset_i arrives
// with every record free and no pending terminal. Poison is Group-owned state;
// a cancel or a new start never clears it. A quiescent_reset while not
// quiescent is refused (ERR_RESET) without clearing anything.
//
// The error-code space is shared with transfer_cancel_tracker.sv by value; see
// CODE_* localparams there. Wire field widths are transport containers, not
// frozen silicon widths.
`timescale 1ns/1ps

module mfe_top
  import c029_params_pkg::*;
  import c032_mfe_pkg::*;
#(
  parameter int unsigned INFLIGHT = MFE_INFLIGHT,  // bounded in-flight set; >= 2
  parameter int unsigned LOCAL_BYTES = 1024,       // power of two; local window bytes
  parameter int unsigned BINDINGS = 4              // installed binding table entries
) (
  input  logic clk,
  input  logic rst_n,

  // direct-move descriptor ingress (one transaction per accepted descriptor)
  input  logic        desc_valid_i,
  output logic        desc_ready_o,
  input  mfe_desc_t   desc_i,

  // binding table install (Frame/descriptor owner; trusted metadata)
  input  logic        bind_valid_i,
  output logic        bind_ready_o,
  input  mfe_binding_t bind_i,

  // cancel request mark: CANCEL_REQUESTED only, never settles
  input  logic        cancel_valid_i,
  input  logic [31:0] cancel_txn_i,

  // Group-owned poison control
  input  logic        isolation_failed_i,
  input  logic        quiescent_reset_i,
  output logic        poisoned_o,

  // local per-transaction terminal (exactly one per accepted descriptor)
  output logic          term_valid_o,
  input  logic          term_ready_i,
  output mfe_terminal_t term_o,

  // C023-style external channel
  output logic         mem_req_valid_o,
  input  logic         mem_req_ready_i,
  output mfe_mem_req_t mem_req_o,
  input  logic         mem_rsp_valid_i,
  output logic         mem_rsp_ready_o,
  input  mfe_mem_rsp_t mem_rsp_i,
  output logic         term_ack_valid_o,
  input  logic         term_ack_ready_i,
  output mfe_term_ack_t term_ack_o,

  // local window observation (unit verification only)
  input  logic [31:0] lpeek_addr_i,
  output logic [7:0]  lpeek_data_o,

  output logic [$clog2(INFLIGHT+1)-1:0] outstanding_o,
  output logic [63:0] sent_o, rsps_o, acks_o, absorbed_o, stale_rsp_o,
  output logic [63:0] tracker_terminals_o, tracker_stale_cancel_o, tracker_refused_o,
  output logic [3:0]  err_code_o,
  output logic        invariant_error_o
);
  localparam int unsigned SW = (INFLIGHT <= 1) ? 1 : $clog2(INFLIGHT);
  localparam int unsigned BW = (BINDINGS <= 1) ? 1 : $clog2(BINDINGS);

  localparam logic [2:0] S_FREE = 3'd0, S_PARSE = 3'd1, S_AUTHORIZE = 3'd2, S_SEGMENT = 3'd3,
                         S_MOVE = 3'd4, S_WAIT = 3'd5, S_TERM = 3'd6, S_DONE = 3'd7;
  localparam logic GO_SEG = 1'b0, GO_DONE = 1'b1;

  // ---- per-record state ----
  logic [2:0] rstate [INFLIGHT];
  logic [1:0] r_src_kind [INFLIGHT], r_dst_kind [INFLIGHT];
  logic r_dir_in [INFLIGHT];
  logic [31:0] r_bid [INFLIGHT], r_bgen [INFLIGHT], r_boff [INFLIGHT], r_bytes [INFLIGHT];
  logic [31:0] r_local [INFLIGHT], r_moved [INFLIGHT], r_seglen [INFLIGHT], r_wtxn [INFLIGHT];
  logic [31:0] r_ctx [INFLIGHT], r_cgen [INFLIGHT], r_arena [INFLIGHT], r_scope [INFLIGHT];
  logic [1:0] r_final [INFLIGHT], r_wirefinal [INFLIGHT];
  logic [3:0] r_err [INFLIGHT];
  logic r_after [INFLIGHT], r_settled_cmd [INFLIGHT];
  logic [7:0] local_mem [LOCAL_BYTES];
  logic [31:0] next_wtxn;
  logic [63:0] sent, rsps, acks, absorbed, stale_rsp;

  // ---- binding table ----
  logic [BINDINGS-1:0] bind_live_gated;
  logic [31:0] bind_id [BINDINGS], bind_gen [BINDINGS], bind_size [BINDINGS];
  logic bind_r [BINDINGS], bind_w [BINDINGS];

  // ---- tracker wiring ----
  logic tk_alloc_ready;
  logic [SW-1:0] tk_alloc_slot;
  logic [INFLIGHT-1:0] tk_free, tk_marked, tk_live, tk_settled, tk_acks_open;
  logic tk_term_valid;
  logic [31:0] tk_term_txn;
  logic [1:0] tk_term_final;
  logic [3:0] tk_term_err;
  logic settle_valid, release_valid;
  logic [SW-1:0] settle_slot, release_slot;

  // ---- selectors ----
  logic desc_have;
  logic [SW-1:0] desc_pick;
  logic bind_free_have, bind_dup;
  logic [BW-1:0] bind_pick;
  logic seg_grant_have;
  logic [SW-1:0] seg_grant;
  logic req_have, term_have, rsp_match;
  logic [SW-1:0] req_pick, term_pick, rsp_idx;
  logic mem_fire, rsp_fire, term_ack_fire;
  logic term_parked;
  logic [SW-1:0] term_parked_idx;
  logic [INFLIGHT-1:0] engine_free_mask;
  logic [LINE_BYTES*8-1:0] req_payload;

  assign mem_fire = mem_req_valid_o && mem_req_ready_i;
  assign rsp_fire = mem_rsp_valid_i && mem_rsp_ready_o;
  assign term_ack_fire = term_ack_valid_o && term_ack_ready_i;

  always_comb begin
    desc_have = 1'b0; desc_pick = '0;
    for (int i = 0; i < INFLIGHT; i++)
      if (rstate[i] == S_FREE && !desc_have) begin
        desc_have = 1'b1; desc_pick = SW'(i);
      end
    engine_free_mask = '0;
    for (int i = 0; i < INFLIGHT; i++)
      if (rstate[i] == S_FREE) engine_free_mask[i] = 1'b1;
  end

  always_comb begin
    bind_free_have = 1'b0; bind_pick = '0; bind_dup = 1'b0;
    for (int j = 0; j < BINDINGS; j++) begin
      if (!bind_live_gated[j] && !bind_free_have) begin
        bind_free_have = 1'b1; bind_pick = BW'(j);
      end
      if (bind_live_gated[j] && bind_id[j] == bind_i.id && bind_gen[j] == bind_i.generation)
        bind_dup = 1'b1;
    end
  end
  assign bind_ready_o = rst_n && bind_free_have && !bind_dup;

  // The wire transaction id is allocated when a record enters MOVE; the
  // request presenter picks the OLDEST allocated id, so a request held without
  // acceptance keeps its fields until the handshake (a cancel retraction is
  // the only withdrawal, and it drops valid rather than swapping payload).
  always_comb begin
    seg_grant_have = 1'b0; seg_grant = '0;
    for (int i = 0; i < INFLIGHT; i++)
      if (rstate[i] == S_SEGMENT && !tk_marked[i] && !seg_grant_have) begin
        seg_grant_have = 1'b1; seg_grant = SW'(i);
      end
  end

  always_comb begin
    req_have = 1'b0; req_pick = '0;
    for (int i = 0; i < INFLIGHT; i++)
      if (rstate[i] == S_MOVE && !tk_marked[i] && (!req_have || r_wtxn[i] < r_wtxn[req_pick])) begin
        req_have = 1'b1; req_pick = SW'(i);
      end
  end

  // The wire TERMINAL_ACK presenter parks its selection. The min-id rescan
  // below runs every cycle, and a reordered MEM_RSP can move an OLDER wire id
  // into S_TERM while a younger one is already offered under
  // term_ack_ready_i=0; without the park the offered payload would be
  // swapped before its handshake. While parked, the parked record is the
  // only selection: its own fire is the only event that can take a record
  // out of S_TERM, so the park can never point at an overwritten or
  // re-allocated slot (r_wtxn/r_wirefinal are held constant in S_TERM).
  always_comb begin
    term_have = 1'b0; term_pick = '0;
    if (term_parked && rstate[term_parked_idx] == S_TERM) begin
      term_have = 1'b1; term_pick = term_parked_idx;
    end else begin
      for (int i = 0; i < INFLIGHT; i++)
        if (rstate[i] == S_TERM && (!term_have || r_wtxn[i] < r_wtxn[term_pick])) begin
          term_have = 1'b1; term_pick = SW'(i);
        end
    end
  end

  // A cancelled record must never have a write accepted after the mark: marked
  // records are excluded from the request presenter above and retracted in
  // S_MOVE below before any handshake can fire.
  assign mem_req_valid_o = req_have;
  always_comb begin
    req_payload = '0;
    for (int unsigned b = 0; b < LINE_BYTES; b++)
      if (b < r_seglen[req_pick])
        req_payload[b*8 +: 8] = local_mem[r_local[req_pick] + r_moved[req_pick] + 32'(b)];
  end
  assign mem_req_o = '{txn_id: r_wtxn[req_pick], slice_id: 32'd0, slice_count: 32'd1,
    binding_id: r_bid[req_pick], binding_generation: r_bgen[req_pick],
    binding_offset: r_boff[req_pick] + r_moved[req_pick], length: r_seglen[req_pick],
    ctx_slot: r_ctx[req_pick], ctx_generation: r_cgen[req_pick],
    arena_generation: r_arena[req_pick], scope_epoch: r_scope[req_pick],
    write: !r_dir_in[req_pick], payload: req_payload};

  assign term_ack_valid_o = term_have;
  assign term_ack_o = '{txn_id: r_wtxn[term_pick], slice_id: 32'd0, slice_count: 32'd1,
                        final_state: r_wirefinal[term_pick]};

  // MEM_RSP matches on (txn_id, slice 0, binding_generation); anything else is
  // a stale response and is counted, never acknowledged.
  always_comb begin
    rsp_match = 1'b0; rsp_idx = '0;
    for (int i = 0; i < INFLIGHT; i++)
      if (rstate[i] == S_WAIT && r_wtxn[i] == mem_rsp_i.txn_id && mem_rsp_i.slice_id == 32'd0 &&
          r_bgen[i] == mem_rsp_i.binding_generation && !rsp_match) begin
        rsp_match = 1'b1; rsp_idx = SW'(i);
      end
  end
  assign mem_rsp_ready_o = rst_n;

  always_comb begin
    settle_valid = 1'b0; settle_slot = '0; release_valid = 1'b0; release_slot = '0;
    for (int i = 0; i < INFLIGHT; i++) begin
      if (rstate[i] == S_DONE && !r_settled_cmd[i] && !settle_valid) begin
        settle_valid = 1'b1; settle_slot = SW'(i);
      end
      if (rstate[i] == S_DONE && r_settled_cmd[i] && !release_valid) begin
        release_valid = 1'b1; release_slot = SW'(i);
      end
    end
  end

  function automatic logic [3:0] auth_code(input int slot);
    logic [3:0] code;
    int bi;
    code = ERR_NONE;
    bi = -1;
    for (int j = 0; j < BINDINGS; j++)
      if (bind_live_gated[j] && bind_id[j] == r_bid[slot] && bind_gen[j] == r_bgen[slot] && bi < 0)
        bi = j;
    if (bi < 0) code = ERR_BINDING;
    else begin
      if (r_dir_in[slot] ? !bind_r[bi] : !bind_w[bi]) code = ERR_PERMISSION;
      if (code == ERR_NONE &&
          (r_boff[slot] > bind_size[bi] || r_bytes[slot] > bind_size[bi] - r_boff[slot]))
        code = ERR_RANGE;
    end
    if (code == ERR_NONE && (r_local[slot] > 32'(LOCAL_BYTES) ||
        r_bytes[slot] > 32'(LOCAL_BYTES) - r_local[slot])) code = ERR_RANGE;
    if (code == ERR_NONE && r_bytes[slot] == 32'd0) code = ERR_RANGE;
    return code;
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      next_wtxn <= 32'd1;
      sent <= 64'd0; rsps <= 64'd0; acks <= 64'd0; absorbed <= 64'd0; stale_rsp <= 64'd0;
      term_parked <= 1'b0; term_parked_idx <= '0;
      for (int i = 0; i < INFLIGHT; i++) begin
        rstate[i] <= S_FREE;
        r_src_kind[i] <= KIND_LOCAL; r_dst_kind[i] <= KIND_LOCAL; r_dir_in[i] <= 1'b0;
        r_bid[i] <= 32'd0; r_bgen[i] <= 32'd0; r_boff[i] <= 32'd0; r_bytes[i] <= 32'd0;
        r_local[i] <= 32'd0; r_moved[i] <= 32'd0; r_seglen[i] <= 32'd0; r_wtxn[i] <= 32'd0;
        r_ctx[i] <= 32'd0; r_cgen[i] <= 32'd0; r_arena[i] <= 32'd0; r_scope[i] <= 32'd0;
        r_final[i] <= MFE_COMPLETED; r_wirefinal[i] <= MFE_COMPLETED; r_err[i] <= ERR_NONE;
        r_after[i] <= GO_DONE; r_settled_cmd[i] <= 1'b0;
      end
      for (int j = 0; j < BINDINGS; j++) begin
        bind_live_gated[j] <= 1'b0; bind_id[j] <= 32'd0; bind_gen[j] <= 32'd0;
        bind_size[j] <= 32'd0; bind_r[j] <= 1'b0; bind_w[j] <= 1'b0;
      end
      for (int b = 0; b < LOCAL_BYTES; b++) local_mem[b] <= 8'd0;
    end else begin
      if (bind_valid_i && bind_ready_o) begin
        bind_live_gated[bind_pick] <= 1'b1;
        bind_id[bind_pick] <= bind_i.id;
        bind_gen[bind_pick] <= bind_i.generation;
        bind_size[bind_pick] <= bind_i.size;
        bind_r[bind_pick] <= bind_i.readable;
        bind_w[bind_pick] <= bind_i.writable;
      end

      if (desc_valid_i && desc_ready_o) begin
        rstate[desc_pick] <= S_PARSE;
        r_src_kind[desc_pick] <= desc_i.src_kind;
        r_dst_kind[desc_pick] <= desc_i.dst_kind;
        r_bid[desc_pick] <= desc_i.binding_id;
        r_bgen[desc_pick] <= desc_i.binding_generation;
        r_boff[desc_pick] <= desc_i.binding_offset;
        r_local[desc_pick] <= desc_i.local_offset;
        r_bytes[desc_pick] <= desc_i.bytes;
        r_ctx[desc_pick] <= desc_i.ctx_slot;
        r_cgen[desc_pick] <= desc_i.ctx_generation;
        r_arena[desc_pick] <= desc_i.arena_generation;
        r_scope[desc_pick] <= desc_i.scope_epoch;
        r_moved[desc_pick] <= 32'd0;
        r_settled_cmd[desc_pick] <= 1'b0;
        r_err[desc_pick] <= ERR_NONE;
        r_final[desc_pick] <= MFE_COMPLETED;
      end

      for (int i = 0; i < INFLIGHT; i++) begin
        case (rstate[i])
          S_PARSE: begin
            if (r_src_kind[i] == KIND_EXT && r_dst_kind[i] == KIND_LOCAL) begin
              r_dir_in[i] <= 1'b1;
              rstate[i] <= S_AUTHORIZE;
            end else if (r_src_kind[i] == KIND_LOCAL && r_dst_kind[i] == KIND_EXT) begin
              r_dir_in[i] <= 1'b0;
              rstate[i] <= S_AUTHORIZE;
            end else begin
              r_dir_in[i] <= 1'b0;
              r_err[i] <= ERR_KIND;
              r_final[i] <= MFE_FAULTED;
              rstate[i] <= S_DONE;
            end
          end
          S_AUTHORIZE: begin
            if (auth_code(i) != ERR_NONE) begin
              r_err[i] <= auth_code(i);
              r_final[i] <= MFE_FAULTED;
              rstate[i] <= S_DONE;
            end else rstate[i] <= S_SEGMENT;
          end
          S_SEGMENT: begin
            // Cancel observed before this step's request was granted: stop
            // issuing — no write leg is accepted after the mark. A completed
            // external write leg already binds the effect (COMPLETED); only a
            // transaction with zero completed write legs may settle ISOLATED.
            if (tk_marked[i]) begin
              r_err[i] <= ERR_NONE;
              r_final[i] <= (!r_dir_in[i] && r_moved[i] != 32'd0)
                            ? MFE_COMPLETED : MFE_ISOLATED;
              rstate[i] <= S_DONE;
            end
            // Otherwise the grant block below moves the lowest record on.
          end
          S_MOVE: begin
            if (mem_fire && req_pick == SW'(i)) rstate[i] <= S_WAIT;
            else if (tk_marked[i]) begin
              // Presented but not accepted: a cancel retracts the request
              // before any handshake. Earlier completed write legs still bind
              // the effect, exactly as in S_SEGMENT.
              r_err[i] <= ERR_NONE;
              r_final[i] <= (!r_dir_in[i] && r_moved[i] != 32'd0)
                            ? MFE_COMPLETED : MFE_ISOLATED;
              rstate[i] <= S_DONE;
            end
          end
          S_WAIT: begin
            if (rsp_fire && rsp_match && rsp_idx == SW'(i)) begin
              if (mem_rsp_i.status != 32'd0) begin
                r_err[i] <= ERR_MEMORY;
                r_final[i] <= MFE_FAULTED;
                r_wirefinal[i] <= MFE_FAULTED;
                r_after[i] <= GO_DONE;
                rstate[i] <= S_TERM;
              end else if (r_dir_in[i]) begin
                if (tk_marked[i]) begin
                  // Cancelled inbound step: the late data is absorbed, never
                  // delivered, and its wire obligation still gets its ACK.
                  absorbed <= absorbed + 64'd1;
                  r_err[i] <= ERR_NONE;
                  r_final[i] <= MFE_ISOLATED;
                  r_wirefinal[i] <= MFE_ISOLATED;
                  r_after[i] <= GO_DONE;
                  rstate[i] <= S_TERM;
                end else begin
                  if (r_local[i] + r_moved[i] + r_seglen[i] > 32'(LOCAL_BYTES))
                    $error("C032_ASSERT mfe_top delivery escapes the local window");
                  for (int b = 0; b < LINE_BYTES; b++)
                    if (32'(b) < r_seglen[i])
                      local_mem[r_local[i] + r_moved[i] + 32'(b)] <= mem_rsp_i.data[b*8 +: 8];
                  r_moved[i] <= r_moved[i] + r_seglen[i];
                  r_wirefinal[i] <= MFE_COMPLETED;
                  if (r_moved[i] + r_seglen[i] == r_bytes[i]) begin
                    r_err[i] <= ERR_NONE;
                    r_final[i] <= MFE_COMPLETED;
                    r_after[i] <= GO_DONE;
                  end else r_after[i] <= GO_SEG;
                  rstate[i] <= S_TERM;
                end
              end else begin
                // A real external write completed: the effect happened. A
                // cancelled transaction stops here and reports COMPLETED
                // after this ACK — never a timeout shortcut, never ISOLATED.
                r_moved[i] <= r_moved[i] + r_seglen[i];
                r_wirefinal[i] <= MFE_COMPLETED;
                if (tk_marked[i] || r_moved[i] + r_seglen[i] == r_bytes[i]) begin
                  r_err[i] <= ERR_NONE;
                  r_final[i] <= MFE_COMPLETED;
                  r_after[i] <= GO_DONE;
                end else r_after[i] <= GO_SEG;
                rstate[i] <= S_TERM;
              end
            end
          end
          S_FREE: begin
            // Allocated only by the descriptor accept above; nothing to do.
          end
          S_TERM: begin
            if (term_ack_fire && term_pick == SW'(i))
              rstate[i] <= r_after[i] ? S_DONE : S_SEGMENT;
          end
          S_DONE: begin
            if (settle_valid && settle_slot == SW'(i)) r_settled_cmd[i] <= 1'b1;
            if (release_valid && release_slot == SW'(i)) rstate[i] <= S_FREE;
          end
        endcase
      end

      // Wire transaction grant: one record per cycle enters MOVE with a fresh
      // monotonic id; retracted ids simply leave gaps in the id space.
      if (seg_grant_have) begin
        r_wtxn[seg_grant] <= next_wtxn;
        r_seglen[seg_grant] <= (r_bytes[seg_grant] - r_moved[seg_grant] > 32'(LINE_BYTES))
                               ? 32'(LINE_BYTES) : (r_bytes[seg_grant] - r_moved[seg_grant]);
        rstate[seg_grant] <= S_MOVE;
        next_wtxn <= next_wtxn + 32'd1;
      end
      // Park an offered-but-stalled S_TERM record at the first stalled edge;
      // its own fire releases the park in the same edge that advances the
      // record, so the tracker's ack_slot and the S_TERM advance below always
      // count exactly the offered record. A record that fires in its first
      // offered cycle is never parked, so a continuously high ready keeps the
      // back-to-back ACK rate with no mandatory bubble.
      if (term_ack_fire) begin
        term_parked <= 1'b0;
      end else if (term_ack_valid_o && !term_ack_ready_i) begin
        term_parked <= 1'b1;
        term_parked_idx <= term_pick;
      end

      if (mem_fire) sent <= sent + 64'd1;
      if (rsp_fire) begin
        rsps <= rsps + 64'd1;
        if (!rsp_match) stale_rsp <= stale_rsp + 64'd1;
      end
      if (term_ack_fire) acks <= acks + 64'd1;
    end
  end

  always_comb begin
    int n;
    n = 0;
    for (int i = 0; i < INFLIGHT; i++)
      if (rstate[i] != S_FREE) n++;
    outstanding_o = n[$clog2(INFLIGHT+1)-1:0];
  end

  assign sent_o = sent;
  assign rsps_o = rsps;
  assign acks_o = acks;
  assign absorbed_o = absorbed;
  assign stale_rsp_o = stale_rsp;

  assign lpeek_data_o = local_mem[lpeek_addr_i[$clog2(LOCAL_BYTES)-1:0]];

  transfer_cancel_tracker #(.ENTRIES(INFLIGHT)) u_tracker (
    .clk, .rst_n,
    .alloc_ready_o(tk_alloc_ready), .alloc_valid_i(desc_valid_i),
    .alloc_txn_i(desc_i.txn_id), .alloc_slot_o(tk_alloc_slot),
    .cancel_valid_i(cancel_valid_i), .cancel_txn_i(cancel_txn_i),
    .obligate_valid_i(mem_fire), .obligate_slot_i(req_pick),
    .ack_valid_i(term_ack_fire), .ack_slot_i(term_pick),
    .settle_valid_i(settle_valid), .settle_slot_i(settle_slot),
    .settle_final_i(r_final[settle_slot]), .settle_err_i(r_err[settle_slot]),
    .release_valid_i(release_valid), .release_slot_i(release_slot),
    .term_valid_o(tk_term_valid), .term_ready_i(term_ready_i),
    .term_txn_o(tk_term_txn), .term_final_o(tk_term_final), .term_err_o(tk_term_err),
    .free_o(tk_free), .live_o(tk_live), .marked_o(tk_marked),
    .settled_o(tk_settled), .acks_open_o(tk_acks_open),
    .poisoned_o(poisoned_o), .isolation_failed_i(isolation_failed_i),
    .quiescent_reset_i(quiescent_reset_i),
    .err_code_o(err_code_o), .invariant_error_o(invariant_error_o),
    .terminals_o(tracker_terminals_o), .stale_cancel_o(tracker_stale_cancel_o),
    .refused_o(tracker_refused_o)
  );

  assign desc_ready_o = rst_n && tk_alloc_ready && desc_have;
  assign term_valid_o = tk_term_valid;
  assign term_o = '{txn_id: tk_term_txn, final_state: tk_term_final, error: tk_term_err};

  // Never-happens invariants in green runs; mutants must die on these or on
  // the bench's C032_ASSERT checks.
  always @(posedge clk) begin
    if (rst_n) begin
      if (engine_free_mask != tk_free || tk_live != ~tk_free)
        $error("C032_ASSERT mfe_top engine/tracker occupancy desync %b %b %b",
               engine_free_mask, tk_free, tk_live);
      if (desc_have && tk_alloc_slot != desc_pick)
        $error("C032_ASSERT mfe_top engine/tracker alloc slot desync");
      if (release_valid && (tk_settled[release_slot] !== 1'b1 || tk_acks_open[release_slot] !== 1'b0))
        $error("C032_ASSERT mfe_top release without settlement closure");
      if (seg_grant_have && tk_marked[seg_grant])
        $error("C032_ASSERT mfe_top marked record granted a wire transaction");
      for (int i = 0; i < INFLIGHT; i++)
        for (int j = i + 1; j < INFLIGHT; j++)
          if ((rstate[i] == S_MOVE || rstate[i] == S_WAIT || rstate[i] == S_TERM) &&
              (rstate[j] == S_MOVE || rstate[j] == S_WAIT || rstate[j] == S_TERM) &&
              r_wtxn[i] == r_wtxn[j])
            $error("C032_ASSERT mfe_top duplicate live wire txn %0d", r_wtxn[i]);
    end
  end
endmodule
