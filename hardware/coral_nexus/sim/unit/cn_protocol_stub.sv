/* cn_protocol_stub.sv — unit-only multi-channel wire-frame stub Tile (C025 / W1-03).
 *
 * `backend_kind == "stub"`: NEVER admissible as CS1 evidence (W1 plan §7.6).
 *
 * Implements all 5 channels (`CTRL=0, EVENT=1, PROGRAM=2, MEMORY=3, DEBUG=4`) at full wire width
 * (`MAX_WORDS = 17` x 64-bit = 136 bytes = 1088 bits, sufficient for 96B `TASK_PLAN`, 104B
 * `TASK_COMMIT`, 128B `PROGRAM_INSTALL_ACK`, and 136B `MEM_REQ`).
 *
 * Behavioral rules:
 *   1. Deterministic per-channel `g2t_ready[c]` lowering from `cfg_ready_seed` (LFSR + bounded
 *      service guarantee so `g2t_ready[c]` also stalls when the channel's 1-deep T2G response
 *      register is still occupied and `t2g_ready[c]` is low).
 *   2. On `posedge clk`, when `g2t_valid[c] && g2t_ready[c]`, decodes the C017 4-word wire header
 *      (`msg_type`, `payload_words`, `message_seq`, `reserved`) using `cn_offsets.svh` constants
 *      and synthesizes a REAL protocol-valid T2G wire frame on `t2g_frame[c]` (`t2g_valid[c] <= 1`
 *      after the clock edge, so it is a STAGED candidate for tick T+1, NEVER accepted in tick T):
 *        - `QUERY_CAPS`       -> `CAPS_RESP` (56 B)
 *        - `TASK_PLAN`        -> `ACK(ACCEPTED or STALE)` (72 B)
 *        - `TASK_COMMIT`      -> `ACK(HOLD_READY or STALE)` (72 B), establishes `live_gen[slot]`
 *        - `TASK_COMMIT_FINAL`-> `ACK(ACCEPTED or STALE)` (72 B)
 *        - `RESET_PHASE_REQ`  -> `RESET_PHASE_ACK` (64 B)
 *        - `EVENT_NOTIFY`     -> `POST_EVENT` (96 B)
 *        - `PROGRAM_INSTALL`  -> `PROGRAM_READ_REQ` (88 B)
 *        - `PROGRAM_READ_RESP`-> `PROGRAM_INSTALL_ACK` (128 B)
 *        - `PROGRAM_INVALIDATE`-> `INVALIDATE_ACK` (88 B)
 *        - `MEM_RSP`          -> `TERMINAL_ACK` (64 B)
 *   3. Optional `cfg_emit_mem_req` emits a Tile-initiated 136-byte `MEM_REQ` (17 words) on
 *      `MEMORY` T2G at reset release to test maximum-width T2G collection.
 *   4. Optional `cfg_corrupt_t2g` flips bit 0 of payload word 0 on every emitted T2G frame so
 *      tests prove Python receives real DUT T2G pins rather than a C-side copy of G2T inputs.
 */
`timescale 1ns/1ps

module cn_protocol_stub (
  input  wire          clk,
  input  wire          rst_n,

  // 5-channel G2T (Group -> Tile) full-width wire frames (17 x 64b = 1088b per channel)
  input  wire [4:0]    g2t_valid,
  output wire [4:0]    g2t_ready,
  input  wire [24:0]   g2t_words_flat,
  input  wire [5439:0] g2t_frame_flat,

  // 5-channel T2G (Tile -> Group) full-width wire frames
  output reg  [4:0]    t2g_valid,
  input  wire [4:0]    t2g_ready,
  output wire [24:0]   t2g_words_flat,
  output wire [5439:0] t2g_frame_flat,

  // configuration / debug / fault injection
  input  wire [31:0]   cfg_ready_seed,
  input  wire [31:0]   cfg_inject_fatal_at_tick,
  input  wire [31:0]   cfg_inject_abort_at_tick,
  input  wire          cfg_corrupt_t2g,
  input  wire          cfg_emit_mem_req,
  input  wire          cfg_glitch_payload,
  output reg  [63:0]   posedge_count,
  output reg  [31:0]   fatal_code,
  output reg  [63:0]   dbg_last_g2t_word0
);

  `include "cn_offsets.svh"

  localparam int MAX_WORDS = 17;

  reg [31:0] lfsr;
  reg [63:0] t2g_seq [0:4];
  reg [63:0] live_ctx_gen [0:CN_CONTEXT_SLOTS-1];
  reg        mem_req_sent;
  wire [1087:0] g2t_frame [0:4];
  reg  [4:0]    t2g_words [0:4];
  reg  [1087:0] t2g_frame [0:4];

  genvar ch;
  generate
    for (ch = 0; ch < 5; ch = ch + 1) begin : gen_ready
      assign g2t_frame[ch] = g2t_frame_flat[ch * 1088 +: 1088];
      assign t2g_words_flat[ch * 5 +: 5] = t2g_words[ch];
      assign t2g_frame_flat[ch * 1088 +: 1088] = t2g_frame[ch];
      wire lfsr_open = lfsr[ch] | (posedge_count[2:0] == 3'b111);
      wire local_busy = (ch == 3) & cfg_emit_mem_req & (!mem_req_sent);
      wire out_space = ((!t2g_valid[ch]) | t2g_ready[ch]) & (!local_busy);
      assign g2t_ready[ch] = rst_n & lfsr_open & out_space;
    end
  endgenerate

  function automatic [63:0] get_word(input [1087:0] frame, input int idx);
    get_word = frame[idx * 64 +: 64];
  endfunction

  function automatic [1087:0] set_word(
    input [1087:0] frame,
    input int idx,
    input [63:0] val
  );
    reg [1087:0] tmp;
    begin
      tmp = frame;
      tmp[idx * 64 +: 64] = val;
      set_word = tmp;
    end
  endfunction

  integer c;
  integer s;
  reg [1087:0] next_frame;
  reg [4:0]    next_words;
  reg          produce;
  reg [63:0]   in_msg_type;
  reg [63:0]   in_pwords;
  reg [63:0]   p0, p1, p2, p3, p4, p5, p6, p7;
  reg [63:0]   corrupt_mask;
  reg [63:0]   ack_outcome;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      posedge_count      <= 64'd0;
      lfsr               <= (cfg_ready_seed == 32'd0) ? 32'h1ACE_B00C : cfg_ready_seed;
      fatal_code         <= 32'd0;
      dbg_last_g2t_word0 <= 64'd0;
      t2g_valid          <= 5'b0;
      mem_req_sent       <= 1'b0;
      for (c = 0; c < 5; c = c + 1) begin
        t2g_words[c] <= 5'd0;
        t2g_frame[c] <= '0;
        t2g_seq[c]   <= 64'd0;
      end
      for (s = 0; s < CN_CONTEXT_SLOTS; s = s + 1) begin
        live_ctx_gen[s] <= 64'd0;
      end
    end else begin
      posedge_count <= posedge_count + 64'd1;
      lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
      corrupt_mask = cfg_corrupt_t2g ? 64'h1 : 64'h0;

      if (cfg_inject_abort_at_tick == posedge_count[31:0]) begin
        $fatal(1, "cn_protocol_stub: injected process-level crash");
      end
      if (cfg_inject_fatal_at_tick == posedge_count[31:0]) begin
        fatal_code <= 32'd1;
      end

      // Clear T2G beats that completed pre-edge handshake `(t2g_valid && t2g_ready)`.
      for (c = 0; c < 5; c = c + 1) begin
        if (t2g_valid[c] && t2g_ready[c]) begin
          t2g_valid[c] <= 1'b0;
        end
      end

      // Optional unsolicited 136-byte MEM_REQ (4 header + 13 payload = 17 words) on MEMORY (ch 3).
      if (cfg_emit_mem_req && !mem_req_sent && !t2g_valid[3]) begin
        next_frame = '0;
        next_frame = set_word(next_frame, 0, CN_MSG_MEM_REQ);
        next_frame = set_word(next_frame, 1, CN_WORDS_MEM_REQ);
        next_frame = set_word(next_frame, 2, t2g_seq[3]);
        next_frame = set_word(next_frame, 3, 64'd0);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_TILE_ID, (64'd0 ^ corrupt_mask));
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_CTX_SLOT, 64'd1);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_CTX_GENERATION, 64'd2);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_TXN_ID, 64'd77);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_SLICE_ID, 64'd0);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_SLICE_COUNT, 64'd1);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_ARENA_ALLOC_GEN, 64'd3);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_BINDING_ID, 64'd4);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_BINDING_GENERATION, 64'd5);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_BINDING_OFFSET, 64'd64);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_SCOPE_EPOCH, 64'd1);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_MEM_OP, 64'd0);
        next_frame = set_word(next_frame, 4 + CN_OFF_MEM_REQ_LENGTH, 64'd128);
        t2g_valid[3] <= 1'b1;
        t2g_words[3] <= 5'd17;
        t2g_frame[3] <= next_frame;
        t2g_seq[3]   <= t2g_seq[3] + 64'd1;
        mem_req_sent <= 1'b1;
      end

      // Process accepted G2T frames per channel.
      for (c = 0; c < 5; c = c + 1) begin
        if (g2t_valid[c] && g2t_ready[c]) begin
          in_msg_type = get_word(g2t_frame[c], 0);
          in_pwords   = get_word(g2t_frame[c], 1);
          p0 = get_word(g2t_frame[c], 4);
          p1 = get_word(g2t_frame[c], 5);
          p2 = get_word(g2t_frame[c], 6);
          p3 = get_word(g2t_frame[c], 7);
          p4 = get_word(g2t_frame[c], 8);
          p5 = get_word(g2t_frame[c], 9);
          p6 = get_word(g2t_frame[c], 10);
          p7 = get_word(g2t_frame[c], 11);

          if (c == 0) begin
            dbg_last_g2t_word0 <= cfg_glitch_payload ? (in_msg_type ^ 64'hDEAD) : in_msg_type;
          end

          produce    = 1'b0;
          next_frame = '0;
          next_words = 5'd0;

          case (in_msg_type[31:0])
            CN_MSG_QUERY_CAPS: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_CAPS_RESP);
              next_frame = set_word(next_frame, 0, CN_MSG_CAPS_RESP);
              next_frame = set_word(next_frame, 1, CN_WORDS_CAPS_RESP);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_TXN_ID, p0 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_CONTEXT_COUNT, 64'd2);
              next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_PROGRAM_SLOTS, 64'd2);
              next_frame = set_word(next_frame, 4 + CN_OFF_CAPS_RESP_STREAM_SUPPORTED, 64'd1);
            end
            CN_MSG_TASK_PLAN: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_ACK);
              next_frame = set_word(next_frame, 0, CN_MSG_ACK);
              next_frame = set_word(next_frame, 1, CN_WORDS_ACK);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_TXN_ID, p5 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_ACK_FOR_MESSAGE_TYPE, CN_MSG_TASK_PLAN);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OP_SEQ, 64'd0);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OUTCOME, 64'd1); // ACCEPTED
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_WAIT_REASON, 64'd0);
            end
            CN_MSG_TASK_COMMIT,
            CN_MSG_TASK_COMMIT_FINAL: begin
              produce     = 1'b1;
              next_words  = 5'(CN_HEADER_WORDS + CN_WORDS_ACK);
              if (p6 < CN_CONTEXT_SLOTS && p7 < live_ctx_gen[p6[5:0]]) begin
                ack_outcome = 64'd4; // STALE
              end else begin
                if (in_msg_type[31:0] == CN_MSG_TASK_COMMIT && p6 < CN_CONTEXT_SLOTS) begin
                  live_ctx_gen[p6[5:0]] <= p7;
                  ack_outcome = 64'd0; // HOLD_READY
                end else begin
                  ack_outcome = 64'd1; // ACCEPTED
                end
              end
              next_frame = set_word(next_frame, 0, CN_MSG_ACK);
              next_frame = set_word(next_frame, 1, CN_WORDS_ACK);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_TXN_ID, p5 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_ACK_FOR_MESSAGE_TYPE, in_msg_type);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OP_SEQ, 64'd0);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_OUTCOME, ack_outcome);
              next_frame = set_word(next_frame, 4 + CN_OFF_ACK_WAIT_REASON, 64'd0);
            end
            CN_MSG_RESET_PHASE_REQ: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_RESET_PHASE_ACK);
              next_frame = set_word(next_frame, 0, CN_MSG_RESET_PHASE_ACK);
              next_frame = set_word(next_frame, 1, CN_WORDS_RESET_PHASE_ACK);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_PHASE, p0 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_RUN_GENERATION, p1);
              next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_PENDING_TO_RESET, 64'd0);
              next_frame = set_word(next_frame, 4 + CN_OFF_RESET_PHASE_ACK_EVIDENCE_MASK, 64'd15);
            end
            CN_MSG_EVENT_NOTIFY: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_POST_EVENT);
              next_frame = set_word(next_frame, 0, CN_MSG_POST_EVENT);
              next_frame = set_word(next_frame, 1, CN_WORDS_POST_EVENT);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + 1, p1);
              next_frame = set_word(next_frame, 4 + 2, p2);
              next_frame = set_word(next_frame, 4 + 3, p3);
              next_frame = set_word(next_frame, 4 + 4, p4);
              next_frame = set_word(next_frame, 4 + 5, p5);
              next_frame = set_word(next_frame, 4 + 6, p6);
              next_frame = set_word(next_frame, 4 + 7, p7);
            end
            CN_MSG_PROGRAM_INSTALL: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_PROGRAM_READ_REQ);
              next_frame = set_word(next_frame, 0, CN_MSG_PROGRAM_READ_REQ);
              next_frame = set_word(next_frame, 1, CN_WORDS_PROGRAM_READ_REQ);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + 1, p1);
              next_frame = set_word(next_frame, 4 + 2, p2);
              next_frame = set_word(next_frame, 4 + 3, p3);
              next_frame = set_word(next_frame, 4 + 4, p4);
              next_frame = set_word(next_frame, 4 + 5, p5);
              next_frame = set_word(next_frame, 4 + 6, p6);
            end
            CN_MSG_PROGRAM_READ_RESP: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_PROGRAM_INSTALL_ACK);
              next_frame = set_word(next_frame, 0, CN_MSG_PROGRAM_INSTALL_ACK);
              next_frame = set_word(next_frame, 1, CN_WORDS_PROGRAM_INSTALL_ACK);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + 1, p1);
              next_frame = set_word(next_frame, 4 + 2, p2);
              next_frame = set_word(next_frame, 4 + 3, p3);
              next_frame = set_word(next_frame, 4 + 4, p4);
              next_frame = set_word(next_frame, 4 + 5, p5);
              next_frame = set_word(next_frame, 4 + 6, p6);
              next_frame = set_word(next_frame, 4 + 7, p7); // installed_len = hbm_chunk_len
              next_frame = set_word(next_frame, 4 + 8, p2);
              next_frame = set_word(next_frame, 4 + 9, p3);
              next_frame = set_word(next_frame, 4 + 10, p4);
              next_frame = set_word(next_frame, 4 + 11, p5);
            end
            CN_MSG_PROGRAM_INVALIDATE: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_INVALIDATE_ACK);
              next_frame = set_word(next_frame, 0, CN_MSG_INVALIDATE_ACK);
              next_frame = set_word(next_frame, 1, CN_WORDS_INVALIDATE_ACK);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + 0, p0 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + 1, p1);
              next_frame = set_word(next_frame, 4 + 2, p2);
              next_frame = set_word(next_frame, 4 + 3, p3);
              next_frame = set_word(next_frame, 4 + 4, p4);
              next_frame = set_word(next_frame, 4 + 5, p5);
              next_frame = set_word(next_frame, 4 + 6, p6);
            end
            CN_MSG_MEM_RSP: begin
              produce    = 1'b1;
              next_words = 5'(CN_HEADER_WORDS + CN_WORDS_TERMINAL_ACK);
              next_frame = set_word(next_frame, 0, CN_MSG_TERMINAL_ACK);
              next_frame = set_word(next_frame, 1, CN_WORDS_TERMINAL_ACK);
              next_frame = set_word(next_frame, 2, t2g_seq[c]);
              next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_TXN_ID, p0 ^ corrupt_mask);
              next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_SLICE_ID, p1);
              next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_SLICE_COUNT, 64'd1);
              next_frame = set_word(next_frame, 4 + CN_OFF_TERMINAL_ACK_FINAL_STATE, 64'd0);
            end
            default: begin
              produce = 1'b0;
            end
          endcase

          if (produce) begin
            t2g_valid[c] <= 1'b1;
            t2g_words[c] <= next_words;
            t2g_frame[c] <= next_frame;
            t2g_seq[c]   <= t2g_seq[c] + 64'd1;
          end
        end
      end
    end
  end

endmodule
