// install_service.sv — C039 (W5-04) Tile-side program install service.
//
// Implements the Tile half of the sealed PROGRAM channel
// (protocol/wire_schema_v1.yaml, generated cn_messages.h / cn_offsets.svh):
// it consumes PROGRAM_INSTALL headers, pulls the real text bytes with
// ordered PROGRAM_READ_REQ / PROGRAM_READ_RESP (+follow-on blob) exchanges,
// measures a real SHA-256 over the staged text plus its length, commits the
// bytes to the C030 program port ONLY after the digest matches the trusted
// expected text digest (stage-verify-commit: a failed verification writes
// nothing to the port, so no half residency is ever published), then emits
// PROGRAM_INSTALL_ACK (installed_len + measured_hash + sideband local
// handle), waits for PROGRAM_READY with that ACK-issued handle, and answers
// PROGRAM_INVALIDATE with INVALIDATE_ACK after execution references drain.
//
// Wire facts honoured here (no v5 change):
//   * a PROGRAM_READ_REQ carries no offset: each request consumes the next
//     chunk of the pending install stream (ordered cursor,
//     pipeline_validator/cosim/program_service.py).
//   * PROGRAM_INSTALL carries only identity + slot; the text length and the
//     trusted expected text digest arrive on the dedicated `setup` sideband,
//     written by the environment from the verified sealed image (never from
//     the wire, never from the blob). The wire `slot` must equal the trusted
//     text_base. An INSTALL with no matching setup faults closed.
//   * PROGRAM_READ_RESP's hbm_chunk_len declares the byte count of the blob
//     words that immediately follow its frame on the same G2T stream; the
//     last blob word is zero-padded and the pad bytes are neither digested
//     nor written. A chunk that would overflow the trusted length
//     (clen > cur_len - received), a wrong identity, or a zero chunk is a
//     fail-stop protocol fault: the pending install drops and the stream
//     re-synchronises on the next header — the service NEVER drains a
//     declared blob of unbounded length.
//   * PROGRAM_INSTALL_ACK's local handle rides the sideband ack_handle_o
//     alongside the ACK frame (v5 ACK has no handle field); PROGRAM_READY
//     must echo that exact handle or it is a protocol fault.
//   * INVALIDATE_ACK is T2G. Invalidation is a per-record serial point, not
//     a global one: accepting an INVALIDATE closes the fetch gate on that
//     identity at once (no new warm hit), the ACK follows once the
//     identity's execution references reach zero, and the record drops when
//     the ACK emits. An INVALIDATE naming the pending install aborts it
//     (staged bytes die unpublished); an INVALIDATE naming a DIFFERENT
//     identity marks only that record and never touches the pending
//     install. While any record is invalidating or any ACK is owed, new
//     INSTALL payload words are deferred (bounded stream backpressure), so
//     a queued INSTALL cannot hit and a re-install is always cold.
//
// Residency is slot-addressed: text_base must be aligned to SLOT_BYTES (the
// sealed C019 slot geometry, 2 x 4096 B for cs1_c2). One slot holds at most
// one resident identity; there is no auto-eviction (an INSTALL whose base
// maps onto a slot held by a different identity is rejected with
// FC_INVALID_DESCRIPTOR - CMP-18 - leaving the resident record and bytes
// untouched). A warm INSTALL of the identical program_identity is answered
// immediately with the stored measurement and handle: ZERO PROGRAM_READ_REQ.
//
// The C030 program SRAM is written through l1_program_port's install
// channel only; this module owns no shadow copy of resident text. The
// staging buffer holds an UNVERIFIED candidate: because the port's
// byte_valid cannot be cleared, publishing before verify would be
// unrecoverable. Commits write ceil(text_len/LINE_BYTES) full fetch lines;
// the tail of the final line is zero-filled under the same identity so a
// line-wide fetch never faults on the pad. text_len <= SLOT_BYTES and
// text_base is slot-aligned, so the pad never leaves the slot.
//
// C030 byte_valid alone is NOT residency: fetch_allow_o is the exported
// gate. It is asserted only inside committed, non-invalidating resident
// slot ranges and masks every address outside them; consumers must gate
// fetches and handles with it (pending / faulted / invalidated text is
// byte_valid but never resident). Execution references (exec_ref_*) are
// single-cycle pulses, forwarded verbatim to the C030 port by the wrapper
// (real FET-06 protection) and counted per identity for the drain.
//
// program_id and program_epoch are the C030 port's narrower identity fields
// (PROG_ID_W / EPOCH_W): an INSTALL whose identity does not fit those
// containers faults instead of aliasing a different program's bytes.
// Parameters are transport containers, not silicon widths: SLOTS/SLOT_BYTES
// default to the sealed cs1 program_slots geometry; SETUP_SLOTS bounds the
// trusted-setup table. Digest convention: byte i sits at [8i +: 8], so the
// wire digest word w is d[w*64 +: 64] (pack() little-endian). Message ids
// and payload word counts come from the GENERATED cn_offsets.svh, never
// from a hand copy.
`timescale 1ns/1ps
module install_service
  import c034_reset_pkg::*;
#(
  parameter int unsigned SLOTS = 2,
  parameter int unsigned SLOT_BYTES = 4096,
  parameter int unsigned SRAM_BYTES = 8192,
  parameter int unsigned SETUP_SLOTS = 8,
  parameter int unsigned WORD_BYTES = 4,
  parameter int unsigned LINE_BYTES = 32,
  parameter int unsigned ADDR_W = 32,
  parameter int unsigned PROG_ID_W = 8,
  parameter int unsigned EPOCH_W = 32,
  localparam int unsigned SLOT_W = (SLOTS <= 1) ? 1 : $clog2(SLOTS)
) (
  input  logic clk,
  input  logic rst_n,

  // ---- trusted setup sideband (one beat; written BEFORE the INSTALL frame
  // of that identity; keyed by the full program_identity). Trusted data from
  // the verified sealed image only; setup_slot_i must equal
  // setup_text_base_i, and a conflicting re-write of an existing key is a
  // protocol fault.
  input  logic         setup_valid_i,
  output logic         setup_ready_o,
  input  logic [63:0]  setup_program_id_i,
  input  logic [63:0]  setup_version_i,
  input  logic [255:0] setup_package_hash_i,   // byte i at [8i +: 8]
  input  logic [63:0]  setup_program_epoch_i,
  input  logic [63:0]  setup_slot_i,           // pre-linked slot address
  input  logic [63:0]  setup_text_base_i,      // SRAM write offset (slot-aligned)
  input  logic [63:0]  setup_text_len_i,
  input  logic [255:0] setup_text_digest_i,

  // ---- G2T word stream: sealed frames; a READ_RESP frame is followed on the
  // same stream by ceil(hbm_chunk_len/8) blob words.
  input  logic         g2t_valid_i,
  output logic         g2t_ready_o,
  input  logic [63:0]  g2t_word_i,

  // ---- T2G word stream: sealed frames out. Header = {msg_type,
  // payload_words, message_seq, 0}; message_seq counts emitted frames 0,1,2..
  output logic         t2g_valid_o,
  input  logic         t2g_ready_i,
  output logic [63:0]  t2g_word_o,

  // ---- sideband riding the INSTALL_ACK frame: the local handle Group must
  // echo in PROGRAM_READY.
  output logic         ack_handle_valid_o,
  output logic [63:0]  ack_handle_o,

  // ---- C030 l1_program_port install channel.
  output logic                    ins_valid_o,
  input  logic                    ins_ready_i,
  output logic [ADDR_W-1:0]       ins_addr_o,
  output logic [WORD_BYTES*8-1:0] ins_data_o,
  output logic [WORD_BYTES-1:0]   ins_be_o,
  output logic [PROG_ID_W-1:0]    ins_prog_id_o,
  output logic [EPOCH_W-1:0]      ins_epoch_o,
  input  logic                    ins_rsp_valid_i,
  output logic                    ins_rsp_ready_o,
  input  logic [2:0]              ins_rsp_code_i,
  input  logic [PROG_ID_W-1:0]    ins_rsp_prog_id_i,
  input  logic [EPOCH_W-1:0]      ins_rsp_epoch_i,

  // ---- execution reference tracking (FET-06): single-cycle pulses,
  // forwarded verbatim to the C030 port by the wrapper, counted per
  // {prog_id, epoch} identity for the invalidate drain.
  input  logic                   exec_ref_inc_i,
  input  logic                   exec_ref_dec_i,
  input  logic [PROG_ID_W-1:0]   exec_ref_prog_id_i,
  input  logic [EPOCH_W-1:0]     exec_ref_epoch_i,

  // ---- exported fetch/residency gate: asserted iff fetch_addr_i lies
  // inside a committed, non-invalidating resident slot range.
  input  logic [ADDR_W-1:0]      fetch_addr_i,
  output logic                   fetch_allow_o,

  // ---- observability.
  output logic [3:0]             state_o,
  output logic [SLOTS-1:0]       resident_o,
  output logic [SLOTS-1:0]       ready_o,
  output logic [SLOTS-1:0]       slot_valid_o,
  output logic                   fault_o,
  output logic                   fault_seen_o,
  output logic [63:0]            fault_code_o,
  output logic [63:0]            read_req_count_o,
  output logic [63:0]            resp_bytes_o,
  output logic [63:0]            ins_beats_o,
  output logic [63:0]            install_count_o,
  output logic [63:0]            warm_hit_count_o,
  output logic [63:0]            inv_count_o
);
  // Sealed wire constants: generated C017 layout (message ids + payload
  // word counts), never a hand copy. The harness stages the committed
  // header under /out; +incdir+/out is required for this slice.
  /* verilator lint_off WIDTHEXPAND */
  `include "cn_offsets.svh"
  /* verilator lint_on WIDTHEXPAND */

  // C030 install response codes (l1_program_port).
  localparam logic [2:0] I_OK = 3'd0, I_ALREADY = 3'd5;

  // Service states (state_o encoding).
  localparam logic [3:0] ST_IDLE       = 4'd0,
                         ST_RECV       = 4'd1,   // pulling the text stream
                         ST_HASH       = 4'd2,   // SHA-256 over staged text
                         ST_COMMIT     = 4'd3,   // writing verified text
                         ST_ACK        = 4'd4,   // INSTALL_ACK emitting
                         ST_WAIT_READY = 4'd5;   // PROGRAM_READY owed

  logic [3:0] state;
  assign state_o = state;

  // -------------------------------------------------------------------
  // Trusted setup table (keyed by full program_identity).
  logic         su_valid  [SETUP_SLOTS];
  logic [63:0]  su_pid    [SETUP_SLOTS];
  logic [63:0]  su_ver    [SETUP_SLOTS];
  logic [63:0]  su_epoch  [SETUP_SLOTS];
  logic [63:0]  su_sadr   [SETUP_SLOTS];   // pre-linked slot address (echo)
  logic [63:0]  su_base   [SETUP_SLOTS];   // SRAM write offset
  logic [63:0]  su_len    [SETUP_SLOTS];
  logic [255:0] su_pkg    [SETUP_SLOTS];
  logic [255:0] su_digest [SETUP_SLOTS];

  function automatic int su_find(input logic [63:0] p, input logic [63:0] v,
                                 input logic [63:0] e, input logic [255:0] h);
    int r;
    r = SETUP_SLOTS;
    for (int i = 0; i < SETUP_SLOTS; i++)
      if (su_valid[i] && su_pid[i] == p && su_ver[i] == v &&
          su_epoch[i] == e && su_pkg[i] == h && r == SETUP_SLOTS)
        r = i;
    return r;
  endfunction

  function automatic int su_free();
    int r;
    r = SETUP_SLOTS;
    for (int i = 0; i < SETUP_SLOTS; i++)
      if (!su_valid[i] && r == SETUP_SLOTS) r = i;
    return r;
  endfunction

  int su_hit, su_slot_i;
  logic su_conflict;
  always_comb begin
    su_hit = su_find(setup_program_id_i, setup_version_i,
                     setup_program_epoch_i, setup_package_hash_i);
    su_slot_i = (su_hit != SETUP_SLOTS) ? su_hit : su_free();
    su_conflict = 1'b0;
    if (su_hit != SETUP_SLOTS)
      su_conflict = su_sadr[SU_W'(su_hit)] != setup_slot_i ||
                    su_base[SU_W'(su_hit)] != setup_text_base_i ||
                    su_len[SU_W'(su_hit)] != setup_text_len_i ||
                    su_digest[SU_W'(su_hit)] != setup_text_digest_i;
  end
  assign setup_ready_o = (su_slot_i != SETUP_SLOTS);
  wire su_take = setup_valid_i && setup_ready_o;

  // -------------------------------------------------------------------
  // Residency records, one per program slot (slot = text_base/SLOT_BYTES).
  logic         r_valid    [SLOTS];
  logic         r_inv      [SLOTS];    // serial point: gate closed, draining
  logic         r_ready    [SLOTS];
  logic [63:0]  r_pid      [SLOTS];
  logic [63:0]  r_ver      [SLOTS];
  logic [63:0]  r_epoch    [SLOTS];
  logic [63:0]  r_handle   [SLOTS];
  logic [63:0]  r_base     [SLOTS];
  logic [63:0]  r_len      [SLOTS];
  logic [255:0] r_pkg      [SLOTS];
  logic [255:0] r_digest   [SLOTS];

  // No-op invalidate (no matching record, or an aborted pending install):
  // answered by an idempotent INVALIDATE_ACK carrying the frame identity.
  logic        noop_pend;
  logic [63:0] noop_pid, noop_ver, noop_epoch;
  logic [255:0] noop_pkg;

  always_comb begin
    for (int k = 0; k < SLOTS; k++) begin
      resident_o[k]   = r_valid[k] && !r_inv[k];
      ready_o[k]      = r_valid[k] && !r_inv[k] && r_ready[k];
      slot_valid_o[k] = r_valid[k];
    end
  end

  // Fetch gate: committed, non-invalidating resident ranges only; every
  // other address (pending / faulted / invalidated / out of the slot mask)
  // is denied even though C030 byte_valid may be set.
  always_comb begin
    fetch_allow_o = 1'b0;
    for (int k = 0; k < SLOTS; k++) begin : gate_rng
      automatic logic [63:0] lo, hi;
      lo = r_base[k];
      hi = r_base[k] + ((r_len[k] + 64'(LINE_BYTES) - 64'd1) / 64'(LINE_BYTES)) *
           64'(LINE_BYTES);
      // Fetchable = committed AND ACK-verified-ready. A committed record
      // whose local_handle was never confirmed by PROGRAM_READY keeps
      // ready_o low AND this gate closed; resident_o (bitmap, committed)
      // is unaffected.
      if (r_valid[k] && !r_inv[k] && r_ready[k] &&
          64'(fetch_addr_i) >= lo && 64'(fetch_addr_i) < hi)
        fetch_allow_o = 1'b1;
    end
  end

  // An invalidation anywhere (record draining or noop ACK owed) holds back
  // new INSTALL payload words: the queued INSTALL misses the warm check and
  // every re-install after an invalidate is a cold pull.
  logic inv_active;
  always_comb begin
    inv_active = noop_pend;
    for (int k = 0; k < SLOTS; k++) inv_active |= r_inv[k];
  end

  // -------------------------------------------------------------------
  // Staging buffer: the unverified candidate text. Never a fetch source.
  logic [7:0] stage [SRAM_BYTES];

  // -------------------------------------------------------------------
  // G2T receive FSM: self-delimiting frames, then a one-cycle DECIDE (the
  // last payload word is only registered after its beat, so frame
  // processing runs one cycle later with pw[] stable).
  localparam logic [2:0] RX_HDR = 3'd0, RX_PAY = 3'd1, RX_DECIDE = 3'd2,
                         RX_BLOB = 3'd3;
  logic [2:0]  rx_state;
  logic [63:0] rx_type;
  logic [5:0]  rx_widx;          // index of the word being consumed
  logic [4:0]  rx_pwords;        // declared payload_words
  logic [63:0] pw [8];           // payload words of the current frame
  logic [63:0] blob_left;        // blob words still owed this chunk
  logic [63:0] blob_bytes;       // real bytes still owed this chunk
  logic [63:0] blob_off;         // real bytes staged so far this chunk

  // Pending install working registers.
  logic [63:0]  cur_pid, cur_ver, cur_epoch, cur_slot, cur_base, cur_len;
  logic [255:0] cur_pkg, cur_digest, cur_measured;
  logic [63:0]  cur_recvd, cur_handle;
  logic [SLOT_W-1:0] cur_sidx;
  logic         req_out;         // a READ_REQ is owed-but-unanswered
  logic         sha_done;

  // Sealed digest mapping: wire digest word w lives at payload word 2+w,
  // digest byte i at [8i +: 8] of the 256-bit field.
  function automatic logic [255:0] pkg_from_pw();
    logic [255:0] r;
    for (int i = 0; i < 32; i++)
      r[8*i +: 8] = pw[2 + i/8][8*(i % 8) +: 8];
    return r;
  endfunction

  function automatic logic [63:0] digest_word(input logic [255:0] d, input int w);
    return d[w*64 +: 64];
  endfunction

  function automatic logic same_id(input logic [63:0] p, input logic [63:0] v,
                                   input logic [63:0] e, input logic [255:0] h,
                                   input logic [63:0] p2, input logic [63:0] v2,
                                   input logic [63:0] e2, input logic [255:0] h2);
    return p == p2 && v == v2 && e == e2 && h == h2;
  endfunction

  function automatic int sidx(input logic [63:0] base);
    return int'(base / 64'(SLOT_BYTES));
  endfunction

  // Frame admissibility: INSTALL/INVALIDATE frames DEFER (stream
  // backpressure) while the service is busy rather than faulting; READ_RESP
  // only while pulling; READY and unknowns are consumed and judged. An
  // INSTALL additionally waits out every pending invalidation so the warm
  // check can never hit a dying record.
  logic rx_admit;
  always_comb begin
    unique case (rx_state)
      RX_HDR:    rx_admit = 1'b1;
      RX_DECIDE: rx_admit = 1'b0;
      RX_BLOB:   rx_admit = 1'b1;
      default: begin
        unique case (rx_type)
          64'(CN_MSG_PROGRAM_INSTALL):
            rx_admit = (state == ST_IDLE) && !inv_active;
          64'(CN_MSG_PROGRAM_INVALIDATE):
            rx_admit = !noop_pend &&
                       ((state == ST_IDLE) || (state == ST_WAIT_READY) ||
                        (state == ST_RECV));
          64'(CN_MSG_PROGRAM_READ_RESP):
            // An unrequested/late READ_RESP is admitted and judged at
            // DECIDE - refusing it here would wedge G2T forever on a
            // partial header (no legal future input can unblock).
            rx_admit = 1'b1;
          default: rx_admit = 1'b1;
        endcase
      end
    endcase
  end
  assign g2t_ready_o = rx_admit;
  wire rx_fire = g2t_valid_i && g2t_ready_o;

  // -------------------------------------------------------------------
  // T2G transmit serializer.
  logic        tx_busy;
  logic [4:0]  tx_len, tx_idx;
  // Derived index containers (transport, not silicon widths).
  localparam int unsigned SA_W = (SRAM_BYTES <= 1) ? 1 : $clog2(SRAM_BYTES);
  localparam int unsigned SU_W = (SETUP_SLOTS <= 1) ? 1 : $clog2(SETUP_SLOTS);
  localparam int unsigned RW = (REF_SLOTS <= 1) ? 1 : $clog2(REF_SLOTS);
  localparam int unsigned TXW = 4;
  logic [63:0] tx_buf [16];
  logic [63:0] tx_seq;
  logic        tx_frame_done;

  assign t2g_valid_o = tx_busy;
  assign t2g_word_o  = tx_buf[TXW'(tx_idx)];
  assign tx_frame_done = tx_busy && t2g_ready_i && (tx_idx + 5'd1 == tx_len);

  assign ack_handle_valid_o = tx_busy && (tx_buf[0] == 64'(CN_MSG_PROGRAM_INSTALL_ACK));
  assign ack_handle_o       = cur_handle;

  task automatic emit_frame(input logic [63:0] mtype, input logic [4:0] nwords);
    tx_buf[0] <= mtype;
    tx_buf[1] <= 64'(nwords);
    tx_buf[2] <= tx_seq;
    tx_buf[3] <= 64'd0;
    tx_len <= 5'(nwords) + 5'd4;
    tx_idx <= '0;
    tx_busy <= 1'b1;
    tx_seq <= tx_seq + 64'd1;
  endtask

  // An invalidating record whose references have all drained.
  function automatic logic inv_due(input int k);
    return r_valid[SLOT_W'(k)] && r_inv[SLOT_W'(k)] &&
           ref_of(PROG_ID_W'(r_pid[SLOT_W'(k)]), EPOCH_W'(r_epoch[SLOT_W'(k)])) == 32'd0;
  endfunction

  // -------------------------------------------------------------------
  // Commit engine: one WORD_BYTES beat per handshake covering
  // ceil(len/LINE)*LINE bytes; the tail of the final line is zero-padded
  // under the same identity. Every install response is consumed; a reject
  // stops new beats and the outstanding ones drain before the fault.
  logic [63:0] commit_total;   // padded byte count to write
  logic [63:0] commit_sent;    // beats issued
  logic [63:0] commit_rsp;     // responses observed
  logic        commit_fail;

  wire [63:0] beats_total = commit_total / 64'(WORD_BYTES);
  wire commit_issue = (state == ST_COMMIT) && !commit_fail &&
                      (commit_sent < beats_total);
  assign ins_valid_o   = commit_issue;
  assign ins_addr_o    = ADDR_W'(cur_base + commit_sent * 64'(WORD_BYTES));
  assign ins_be_o      = '1;
  assign ins_prog_id_o = PROG_ID_W'(cur_pid);
  assign ins_epoch_o   = EPOCH_W'(cur_epoch);
  always_comb begin
    ins_data_o = '0;
    for (int b = 0; b < WORD_BYTES; b++)
      if (commit_sent * 64'(WORD_BYTES) + 64'(b) < cur_len)
        ins_data_o[b*8 +: 8] = stage[SA_W'(commit_sent * 64'(WORD_BYTES) + 64'(b))];
  end
  assign ins_rsp_ready_o = (state == ST_COMMIT);
  wire ins_beat_fire = commit_issue && ins_ready_i;
  wire ins_rsp_fire  = ins_rsp_valid_i && ins_rsp_ready_o;
  wire commit_done   = (commit_sent == beats_total) && (commit_rsp == beats_total);

  // -------------------------------------------------------------------
  // SHA-256 measurement (iterative; one round per cycle; digests the staged
  // bytes strictly over cur_len — the line pad never participates).
  logic [31:0] H [8];
  logic [31:0] W [64];
  logic [31:0] a_r, b_r, c_r, d_r, e_r, f_r, g_r, h_r;
  logic [6:0]  rnd;          // 0..65 per block
  logic [63:0] blk;
  logic [63:0] nblocks;
  logic        sha_busy;

  localparam logic [31:0] K [64] = '{
    32'h428a2f98, 32'h71374491, 32'hb5c0fbcf, 32'he9b5dba5,
    32'h3956c25b, 32'h59f111f1, 32'h923f82a4, 32'hab1c5ed5,
    32'hd807aa98, 32'h12835b01, 32'h243185be, 32'h550c7dc3,
    32'h72be5d74, 32'h80deb1fe, 32'h9bdc06a7, 32'hc19bf174,
    32'he49b69c1, 32'hefbe4786, 32'h0fc19dc6, 32'h240ca1cc,
    32'h2de92c6f, 32'h4a7484aa, 32'h5cb0a9dc, 32'h76f988da,
    32'h983e5152, 32'ha831c66d, 32'hb00327c8, 32'hbf597fc7,
    32'hc6e00bf3, 32'hd5a79147, 32'h06ca6351, 32'h14292967,
    32'h27b70a85, 32'h2e1b2138, 32'h4d2c6dfc, 32'h53380d13,
    32'h650a7354, 32'h766a0abb, 32'h81c2c92e, 32'h92722c85,
    32'ha2bfe8a1, 32'ha81a664b, 32'hc24b8b70, 32'hc76c51a3,
    32'hd192e819, 32'hd6990624, 32'hf40e3585, 32'h106aa070,
    32'h19a4c116, 32'h1e376c08, 32'h2748774c, 32'h34b0bcb5,
    32'h391c0cb3, 32'h4ed8aa4a, 32'h5b9cca4f, 32'h682e6ff3,
    32'h748f82ee, 32'h78a5636f, 32'h84c87814, 32'h8cc70208,
    32'h90befffa, 32'ha4506ceb, 32'hbef9a3f7, 32'hc67178f2
  };
  localparam logic [31:0] IV [8] = '{
    32'h6a09e667, 32'hbb67ae85, 32'h3c6ef372, 32'ha54ff53a,
    32'h510e527f, 32'h9b05688c, 32'h1f83d9ab, 32'h5be0cd19
  };

  // SHA-256 byte at absolute stream position g (0 <= g < nblocks*64): real
  // text from stage, then the 0x80 terminator, zeros, and the 64-bit
  // big-endian bit length in the final 8 bytes.
  function automatic logic [7:0] mbyte(input logic [63:0] g);
    logic [7:0] r;
    logic [63:0] lbits, lpos;
    lbits = cur_len * 64'd8;
    lpos  = nblocks * 64'd64 - 64'd8;
    if (g < cur_len) r = stage[SA_W'(g)];
    else if (g == cur_len) r = 8'h80;
    else if (g >= lpos) r = 8'(lbits >> ((63 - (g - lpos) * 8 - 7)));
    else r = 8'h00;
    return r;
  endfunction

  function automatic logic [31:0] mword(input int t);
    return {mbyte(blk * 64'd64 + 64'(t) * 4), mbyte(blk * 64'd64 + 64'(t) * 4 + 1),
            mbyte(blk * 64'd64 + 64'(t) * 4 + 2), mbyte(blk * 64'd64 + 64'(t) * 4 + 3)};
  endfunction

  function automatic logic [31:0] rr(input logic [31:0] x, input int n);
    return (x >> n) | (x << (32 - n));
  endfunction

  // Round inputs for round index t = rnd-1. For t < 16 the word comes
  // straight from the padded stream; for t >= 16 it is the registered
  // message schedule W[t].
  wire [6:0] t_r = (rnd == 7'd0) ? 7'd0 : (rnd - 7'd1);
  logic [31:0] wt, t1, t2;
  always_comb begin
    if (t_r < 7'd16)
      wt = mword(int'(t_r));
    else if (t_r < 7'd64)
      wt = (rr(W[t_r[5:0] - 6'd2], 17) ^ rr(W[t_r[5:0] - 6'd2], 19) ^ (W[t_r[5:0] - 6'd2] >> 10)) +
           W[t_r[5:0] - 6'd7] +
           (rr(W[t_r[5:0] - 6'd15], 7) ^ rr(W[t_r[5:0] - 6'd15], 18) ^ (W[t_r[5:0] - 6'd15] >> 3)) +
           W[t_r[5:0] - 6'd16];
    else
      wt = 32'd0;
    t1 = h_r + (rr(e_r, 6) ^ rr(e_r, 11) ^ rr(e_r, 25)) +
         ((e_r & f_r) ^ (~e_r & g_r)) + K[t_r[5:0]] + wt;
    t2 = (rr(a_r, 2) ^ rr(a_r, 13) ^ rr(a_r, 22)) +
         ((a_r & b_r) ^ (a_r & c_r) ^ (b_r & c_r));
  end

  // Byte-swap per 32-bit word so digest byte i lands at [8i +: 8].
  function automatic logic [31:0] bs32(input logic [31:0] x);
    return {x[7:0], x[15:8], x[23:16], x[31:24]};
  endfunction
  function automatic logic [255:0] digest_of();
    return {bs32(H[7]), bs32(H[6]), bs32(H[5]), bs32(H[4]),
            bs32(H[3]), bs32(H[2]), bs32(H[1]), bs32(H[0])};
  endfunction

  localparam int unsigned REF_SLOTS = 8;
  // -------------------------------------------------------------------
  // Execution reference counts per {prog_id8, epoch32} identity (mirrors
  // the port's model so the invalidate drain has a real count; the pulses
  // are forwarded to the C030 port by the wrapper).
  logic [15:0]         ref_cnt [REF_SLOTS];
  logic [PROG_ID_W-1:0] ref_pid [REF_SLOTS];
  logic [EPOCH_W-1:0]   ref_ep  [REF_SLOTS];

  function automatic logic [31:0] ref_of(input logic [PROG_ID_W-1:0] p,
                                         input logic [EPOCH_W-1:0] e);
    logic [31:0] r;
    r = 32'd0;
    for (int i = 0; i < REF_SLOTS; i++)
      if (ref_cnt[i] != 16'd0 && ref_pid[i] == p && ref_ep[i] == e)
        r = r + 32'(ref_cnt[i]);
    return r;
  endfunction

  // -------------------------------------------------------------------
  // Counters and fault reporting (one pulse plus a sticky record; the Group
  // resets the Tile on a real fault, the bench keeps observing).
  logic        fault_pulse;
  logic        fault_seen;
  logic [63:0] fault_code_q;
  logic [63:0] cnt_read_req, cnt_resp_bytes, cnt_ins, cnt_install,
               cnt_warm, cnt_inv;
  logic [63:0] handle_ctr;

  assign fault_o         = fault_pulse;
  assign fault_seen_o    = fault_seen;
  assign fault_code_o    = fault_code_q;
  assign read_req_count_o  = cnt_read_req;
  assign resp_bytes_o      = cnt_resp_bytes;
  assign ins_beats_o       = cnt_ins;
  assign install_count_o   = cnt_install;
  assign warm_hit_count_o  = cnt_warm;
  assign inv_count_o       = cnt_inv;

  task automatic raise_fault(input logic [63:0] code);
    fault_pulse  <= 1'b1;
    fault_seen   <= 1'b1;
    fault_code_q <= code;
  endtask

  // -------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= ST_IDLE;
      rx_state <= RX_HDR; rx_widx <= '0; rx_type <= '0; rx_pwords <= '0;
      blob_left <= '0; blob_bytes <= '0; blob_off <= '0;
      cur_pid <= '0; cur_ver <= '0; cur_epoch <= '0; cur_slot <= '0;
      cur_base <= '0; cur_len <= '0; cur_pkg <= '0; cur_digest <= '0;
      cur_measured <= '0; cur_recvd <= '0; cur_handle <= '0; cur_sidx <= '0;
      req_out <= 1'b0; sha_done <= 1'b0;
      noop_pend <= 1'b0;
      noop_pid <= '0; noop_ver <= '0; noop_epoch <= '0; noop_pkg <= '0;
      tx_busy <= 1'b0; tx_len <= '0; tx_idx <= '0; tx_seq <= '0;
      commit_total <= '0; commit_sent <= '0; commit_rsp <= '0; commit_fail <= 1'b0;
      sha_busy <= 1'b0; rnd <= '0; blk <= '0; nblocks <= '0;
      for (int i = 0; i < 8; i++) H[i] <= '0;
      for (int i = 0; i < SETUP_SLOTS; i++) su_valid[i] <= 1'b0;
      for (int i = 0; i < SLOTS; i++) begin
        r_valid[i] <= 1'b0; r_inv[i] <= 1'b0; r_ready[i] <= 1'b0;
        r_pid[i] <= '0; r_ver[i] <= '0; r_epoch[i] <= '0; r_handle[i] <= '0;
        r_base[i] <= '0; r_len[i] <= '0; r_pkg[i] <= '0; r_digest[i] <= '0;
      end
      for (int i = 0; i < REF_SLOTS; i++) begin
        ref_cnt[i] <= '0; ref_pid[i] <= '0; ref_ep[i] <= '0;
      end
      fault_pulse <= 1'b0; fault_seen <= 1'b0; fault_code_q <= '0;
      cnt_read_req <= '0; cnt_resp_bytes <= '0; cnt_ins <= '0;
      cnt_install <= '0; cnt_warm <= '0; cnt_inv <= '0; handle_ctr <= '0;
    end else begin
      fault_pulse <= 1'b0;

      // ---------------- trusted setup beat ----------------
      if (su_take) begin
        if (su_conflict || setup_slot_i != setup_text_base_i) begin
          raise_fault(FC_STREAM_PROTOCOL_ERROR);
        end else if (su_hit == SETUP_SLOTS) begin
          su_valid[SU_W'(su_slot_i)]  <= 1'b1;
          su_pid[SU_W'(su_slot_i)]    <= setup_program_id_i;
          su_ver[SU_W'(su_slot_i)]    <= setup_version_i;
          su_pkg[SU_W'(su_slot_i)]    <= setup_package_hash_i;
          su_epoch[SU_W'(su_slot_i)]  <= setup_program_epoch_i;
          su_sadr[SU_W'(su_slot_i)]   <= setup_slot_i;
          su_base[SU_W'(su_slot_i)]   <= setup_text_base_i;
          su_len[SU_W'(su_slot_i)]    <= setup_text_len_i;
          su_digest[SU_W'(su_slot_i)] <= setup_text_digest_i;
        end
      end

      // ---------------- G2T receive ----------------
      if (rx_fire) begin
        unique case (rx_state)
          RX_HDR: begin
            rx_type <= g2t_word_i;
            rx_widx <= 6'd1;
            rx_state <= RX_PAY;
          end
          RX_PAY: begin
            if (rx_widx == 6'd1) begin : pwords_chk
              // word1 = payload_words; must equal the sealed layout.
              automatic logic [4:0] need;
              unique case (rx_type)
                64'(CN_MSG_PROGRAM_INSTALL):     need = 5'(CN_WORDS_PROGRAM_INSTALL);
                64'(CN_MSG_PROGRAM_READ_RESP):   need = 5'(CN_WORDS_PROGRAM_READ_RESP);
                64'(CN_MSG_PROGRAM_READY):       need = 5'(CN_WORDS_PROGRAM_READY);
                64'(CN_MSG_PROGRAM_INVALIDATE):  need = 5'(CN_WORDS_PROGRAM_INVALIDATE);
                default:                         need = 5'd0;
              endcase
              rx_pwords <= g2t_word_i[4:0];
              if (need == 5'd0 || g2t_word_i != 64'(need))
                raise_fault(FC_STREAM_PROTOCOL_ERROR);
            end else if (rx_widx == 6'd3) begin
              // word3 = header_reserved: must be zero.
              if (g2t_word_i != 64'd0)
                raise_fault(FC_STREAM_PROTOCOL_ERROR);
            end else if (rx_widx >= 6'd4 && rx_widx - 6'd4 < 6'd8) begin
              pw[3'(rx_widx - 6'd4)] <= g2t_word_i;
            end
            rx_widx <= rx_widx + 6'd1;
            // Frame ends when its last word (index pwords+3) is consumed.
            if (rx_widx == {1'b0, rx_pwords} + 6'd3)
              rx_state <= RX_DECIDE;
          end
          RX_BLOB: begin : blob_take
            automatic logic [63:0] n;
            n = (blob_bytes >= 64'd8) ? 64'd8 : blob_bytes;
            for (int b = 0; b < 8; b++)
              if (64'(b) < n)
                stage[SA_W'(cur_recvd + blob_off + 64'(b))] <= g2t_word_i[b*8 +: 8];
            cnt_resp_bytes <= cnt_resp_bytes + n;
            blob_bytes <= blob_bytes - n;
            blob_off   <= blob_off + n;
            if (blob_left == 64'd1) begin
              rx_state <= RX_HDR;
              cur_recvd <= cur_recvd + blob_off + n;
              req_out <= 1'b0;   // this chunk's request is fully served
              if (cur_recvd + blob_off + n == cur_len) begin
                // Stream complete: arm the measurement.
                state <= ST_HASH;
                for (int i = 0; i < 8; i++) H[i] <= IV[i];
                sha_busy <= 1'b1; rnd <= '0; blk <= '0; sha_done <= 1'b0;
              end
            end else begin
              blob_left <= blob_left - 64'd1;
            end
          end
          default: ;
        endcase
      end

      // ---------------- RX_DECIDE: process the completed frame ----------
      if (rx_state == RX_DECIDE) begin : rx_decide
        automatic logic [63:0] p, v, e, sl, hd, clen;
        automatic logic [255:0] h;
        automatic int si, k;
        // Default back to the header; RX_BLOB (below) overrides it.
        rx_state <= RX_HDR;
        p = pw[0]; v = pw[1]; e = pw[6]; h = pkg_from_pw();
        unique case (rx_type)
          64'(CN_MSG_PROGRAM_INSTALL): begin
            sl = pw[7];
            si = su_find(p, v, e, h);
            if (si == SETUP_SLOTS) begin
              raise_fault(FC_INVALID_DESCRIPTOR);   // no trusted setup: closed
            end else if (sl != su_base[si] || sl != su_sadr[si]) begin
              raise_fault(FC_INVALID_DESCRIPTOR);   // wire slot != trusted base
            end else if (p >= 64'd1 << PROG_ID_W || e >= 64'd1 << EPOCH_W) begin
              // The identity must fit the C030 port's narrower containers;
              // otherwise the byte-owner tag would alias another program.
              raise_fault(FC_ADDRESS_FAULT);
            end else if (su_len[si] == 64'd0 || su_base[si] >= 64'(SRAM_BYTES) ||
                         (su_base[si] % 64'(SLOT_BYTES)) != 64'd0 ||
                         su_base[si] + 64'(SLOT_BYTES) > 64'(SRAM_BYTES)) begin
              raise_fault(FC_ADDRESS_FAULT);
            end else if (su_len[si] > 64'(SLOT_BYTES)) begin
              raise_fault(FC_L2_CAPACITY_FAULT);
            end else begin
              k = sidx(su_base[si]);
              if (k >= SLOTS) begin
                raise_fault(FC_ADDRESS_FAULT);
              end else if (r_valid[k] && !r_inv[k] &&
                  same_id(r_pid[k], r_ver[k], r_epoch[k], r_pkg[k], p, v, e, h)) begin
                // WARM HIT: identical identity resident - re-ACK the stored
                // measurement and handle. ZERO READ_REQ.
                cur_pid <= p; cur_ver <= v; cur_epoch <= e; cur_pkg <= h;
                cur_slot <= sl; cur_base <= su_base[si];
                cur_len <= r_len[k]; cur_digest <= r_digest[k];
                cur_measured <= r_digest[k]; cur_handle <= r_handle[k];
                cur_sidx <= SLOT_W'(k);
                cnt_warm <= cnt_warm + 64'd1;
                state <= ST_ACK;
              end else if (r_valid[k]) begin
                // Slot held by a different (or draining) identity: CMP-18.
                raise_fault(FC_INVALID_DESCRIPTOR);
              end else begin
                // COLD INSTALL: pull the stream, then measure and commit.
                cur_pid <= p; cur_ver <= v; cur_epoch <= e; cur_pkg <= h;
                cur_slot <= sl; cur_base <= su_base[si];
                cur_len <= su_len[si]; cur_digest <= su_digest[si];
                cur_measured <= '0; cur_recvd <= '0; cur_handle <= '0;
                cur_sidx <= SLOT_W'(k); req_out <= 1'b0;
                commit_total <= ((su_len[si] + 64'(LINE_BYTES) - 64'd1) /
                                 64'(LINE_BYTES)) * 64'(LINE_BYTES);
                commit_sent <= '0; commit_rsp <= '0; commit_fail <= 1'b0;
                nblocks <= (su_len[si] + 64'd72) / 64'd64;
                state <= ST_RECV;
              end
            end
          end

          64'(CN_MSG_PROGRAM_READ_RESP): begin
            clen = pw[7];
            if (state != ST_RECV || !req_out ||
                !same_id(p, v, e, h, cur_pid, cur_ver, cur_epoch, cur_pkg) ||
                clen == 64'd0 || clen > cur_len - cur_recvd) begin
              // Fail stop: the pending install drops and the stream
              // re-synchronises at the next header. The declared blob is
              // NEVER drained on a header fault - an unbounded chunk_len
              // cannot hang the service (any blob words that still arrive
              // parse as garbage frames and fault themselves out).
              raise_fault(FC_STREAM_PROTOCOL_ERROR);
              cur_recvd <= '0; req_out <= 1'b0;
              blob_left <= '0; blob_bytes <= '0; blob_off <= '0;
              state <= ST_IDLE;
            end else begin
              blob_bytes <= clen;
              blob_left <= (clen + 64'd7) / 64'd8;
              blob_off <= '0;
              rx_state <= RX_BLOB;
            end
          end

          64'(CN_MSG_PROGRAM_READY): begin
            hd = pw[7];
            if (state != ST_WAIT_READY ||
                !same_id(p, v, e, h, cur_pid, cur_ver, cur_epoch, cur_pkg) ||
                hd != cur_handle) begin
              raise_fault(FC_STREAM_PROTOCOL_ERROR);
            end else begin
              r_ready[cur_sidx] <= 1'b1;
              state <= ST_IDLE;
            end
          end

          64'(CN_MSG_PROGRAM_INVALIDATE): begin
            // A matching record goes invalidating at once (the gate closes
            // now, the ACK follows the reference drain, the record drops at
            // ACK emit). An INVALIDATE naming the pending install aborts it.
            // Every ACK comes from exactly one source - a draining record,
            // or the idempotent noop queue - so nothing double-answers, and
            // no path touches an unrelated pending install's cursor or
            // state.
            begin : inv_case
            automatic logic abort_pending, any_hit;
            abort_pending = 1'b0;
            if (state == ST_RECV || state == ST_WAIT_READY)
              abort_pending = same_id(p, v, e, h,
                                      cur_pid, cur_ver, cur_epoch, cur_pkg);
            any_hit = 1'b0;
            for (k = 0; k < SLOTS; k++)
              if (r_valid[k] && !r_inv[k] &&
                  same_id(r_pid[k], r_ver[k], r_epoch[k], r_pkg[k], p, v, e, h)) begin
                r_inv[k] <= 1'b1;          // serial point: fetch gate closes now
                any_hit = 1'b1;
              end
            if (abort_pending) begin
              // The pending install dies unpublished. In WAIT_READY its
              // record was marked above and answers itself; a RECV abort
              // has no record, so the noop queue carries the ACK.
              cur_recvd <= '0; req_out <= 1'b0;
              sha_busy <= 1'b0;
              blob_left <= '0; blob_bytes <= '0; blob_off <= '0;
              state <= ST_IDLE;
            end
            if (!any_hit) begin
              // Nothing resident and (if abort_pending) nothing committed:
              // the idempotent ACK carries the frame's identity.
              noop_pend <= 1'b1;
              noop_pid <= p; noop_ver <= v; noop_epoch <= e; noop_pkg <= h;
            end
            end : inv_case
          end

          default: raise_fault(FC_STREAM_PROTOCOL_ERROR);
        endcase
      end

      // ---------------- T2G emit ----------------
      if (tx_busy && t2g_ready_i) begin
        if (tx_frame_done) tx_busy <= 1'b0;
        else tx_idx <= tx_idx + 5'd1;
      end
      if (!tx_busy) begin : tx_emit
        automatic logic inv_emitted;
        inv_emitted = 1'b0;
        if (state == ST_RECV && !req_out) begin
          emit_frame(64'(CN_MSG_PROGRAM_READ_REQ), 5'(CN_WORDS_PROGRAM_READ_REQ));
          tx_buf[4] <= cur_pid; tx_buf[5] <= cur_ver;
          tx_buf[6] <= digest_word(cur_pkg, 0);
          tx_buf[7] <= digest_word(cur_pkg, 1);
          tx_buf[8] <= digest_word(cur_pkg, 2);
          tx_buf[9] <= digest_word(cur_pkg, 3);
          tx_buf[10] <= cur_epoch;
          cnt_read_req <= cnt_read_req + 64'd1;
          req_out <= 1'b1;
        end else if (state == ST_ACK) begin
          emit_frame(64'(CN_MSG_PROGRAM_INSTALL_ACK), 5'(CN_WORDS_PROGRAM_INSTALL_ACK));
          tx_buf[4] <= cur_pid; tx_buf[5] <= cur_ver;
          tx_buf[6] <= digest_word(cur_pkg, 0);
          tx_buf[7] <= digest_word(cur_pkg, 1);
          tx_buf[8] <= digest_word(cur_pkg, 2);
          tx_buf[9] <= digest_word(cur_pkg, 3);
          tx_buf[10] <= cur_epoch;
          tx_buf[11] <= cur_len;
          tx_buf[12] <= digest_word(cur_measured, 0);
          tx_buf[13] <= digest_word(cur_measured, 1);
          tx_buf[14] <= digest_word(cur_measured, 2);
          tx_buf[15] <= digest_word(cur_measured, 3);
          state <= ST_WAIT_READY;
        end else begin
          // A drained invalidating record emits its INVALIDATE_ACK and the
          // record drops with the frame; a queued noop ACK follows.
          for (int k = 0; k < SLOTS; k++) begin
            if (!inv_emitted && inv_due(k)) begin
              inv_emitted = 1'b1;
              emit_frame(64'(CN_MSG_INVALIDATE_ACK), 5'(CN_WORDS_INVALIDATE_ACK));
              tx_buf[4] <= r_pid[k]; tx_buf[5] <= r_ver[k];
              tx_buf[6] <= digest_word(r_pkg[k], 0);
              tx_buf[7] <= digest_word(r_pkg[k], 1);
              tx_buf[8] <= digest_word(r_pkg[k], 2);
              tx_buf[9] <= digest_word(r_pkg[k], 3);
              tx_buf[10] <= r_epoch[k];
              r_valid[k] <= 1'b0; r_ready[k] <= 1'b0; r_inv[k] <= 1'b0;
              cnt_inv <= cnt_inv + 64'd1;
            end
          end
          if (!inv_emitted && noop_pend) begin
            emit_frame(64'(CN_MSG_INVALIDATE_ACK), 5'(CN_WORDS_INVALIDATE_ACK));
            tx_buf[4] <= noop_pid; tx_buf[5] <= noop_ver;
            tx_buf[6] <= digest_word(noop_pkg, 0);
            tx_buf[7] <= digest_word(noop_pkg, 1);
            tx_buf[8] <= digest_word(noop_pkg, 2);
            tx_buf[9] <= digest_word(noop_pkg, 3);
            tx_buf[10] <= noop_epoch;
            noop_pend <= 1'b0;
            cnt_inv <= cnt_inv + 64'd1;
          end
        end
      end

      // ---------------- SHA-256 engine ----------------
      if (sha_busy) begin
        if (rnd == 7'd0) begin
          for (int t = 0; t < 16; t++) W[t] <= mword(t);
          a_r <= H[0]; b_r <= H[1]; c_r <= H[2]; d_r <= H[3];
          e_r <= H[4]; f_r <= H[5]; g_r <= H[6]; h_r <= H[7];
          rnd <= 7'd1;
        end else if (rnd <= 7'd64) begin
          if (t_r >= 7'd16) W[t_r[5:0]] <= wt;
          h_r <= g_r; g_r <= f_r; f_r <= e_r; e_r <= d_r + t1;
          d_r <= c_r; c_r <= b_r; b_r <= a_r; a_r <= t1 + t2;
          rnd <= rnd + 7'd1;
        end else begin
          H[0] <= H[0] + a_r; H[1] <= H[1] + b_r; H[2] <= H[2] + c_r;
          H[3] <= H[3] + d_r; H[4] <= H[4] + e_r; H[5] <= H[5] + f_r;
          H[6] <= H[6] + g_r; H[7] <= H[7] + h_r;
          rnd <= '0;
          if (blk + 64'd1 == nblocks) begin
            sha_busy <= 1'b0;
            sha_done <= 1'b1;
          end else blk <= blk + 64'd1;
        end
      end

      if (state == ST_HASH && sha_done) begin
        sha_done <= 1'b0;
        cur_measured <= digest_of();
        if (digest_of() != cur_digest) begin
          raise_fault(FC_INVALID_DESCRIPTOR);
          cur_recvd <= '0;
          state <= ST_IDLE;
        end else begin
          state <= ST_COMMIT;
        end
      end

      // ---------------- commit responses ----------------
      if (ins_beat_fire) begin
        commit_sent <= commit_sent + 64'd1;
        cnt_ins <= cnt_ins + 64'd1;
      end
      if (ins_rsp_fire) begin
        commit_rsp <= commit_rsp + 64'd1;
        if (ins_rsp_code_i != I_OK && ins_rsp_code_i != I_ALREADY)
          commit_fail <= 1'b1;
      end
      if (state == ST_COMMIT && commit_done) begin
        if (commit_fail) begin
          raise_fault(FC_INVALID_DESCRIPTOR);
          cur_recvd <= '0;
          state <= ST_IDLE;
        end else begin
          // Publish residency exactly once, at commit completion.
          r_valid[cur_sidx]  <= 1'b1;
          r_inv[cur_sidx]    <= 1'b0;
          r_ready[cur_sidx]  <= 1'b0;
          r_pid[cur_sidx]    <= cur_pid;
          r_ver[cur_sidx]    <= cur_ver;
          r_epoch[cur_sidx]  <= cur_epoch;
          r_pkg[cur_sidx]    <= cur_pkg;
          r_handle[cur_sidx] <= handle_ctr;
          r_base[cur_sidx]   <= cur_base;
          r_len[cur_sidx]    <= cur_len;
          r_digest[cur_sidx] <= cur_measured;
          cur_handle <= handle_ctr;
          handle_ctr <= handle_ctr + 64'd1;
          cnt_install <= cnt_install + 64'd1;
          state <= ST_ACK;
        end
      end

      // ---------------- execution reference bookkeeping ---------------
      if (exec_ref_inc_i && exec_ref_dec_i) begin
        raise_fault(FC_STREAM_PROTOCOL_ERROR);
      end else if (exec_ref_inc_i) begin : ref_inc
        automatic int hit, free_i;
        hit = REF_SLOTS; free_i = REF_SLOTS;
        for (int i = 0; i < REF_SLOTS; i++) begin
          if (ref_cnt[i] != 16'd0 && ref_pid[i] == exec_ref_prog_id_i &&
              ref_ep[i] == exec_ref_epoch_i && hit == REF_SLOTS) hit = i;
          if (ref_cnt[i] == 16'd0 && free_i == REF_SLOTS) free_i = i;
        end
        if (hit != REF_SLOTS) ref_cnt[RW'(hit)] <= ref_cnt[RW'(hit)] + 16'd1;
        else if (free_i != REF_SLOTS) begin
          ref_cnt[RW'(free_i)] <= 16'd1;
          ref_pid[RW'(free_i)] <= exec_ref_prog_id_i;
          ref_ep[RW'(free_i)] <= exec_ref_epoch_i;
        end else begin
          raise_fault(FC_ENGINE_INTERNAL_FAULT);
        end
      end else if (exec_ref_dec_i) begin : ref_dec
        automatic int hit;
        hit = REF_SLOTS;
        for (int i = 0; i < REF_SLOTS; i++)
          if (ref_cnt[i] != 16'd0 && ref_pid[i] == exec_ref_prog_id_i &&
              ref_ep[i] == exec_ref_epoch_i && hit == REF_SLOTS) hit = i;
        if (hit != REF_SLOTS) ref_cnt[RW'(hit)] <= ref_cnt[RW'(hit)] - 16'd1;
        else raise_fault(FC_STREAM_PROTOCOL_ERROR);
      end
    end
  end

  // Elaboration-time guards: transport containers must hold the sealed
  // geometry; a wrong container fails the build, never the protocol.
  initial begin
    if (SLOT_BYTES % LINE_BYTES != 0)
      $fatal(1, "C039_ASSERT install_service SLOT_BYTES must be a fetch-line multiple");
    if (SLOTS * SLOT_BYTES > SRAM_BYTES)
      $fatal(1, "C039_ASSERT install_service slot mask exceeds the program SRAM");
    if (WORD_BYTES != 4 || LINE_BYTES != 32)
      $fatal(1, "C039_ASSERT install_service built for the C030 beat/line geometry");
  end

  // Live invariants a mutation cannot mask.
  always @(posedge clk) begin
    if (rst_n) begin
      if (commit_rsp > commit_sent)
        $error("C039_ASSERT install_service install responses outnumber beats");
      if (state == ST_ACK && cur_len == 64'd0)
        $error("C039_ASSERT install_service ACK would report zero length");
      for (int k = 0; k < SLOTS; k++) begin
        if (r_ready[k] && !r_valid[k])
          $error("C039_ASSERT install_service ready bit on an absent record");
        if (r_valid[k] && r_base[k] % 64'(SLOT_BYTES) != 64'd0)
          $error("C039_ASSERT install_service resident base escapes its slot");
        if (r_inv[k] && !r_valid[k])
          $error("C039_ASSERT install_service invalidating an absent record");
      end
      if (tx_busy && tx_idx >= tx_len)
        $error("C039_ASSERT install_service tx index escaped its frame");
      if (rx_state == RX_BLOB && blob_left == 64'd0)
        $error("C039_ASSERT install_service blob drain underflow");
      if (state == ST_RECV && !req_out && cur_recvd >= cur_len)
        $error("C039_ASSERT install_service pull overshot the trusted length");
    end
  end
endmodule
