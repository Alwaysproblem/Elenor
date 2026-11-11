// SystemVerilog receive model of one Tile link (C017). It includes ONLY the generated cn_offsets.svh and
// re-implements the rules of the generated Python Rx and the C consumer. Run with Verilator:
//   +cases=<file>  one message per line "<channel index> <dir index> <hex bytes>"
// It prints one canonical line per message, identical to wire.Rx.line().
module cn_rx;
`include "cn_offsets.svh"

  localparam int MAX_BYTES = 4096;
  typedef longint unsigned u64;
  typedef byte unsigned u8;

  longint unsigned next_seq [CN_LINKS];
  longint unsigned dd_txn   [CN_DEDUP_ENTRIES];
  longint unsigned dd_stage [CN_DEDUP_ENTRIES];
  int              dd_msg   [CN_DEDUP_ENTRIES];
  int              dd_n;
  longint unsigned rq_txn   [CN_DEDUP_ENTRIES];
  int              rq_msg   [CN_DEDUP_ENTRIES];
  int              rq_n;
  longint unsigned live     [CN_CONTEXT_SLOTS];
  longint unsigned effects;

  byte unsigned    blob [MAX_BYTES];
  longint unsigned w    [64];
  int              nw;

  function automatic longint unsigned word(input int i);
    longint unsigned v;
    v = 0;
    for (int k = 7; k >= 0; k--) v = (v << 8) | u64'(blob[i*8 + k]);
    return v;
  endfunction

  function automatic bit has_req(input longint unsigned txn, input int msg);
    for (int i = 0; i < rq_n; i++) if (rq_txn[i] == txn && rq_msg[i] == msg) return 1'b1;
    return 1'b0;
  endfunction

  function automatic int hexval(input byte c);
    if (c >= "0" && c <= "9") return c - "0";
    if (c >= "a" && c <= "f") return c - "a" + 10;
    if (c >= "A" && c <= "F") return c - "A" + 10;
    return -1;
  endfunction

  function automatic int deliver(input int ch, input int dir, input int n);
    longint unsigned type_, payload, seq, reserved, v, gen, txn, stage;
    int m, total, link, slot;
    nw = 0;
    if (n < CN_HEADER_WORDS * 8) return CN_ERR_TRUNCATED;
    type_ = word(0); payload = word(1); seq = word(2); reserved = word(3);
    if (type_ >= u64'(CN_MSG_COUNT)) return CN_ERR_UNKNOWN_MSG;
    if (reserved != 0) return CN_ERR_RESERVED;
    m = int'(type_);
    if (payload != u64'(cn_msg_words(m))) return CN_ERR_LENGTH_MISMATCH;
    total = (CN_HEADER_WORDS + cn_msg_words(m)) * 8;
    if (n < total) return CN_ERR_TRUNCATED;
    if (n > total) return CN_ERR_TRAILING;
    if (cn_msg_chan(m) != ch || cn_msg_dir(m) != dir) return CN_ERR_WRONG_PORT;
    for (int i = 0; i < cn_msg_words(m); i++) w[i] = word(CN_HEADER_WORDS + i);
    nw = cn_msg_words(m);
    for (int i = 0; i < cn_msg_nfields(m); i++) begin
      v = w[cn_field_off(m, i)];
      case (cn_field_kind(m, i))
        1: if (v >= cn_field_limit(m, i)) return CN_ERR_ENUM_RANGE;
        2: if (v > 1) return CN_ERR_BOOL_RANGE;
        3: if ((v & ~cn_field_limit(m, i)) != 0) return CN_ERR_BITSET_RANGE;
        default: ;
      endcase
    end
    for (int i = 0; i < cn_msg_nconds(m); i++) begin
      if (cn_cond_kind(m, i) == 0 && w[cn_cond_a(m, i)] != w[cn_cond_b(m, i)]) return CN_ERR_CONDITION;
      if (cn_cond_kind(m, i) == 1 &&
          ((w[cn_cond_a(m, i)] != 0) != (w[cn_cond_b(m, i)] == u64'(cn_cond_value(m, i)))))
        return CN_ERR_CONDITION;
    end
    slot = -1;
    if (cn_msg_ctx_slot(m) >= 0) begin
      if (w[cn_msg_ctx_slot(m)] >= u64'(CN_CONTEXT_SLOTS)) return CN_ERR_CAPACITY;
      slot = int'(w[cn_msg_ctx_slot(m)]);
    end
    link = ch * CN_DIRS + dir;
    if (seq < next_seq[link]) return CN_ERR_TRANSPORT_REPLAY;
    if (seq > next_seq[link]) return CN_ERR_INCOMPLETE;
    next_seq[link] = next_seq[link] + 1;
    gen = (cn_msg_ctx_gen(m) >= 0) ? w[cn_msg_ctx_gen(m)] : 0;
    if (slot >= 0 && gen < live[slot]) return CN_ERR_STALE_GENERATION;
    txn = (cn_msg_txn(m) >= 0) ? w[cn_msg_txn(m)] : 0;
    if (cn_msg_txn(m) >= 0) begin
      if (cn_msg_ack_for(m) >= 0 && !has_req(txn, int'(w[cn_msg_ack_for(m)]))) return CN_ERR_ACK_MISMATCH;
      stage = (cn_msg_stage(m) >= 0) ? w[cn_msg_stage(m)] : 0;
      for (int i = 0; i < dd_n; i++)
        if (dd_txn[i] == txn && dd_msg[i] == m && dd_stage[i] == stage) return CN_ERR_DUPLICATE_STAGE;
      if (dd_n >= CN_DEDUP_ENTRIES) return CN_ERR_CAPACITY;
      dd_txn[dd_n] = txn; dd_msg[dd_n] = m; dd_stage[dd_n] = stage; dd_n++;
      if (!has_req(txn, m)) begin
        rq_txn[rq_n] = txn; rq_msg[rq_n] = m; rq_n++;
      end
    end
    if (slot >= 0 && cn_msg_establishes(m) != 0) live[slot] = gen;
    effects = effects + 1;
    return CN_ERR_OK;
  endfunction

  string path, line, hex, out;
  int fd, ch, dir, n, code, hi, lo, r;
  longint unsigned sum;

  initial begin
    for (int i = 0; i < CN_LINKS; i++) next_seq[i] = 0;
    for (int i = 0; i < CN_CONTEXT_SLOTS; i++) live[i] = 0;
    dd_n = 0; rq_n = 0; effects = 0;
    if (!$value$plusargs("cases=%s", path)) begin
      $display("cn_rx: +cases=<file> required");
      $finish;
    end
    fd = $fopen(path, "r");
    if (fd == 0) begin
      $display("cn_rx: cannot open %s", path);
      $finish;
    end
    while (!$feof(fd)) begin
      line = "";
      r = $fgets(line, fd);
      if (r == 0) break;
      hex = "";
      r = $sscanf(line, "%d %d", ch, dir);
      if (r < 2) continue;
      void'($sscanf(line, "%d %d %s", ch, dir, hex));
      n = 0;
      for (int i = 0; i + 1 < hex.len(); i += 2) begin
        hi = hexval(hex[i]); lo = hexval(hex[i+1]);
        if (hi < 0 || lo < 0) break;
        blob[n] = u8'(hi * 16 + lo);
        n++;
      end
      code = deliver(ch, dir, n);
      sum = 0;
      for (int i = 0; i < CN_CONTEXT_SLOTS; i++) sum = sum + live[i];
      out = $sformatf("%s eff=%0d dd=%0d seq=", cn_err_name(code), effects, dd_n);
      for (int i = 0; i < CN_LINKS; i++) out = {out, (i == 0) ? "" : ",", $sformatf("%0d", next_seq[i])};
      out = {out, $sformatf(" live=%0d f=", sum)};
      if (code != CN_ERR_OK) out = {out, "-"};
      else for (int i = 0; i < nw; i++) out = {out, (i == 0) ? "" : ",", $sformatf("%0x", w[i])};
      $display("%s", out);
    end
    $finish;
  end
endmodule
