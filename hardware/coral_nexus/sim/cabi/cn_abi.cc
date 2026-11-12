/* cn_abi.cc — CN C ABI multi-slot full-wire implementation over
 * Vcn_protocol_stub (C025 / W1-03).
 *
 * Phase discipline (W1 plan §9.1-§9.2):
 *   - bp_layout: pure read-only query of `cn_slot_table` (derived from
 * `cn_layout.json`); never mutates functional state and may be called
 * repeatedly.
 *   - bp_drive (P0): freezes per-slot inputs into preallocated channel buffers
 * without calling `eval()`. Mutating a G2T slot whose previous frame is still
 * stalled (`valid && !ready`) records `CN_EP_PAYLOAD_READY_MISMATCH` and
 * returns `CN_ERR_STATE`.
 *   - bp_tick (P1..P3):
 *       P1: apply frozen pins, `clk=0`, `eval()`, sample `g2t_ready` and
 * `t2g_valid`/`t2g_frame` at the SOLE pre-edge handshake decision point. P2:
 * `clk=1`, `eval()` (the ONLY rising edge in the tick). P3: `clk=0`, `eval()`;
 * commit ONLY the P1-accepted set into the bounded collect ring. Outputs
 * asserted by the DUT at P2 are staged candidates for the NEXT tick only.
 *   - bp_collect: copies committed records into caller's buffer (`CN_ERR_RANGE`
 * without truncation or loss when `buf_len` is too small; `CN_ERR_STATE` on
 * duplicate collect in the same tick; uncollected records survive across
 * subsequent `bp_drive -> bp_tick` cycles).
 */

#include "cn_abi.h"

#include <cstdlib>
#include <cstring>

#include "Vcn_protocol_stub.h"
#include "cn_slot_table.h"
#include "verilated.h"

namespace {

constexpr uint32_t kChannelCount = CN_CHANNEL_COUNT;
constexpr uint32_t kWords32PerChannel =
    34u;  // 17 x 64-bit words = 136 bytes = 1088 bits
constexpr uint32_t kMaxCollectRecords = 256u;

struct G2tChannelBuf {
  bool valid = false;
  uint32_t slot_id = 0;
  uint32_t byte_len = 0;
  uint8_t bytes[CN_MAX_FRAME_BYTES]{};
};

struct P1T2gSample {
  bool accepted = false;
  uint32_t slot_id = 0;
  uint32_t byte_len = 0;
  uint8_t bytes[CN_MAX_FRAME_BYTES]{};
};

struct CollectRecord {
  uint32_t slot_id = 0;
  uint32_t byte_len = 0;
  uint8_t is_t2g =
      0;  // 0 = G2T acceptance receipt (0 trailing bytes), 1 = T2G wire frame
  uint8_t bytes[CN_MAX_FRAME_BYTES]{};
};

int find_slot(uint32_t slot_id) {
  for (uint32_t i = 0; i < CN_SLOT_COUNT; ++i) {
    if (cn_slot_table[i].slot_id == slot_id) return static_cast<int>(i);
  }
  return -1;
}

uint32_t slot_for_link(uint32_t ch, uint32_t dir) {
  for (uint32_t i = 0; i < CN_SLOT_COUNT; ++i) {
    if (cn_slot_table[i].channel_index == ch &&
        cn_slot_table[i].dir_index == dir) {
      return cn_slot_table[i].slot_id;
    }
  }
  return 0;
}

void write_g2t_channel_pins(Vcn_protocol_stub* top, uint32_t ch,
                            const G2tChannelBuf& buf) {
  const uint32_t base_w32 = ch * kWords32PerChannel;
  for (uint32_t i = 0; i < kWords32PerChannel; ++i) {
    top->g2t_frame_flat[base_w32 + i] = 0u;
  }
  if (!buf.valid || buf.byte_len == 0) {
    return;
  }
  uint32_t n_w32 = (buf.byte_len + 3u) / 4u;
  if (n_w32 > kWords32PerChannel) n_w32 = kWords32PerChannel;
  for (uint32_t i = 0; i < n_w32; ++i) {
    uint32_t w = 0;
    const uint32_t off = i * 4u;
    const uint32_t rem = (buf.byte_len > off) ? (buf.byte_len - off) : 0u;
    const uint32_t copy_n = rem < 4u ? rem : 4u;
    std::memcpy(&w, buf.bytes + off, copy_n);
    top->g2t_frame_flat[base_w32 + i] = w;
  }
}

void read_t2g_channel_pins(const Vcn_protocol_stub* top, uint32_t ch,
                           uint32_t byte_len, uint8_t out[CN_MAX_FRAME_BYTES]) {
  std::memset(out, 0, CN_MAX_FRAME_BYTES);
  const uint32_t base_w32 = ch * kWords32PerChannel;
  uint32_t n_w32 = (byte_len + 3u) / 4u;
  if (n_w32 > kWords32PerChannel) n_w32 = kWords32PerChannel;
  for (uint32_t i = 0; i < n_w32; ++i) {
    const uint32_t w = top->t2g_frame_flat[base_w32 + i];
    const uint32_t off = i * 4u;
    const uint32_t rem = (byte_len > off) ? (byte_len - off) : 0u;
    const uint32_t copy_n = rem < 4u ? rem : 4u;
    std::memcpy(out + off, &w, copy_n);
  }
}

uint32_t popcount5(uint32_t mask) {
  uint32_t c = 0;
  for (uint32_t i = 0; i < kChannelCount; ++i) {
    if ((mask >> i) & 1u) ++c;
  }
  return c;
}

}  // namespace

// Defined at global scope to match `struct CnSim` forward declaration in
// `cn_abi.h`.
struct CnSim {
  CnBeginCfg cfg{};
  VerilatedContext* ctx = nullptr;
  Vcn_protocol_stub* top = nullptr;
  uint8_t* slab = nullptr;
  G2tChannelBuf g2t[kChannelCount]{};
  bool g2t_stalled[kChannelCount]{};  // true when previous tick left `valid &&
                                      // !ready`
  uint8_t t2g_ready_cfg[kChannelCount]{};
  CollectRecord* ring = nullptr;
  uint32_t ring_head = 0;
  uint32_t ring_count = 0;
  bool drive_frozen = false;
  bool collect_armed = false;
  uint64_t ticks = 0;
  uint64_t posedges = 0;
  uint64_t drives = 0;
  uint64_t collects = 0;
  uint64_t accepted = 0;
  uint64_t returned = 0;
  uint32_t fatal_code = 0;
  uint32_t proto_err = 0;
  uint32_t seq_gap_seen = 0;
  uint64_t staged_after_p3 = 0;
};

namespace {

bool push_record(CnSim* sim, const CollectRecord& rec) {
  if (sim->ring_count >= kMaxCollectRecords) {
    return false;
  }
  const uint32_t idx = (sim->ring_head + sim->ring_count) % kMaxCollectRecords;
  sim->ring[idx] = rec;
  sim->ring_count += 1;
  return true;
}

uint32_t serialized_collect_bytes(const CnSim* sim) {
  uint32_t total = 8u;  // record_count (u32) + total_bytes (u32)
  for (uint32_t i = 0; i < sim->ring_count; ++i) {
    const CollectRecord& r =
        sim->ring[(sim->ring_head + i) % kMaxCollectRecords];
    total += 8u + (r.is_t2g ? r.byte_len
                            : 0u);  // slot_id (u32) + flag_len (u32) + payload
  }
  return total;
}

}  // namespace

extern "C" {

uint32_t bp_abi_info(uint8_t out_hash[129]) {
  if (out_hash == nullptr) return CN_ABI_VERSION;
  std::memcpy(out_hash, kAbiHash128, 129);
  return CN_ABI_VERSION;
}

struct CnSim* bp_begin(const struct CnBeginCfg* cfg) {
  if (cfg == nullptr || cfg->abi_version != CN_ABI_VERSION ||
      cfg->tile_count < 1) {
    return nullptr;
  }
  auto* sim = new (std::nothrow) CnSim();
  if (sim == nullptr) return nullptr;
  sim->cfg = *cfg;
  sim->ctx = new (std::nothrow) VerilatedContext();
  sim->top = new (std::nothrow) Vcn_protocol_stub(sim->ctx);
  sim->slab = new (std::nothrow) uint8_t[CN_SLAB_TOTAL_BYTES]();
  sim->ring = new (std::nothrow) CollectRecord[kMaxCollectRecords]();
  if (sim->ctx == nullptr || sim->top == nullptr || sim->slab == nullptr ||
      sim->ring == nullptr) {
    bp_destroy(sim);
    return nullptr;
  }
  sim->ctx->debug(0);
  sim->ctx->randReset(0);
  for (uint32_t c = 0; c < kChannelCount; ++c) {
    sim->g2t[c].slot_id = slot_for_link(c, 0u);
    sim->t2g_ready_cfg[c] = 1u;
  }

  const uint8_t flags = cfg->reserved[0];
  sim->top->clk = 0;
  sim->top->g2t_valid = 0;
  sim->top->g2t_words_flat = 0;
  for (uint32_t i = 0; i < kChannelCount * kWords32PerChannel; ++i) {
    sim->top->g2t_frame_flat[i] = 0u;
  }
  sim->top->t2g_ready = 0x1Fu;
  sim->top->cfg_ready_seed = cfg->ready_seed;
  sim->top->cfg_inject_fatal_at_tick =
      (cfg->inject_fatal_at_tick < 0)
          ? 0xFFFFFFFFu
          : static_cast<uint32_t>(cfg->inject_fatal_at_tick);
  sim->top->cfg_inject_abort_at_tick =
      (cfg->inject_abort_at_tick < 0)
          ? 0xFFFFFFFFu
          : static_cast<uint32_t>(cfg->inject_abort_at_tick);
  sim->top->cfg_corrupt_t2g = (flags & CN_STUB_FLAG_CORRUPT_T2G) ? 1u : 0u;
  sim->top->cfg_emit_mem_req = (flags & CN_STUB_FLAG_EMIT_MEM_REQ) ? 1u : 0u;
  sim->top->cfg_glitch_payload =
      (flags & CN_STUB_FLAG_GLITCH_PAYLOAD) ? 1u : 0u;
  // Drive a real 1 -> 0 -> 1 `rst_n` transition so `always @(posedge clk or
  // negedge rst_n)` fires and loads `cfg_ready_seed` without advancing
  // `posedge_count`.
  sim->top->rst_n = 1;
  sim->top->eval();
  sim->top->rst_n = 0;
  sim->top->eval();
  sim->top->rst_n = 1;
  sim->top->eval();
  return sim;
}

int bp_layout(struct CnSim* sim, struct CnSlotLayout* slabs, uint32_t count) {
  if (sim == nullptr || slabs == nullptr) return CN_ERR_ARGS;
  if (count < CN_SLOT_COUNT) return CN_ERR_RANGE;
  for (uint32_t i = 0; i < CN_SLOT_COUNT; ++i) {
    slabs[i].slot_id = cn_slot_table[i].slot_id;
    slabs[i].offset = cn_slot_table[i].offset;
    slabs[i].len = cn_slot_table[i].len;
  }
  return CN_OK;
}

int bp_drive(struct CnSim* sim, const struct CnIoVec* vec) {
  if (sim == nullptr || vec == nullptr) return CN_ERR_ARGS;
  if (sim->fatal_code != 0) return CN_ERR_FATAL_DUT;
  if (vec->slot_count > 0 && vec->slots == nullptr) return CN_ERR_ARGS;

  for (uint32_t i = 0; i < vec->slot_count; ++i) {
    const struct CnSlot* s = &vec->slots[i];
    const int idx = find_slot(s->slot_id);
    if (idx < 0) return CN_ERR_ARGS;
    const CnSlotSpec& spec = cn_slot_table[idx];
    if (s->len > 0 && s->bytes == nullptr) return CN_ERR_ARGS;
    if (s->len > spec.len) return CN_ERR_RANGE;

    const uint32_t ch = spec.channel_index;
    if (spec.dir_index == 0u) {
      // G2T slot: must be 8-byte word aligned when non-empty.
      if (s->len != 0 && (s->len % 8u != 0 || s->len < 32u)) {
        return CN_ERR_ARGS;
      }
      if (sim->g2t_stalled[ch]) {
        // Payload stability invariant: while `valid && !ready`, payload must
        // remain byte-identical!
        if (s->len != sim->g2t[ch].byte_len ||
            (s->len > 0 &&
             std::memcmp(sim->g2t[ch].bytes, s->bytes, s->len) != 0)) {
          sim->proto_err = CN_EP_PAYLOAD_READY_MISMATCH;
          return CN_ERR_STATE;
        }
      } else if (s->len == 0) {
        sim->g2t[ch].valid = false;
        sim->g2t[ch].byte_len = 0;
      } else {
        sim->g2t[ch].valid = true;
        sim->g2t[ch].slot_id = spec.slot_id;
        sim->g2t[ch].byte_len = s->len;
        std::memcpy(sim->g2t[ch].bytes, s->bytes, s->len);
        std::memcpy(sim->slab + spec.offset, s->bytes, s->len);
      }
    } else {
      // T2G slot: optional 1-byte ready control (`0` = backpressure T2G,
      // non-zero = ready).
      if (s->len == 1u) {
        sim->t2g_ready_cfg[ch] = (s->bytes[0] != 0u) ? 1u : 0u;
      } else if (s->len != 0u) {
        return CN_ERR_ARGS;
      }
    }
  }

  sim->drive_frozen = true;
  sim->drives += 1;
  return CN_OK;
}

int bp_tick(struct CnSim* sim) {
  if (sim == nullptr) return CN_ERR_ARGS;
  if (!sim->drive_frozen) return CN_ERR_STATE;
  if (sim->fatal_code != 0) return CN_ERR_FATAL_DUT;

  Vcn_protocol_stub* top = sim->top;

  // ---- P1: apply frozen P0 pins, `clk = 0`, `eval()`, sole pre-edge handshake
  // sample ----
  uint32_t g2t_valid_mask = 0;
  uint32_t g2t_words_packed = 0;
  uint32_t t2g_ready_mask = 0;
  for (uint32_t c = 0; c < kChannelCount; ++c) {
    if (sim->g2t[c].valid) {
      g2t_valid_mask |= (1u << c);
      const uint32_t words = (sim->g2t[c].byte_len / 8u) & 0x1Fu;
      g2t_words_packed |= (words << (c * 5u));
    }
    if (sim->t2g_ready_cfg[c]) {
      t2g_ready_mask |= (1u << c);
    }
    write_g2t_channel_pins(top, c, sim->g2t[c]);
  }
  top->clk = 0;
  top->g2t_valid = g2t_valid_mask;
  top->g2t_words_flat = g2t_words_packed;
  top->t2g_ready = t2g_ready_mask;
  top->eval();

  bool p1_g2t_acc[kChannelCount]{};
  uint64_t p1_ctrl_word0 = 0;
  if (sim->g2t[0].valid && sim->g2t[0].byte_len >= 8u) {
    std::memcpy(&p1_ctrl_word0, sim->g2t[0].bytes, 8u);
  }
  P1T2gSample p1_t2g[kChannelCount]{};
  const uint32_t p1_g2t_ready = top->g2t_ready;
  const uint32_t p1_t2g_valid = top->t2g_valid;
  const uint32_t p1_t2g_words_flat = top->t2g_words_flat;

  for (uint32_t c = 0; c < kChannelCount; ++c) {
    const bool gv = ((g2t_valid_mask >> c) & 1u) != 0;
    const bool gr = ((p1_g2t_ready >> c) & 1u) != 0;
    p1_g2t_acc[c] = gv && gr;

    const bool tv = ((p1_t2g_valid >> c) & 1u) != 0;
    const bool tr = ((t2g_ready_mask >> c) & 1u) != 0;
    if (tv && tr) {
      const uint32_t words = (p1_t2g_words_flat >> (c * 5u)) & 0x1Fu;
      const uint32_t byte_len = words * 8u;
      p1_t2g[c].accepted = (byte_len >= 32u && byte_len <= CN_MAX_FRAME_BYTES);
      p1_t2g[c].slot_id = slot_for_link(c, 1u);
      p1_t2g[c].byte_len = byte_len;
      if (p1_t2g[c].accepted) {
        read_t2g_channel_pins(top, c, byte_len, p1_t2g[c].bytes);
      }
    }
  }

  // ---- P2: single rising clock edge ----
  top->clk = 1;
  top->eval();
  sim->posedges += 1;

  // ---- P3: `clk = 0`, `eval()`, commit ONLY the P1-accepted set ----
  top->clk = 0;
  top->eval();

  if (p1_g2t_acc[0] && top->dbg_last_g2t_word0 != p1_ctrl_word0) {
    sim->proto_err = CN_EP_PAYLOAD_READY_MISMATCH;
    sim->drive_frozen = false;
    return CN_ERR_STATE;
  }

  for (uint32_t c = 0; c < kChannelCount; ++c) {
    if (p1_g2t_acc[c]) {
      CollectRecord rec{};
      rec.slot_id = sim->g2t[c].slot_id;
      rec.byte_len = sim->g2t[c].byte_len;
      rec.is_t2g = 0u;
      if (!push_record(sim, rec)) {
        sim->drive_frozen = false;
        return CN_ERR_RANGE;
      }
      sim->g2t[c].valid = false;
      sim->g2t[c].byte_len = 0;
      sim->g2t_stalled[c] = false;
      sim->accepted += 1;
    } else if (sim->g2t[c].valid) {
      sim->g2t_stalled[c] = true;
    } else {
      sim->g2t_stalled[c] = false;
    }

    if (p1_t2g[c].accepted) {
      CollectRecord rec{};
      rec.slot_id = p1_t2g[c].slot_id;
      rec.byte_len = p1_t2g[c].byte_len;
      rec.is_t2g = 1u;
      std::memcpy(rec.bytes, p1_t2g[c].bytes, p1_t2g[c].byte_len);
      if (!push_record(sim, rec)) {
        sim->drive_frozen = false;
        return CN_ERR_RANGE;
      }
      sim->accepted += 1;
    }
  }

  // Any `top->t2g_valid` asserted at P2 is ONLY a next-tick staged candidate,
  // never committed now.
  sim->staged_after_p3 = popcount5(top->t2g_valid);

  if (top->fatal_code != 0) {
    sim->fatal_code = top->fatal_code;
    sim->drive_frozen = false;
    return CN_ERR_FATAL_DUT;
  }

  sim->ticks += 1;
  sim->drive_frozen = false;
  sim->collect_armed = true;
  return CN_OK;
}

int bp_collect(struct CnSim* sim, const struct CnCollect* out) {
  if (sim == nullptr || out == nullptr) return CN_ERR_ARGS;
  if (!sim->collect_armed) return CN_ERR_STATE;
  const uint32_t needed = serialized_collect_bytes(sim);
  if (out->buf == nullptr) return CN_ERR_ARGS;
  if (out->buf_len < needed)
    return CN_ERR_RANGE;  // never truncate, preserve ring

  const uint32_t count = sim->ring_count;
  std::memcpy(out->buf + 0, &count, 4u);
  std::memcpy(out->buf + 4, &needed, 4u);
  uint32_t cursor = 8u;
  for (uint32_t i = 0; i < count; ++i) {
    const CollectRecord& r =
        sim->ring[(sim->ring_head + i) % kMaxCollectRecords];
    // High bit of word 1 distinguishes T2G payload-bearing records (0x80000000
    // | byte_len) from G2T acceptance receipts (byte_len only, 0 trailing
    // payload bytes).
    const uint32_t tagged_len = r.byte_len | (r.is_t2g ? 0x80000000u : 0u);
    std::memcpy(out->buf + cursor, &r.slot_id, 4u);
    std::memcpy(out->buf + cursor + 4u, &tagged_len, 4u);
    cursor += 8u;
    if (r.is_t2g && r.byte_len > 0) {
      std::memcpy(out->buf + cursor, r.bytes, r.byte_len);
      cursor += r.byte_len;
    }
    sim->returned += 1;
  }
  sim->ring_head = 0;
  sim->ring_count = 0;
  sim->collect_armed = false;
  sim->collects += 1;
  return CN_OK;
}

int bp_status(struct CnSim* sim, struct CnStatus* out) {
  if (sim == nullptr || out == nullptr) return CN_ERR_ARGS;
  uint32_t fatal = sim->fatal_code;
  uint64_t posedges = sim->posedges;
  if (sim->top != nullptr) {
    if (sim->top->fatal_code != 0) fatal = sim->top->fatal_code;
    posedges = sim->top->posedge_count;
  }
  out->abi_version = CN_ABI_VERSION;
  out->fatal_code = fatal;
  out->proto_err = sim->proto_err;
  out->message_seq_gap_seen = sim->seq_gap_seen;
  out->accepted_but_unreturned_credit = static_cast<uint32_t>(sim->ring_count);
  out->reserved = 0;
  out->tick_count = sim->ticks;
  out->posedge_count = posedges;
  out->drive_count = sim->drives;
  out->collect_count = sim->collects;
  out->accepted_beats = sim->accepted;
  out->returned_beats = sim->returned;
  out->staged_pending = sim->staged_after_p3;
  return CN_OK;
}

void bp_destroy(struct CnSim* sim) {
  if (sim == nullptr) return;
  delete sim->top;
  delete sim->ctx;
  delete[] sim->slab;
  delete[] sim->ring;
  delete sim;
}

}  // extern "C"
