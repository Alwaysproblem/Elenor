/* cn_abi.cc — CN C ABI stub implementation over the Verilator stub top (C025 /
 * W1-03).
 *
 * Phase discipline (W1 plan §9.2):
 *   bp_drive = P0 only (freeze inputs onto pins; no eval, no state change).
 *   bp_tick  = P1 (eval, sole pre-edge handshake decision) -> P2 (exactly one
 * rising edge) -> P3 (eval, commit ONLY the P1-accepted set; staged outputs are
 * next-tick candidates) -> clk back low. bp_collect = once, P3..P4, copies
 * accepted beats into the caller buffer (no truncation).
 *
 * The stub library is unit-only; the real Tile top replaces Vcn_protocol_stub
 * in C040 with the same ABI surface.
 */

#include "cn_abi.h"

#include <cstdlib>
#include <cstring>

#include "Vcn_protocol_stub.h"
#include "cn_slot_table.h"
#include "verilated.h"

// Defined at global scope so the extern "C" forward declaration in cn_abi.h
// names the same type (an anonymous-namespace definition would create a second,
// distinct CnSim).
struct CnSim {
  CnBeginCfg cfg{};
  VerilatedContext* ctx = nullptr;
  Vcn_protocol_stub* top = nullptr;
  uint8_t* slab =
      nullptr;  // CN_SLAB_TOTAL_BYTES shared input slab (frozen at P0)
  uint8_t* pending =
      nullptr;  // accepted-beat buffer (header+payload words, 8B each)
  uint32_t pending_bytes = 0;
  bool layout_done = false;
  bool drive_frozen = false;  // set by bp_drive, cleared by bp_tick
  bool collect_armed =
      false;  // a tick completed and no collect has happened yet
  // counters
  uint64_t ticks = 0, posedges = 0, drives = 0, collects = 0;
  uint64_t accepted = 0, returned = 0;
  uint32_t fatal_code = 0, proto_err = 0;
  uint32_t seq_gap_seen = 0;
  // P1 sample of the G2T beat
  bool p1_accepted = false;
  uint64_t p1_payload = 0;
};

static const uint32_t BEAT_BYTES = 8;  // one 64-bit payload word per stub beat

static int find_slot(uint32_t slot_id) {
  for (uint32_t i = 0; i < CN_SLOT_COUNT; ++i) {
    if (cn_slot_table[i].slot_id == slot_id) return (int)i;
  }
  return -1;
}

extern "C" {

uint32_t bp_abi_info(uint8_t out_hash[129]) {
  if (out_hash == nullptr) return CN_ABI_VERSION;
  memcpy(out_hash, kAbiHash128, 129);
  return CN_ABI_VERSION;
}

struct CnSim* bp_begin(const struct CnBeginCfg* cfg) {
  if (cfg == nullptr || cfg->abi_version != CN_ABI_VERSION ||
      cfg->tile_count < 1)
    return nullptr;
  CnSim* sim = new (std::nothrow) CnSim();
  if (sim == nullptr) return nullptr;
  sim->cfg = *cfg;
  sim->ctx = new (std::nothrow) VerilatedContext();
  sim->top = new (std::nothrow) Vcn_protocol_stub(sim->ctx);
  sim->slab = new (std::nothrow) uint8_t[CN_SLAB_TOTAL_BYTES]();
  sim->pending = new (std::nothrow) uint8_t[BEAT_BYTES]();
  if (sim->ctx == nullptr || sim->top == nullptr || sim->slab == nullptr ||
      sim->pending == nullptr) {
    bp_destroy(sim);
    return nullptr;
  }
  sim->ctx->debug(0);
  // Deterministic reset: rst_n low, eval, then release.
  sim->top->rst_n = 0;
  sim->top->g2t_valid = 0;
  sim->top->g2t_payload = 0;
  sim->top->t2g_ready = 1;
  sim->top->cfg_ready_seed = cfg->ready_seed;
  sim->top->cfg_inject_fatal_at_tick =
      (cfg->inject_fatal_at_tick < 0) ? 0xFFFFFFFFu
                                      : (uint32_t)cfg->inject_fatal_at_tick;
  sim->top->cfg_inject_abort_at_tick =
      (cfg->inject_abort_at_tick < 0) ? 0xFFFFFFFFu
                                      : (uint32_t)cfg->inject_abort_at_tick;
  sim->top->glitch_enable = 0;
  sim->top->clk = 0;
  sim->top->eval();
  sim->top->rst_n = 1;
  sim->top->eval();
  return sim;
}

int bp_layout(struct CnSim* sim, struct CnSlotLayout* slabs, uint32_t count) {
  if (sim == nullptr || slabs == nullptr) return CN_ERR_ARGS;
  if (sim->layout_done) return CN_ERR_STATE;
  if (count < CN_SLOT_COUNT) return CN_ERR_RANGE;
  for (uint32_t i = 0; i < CN_SLOT_COUNT; ++i) {
    slabs[i].slot_id = cn_slot_table[i].slot_id;
    slabs[i].offset = cn_slot_table[i].offset;
    slabs[i].len = cn_slot_table[i].len;
  }
  sim->layout_done = true;
  return CN_OK;
}

int bp_drive(struct CnSim* sim, const struct CnIoVec* vec) {
  if (sim == nullptr || vec == nullptr) return CN_ERR_ARGS;
  if (sim->fatal_code != 0) return CN_ERR_FATAL_DUT;
  if (vec->slot_count > 0 && vec->slots == nullptr) return CN_ERR_ARGS;
  for (uint32_t i = 0; i < vec->slot_count; ++i) {
    const struct CnSlot* s = &vec->slots[i];
    int idx = find_slot(s->slot_id);
    if (idx < 0) return CN_ERR_ARGS;
    if (s->bytes == nullptr && s->len != 0) return CN_ERR_ARGS;
    if (s->len > cn_slot_table[idx].len) return CN_ERR_RANGE;
    memcpy(sim->slab + cn_slot_table[idx].offset, s->bytes, s->len);
  }
  sim->drive_frozen = true;
  sim->drives += 1;
  return CN_OK;
}

int bp_tick(struct CnSim* sim) {
  if (sim == nullptr) return CN_ERR_ARGS;
  if (!sim->drive_frozen)
    return CN_ERR_STATE;  // P1 input must be frozen drive content
  if (sim->fatal_code != 0) return CN_ERR_FATAL_DUT;

  Vcn_protocol_stub* top = sim->top;

  // ---- P1: pre-edge eval; the ONLY handshake decision point
  // --------------------
  top->clk = 0;
  top->g2t_valid = 1;
  top->g2t_payload =
      *(const uint64_t*)(sim->slab);  // slot 0 = CTRL/G2T first 8 bytes
  top->t2g_ready = 1;
  top->eval();
  const bool p1_ready = top->g2t_ready != 0;
  const uint64_t p1_payload = top->g2t_payload;
  sim->p1_accepted = (top->g2t_valid != 0) && p1_ready;
  sim->p1_payload = p1_payload;

  // ---- P2: exactly one rising edge
  // ----------------------------------------------
  top->clk = 1;
  top->eval();
  sim->posedges += 1;

  // ---- P3: post-edge eval; commit ONLY the P1-accepted set
  // ----------------------
  top->clk = 0;
  top->eval();

  if (sim->p1_accepted) {
    // Payload stability across the edge: the DUT's registered view must equal
    // the P1 sample.
    if (top->dbg_payload_seen != p1_payload) {
      sim->proto_err = CN_EP_PAYLOAD_READY_MISMATCH;
      sim->drive_frozen = false;
      return CN_ERR_STATE;
    }
    memcpy(sim->pending, &p1_payload, BEAT_BYTES);
    sim->pending_bytes = BEAT_BYTES;
    sim->accepted += 1;
  } else {
    sim->pending_bytes = 0;
  }

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
  if (!sim->collect_armed)
    return CN_ERR_STATE;  // no completed tick, or already collected
  if (out->buf == nullptr && sim->pending_bytes != 0) return CN_ERR_ARGS;
  if (out->buf_len < sim->pending_bytes) return CN_ERR_RANGE;  // never truncate
  if (sim->pending_bytes != 0) {
    memcpy(out->buf, sim->pending, sim->pending_bytes);
    sim->returned += 1;
  }
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
  out->accepted_but_unreturned_credit =
      (uint32_t)(sim->accepted - sim->returned -
                 (sim->collect_armed && sim->pending_bytes ? 1u : 0u));
  out->reserved = 0;
  out->tick_count = sim->ticks;
  out->posedge_count = posedges;
  out->drive_count = sim->drives;
  out->collect_count = sim->collects;
  out->accepted_beats = sim->accepted;
  out->returned_beats = sim->returned;
  out->staged_pending = sim->pending_bytes / BEAT_BYTES;
  return CN_OK;
}

void bp_destroy(struct CnSim* sim) {
  if (sim == nullptr) return;
  delete sim->top;
  delete sim->ctx;
  delete[] sim->slab;
  delete[] sim->pending;
  delete sim;
}

}  // extern "C"
