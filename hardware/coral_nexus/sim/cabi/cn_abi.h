/* cn_abi.h — CN co-simulation C ABI, authoritative naming (C025 / W1-03).
 *
 * Contract source:
 * design/coral_nexus_tile/implementation/W1_Boundary_Cosimulation_Plan.md §9.1
 * (bp_end / FREEZE_INDEX / cn_wire.h / CnWire.scala / libcn_sim naming is
 * OBSOLETE).
 *
 * Generated protocol artifacts consumed at build time:
 *   hardware/coral_nexus/protocol/generated/{cn_messages.h, cn_offsets.svh,
 * cn_layout.json} Python side: pipeline_validator/cosim/{abi.py, wire.py}
 * Libraries:   libcn_cosim.so (real Tile top, C040) / libcn_cosim_stub.so
 * (unit-only stub). Teardown:    bp_destroy.
 *
 * Single-threaded; Python is the sole time authority; C never calls back into
 * Python and never executes P4. Fixed-capacity buffers are preallocated by the
 * caller; the C side performs no steady-state allocation.
 */

#ifndef CN_ABI_H_
#define CN_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CN_ABI_VERSION 1u

/* Error codes (int return, 0 = success). */
#define CN_OK 0
#define CN_ERR_ARGS 1 /* invalid argument */
#define CN_ERR_STATE \
  2 /* call-order error: missing bp_drive, double collect, ... */
#define CN_ERR_RANGE \
  3 /* collect buffer too small / slot length exceeds capacity */
#define CN_ERR_ABI 4 /* hash/layout verification failed */
#define CN_ERR_FATAL_DUT \
  5 /* catchable DUT fatal (uncatchable crash terminates the process) */

/* Protocol error kinds reported through CnStatus.proto_err. */
#define CN_EP_NONE 0
#define CN_EP_PAYLOAD_READY_MISMATCH \
  1 /* payload mutated while valid&&!ready or post-edge mismatch */

/* CnBeginCfg.reserved[0] unit-stub test flags (ignored by production tops). */
#define CN_STUB_FLAG_CORRUPT_T2G \
  0x01u /* flip bit 0 of DUT T2G payload word 0 */
#define CN_STUB_FLAG_EMIT_MEM_REQ \
  0x02u /* emit a full 136-byte MEM_REQ on MEMORY T2G at tick 0 */
#define CN_STUB_FLAG_GLITCH_PAYLOAD \
  0x04u /* corrupt post-edge G2T snapshot inside stub */

struct CnSim; /* opaque handle: C side is the sole owner */

/* One tile x channel x direction port slot; the slot table is generated from
 * protocol/generated/cn_layout.json (never hand-written).
 * For G2T slots: `bytes[0..len)` carries a complete wire.pack frame (`len == 0`
 * clears idle). For T2G slots: optional 1-byte flow-control override (`len ==
 * 1`: `bytes[0] == 0` lowers `t2g_ready` for that channel on the upcoming tick;
 * `bytes[0] != 0` raises `t2g_ready`). */
struct CnSlot {
  uint32_t slot_id;
  const uint8_t* bytes; /* caller memory; consumed/copied inside bp_drive */
  uint32_t len; /* byte length; exceeding slot capacity -> CN_ERR_RANGE */
};

/* bp_drive input batch (caller-allocated; the C side copies within the call).
 */
struct CnIoVec {
  struct CnSlot* slots;
  uint32_t slot_count;
};

/* bp_collect output buffer (caller-allocated; C copies accepted records in).
 * Binary layout written into `buf`:
 *   uint32_t record_count;
 *   uint32_t total_bytes_written;
 *   followed by `record_count` records, each:
 *     uint32_t slot_id;   (even slot = G2T beat accepted at P1; odd slot = T2G
 * frame accepted at P1) uint32_t byte_len;  (for G2T ack record: accepted wire
 * length, followed by 0 payload bytes; for T2G frame record: full wire byte
 * length, followed by `byte_len` raw wire bytes read from the DUT T2G pins) */
struct CnCollect {
  uint8_t* buf;
  uint32_t buf_len;
};

/* One slot's shared-slab geometry (read-only query; never mutates functional
 * state). */
struct CnSlotLayout {
  uint32_t slot_id;
  uint32_t offset;
  uint32_t len;
};

/* Session configuration; bp_begin deep-copies the whole struct and retains no
 * caller pointer. */
struct CnBeginCfg {
  uint32_t abi_version; /* must equal CN_ABI_VERSION */
  uint32_t tile_count;  /* 1 for the unit stub */
  uint32_t ready_seed;  /* deterministic ready-lowering seed */
  int32_t
      inject_fatal_at_tick; /* -1 = never; else catchable fatal at that tick */
  int32_t inject_abort_at_tick; /* -1 = never; else process-level crash at that
                                   tick */
  uint8_t trace;                /* 0 = no VCD */
  uint8_t
      reserved[7]; /* reserved[0]: unit-stub test flags (0 in normal runs) */
};

/* Read-only status POD; readable at any time without mutating functional state.
 */
struct CnStatus {
  uint32_t abi_version;
  uint32_t fatal_code;
  uint32_t proto_err;
  uint32_t message_seq_gap_seen;
  uint32_t accepted_but_unreturned_credit;
  uint32_t reserved;
  uint64_t tick_count;
  uint64_t posedge_count;
  uint64_t drive_count;
  uint64_t collect_count;
  uint64_t accepted_beats;
  uint64_t returned_beats;
  uint64_t staged_pending;
};

/* (0) ABI/protocol identity: returns CN_ABI_VERSION and fills out_hash with 128
 * hex chars = sha256(cn_protocol_v1.yaml) ++ sha256(cn_layout.json),
 * NUL-terminated at byte 128. */
uint32_t bp_abi_info(uint8_t out_hash[129]);

/* (1) Session start: deep-copies cfg, allocates the model and all shared slabs
 * (zero steady-state allocation afterwards). Returns NULL on invalid arguments
 * / missing dependencies. */
struct CnSim* bp_begin(const struct CnBeginCfg* cfg);

/* (2) Read-only layout query: fills `slabs[0..CN_SLOT_COUNT)` without mutating
 * functional state. */
int bp_layout(struct CnSim* sim, struct CnSlotLayout* slabs, uint32_t count);

/* (3) P0, alone: once per tick, BEFORE bp_tick; freezes drive inputs without
 * evaluating or changing model state. bp_tick without a prior bp_drive ->
 * CN_ERR_STATE. */
int bp_drive(struct CnSim* sim, const struct CnIoVec* vec);

/* (4) P1..P3: eval + sole pre-edge handshake decision + exactly one rising edge
 * + eval committing only the P1-accepted set (staged outputs are next-tick
 * candidates only). No drive, no P4. */
int bp_tick(struct CnSim* sim);

/* (5) Between P3 and P4 (at most once per tick): copies committed P1-accepted
 * records into `out`; insufficient `buf_len` -> CN_ERR_RANGE without truncation
 * or loss. */
int bp_collect(struct CnSim* sim, const struct CnCollect* out);

/* (6) Non-blocking read-only status export. */
int bp_status(struct CnSim* sim, struct CnStatus* out);

/* (7) Teardown: frees everything the C side owns. */
void bp_destroy(struct CnSim* sim);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CN_ABI_H_ */
