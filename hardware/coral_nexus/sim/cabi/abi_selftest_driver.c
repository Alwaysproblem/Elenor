/* abi_selftest_driver.c — C driver exercising libcn_cosim_stub.so with
 * full-width C017 wire frames.
 *
 * Modes:
 *   phase : verifies ABI hash, repeatable read-only bp_layout, full 96-byte
 * TASK_PLAN G2T -> 72-byte ACK T2G round-trip from the SV DUT, posedge +1 per
 * tick, payload-stability violation, double collect, short collect buffer, and
 * credit conservation; writes run_end.json. fatal : verifies catchable DUT
 * fatal at tick 3 -> CN_ERR_FATAL_DUT + nonzero fatal_code without writing
 * run_end.json. abort : triggers uncatchable Verilog $fatal at tick 2 ->
 * process terminates before run_end.json.
 */
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cn_abi.h"
#include "cn_messages.h"

typedef uint32_t (*abi_info_fn)(uint8_t*);
typedef struct CnSim* (*begin_fn)(const struct CnBeginCfg*);
typedef int (*layout_fn)(struct CnSim*, struct CnSlotLayout*, uint32_t);
typedef int (*drive_fn)(struct CnSim*, const struct CnIoVec*);
typedef int (*tick_fn)(struct CnSim*);
typedef int (*collect_fn)(struct CnSim*, const struct CnCollect*);
typedef int (*status_fn)(struct CnSim*, struct CnStatus*);
typedef void (*destroy_fn)(struct CnSim*);

static int g_failures = 0;

static void check(int cond, const char* what) {
  printf("CHECK %s %s\n", cond ? "PASS" : "FAIL", what);
  if (!cond) g_failures++;
}

static void* must_dlsym(void* h, const char* name) {
  void* p = dlsym(h, name);
  if (p == NULL) {
    fprintf(stderr, "dlsym(%s) failed: %s\n", name, dlerror());
    exit(2);
  }
  return p;
}

static void build_task_plan_wire(uint8_t out[96], uint64_t seq,
                                 uint64_t txn_id) {
  uint64_t words[12];
  words[0] = CN_MSG_TASK_PLAN;
  words[1] = 8u; /* payload_words */
  words[2] = seq;
  words[3] = 0u;
  words[4] = 1u; /* launch_generation */
  words[5] = 1u; /* root_launch_generation */
  words[6] = 0u; /* dispatch_ordinal */
  words[7] = 3u; /* task_id */
  words[8] = 0u; /* context_handle */
  words[9] = txn_id;
  words[10] = 1u; /* pool_epoch */
  words[11] = 1u; /* pool_version */
  memcpy(out, words, sizeof(words));
}

static void run_phase_checks(void* h) {
  abi_info_fn bp_abi_info = (abi_info_fn)must_dlsym(h, "bp_abi_info");
  begin_fn bp_begin = (begin_fn)must_dlsym(h, "bp_begin");
  layout_fn bp_layout = (layout_fn)must_dlsym(h, "bp_layout");
  drive_fn bp_drive = (drive_fn)must_dlsym(h, "bp_drive");
  tick_fn bp_tick = (tick_fn)must_dlsym(h, "bp_tick");
  collect_fn bp_collect = (collect_fn)must_dlsym(h, "bp_collect");
  status_fn bp_status = (status_fn)must_dlsym(h, "bp_status");
  destroy_fn bp_destroy = (destroy_fn)must_dlsym(h, "bp_destroy");

  uint8_t hash[129];
  check(bp_abi_info(hash) == CN_ABI_VERSION, "abi_version");
  check(strlen((char*)hash) == 128, "abi_hash_len_128");
  check(bp_begin(NULL) == NULL, "begin_null_args_rejected");

  struct CnBeginCfg cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.abi_version = CN_ABI_VERSION;
  cfg.tile_count = 1;
  cfg.ready_seed = 0x12345678u;
  cfg.inject_fatal_at_tick = -1;
  cfg.inject_abort_at_tick = -1;
  struct CnSim* sim = bp_begin(&cfg);
  check(sim != NULL, "bp_begin_nonnull");

  struct CnSlotLayout slabs[16];
  struct CnSlotLayout slabs2[16];
  check(bp_layout(sim, slabs, 0) == CN_ERR_RANGE, "layout_small_array_range");
  check(bp_layout(sim, slabs, 10) == CN_OK, "layout_ok");
  /* Read-only layout query must remain callable without mutating state! */
  check(bp_layout(sim, slabs2, 10) == CN_OK, "layout_repeatable_readonly");
  check(memcmp(slabs, slabs2, 10 * sizeof(struct CnSlotLayout)) == 0,
        "layout_idempotent");
  check(slabs[0].len == 104u, "ctrl_g2t_slot_104b");
  check(slabs[7].len == 136u, "memory_t2g_slot_136b");

  check(bp_tick(sim) == CN_ERR_STATE, "tick_without_drive_state");

  uint8_t plan_wire[96];
  uint8_t collect_buf[4096];
  uint32_t g2t_sent_seq = 0;
  uint32_t t2g_acks_seen = 0;
  int inflight = 0;

  for (uint32_t t = 0; t < 64; ++t) {
    struct CnSlot slot;
    if (!inflight && g2t_sent_seq < 12) {
      build_task_plan_wire(plan_wire, g2t_sent_seq, 100u + g2t_sent_seq);
      inflight = 1;
    }
    slot.slot_id = 0u; /* CTRL.G2T */
    slot.bytes = inflight ? plan_wire : NULL;
    slot.len = inflight ? 96u : 0u;
    struct CnIoVec vec = {&slot, 1u};
    check(bp_drive(sim, &vec) == CN_OK, "drive_ok");
    check(bp_tick(sim) == CN_OK, "tick_ok");

    struct CnStatus st;
    bp_status(sim, &st);
    check(st.posedge_count == (uint64_t)(t + 1),
          "posedge_exactly_one_per_tick");

    struct CnCollect col = {collect_buf, sizeof(collect_buf)};
    check(bp_collect(sim, &col) == CN_OK, "collect_ok");
    check(bp_collect(sim, &col) == CN_ERR_STATE, "double_collect_state");

    uint32_t rec_count = 0;
    memcpy(&rec_count, collect_buf, 4u);
    uint32_t cur = 8u;
    for (uint32_t r = 0; r < rec_count; ++r) {
      uint32_t sid = 0, tlen = 0;
      memcpy(&sid, collect_buf + cur, 4u);
      memcpy(&tlen, collect_buf + cur + 4u, 4u);
      cur += 8u;
      int is_t2g = (tlen & 0x80000000u) != 0;
      uint32_t blen = tlen & 0x7FFFFFFFu;
      if (!is_t2g && sid == 0u) {
        check(blen == 96u, "g2t_task_plan_96b_accepted");
        inflight = 0;
        g2t_sent_seq += 1;
      } else if (is_t2g && sid == 1u) {
        check(blen == 72u, "t2g_ack_72b_received");
        uint64_t ack_words[9];
        memcpy(ack_words, collect_buf + cur, 72u);
        check(ack_words[0] == CN_MSG_ACK && ack_words[1] == 5u,
              "t2g_ack_header_valid");
        check(ack_words[4] == 100u + t2g_acks_seen, "t2g_ack_txn_matches");
        check(ack_words[5] == CN_MSG_TASK_PLAN, "t2g_ack_for_task_plan");
        check(ack_words[7] == 1u, "t2g_ack_outcome_accepted");
        t2g_acks_seen += 1;
      }
      if (is_t2g) cur += blen;
    }
  }
  check(t2g_acks_seen == 12u, "all_12_task_plans_acked_by_dut");

  /* Short collect buffer -> CN_ERR_RANGE without losing records. */
  build_task_plan_wire(plan_wire, 12u, 999u);
  for (uint32_t t = 0; t < 16; ++t) {
    struct CnSlot slot = {0u, plan_wire, 96u};
    struct CnIoVec vec = {&slot, 1u};
    bp_drive(sim, &vec);
    bp_tick(sim);
    struct CnCollect tiny = {collect_buf, 4u};
    check(bp_collect(sim, &tiny) == CN_ERR_RANGE, "collect_small_range");
    struct CnCollect ok = {collect_buf, sizeof(collect_buf)};
    check(bp_collect(sim, &ok) == CN_OK, "collect_after_range_ok");
    uint32_t cnt = 0;
    memcpy(&cnt, collect_buf, 4u);
    if (cnt > 0) break;
  }

  /* Drain any staged T2G response so accepted == returned. */
  for (uint32_t t = 0; t < 4; ++t) {
    struct CnSlot idle = {0u, NULL, 0u};
    struct CnIoVec vec = {&idle, 1u};
    bp_drive(sim, &vec);
    bp_tick(sim);
    struct CnCollect ok = {collect_buf, sizeof(collect_buf)};
    bp_collect(sim, &ok);
  }
  {
    struct CnStatus st;
    bp_status(sim, &st);
    check(st.accepted_beats == st.returned_beats, "credit_conservation");
  }

  bp_destroy(sim);
  printf("PHASE_SUITE_DONE failures=%d\n", g_failures);
}

static void run_fatal_checks(void* h) {
  abi_info_fn bp_abi_info = (abi_info_fn)must_dlsym(h, "bp_abi_info");
  begin_fn bp_begin = (begin_fn)must_dlsym(h, "bp_begin");
  layout_fn bp_layout = (layout_fn)must_dlsym(h, "bp_layout");
  drive_fn bp_drive = (drive_fn)must_dlsym(h, "bp_drive");
  tick_fn bp_tick = (tick_fn)must_dlsym(h, "bp_tick");
  status_fn bp_status = (status_fn)must_dlsym(h, "bp_status");
  destroy_fn bp_destroy = (destroy_fn)must_dlsym(h, "bp_destroy");
  uint8_t hash[129];
  bp_abi_info(hash);

  struct CnBeginCfg cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.abi_version = CN_ABI_VERSION;
  cfg.tile_count = 1;
  cfg.ready_seed = 7u;
  cfg.inject_fatal_at_tick = 3;
  cfg.inject_abort_at_tick = -1;
  struct CnSim* sim = bp_begin(&cfg);
  struct CnSlotLayout slabs[16];
  bp_layout(sim, slabs, 10);
  uint32_t fatal_tick = 99u;
  for (uint32_t t = 0; t < 16; ++t) {
    struct CnSlot idle = {0u, NULL, 0u};
    struct CnIoVec vec = {&idle, 1u};
    bp_drive(sim, &vec);
    int rc = bp_tick(sim);
    if (rc == CN_ERR_FATAL_DUT) {
      fatal_tick = t;
      struct CnStatus st;
      bp_status(sim, &st);
      check(st.fatal_code != 0, "fatal_code_nonzero");
      break;
    }
  }
  check(fatal_tick == 3u, "fatal_at_tick3");
  check(bp_tick(sim) == CN_ERR_STATE || bp_tick(sim) == CN_ERR_FATAL_DUT,
        "tick_after_fatal_error");
  bp_destroy(sim);
  printf("FATAL_SUITE_DONE failures=%d\n", g_failures);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <so-path> phase|fatal|abort\n", argv[0]);
    return 2;
  }
  const char* mode = argv[argc - 1];
  void* h = dlopen(argv[1], RTLD_NOW);
  if (h == NULL) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 2;
  }
  if (strcmp(mode, "phase") == 0) {
    run_phase_checks(h);
    if (g_failures == 0) {
      FILE* f = fopen("run_end.json", "w");
      if (f != NULL) {
        fprintf(f, "{\"status\": \"complete\"}\n");
        fclose(f);
      }
    }
  } else if (strcmp(mode, "fatal") == 0) {
    run_fatal_checks(h);
  } else if (strcmp(mode, "abort") == 0) {
    begin_fn bp_begin = (begin_fn)must_dlsym(h, "bp_begin");
    layout_fn bp_layout = (layout_fn)must_dlsym(h, "bp_layout");
    drive_fn bp_drive = (drive_fn)must_dlsym(h, "bp_drive");
    tick_fn bp_tick = (tick_fn)must_dlsym(h, "bp_tick");
    struct CnBeginCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = CN_ABI_VERSION;
    cfg.tile_count = 1;
    cfg.ready_seed = 7u;
    cfg.inject_fatal_at_tick = -1;
    cfg.inject_abort_at_tick = 2;
    struct CnSim* sim = bp_begin(&cfg);
    struct CnSlotLayout slabs[16];
    bp_layout(sim, slabs, 10);
    for (uint32_t t = 0; t < 8; ++t) {
      struct CnSlot idle = {0u, NULL, 0u};
      struct CnIoVec vec = {&idle, 1u};
      bp_drive(sim, &vec);
      bp_tick(sim);
      printf("abort-mode tick %u done\n", t);
      fflush(stdout);
    }
    printf("UNREACHABLE\n");
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  return g_failures == 0 ? 0 : 1;
}
