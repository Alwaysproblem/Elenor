/* abi_selftest_driver.c — in-container C driver exercising the full C025
 * positive + negative sequence against libcn_cosim_stub.so. Prints one CHECK
 * line per assertion; exits non-zero on any failure. Writes run_end.json only
 * on a clean full run (the crash driver reuses this contract).
 *
 * Usage: abi_selftest_driver <libcn_cosim_stub.so-dir> [--mode
 * phase|fatal|abort]
 */
#include <dlfcn.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cn_abi.h"

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

static void run_phase_checks(void* h) {
  abi_info_fn bp_abi_info = (abi_info_fn)must_dlsym(h, "bp_abi_info");
  begin_fn bp_begin = (begin_fn)must_dlsym(h, "bp_begin");
  layout_fn bp_layout = (layout_fn)must_dlsym(h, "bp_layout");
  drive_fn bp_drive = (drive_fn)must_dlsym(h, "bp_drive");
  tick_fn bp_tick = (tick_fn)must_dlsym(h, "bp_tick");
  collect_fn bp_collect = (collect_fn)must_dlsym(h, "bp_collect");
  status_fn bp_status = (status_fn)must_dlsym(h, "bp_status");
  destroy_fn bp_destroy = (destroy_fn)must_dlsym(h, "bp_destroy");

  // ABI identity
  uint8_t hash[129];
  check(bp_abi_info(hash) == CN_ABI_VERSION, "abi_version");
  check(strlen((char*)hash) == 128, "abi_hash_len_128");

  // begin
  struct CnBeginCfg cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.abi_version = CN_ABI_VERSION;
  cfg.tile_count = 1;
  cfg.ready_seed = 0x12345678u;
  cfg.inject_fatal_at_tick = -1;
  cfg.inject_abort_at_tick = -1;
  struct CnSim* sim = bp_begin(&cfg);
  check(sim != NULL, "bp_begin_nonnull");
  check(bp_begin(NULL) == NULL, "begin_null_args_rejected");

  // layout: exactly once; too-small array -> RANGE
  struct CnSlotLayout slabs[64];
  check(bp_layout(sim, slabs, 0) == CN_ERR_RANGE, "layout_small_array_range");
  check(bp_layout(sim, slabs, 64) == CN_OK, "layout_ok");
  check(bp_layout(sim, slabs, 64) == CN_ERR_STATE, "layout_twice_state");

  // missing bp_drive -> tick must refuse
  check(bp_tick(sim) == CN_ERR_STATE, "tick_without_drive_state");

  // happy loop with deterministic ready lowering
  uint64_t posedges_before = 0;
  {
    struct CnStatus st;
    bp_status(sim, &st);
    posedges_before = st.posedge_count;
  }
  uint32_t accepted_total = 0;
  uint8_t payload[16];
  for (uint32_t t = 0; t < 48; ++t) {
    uint64_t word = ((uint64_t)t << 32) | 0xCAFE0000u | t;
    memcpy(payload, &word, 8);
    struct CnSlot slot = {0u, payload, 8u};
    struct CnIoVec vec = {&slot, 1u};
    check(bp_drive(sim, &vec) == CN_OK, "drive_ok");
    int rc = bp_tick(sim);
    check(rc == CN_OK || rc == CN_ERR_FATAL_DUT, "tick_ok");
    struct CnStatus st;
    bp_status(sim, &st);
    check(st.posedge_count == posedges_before + t + 1,
          "posedge_exactly_one_per_tick");
    uint8_t out[16];
    struct CnCollect col = {out, sizeof(out)};
    int crc = bp_collect(sim, &col);
    check(crc == CN_OK, "collect_ok");
    // double collect -> STATE
    check(bp_collect(sim, &col) == CN_ERR_STATE, "double_collect_state");
    if (st.staged_pending != 0) {
      uint64_t got = 0;
      memcpy(&got, out, 8);
      check(got == word, "echo_payload_exact");
      accepted_total += 1;
    }
  }
  check(accepted_total > 0, "echo_drain_nonempty");

  // collect buffer too small (accept a beat then collect into 0 bytes) ->
  // RANGE, no truncation
  {
    uint8_t out8[16];
    for (uint32_t t = 0; t < 64; ++t) {
      uint64_t word = 0xDEAD0000u + t;
      memcpy(payload, &word, 8);
      struct CnSlot slot = {0u, payload, 8u};
      struct CnIoVec vec = {&slot, 1u};
      bp_drive(sim, &vec);
      bp_tick(sim);
      struct CnStatus st;
      bp_status(sim, &st);
      struct CnCollect small = {out8, 0u};
      if (st.staged_pending != 0) {
        check(bp_collect(sim, &small) == CN_ERR_RANGE, "collect_small_range");
        uint8_t out2[16];
        struct CnCollect ok = {out2, sizeof(out2)};
        check(bp_collect(sim, &ok) == CN_OK, "collect_after_range_ok");
        break;
      }
      struct CnCollect ok = {out8, sizeof(out8)};
      bp_collect(sim, &ok);
    }
  }

  // credit conservation: accepted == returned after draining everything
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
  collect_fn bp_collect = (collect_fn)must_dlsym(h, "bp_collect");
  status_fn bp_status = (status_fn)must_dlsym(h, "bp_status");
  destroy_fn bp_destroy = (destroy_fn)must_dlsym(h, "bp_destroy");
  uint8_t hash[129];
  bp_abi_info(hash);

  struct CnBeginCfg cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.abi_version = CN_ABI_VERSION;
  cfg.tile_count = 1;
  cfg.ready_seed = 7u;
  cfg.inject_fatal_at_tick = 3;  // catchable fatal at tick 3
  cfg.inject_abort_at_tick = -1;
  struct CnSim* sim = bp_begin(&cfg);
  struct CnSlotLayout slabs[64];
  bp_layout(sim, slabs, 64);
  uint8_t payload[16] = {0};
  uint32_t fatal_tick = 0;
  for (uint32_t t = 0; t < 16; ++t) {
    struct CnSlot slot = {0u, payload, 8u};
    struct CnIoVec vec = {&slot, 1u};
    bp_drive(sim, &vec);
    int rc = bp_tick(sim);
    uint8_t out[16];
    struct CnCollect col = {out, sizeof(out)};
    int crc = bp_collect(sim, &col);
    if (rc == CN_ERR_FATAL_DUT) {
      fatal_tick = t;
      struct CnStatus st;
      bp_status(sim, &st);
      check(st.fatal_code != 0, "fatal_code_nonzero");
      check(crc == CN_OK || crc == CN_ERR_STATE, "collect_after_fatal_bounded");
      break;
    }
  }
  check(fatal_tick == 3u, "fatal_at_tick3");
  // further ticks keep reporting the fatal, they do not crash
  check(bp_tick(sim) == CN_ERR_STATE || bp_tick(sim) == CN_ERR_FATAL_DUT,
        "tick_after_fatal_error");
  bp_destroy(sim);
  printf("FATAL_SUITE_DONE failures=%d\n", g_failures);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <so-path> --mode phase|fatal|abort\n", argv[0]);
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
    // real process-level crash: no run_end.json is written by design
    abi_info_fn bp_abi_info = (abi_info_fn)must_dlsym(h, "bp_abi_info");
    begin_fn bp_begin = (begin_fn)must_dlsym(h, "bp_begin");
    layout_fn bp_layout = (layout_fn)must_dlsym(h, "bp_layout");
    drive_fn bp_drive = (drive_fn)must_dlsym(h, "bp_drive");
    tick_fn bp_tick = (tick_fn)must_dlsym(h, "bp_tick");
    uint8_t hash[129];
    bp_abi_info(hash);
    struct CnBeginCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.abi_version = CN_ABI_VERSION;
    cfg.tile_count = 1;
    cfg.ready_seed = 7u;
    cfg.inject_fatal_at_tick = -1;
    cfg.inject_abort_at_tick = 2;  // $fatal -> process dies here
    struct CnSim* sim = bp_begin(&cfg);
    struct CnSlotLayout slabs[64];
    bp_layout(sim, slabs, 64);
    uint8_t payload[16] = {0};
    for (uint32_t t = 0; t < 8; ++t) {
      struct CnSlot slot = {0u, payload, 8u};
      struct CnIoVec vec = {&slot, 1u};
      bp_drive(sim, &vec);
      bp_tick(sim);
      printf("abort-mode tick %u done\n", t);
      fflush(stdout);
    }
    printf("UNREACHABLE\n");
  } else {
    fprintf(stderr, "unknown mode\n");
    return 2;
  }
  return g_failures == 0 ? 0 : 1;
}
