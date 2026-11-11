/* Bootstrap control sequence + helper bodies (C018 / W5-02).
 *
 * This image exists to MEASURE bounded resources of a real helper-bearing kernel: text, static data and a
 * provable stack bound. It is NOT a loadable CS1 package (C020 seals those) and it is not executed here.
 *
 * Auditability rules the measurer depends on:
 *   * every helper is noinline volatile-asm with a memory clobber, so exactly one csrw per call appears in
 *     the disassembly and the compiler cannot reorder it against the memory sequence;
 *   * the control sequence is a fixed, straight-line order (START, FIRST, CONTINUE x CN_BOOTSTRAP_STEPS,
 *     FINAL, END) with a static trip count -- no recursion, no indirect calls, no VLAs;
 *   * the loop accumulator lives in a scalar so the data footprint stays static data only.
 */
#include <stdint.h>

#include "bootstrap_helpers.h"

#define CN_BOOTSTRAP_STEPS 4

__attribute__((noinline)) void cn_region_start(uint32_t v) {
  __asm__ volatile(CN_CSRW(CN_CSR_NEXUS_MREGION_START) ::"r"(v)
                   : "memory");
}

__attribute__((noinline)) void cn_region_end(uint32_t v) {
  __asm__ volatile(CN_CSRW(CN_CSR_NEXUS_MREGION_END) ::"r"(v)
                   : "memory");
}

__attribute__((noinline)) void cn_chain_first(uint32_t v) {
  __asm__ volatile(CN_CSRW(CN_CSR_NEXUS_MREGION_FIRST) ::"r"(v)
                   : "memory");
}

__attribute__((noinline)) void cn_chain_continue(uint32_t v) {
  __asm__ volatile(CN_CSRW(CN_CSR_NEXUS_MREGION_CONTINUE) ::"r"(v)
                   : "memory");
}

__attribute__((noinline)) void cn_chain_final(uint32_t v) {
  __asm__ volatile(CN_CSRW(CN_CSR_NEXUS_MREGION_FINAL) ::"r"(v)
                   : "memory");
}

/* Entry of the measured image. `_start` (bootstrap_start.S) only sets gp/sp and tail-calls this. */
void cn_bootstrap_entry(void) {
  uint32_t step = 0x10;
  cn_region_start(step++);
  cn_chain_first(step++);
  for (int i = 0; i < CN_BOOTSTRAP_STEPS; i++) {
    cn_chain_continue(step++);
  }
  cn_chain_final(step++);
  cn_region_end(step);
}
