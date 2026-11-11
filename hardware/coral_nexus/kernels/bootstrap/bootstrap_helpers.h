/* Bootstrap helper declarations (C018 / W5-02).
 *
 * CSR numbers come ONLY from the generated C017 header (protocol/generated/cn_messages.h); a handwritten
 * numeric CSR address is a build error, not a style issue. Each helper is the observable-semantics +
 * reordering-barrier carrier the W5 contract (helpers entry) requires: volatile asm with a memory clobber,
 * never a plain MMIO store the compiler may sink.
 */
#ifndef CN_BOOTSTRAP_HELPERS_H_
#define CN_BOOTSTRAP_HELPERS_H_

#include <stdint.h>

#include "cn_messages.h"

#define CN_STR(x) #x
#define CN_XSTR(x) CN_STR(x)
#define CN_CSRW(csr) "csrw " CN_XSTR(csr) ", %0"

/* One declaration value per pending CSR write. The value domain belongs to W3; the bootstrap kernel only
 * needs distinct, increasing values so the emitted control sequence is auditable in the disassembly. */
__attribute__((noinline)) void cn_region_start(uint32_t v);
__attribute__((noinline)) void cn_region_end(uint32_t v);
__attribute__((noinline)) void cn_chain_first(uint32_t v);
__attribute__((noinline)) void cn_chain_continue(uint32_t v);
__attribute__((noinline)) void cn_chain_final(uint32_t v);

#endif /* CN_BOOTSTRAP_HELPERS_H_ */
