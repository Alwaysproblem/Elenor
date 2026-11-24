/* K-SCALAR-ADD7 / scalar_ctl_v1: the first real sealed scalar control kernel (C038 / W5-02).
 *
 * Task chain (design/coral_nexus_tile/execution/G2_scalar.md#c038):
 *   MFE L2->Frame -> WAIT -> scalar load / +7 / store (with the fflags path) -> MFE Frame->L2 -> WAIT
 *   -> phase signals -> normal retirement chain.
 *
 * No RVV or VME instruction and no matrix region: the task ends through task_ret, never an END helper.
 * The scalar code never moves a byte across L1/L2 itself. Every byte it loads or stores is Frame-relative
 * to the launch register a0; the two L1<->L2 moves are MFE descriptors that the kernel only selects by
 * index. scripts/lib/scalar_kernel_build.py decodes the linked bytes and refuses anything else.
 *
 * Helper ABI (protocol/scalar_helper_abi_v1.json): every helper is `csrrwi x0, CSR, imm` (rd = x0). A
 * helper CSR appears only through its generated CN_CSR_NEXUS_* macro; cn_csrs.inc is cut verbatim from
 * protocol/generated/cn_messages.h at build time. A handwritten CSR number in this file is a build error.
 *
 * Launch state: a0 = Frame base of this Context (the per-Context mutable area; text is shared across
 * Contexts, so no absolute data address may exist in text). sp, gp and every other register are unused.
 * Frame layout = kernels/bindings/scalar_ctl_v1.yaml launch_descriptors, the same placement the compiled
 * control plane uses (one 1024-byte stripe round = 16 banks x 64-byte line between the two buffers):
 *   [a0+0x000, a0+0x020)  input   8 x i32   descriptor 0, l2_to_frame
 *   [a0+0x400, a0+0x420)  output  8 x i32   descriptor 1, frame_to_l2
 *
 * 36 instructions, straight-line, 144 bytes, identical for both link slots.
 */
#include "cn_csrs.inc"

  .section .text._start,"ax",@progbits
  .option norvc
  .globl _start
  .type _start, @function
_start:
  /* input: one MFE direct move L2 -> Frame (descriptor 0), then WAIT for its completion event */
  csrrwi x0, CN_CSR_NEXUS_MFE_SUBMIT, 0
  csrrwi x0, CN_CSR_NEXUS_EVENT_WAIT, 0

  /* scalar section inside the Frame: load, +7, store; eight little-endian i32 */
  lw t0, 0(a0)
  addi t0, t0, 7
  sw t0, 1024(a0)
  lw t0, 4(a0)
  addi t0, t0, 7
  sw t0, 1028(a0)
  lw t0, 8(a0)
  addi t0, t0, 7
  sw t0, 1032(a0)
  lw t0, 12(a0)
  addi t0, t0, 7
  sw t0, 1036(a0)
  lw t0, 16(a0)
  addi t0, t0, 7
  sw t0, 1040(a0)
  lw t0, 20(a0)
  addi t0, t0, 7
  sw t0, 1044(a0)
  lw t0, 24(a0)
  addi t0, t0, 7
  sw t0, 1048(a0)
  lw t0, 28(a0)
  addi t0, t0, 7
  sw t0, 1052(a0)

  /* fflags path: one FP instruction raises a deterministic flag delta and the flags are read with csrr.
   * t0 = 0x00800000 = 2^-126 (the smallest normal); the product 2^-252 rounds to +0: UF|NX = 0x3. */
  /* fflags path: one FP instruction raises a deterministic flag delta and the flags are read with csrr.
   * t0 = 0x00800000 = 2^-126 (the smallest normal); the product 2^-252 rounds to +0 under the encoded
   * RNE: UF|NX = 0x3 whatever frm holds (a dynamic rounding mode, frm=DYN, is a build refusal). */
  lui t0, 0x80
  fmv.w.x ft0, t0
  fmul.s ft1, ft0, ft0, rne
  csrr t1, fflags
  csrrw x0, fflags, x0

  /* output: one MFE direct move Frame -> L2 (descriptor 1), WAIT, then the two phase signals */
  csrrwi x0, CN_CSR_NEXUS_MFE_SUBMIT, 1
  csrrwi x0, CN_CSR_NEXUS_EVENT_WAIT, 1
  csrrwi x0, CN_CSR_NEXUS_PHASE_SIGNAL, 0
  csrrwi x0, CN_CSR_NEXUS_PHASE_SIGNAL, 1

  /* normal retirement chain: stop new issue, wait for the local events, EXEC_DRAINED. Last word of .text. */
  csrrwi x0, CN_CSR_NEXUS_TASK_RET, 0
  .size _start, . - _start
