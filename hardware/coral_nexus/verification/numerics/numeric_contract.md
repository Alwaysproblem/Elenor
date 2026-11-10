# W3 BF16 numeric contract (C011) — DRAFT, experimental

Machine-readable form: `protocol/w3_numeric_profile_v1.yaml` (rules, sources, open items).
Executable form: `scripts/lib/numeric_ref.py`. Vectors: `known_vectors.json`.

This pins what the **unmodified pinned upstream** does for the BF16 `vtfmm` matrix multiply and the
final narrowing. It is a DUT contract check. It is **not** the independent oracle (C047/W7-02), and
nothing here marks a numeric capability `supported`.

## What the hardware computes

One output element of `vtfmm` with `tk` K lanes (`tk` is 1 or 2 for BF16):

1. **Mask first.** A masked lane has both operands forced to 0 _before_ classification, so it is
   `0 x 0`: it raises no NV and never contributes an Inf or NaN (rule MSK-01).
2. **Multiply and tree.** Each live lane forms the exact product of two BF16 significands. Lanes are
   aligned to the largest normalised product exponent on a 26-bit grid; bits that fall below the grid
   are OR-ed into the LSB (**jam**, rule JAM-01). The aligned values are added as signed integers.
3. **One ROD rounding to FP32.** The tree sum is rounded once with round-to-odd (rule ROD-01). A
   widening multiply forces ROD whatever `frm` says.
4. **Accumulate.** The result is added to the tile's FP32 accumulator and rounded again with ROD
   (rules ROD-01, OVF-ACC-01, CAN-01).
5. **Flags.** Only OF and NV leave the VME; NX, UF and DZ do not (rule FLG-01). This limits the matrix
   instruction only: other RVV instructions keep their own IEEE flags.

## Two overflow behaviours that are deliberately different

| Stage      | Condition                                   | Result                          | Rule       |
| ---------- | ------------------------------------------- | ------------------------------- | ---------- |
| Multiplier | BF16 → FP32 widening, source ≠ destination  | **+Inf**, OF                    | OVF-MUL-01 |
| Accumulate | ROD saturation (`round_of_to_max_norm` = 1) | **max-normal** `0x7F7FFFFF`, OF | OVF-ACC-01 |

A single BF16×BF16 product cannot overflow FP32, so multiplier-stage overflow needs the two-lane sum
to exceed FP32 max (vector `mul_overflow_two_lane_sum`).

KD-01: raw Spike/MPACT `F32AddOdd` returns Inf for the accumulate case. The pinned RTL does not. The
RTL contract is **not** changed to match the reference.

## Things this card found that the design text does not say

- **The jam is invisible under ROD for two lanes.** A lane only loses bits when its exponent gap to the
  largest lane is at least 11, which limits cancellation to one leading bit; the jam bit then behaves
  as the sticky bit and ROD only reads `round | sticky`. Measured: 40000 random two-lane trees, 0
  differences from one exact ROD rounding. The finite-width tree is therefore **not** observably
  different from exact math here, contrary to the design's "not mathematically exact" framing. The jam
  is still pinned, by ablation: plain truncation differs on 25247 of 40000 trees and an internal RNE
  differs on 1708 of 40000.
- **Signed zero is not IEEE.** `-0 + (-0)` yields `+0`, because the tree forces the product sign to `+`
  when it is zero (`zvt_pe_mulbulk_fp_lane.sv` L478) and the adder then sees an effective subtraction.
- **ROD never overflows at exactly max + half an ulp**: the max mantissa has guard bit 1, so ROD does
  not round up. The first value that overflows under every mode is `2^128`.
- **frm.** `5` is ROD, `7` is DYN; both and the unassigned `6` are rejected by the profile.
- **cvfpu line citations in the design drift.** The vendor copy confirms `ROD` at L146 and the FP16ALT
  enum at L45; the design's L55 is not the FP16ALT encoding.

## What is and is not proven

| Claim                                                                                       | Evidence                                                        |
| ------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Every cited RTL line says what the profile claims                                           | validator reads each `file:line` back (39 anchors)              |
| Generator parameters (VEC_LEN 4, guard 2, 26-bit grid)                                      | re-derived from the format table and compared                   |
| Reference matches real RTL on the four C006 KATs                                            | 1024 cells, byte-identical pinned record                        |
| Each rule has a directed vector, expected bits derived by hand                              | `known_vectors.json`; model checked against them                |
| Jam, ROD vs RNE, overflow asymmetry, mask order are enforced                                | ablations and 14 reference mutants each fail a test             |
| Result bits of 30 boundary vectors (overflow, specials, mask, cancellation, subnormal, ROD) | real RTL probe, 30/30 equal to the hand-derived bits            |
| OF/NV flag values                                                                           | **model and source lines only**; RTL flags unobserved (OPEN-04) |
| Tail Tm/Tn                                                                                  | C006 case 3 only; the probe runs tm=tn=16                       |
| Narrowing `vfncvtbf16` RTL path                                                             | **not read back** (OPEN-02); rule is decided by card            |

The flag, tail-in-probe and narrowing rows are the open items: see `open_items` in the profile. The C006 KATs are
quarter-integer exact cases and do not reach rounding, overflow or specials; the C011 probe does, but only for
result bits.
