# Known Oracle Deviations (C021 / W7-02)

This file records verified semantic differences between pinned upstream ISA simulators
(`riscv-isa-sim@93a10ae6` + CoralNPU patches, `mpact-riscv@e4f1e9c1`) and the sealed
Coral Nexus contracts (`protocol/PROTOCOL_FREEZE.json`).

Invariants enforced by `oracle_main.py`:

- Known deviations NEVER relax comparison tolerances (no ULP window, no `float` cast, no DUT
  output copying).
- Where raw Spike disagrees with a sealed Coral Nexus contract, the normative source is the sealed
  contract, and the class is explicitly routed either to `bitwise_scalar.py` / `nexus_extend.py`
  (scalar / CSR / helper scope in C021) or marked `BLOCKED(归 C047)` (BF16/ROD/VME matrix scope in
  CN-G3).

| ID      | Surface                                                                         | Upstream Spike / MPACT Behavior                                                                 | Sealed Coral Nexus Normative Rule                                                                                                             | C021 Handling & Owner                                                                                                                                       |
| ------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `KD-01` | VME BF16 accumulate-stage ROD overflow (`vtfmm.alt.tvv`, `OVF-ACC-01`)          | Raw Spike `f32_add_odd` and patched MPACT `F32AddOdd` increment `FLT_MAX` to `±Inf` on overflow | Pinned Coral RTL and sealed `w3_numeric_profile_v1.yaml` saturate accumulate-stage ROD overflow to `max-normal` (`0x7f7fffff` / `0xff7fffff`) | Registered as `oracle_kind: NONE` -> `BLOCKED(归 C047)` in C021; independent bitwise VME oracle in C047 is normative, never raw Spike                       |
| `KD-02` | VME lane mask before special-value classification (`MSK-01`)                    | Raw Spike `vtfmm` has no lane-mask input                                                        | Sealed `w3_numeric_profile_v1.yaml` `MSK-01` masks inactive lanes to `0 x 0` before NaN/Inf classification                                    | Registered as `oracle_kind: NONE` -> `BLOCKED(归 C047)` in C021; covered by C047 bitwise VME oracle                                                         |
| `KD-03` | Per-instruction `fflags` / `vxsat` retire delta (`ctx_ctrl.yaml` `flags_delta`) | Spike `STATE.fflags` / `STATE.vxsat` are sticky architectural CSRs accumulated immediately      | Coral Nexus `retire_record.yaml` carries 6-bit per-instruction `flags_delta` (`fflags[4:0] \| (vxsat<<5)`) merged in program order at retire  | `spike_ref_driver.py` isolates per-step sticky deltas and `bitwise_scalar.py` independently computes exact integer bit-level FP32 `fflags_delta` (`BOTH`)   |
| `KD-04` | Nexus-private CSRs `0x7c8..0x7cf` (`csr_name_map.yaml`, `cn_layout.json`)       | Unallocated in upstream Spike -> raises illegal-instruction trap (`mcause = 2`)                 | Sealed `cn_layout.json` allocates `nexus.mctx.*`, `nexus.msnap.*`, `nexus.mregion.*`, `nexus.missue.*` in `0x7c8..0x7cf`                      | Raw Spike probe records `upstream_oracle_kind: NONE` (never faked as upstream support); explicit `nexus_extend.py` adapter handles C012/C014/C038 semantics |
