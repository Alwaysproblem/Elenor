# Coral Nexus hardware workspace maintenance

Operational handbook for the `hardware/coral_nexus` workspace (CN-G0 onward).
Verification evidence lives under `out/coral_nexus/<CARD>/attempt-NNN/`;
`out/` is gitignored and never constitutes an archive by itself.

## Fresh-machine bootstrap (full sequence)

Every step below is a verified entry on a new clone. The G0 gate only turns
green after the evidence producers have actually run on THIS machine (or
their archived outputs were restored and re-pinned — the pin verifies bytes,
it does not create them).

```bash
git clone --recurse-submodules <repo> && cd coral-npu-nexus
alias cn="$PWD/hardware/coral_nexus/scripts/cn.sh"

cn env build        # or: docker load < archived image, then cn env verify
cn sources verify   # 19 checks; the bare form defaults --lock/--out

# Evidence producers (fresh attempts under out/coral_nexus/...):
cn baseline --config hardware/coral_nexus/configs/p0_coral.yaml \
  --out out/coral_nexus/C005/baseline
cn bf16-baseline --out out/coral_nexus/C006/attempt-001  # ~10 min, two-seed KAT

# Offline archive: needs ONE online build first to produce the vendor source
# tree (see Offline seeds), then export and the two proofs:
cn env export --out out/coral_nexus/C004/export-001
python3 hardware/coral_nexus/scripts/lib/sources_tool.py materialize \
  --lock hardware/coral_nexus/env/sources.lock \
  --out out/coral_nexus/C004/offline/work
cn env offline-rebuild --export out/coral_nexus/C004/export-001 \
  --work out/coral_nexus/C004/offline/work \
  --out out/coral_nexus/C004/offline/rebuild
cn env devcontainer-probe --out out/coral_nexus/C004/devcontainer

cn gate --stage G0 --out out/coral_nexus/C007/gate  # 31 checks, exit 0
```

Known gap on a new machine: the ~7 GB vendor archive bytes are NOT in git
(only the pinned digest is). Rebuilding them requires one online C005-style
build followed by `bazel vendor`, or fetching the archive from an artifact
store that does not exist yet.

## Two dispatchers, one registry

| Where     | Entry point                                  | Behaviour                                                                                    |
| --------- | -------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Host      | `hardware/coral_nexus/scripts/cn.sh`         | runs host commands (`env`, `sources`, `gate`, `unit`, `baseline`, `bf16-baseline`, `verify`) |
| Container | `/workspace/hardware/coral_nexus/scripts/cn` | refuses any command whose `executor` is not `container` (exit 3)                             |

Both read the same registry (`scripts/commands.yaml`); there is no shadow
list. For convenience on the host:

```bash
alias cn="$PWD/hardware/coral_nexus/scripts/cn.sh"   # or: export PATH
```

Exit codes are part of the contract: `0` success, `1` checker/assertion
failure, `2` usage error, `3` unknown command or suite, `4` registered but
not implemented.

## Environment

- One locked build image: `coral-nexus-builder:c002` built from
  `env/Dockerfile` (base digest, tool versions and deviations recorded in
  `env/image.lock`). Build and verify:

  ```bash
  cn env build
  cn env verify --out out/coral_nexus/C002/verify
  ```

- Jobs always run as the caller UID/GID; no docker socket, no
  `--privileged`, repository mounted read-only.
- Python split: validator in conda env `elenor-validator`
  (`pipeline_validator/environment.yml`, hash locked in
  `env/requirements.lock`); container python is distro-managed only.

## Sources

- Upstreams are pristine submodules pinned in `env/sources.lock`
  (gitlink + git-archive sha256 per source; compiler-embedded pins recorded,
  never aligned). Verify: `cn sources verify --out <dir>`.
- Derived builds materialize into fresh work roots under
  `out/coral_nexus/work/` via `scripts/lib/materialize.sh`; overlay entries
  (`env/overlay.yaml`) and the patch series (`patches/series`,
  single-owner enforced by `patches/file_owners.yaml`) apply strictly there.

## Focused C027 acceptance

The `w2` suite contains C026, C027 and C028. To verify C027 without rerunning
the other cards, use its independent pytest entry:

```bash
EVIDENCE=out/coral_nexus/C027/attempt-NNN
mkdir -p "$EVIDENCE"
conda run --no-capture-output -n elenor-validator python -m pytest \
  hardware/coral_nexus/verification/unit/w2/test_w2_c027.py \
  -n0 -v --basetemp="$EVIDENCE/pytest"
```

Requires the locked Docker builder, the populated `cache/w2_bazel` offline
caches, and the C021 pinned Spike/toolchain cache. The runner materializes
strict patches, builds the hermetic Bazel Verilator, executes native SV
cases and Scala RTL scenarios, then compares cancel-time retirement records
with independent Spike runs. It preserves logs, JUnit, native results and
oracle comparison output under the attempt. A mutation is accepted only
for a real behavioral assertion failure; build errors and skipped cases fail.
`PROTOCOL_FREEZE.json` must remain byte-identical.

Patch files are excluded from whitespace/EOF fixers: removing a trailing
blank context line can corrupt a unified diff. Regenerate patch artifacts
from the derived tree rather than formatting them.

Prettier also excludes C012 declaration artifacts and the sealed W3 field
requests. `clang-format` excludes C018's `bootstrap_control.c` and
`bootstrap_helpers.h`: `kernel_measurement.json` pins their source bytes.
Restore formatter-only changes to these files; do not change the protocol
seal or measurement merely to accept formatting.

ChiselSim test titles become build-directory names: use shell-safe words,
not semicolons, pipes or ampersands. The simulation launcher adds ChiselSim's
`support-artifacts` include directory and parses macro headers before SV
consumers; FPnew definitions must not depend on resource enumeration order.
Native-resource simulations retain warnings under the upstream
`fpga/coralnpu_soc.core` `-Wno-fatal` policy; this is functional acceptance,
not a warning-free lint claim. Chisel-only cases keep fatal warnings.

## Focused C028 acceptance

C028 consumes the reopened C019 target field `fetch_outstanding_per_context`.
Generate its Scala consumer after an authorized target update; the acceptance
entry rejects a stale consumer or an invalid joint seal before building.

```bash
python3 hardware/coral_nexus/scripts/lib/gen_frontend_params.py
EVIDENCE=out/coral_nexus/C028/attempt-NNN
mkdir -p "$EVIDENCE"
conda run --no-capture-output -n elenor-validator python -m pytest \
  hardware/coral_nexus/verification/unit/w2/test_w2_c028.py \
  -n0 -v --basetemp="$EVIDENCE/pytest"
```

The isolated entry executes the stale-fetch and non-prefix mutants before the
green fetch, dispatch and real-Core specs, retaining JUnit and per-target logs.
It does not execute C026/C027. Build failures are not mutation kills. The
frontend's tagged-port tests cover delayed/out-of-order responses; the native
Core adapter retains the existing fixed-next-cycle IBus timing. Neither result
claims C030 physical-port arbitration or C040 full-Tile assembly is complete.

## Focused C029 acceptance

C029 consumes the sealed C013 memory bounds and C019 C1/C2 targets through
`scripts/lib/gen_memory_params.py`. Its generated `c029_params_pkg.sv` is an
experimental geometry consumer, not a new capacity authority or silicon-width
freeze. Regeneration must not modify the protocol seal.

```bash
python3 hardware/coral_nexus/scripts/lib/gen_memory_params.py --check
EVIDENCE=out/coral_nexus/C029/attempt-NNN
bash hardware/coral_nexus/scripts/cn.sh unit --suite w4 \
  --config hardware/coral_nexus/configs/cs1_c2.yaml --out "$EVIDENCE/unit"
conda run -n elenor-validator python -m pytest \
  pipeline_validator/tests/test_memory_invariants.py -n0 -q \
  -k "double_release or stale_generation"
```

The unit handler forwards `--config` as `CN_UNIT_CONFIG` and retains pytest
temporary artifacts under `<out>/pytest`. The W4 runner rejects a config that
differs from its sealed target; C2 also exercises the C1 same-RTL degradation
direction. It builds the pinned Verilator from the existing offline
`cache/w2_bazel` caches inside the locked builder, then runs behavioral mutants
before the bank, SlotFrame and integrated arena-ledger benches. A mutation
must compile and reach an RTL assertion; build failures and process timeouts
are not mutation kills.

Bank clients capture `req_generation_o` on acceptance and echo that incarnation
in `fin_generation_i`; caller txn IDs alone are not lifetime identities.
Invalidation waits for bank ownership, clears one sourced data word per cycle,
and ACKs only after the final clear. It never rewrites the SRAM bytes.

Frame PREPARE streams immutable layout records without imposing a whole-program
buffer-count limit. VIEW_BIND consumes the resident descriptor owner's immutable
per-buffer lookup; Frame storage remains bounded by live Slots. The ledger
returns a fresh `rsp_view_alloc_id`, and its engine-facing shadow query exposes
the actual handle. Closing views are no longer query-valid.

`exec_drained` is a current-Task retirement handshake, not merely an empty ROB
bit; `exec_drained_generation_i` must match the current Context. The ledger uses
a selected sourced profile (`ACTIVE_PROFILE_IDX`); integrated runtime profile
sequencing remains C058. Reference events must arrive exactly once from their
producer, which retains transaction de-duplication ownership.

The 48-byte indexed stimulus verifies actual bank bytes and ordering, not an
unimplemented Gather/Scatter FSM. Python lifecycle tests use existing reference
objects only; they do not create a second L1 execution ledger. This entry does
not execute C026/C027/C028 or claim C030/C040 completion.

## Focused C030 acceptance

C030 (MSHR, banked cache, program port, native fetch integration) runs under
the same `cn unit --suite w4` entry as C029. All four RTL runners share
`verification/unit/w4/w4_rtl_harness.py` (sealed-target fence, source/seal
hashes, offline Verilator build, Docker run, mutant/green verdict rules); a
runner only declares its bench, sources and mutant globs. A slice's mutants
must match every glob it names, and each must compile and then die on an
executed `C030_ASSERT`.

```bash
EVIDENCE=out/coral_nexus/C030/attempt-NNN
bash hardware/coral_nexus/scripts/cn.sh unit --suite w4 \
  --config "$PWD/hardware/coral_nexus/configs/cs1_c2.yaml" --out "$EVIDENCE/unit"
```

`--config` must be a path, not a bare name. Tests run only if registered in
`scripts/commands.yaml` under `unit.suites.w4`; an unregistered runner is
silently skipped by this entry. Editing `cache.sv` or `transfer_outbridge.sv`
requires regenerating the matching `cache-*`/`bridge-*` patches, since a patch
that no longer applies fails the whole runner rather than a single mutant.
Benches must stay valid at CONTEXT_COUNT 1 and 2: index peer Contexts through
a clamped constant, never a bare `[1]`.

## Focused C031 acceptance

C031 is the scalar LSU module-level integration, not the complete C040 Tile
top. `rtl/memory/lsu_mslot.sv` owns bounded per-Context requests, real bank
accesses and reliable owner-tagged completions. `lsu_cache_path.sv` drives the
C030 cache/MSHR/outbridge; cached windows are trusted read-only bindings, not
free global pointers. The per-Context limit is consumed from the sealed
`configs/w4_capacity_decisions.yaml`; do not substitute the upstream
single-Context transaction-ID count.

```bash
EVIDENCE=out/coral_nexus/C031/attempt-NNN
CN_UNIT_CONFIG="$PWD/hardware/coral_nexus/configs/cs1_c2.yaml" \
  conda run --no-capture-output -n elenor-validator python -m pytest \
  hardware/coral_nexus/verification/unit/w4/test_w4_c031.py \
  -n0 -v --basetemp="$EVIDENCE/pytest"
```

The integration bench exchanges actual requests and terminal ACKs over local
FIFOs with `pipeline_validator.cosim.memory_service.MemoryService`. That service
owns only external bytes and terminal references; Python never executes L1.
Each mutant/green run gets an isolated service, with `memory-service.json` and
`*.service.log` retaining the request/terminal history. Delayed, reordered and
duplicate transport responses must not duplicate completion or return credit
before terminal settlement. The native-return proof uses the generated full
C026 retirement bank; the mini-ROB emitted by the default CoreMiniVerification
target is not sufficient evidence.

## Focused C032 acceptance

C032 covers the MFE direct-move engine plus per-transaction settlement. The
bench drives real read and write payloads against the C023
`pipeline_validator.cosim.memory_service.MemoryService` over FIFO endpoints, so
`c032_mfe_peer.py` owns only external bytes and terminal references — no L1
execution and no completion timers. Run it through the registered slice:

```bash
EVIDENCE=out/coral_nexus/C032/attempt-NNN
CN_UNIT_CONFIG="$PWD/hardware/coral_nexus/configs/cs1_c2.yaml" \
  conda run --no-capture-output -n elenor-validator python -m pytest \
  hardware/coral_nexus/verification/unit/w4/test_w4_c032.py \
  -n0 -v --basetemp="$EVIDENCE/pytest"
```

The MFE has no Context dimension; the bench declares only a registration shim,
so do not add it to a `contexts=(1, 2)` list. Settlement rule shared with the
Python manager: a cancel settles ISOLATED only when no external write leg was
accepted for that transaction; an already-completed write binds COMPLETED, and
the tracker/cancel_tracker poison may only be cleared by an explicit quiescent
reset. `test_w4_cancel_isolation.py` pins the Python half of that rule and is
registered in the w4 suite, so a Python-only regression also fails the suite.

Wire ACK versus local visibility: each accepted MEM_REQ leg gets exactly one
wire TERMINAL_ACK (`term_ack_o`), which settles only the transport obligation,
while each accepted descriptor gets exactly one local terminal (`term_o`, via
the tracker). Only a successful, non-absorbed inbound MEM_RSP writes the local
window — cancelled (absorbed) or faulted inbound responses write nothing — and
those bytes commit on the MEM_RSP itself, before that leg's wire ACK, so they
are peekable while the ACK stalls.

The `S_TERM` presenter in `rtl/memory/mfe_top.sv` parks its selected record, so
a reordered older MEM_RSP arriving while a younger ACK stalls under
`term_ack_ready_i=0` cannot swap the offered payload: valid and payload stay
stable through the fire cycle and the tracker counts exactly that record. A
surviving mutant against this logic needs a distinguishing legal input (a
reordered MEM_RSP under sustained ACK backpressure, whole offered ACK stable,
each handshake exactly once) or an equivalence proof plus a rebase onto
observable behavior — never a watchdog, tautological or source-only assertion.

## Focused C033 acceptance

C033 is W6-02 unit evidence: `tb_c033` instantiates the real `rtl/w6`
`event_table` and `terminal_store` (with their internal `wait_record` and
`completion_bus`). It is not full-Tile assembly (C040), not the retirement
chain (C035), and not the reset/drain FSM (C034).

```bash
EVIDENCE=out/coral_nexus/C033/attempt-NNN
bash hardware/coral_nexus/scripts/cn.sh unit --suite w6 \
  --config "$PWD/hardware/coral_nexus/configs/cs1_c2.yaml" --out "$EVIDENCE/unit"
```

The suite is registered in `scripts/commands.yaml` under `unit.suites.w6`;
an unregistered runner is silently skipped by this entry. The runner shares
`verification/unit/w4/w4_rtl_harness.py` (sealed-target fence, source/seal
hash fence, offline Verilator build in the locked builder, Docker run,
mutant/green verdict rules) with `rtl_dir`/`mutants_dir` pointed at the W6
roots; the W4 defaults are unchanged.

Geometry: CONTEXT_COUNT is generated from the sealed target (C1 and C2 both
run). FRONTIER=2, OPERANDS=3 and the terminal reserve of 1 follow the C015
`make` test fixture; COUNTER_BITS=4 is deliberately narrowed so stress runs
saturate the counters, and SOURCES=4 stands for the four producer classes
(engine/MFE/VME/Group). All are verification geometry instantiated inside the
bench, not frozen silicon values; identities are full 64-bit transport words.
Greens are exactly `tb_c033` at the C1 and requested counts plus
`tb_c033_min` — the same fixture file instantiated with OPERANDS=2, this
bench's configured WAITALL ceiling (it matches the explicit
`make(..., waitall_max=2)` C015 boundary test; the sealed contract makes 2 the
WAITALL minimum and leaves the maximum unresolved, owner C018) — at C1 only;
no mutant runs against the alternate top.

The runner loads the nine explicit scenario YAMLs
(`verification/scenarios/w6/u2?_*.yaml`, cases `U2-a`..`U2-i`, each with a
`seed` and an `expect` mapping) fail-closed, and materializes them as twelve
space-separated lowercase-hex 64-bit words per row (four seed words then
eight expected: armed, success, failure, consumer_refs, future_uses, phases,
retire_ready, completions), rows sorted a..i, into
`evidence/c033-vectors.txt`. Every simulator invocation — mutants included —
receives `+VECTORS=/out/c033-vectors.txt`; the bench owns comparing real RTL
state against those words and emits exactly one `C033_CASE U2-x` line per case
plus a single `C033_PASS <top> contexts=N checks=M` record. The runner never
hard-codes green output: it requires the nine case lines exactly once per green
run, binds each PASS record's top and `contexts=N` to that run's planned
identity, asserts the exact (top, Context count) green pairs, replays each
green's PHASE_SIGNAL handshakes through the live C024
EventService/Group (`c033_phase_replay.py`), and derives C1/C2 from the real
sealed config.

Receipts retained per attempt: `requested-target.json`, `source-sha256.json`
(nine YAMLs, runner, harness, phase replay, EventService and its Group replay
inputs `tile_group.py`/`test_runtime.py`, both C015 contract fragments, W6 RTL
sources and mutant patches — fenced before and after), `c033-scenarios.json`
(per-case file hash plus validated seed/expect) and `c015-fragments.json`
(both fragment hashes cross-checked against the actual C020
`PROTOCOL_FREEZE.json`; the seal itself must stay byte-identical).
Behavioral mutants live under `verification/unit/w6/mutants/event-*.patch`;
at least nine must apply, compile, and die on an executed `C033_ASSERT` —
build failures, timeouts and watchdogs are not mutation kills.

## Focused C034 acceptance

C034 is W6-03 unit evidence: `tb_c034` instantiates the real `rtl/w6`
`event_table`, `terminal_store`, `first_fault_latch` and `reset_endpoint`
next to the real C032 `mfe_top` (internal `transfer_cancel_tracker`) and the
real C023 MemoryService, so reset/drain behavior is checked against live
Group cycles and live memory legs rather than a stub. It is not full-Tile
assembly (C040), not the retirement chain (C035), and not C035 reset policy.

```bash
EVIDENCE=out/coral_nexus/C034/attempt-NNN
bash hardware/coral_nexus/scripts/cn.sh unit --suite w6 \
  --config "$PWD/hardware/coral_nexus/configs/cs1_c2.yaml" --out "$EVIDENCE/unit"
```

The `unit.suites.w6` registry entry keeps the unmodified C033 runner, adds
`verification/unit/w6/test_w6_c034.py`, and registers the focused Python
regression `pipeline_validator/tests/test_w6_reset_endpoint.py`; the gate
therefore runs both C033 and C034. The runner reuses
`verification/unit/w4/w4_rtl_harness.py` with `rtl_dir` pointed at
`hardware/coral_nexus/rtl` so the W6 reset sources, the C032 tracker and
`mfe_top` compile in one bench; sources are ordered `w6/c033_event_pkg.sv`,
`w6/c034_reset_pkg.sv`, `w6/wait_record.sv`, `w6/completion_bus.sv`,
`w6/event_table.sv`, `w6/terminal_store.sv`, `memory/c029_params_pkg.sv`,
`memory/c032_mfe_pkg.sv`, `memory/transfer_cancel_tracker.sv`,
`memory/mfe_top.sv`, `w6/first_fault_latch.sv`, `w6/reset_endpoint.sv`,
`w6/tb/tb_c034_memory.sv`, `w6/tb/tb_c034.sv` — the memory testbench file
compiles immediately before the reset top and carries the C032 MFE / C023
service tasks, splitting them from the reset behavioral checks. Greens are
exactly `tb_c034` at the C1 and requested Context counts; mutants run
against the same bench.

The runner loads the nine scenario YAMLs
(`verification/scenarios/w6/u3?_*.yaml`, cases `U3-a`..`U3-i`, each an
`inject` mapping of generation/tag/hold_cycles and an `expect` mapping of
winner/suppressed/marked/terminals/stale/poisoned — all strict unsigned
words) fail-closed and materializes them as nine lowercase-hex words per row
(three inject then six expect), rows sorted a..i, behind
`+VECTORS=/out/c034-vectors.txt`. Scenario loading, vector serialization and
the C015-fragment seal receipts are shared with the C033 runner through
`verification/unit/w6/w6_scenario_io.py`, fenced in both host_sources. Every
simulator invocation also receives `+PEER=/out/<label>`.

`c034_group_peer.py` is the live boundary behind `+PEER`; its module
docstring is the command reference (every reply exactly three words, errors
nonzero, never a hang). In brief: U3-f/U3-h/U3-i run real Group cycles —
genuine Simulator/TileGroup, production `accept_tile_fault` (GROUP),
per-phase requests from stepping the actual ResetDomain (bounded 64 steps
per the C034 contract), observed RESET_PHASE_ACKs, the production isolation
poison, explicit `TileGroup.reset()` recovery with ordered receipts, and a
`Z` that demands the real DONE observation plus zero residue. U3-e/U3-f open
real C023 windows through the reused `c032_mfe_peer.Peer`, and `G` observes
a real bounded FaultRing (retention, oldest source, physical index).

Receipts retained per attempt: the shared harness set plus `group-peer.json`
(per-label live cycles, memory windows, ring observations, open-state
residue, errors) and the per-label `.peer.log` transcripts. Greens must show
three completed live Group resets with the U3-f ordered recovery exactly
`(refused, poisoned)` then `(allowed, cleared)`, two closed memory windows
whose ordered W4 terminal classes are exactly [0,1,2,0] (U3-e, with exactly
one 8-byte write) and [1] (U3-f inbound), and real FaultRing observations.
Behavioral mutants live under `verification/unit/w6/mutants/reset-*.patch`;
at least eight distinct patches must apply, compile and die on an executed
`C034_ASSERT` — build failures, timeouts and watchdogs are not kills.

C015 anchor drift: the frozen contracts cite ten Python statements in
`pipeline_validator/tile_group.py` and
`pipeline_validator/runtime/reset_domain.py` whose lines moved (six had
already drifted at the C033 base; four moved with the C034 ResetDomain
rework; all statements are unchanged and remain in the same owning methods).
`validate_w6_contract.py` binds exactly those ten citations by their contract
identity (label + field location) AND their sealed (file, original line,
needle) triple, in a fail-closed author-time table (`ANCHOR_RELOCATIONS`,
derived from seal revision `5ed2217`). The sealed identity is verified
FIRST: an altered line, needle, file or a swapped citation fails with the
ordinary SOURCE_LINE_MISMATCH before any text is read back, so pointing a
citation at today's moved text — or at a comment or docstring — cannot
bypass the pin. The pin is then always AST-verified against the current
source: exactly one statement/header with the exact original text in the
single top-level class with the pinned name and its unique direct method
(a shadowed class re-binding rejects), walking every descendant except
nested lexical scopes (try/except and match-case bodies count; an inner
def/class/lambda and anything inside it does not — nor does a docstring or
a neighbouring method). A pinned citation deleted or retyped so the walk
never sees it fails with MISSING_SOURCE. Root/ownership mismatches and
missing, ambiguous or edited statements are rejected. Relocations are
reported additively as `relocated_sources` (original and verified current
line, source and owner) only when the live line actually moved. Every other
citation keeps its exact-line check; the validator never calls git at
runtime, there is no fuzzy needle search, and source lines are never padded
to preserve numbers.

## Offline seeds (C004)

- Inventory: `env/seed_targets.json` — every seed has URL/path + sha256 +
  license row in `env/notices.md`. `bazel_vendor_cache` is a `local-dir`
  seed whose sha256 is the tree digest of the **archived** copy.
- Directory seeds are archived by copying the tree and re-hashing it. In-tree
  relative symlinks are preserved (bazel uses them for interpreters and tool
  aliases); absolute or escaping links are dropped and listed under
  `dropped_links`, because they point at a producer-local path and would make
  the archive non-self-contained. A missing source or a digest mismatch is a
  hard failure that names the seed — the exporter never refills silently.
- Export into a fresh versioned directory (refuses a non-empty one):

  ```bash
  cn env export --out out/coral_nexus/C004/export-<stamp>
  ```

  Export reuses `out/coral_nexus/cache/seeds/`, verifies every sha256, and
  fails if any seed is missing or mismatched. Report: `export_manifest.json`.

### Offline rebuild proof (C004)

Rebuild the P0 targets with no network and empty build caches, using only the
exported vendor archive:

```bash
python3 hardware/coral_nexus/scripts/lib/sources_tool.py materialize \
  --lock hardware/coral_nexus/env/sources.lock --out out/coral_nexus/C004/offline/work
cn env offline-rebuild --export out/coral_nexus/C004/export-<stamp> \
  --work out/coral_nexus/C004/offline/work --out out/coral_nexus/C004/offline/rebuild
```

The probe runs `docker --network none`, asserts the bazel output root, disk
cache and repository cache are empty, re-verifies the archive digest, then
builds `-c opt`. It copies the two ELFs into `artifacts/` and records their
hashes; the G0 gate refuses an offline rebuild whose ELF bytes differ from the
recorded baseline (`offline_rebuild.matches_baseline`).

### Devcontainer proof (C004)

```bash
cn env devcontainer-probe --out out/coral_nexus/C004/devcontainer
```

Launches the container through `@devcontainers/cli` (fresh, from the current
JSON), then probes the non-root user, tool versions, the `/workspace` mount
and a real `bazel build` inside it. A static JSON check is not acceptance:
when the CLI is missing the command records a blocker and exits 3.

## Gates (C007 / C008)

```bash
cn gate --stage G0 --out out/coral_nexus/C007/gate
```

G0 verifies five evidence sets (`sources_verify`, `environment_verify`,
`seed_export`, `baseline_run`, `bf16_baseline`) against
**`verification/expected_evidence.json`** — a committed pin holding the
sha256 of every required report _and_ artifact plus the expected verdict.
The pin is the root of trust: re-hashing against a manifest rebuilt from
the same directory proves nothing, and a directory containing only
`{"pass": true}` fails. A missing or incomplete pin fails closed
(`pin.present` / `pin.coverage`).

Exit codes: `0` pass, `1` evidence mismatch, `2` usage, `4` stage not
implemented. `--evidence-index DIR` and `--pin FILE` override the
defaults (used by the negative tests in
`verification/contracts/test_g0_gate.py`).

### Refreshing the pin

After a legitimate re-run of any evidence producer, the recorded hashes
change and the gate fails until the pin is updated **in a reviewed
commit**:

```bash
cn gate --stage G0 --out out/coral_nexus/C007/gate            # see which hashes moved
python3 hardware/coral_nexus/verification/gates.py \
  --evidence-index out/coral_nexus/C007/evidence_index \
  --pin hardware/coral_nexus/verification/expected_evidence.json --write-pin
```

Never refresh the pin merely to make a red gate green: an unchanged
verdict with changed bytes means the producer changed, and that is a
review event.

## CN-G1 gate and the capacity decisions (C020)

`cn gate --stage G1 --out <dir>` re-runs every producer validator (C009-C016), re-renders every generated
artifact, checks the measured resource record and verifies the joint seal. Exit codes: 11 contracts,
12 codec artifacts, 13 CSR bindings, 14 open capacity budget, 15 measured resources, 16 targets, 17 seal,
18 gate spec drift.

Two capacities have no derivable bound and are therefore **decisions**, recorded in
`configs/w4_capacity_decisions.yaml` (read by `freeze_check.open_budgets`, sealed by hash):

- `per_ctx_outstanding_credit_cap`: per-Context CreditManager ceiling, in entries.
- `response_record_budget`: waiter response records per MSHR refill entry, in entries.

A budget is closed only when `value` is a positive integer in `entries` and `decided_by`, `rationale` and
`evidence` are non-empty (for the response budget the rationale must say the hardware stalls when full).
`null`, 0, a bool, a float or a blank field keeps it open. Recording a decision does not edit
`w4_memory_constraints.yaml`: both stay `derivable: false` there. After filling the file: run
`python hardware/coral_nexus/protocol/freeze_check.py seal`, then `cn gate --stage G1`. Changing a recorded
value later changes the file hash, so the seal stops verifying until `seal --bump`.

## Devcontainer

`.devcontainer/coral_nexus/devcontainer.json` builds the same locked image
(same Dockerfile, same digest) and runs as the non-root `builder` user with
the repository mounted; it mounts no docker socket and adds no privileges.
Static policy violations (privileged, socket mount, `build` section) are
rejected by `test_environment.py`.

## Upgrade path

1. Change the pin/hash in the relevant lock (`sources.lock`,
   `tools.lock`, `image.lock`, `seed_targets.json`); add/adjust the
   notices.md row in the same commit.
2. Rebuild the image and rerun `cn env verify`; re-materialize work roots
   and rerun affected suites — old PASS never carries over.
3. Bazel vendor seeds: after a green in-container build, refresh
   `bazel_vendor_cache` in `seed_targets.json` and re-export; offline
   rebuild verification is C073's exec-repro gate.
