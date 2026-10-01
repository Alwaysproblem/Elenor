# ELENOR Runtime Pipeline Validator

The validator is a cycle-stepped CPU + accelerator simulator for one
FPGA-oriented Tile Group with four Compute Tiles. It models the
`Graph → Context → Grid → Task → Tile Program → Engine` control path,
finite controller/storage resources, profiled L1/L2 SRAM partitioning,
HBM/NoC/DMA traffic, Stream Queue credit, and PMU attribution.

The supported execution path is deliberately explicit:

```text
source xDSL ModuleOp
  └─ compiler.compile_program(...)
       └─ immutable CompiledProgram (strict persistent artifact)
            └─ loader.load_program(..., actual_bindings=...)
                 └─ LoadedProgram
                      └─ Simulator.run(LoadedProgram)
```

`Simulator.run` accepts only `LoadedProgram`. It rejects source `ModuleOp`,
private execution DTOs, and an unloaded `CompiledProgram`; there is no
runtime source-lowering fallback. Compiler lowering helpers are private to
`pipeline_validator.compiler`; the only public compiler entry is
`pipeline_validator.compiler.compile_program`.

## Explicit compile, load, and run API

```python
from pipeline_validator import (
    GlobalBinding,
    HardwareConfig,
    SimConfig,
    Simulator,
    load_program,
    parse_compiled_program,
    serialize_compiled_program,
)
from pipeline_validator.compiler import compile_program
from pipeline_validator.workload_ir import load_workload_ir

hw = HardwareConfig()
sim = SimConfig(context_count=1, device_context_count=1)
module = load_workload_ir("examples/fixtures/pow_single_context.mlir")
bindings = {"Y": GlobalBinding("Y", 0x100000, 524288, "rw")}

compiled = compile_program(
    module,
    hw,
    sim,
    binding_assumptions=bindings,
    source_name="examples/fixtures/pow_single_context.mlir",
)
text = serialize_compiled_program(compiled)
parsed = parse_compiled_program(text)
loaded = load_program(parsed, hw, sim, actual_bindings=bindings)
result = Simulator(hw, sim).run(loaded)
```

`CompiledProgram` is deeply immutable and contains the canonical source,
Registry, target/artifact hashes, frozen executable, call bindings,
relocations, source map, dependency proofs, readonly import references, entry/exit
Profiles, resource budgets, binding guards, and workload metadata. The compiled
package format currently has `schema_version=2` and `compiler_abi="v2"`; this is
distinct from hardware YAML schema 2. Schema 1 and ABI `v0`/`v1` artifacts are
rejected and must be rebuilt from source. Parsing uses an explicit type/opcode
allowlist and rejects unknown fields/types, duplicate JSON keys, non-finite
numbers, unknown versions, and a mismatched artifact hash.

`load_program` does not import or invoke the compiler. It verifies artifact
integrity and executable semantics, matches the embedded Registry/static
target fingerprint to `HardwareConfig` and `SimConfig`, and validates actual
binding names, ranges, alias guards, and permissions. Loading never repairs
dependencies, inserts waits, chooses a different Profile, or rebuilds source
IR.

## CPU / hardware boundary and scheduling

- `device.py` interprets the compiled `nexus.program`, manages dependency-ready
  pending descriptors, explicit awaits, and bounded outstanding/completion
  credits.
- `runtime/group_port.py` adapts messages: a CPU-accepted root first consumes
  bounded pending metadata, not a Group slot or L2 Arena. Full root admission
  later commits slot, Arena, event/control budget, and ownership.
- `tile_group_sequencer.py` owns each root's registration cursor, fence, and
  completion; the shared `group_scheduler.py` polls completion/control, issues
  at most one action, then registers at most one per cycle. S0 considers only
  each Context's oldest action, S1 may pass PENDING dependencies within
  `scan_width`; S2 currently runs the same implementation as S1.
- `tile_group.py` registers a bounded Grid Route, then each Tile admits at
  most one Task per Tick. One blocked Tile does not undo committed work on
  another Tile; failed preparation can abort/roll back before exposure.
- `tile.py` owns exact simulated UCE context pins, L1 Arena/Frame planning,
  and eligible-head round-robin single-instruction issue. Engine queue
  credit, streams, and event waits may block one head without blocking all.

`--device-context-mode N` is the CPU-delivered outstanding-request limit,
not the Group execution-slot capacity (`group.active_context_capacity`, default
8). The CPU may mark a request ACTIVE before its actual Group `active_cycle`.
`--context-mode N` selects exactly N UCE execution contexts per Tile
(simulation default 1, supported 1..8); silicon count is not frozen.
Optional `nest.context context=N` pins a Group execution slot; dispatch
`context=N` pins that UCE index on every selected Tile. A Frame Slot,
resident program slot, and ABI isolation `context_id` are different resources.

Root admission keeps independent SAME and COMPATIBLE FIFO heads for the active
L2 Profile. SAME is tried first; if its head cannot fully commit, the
COMPATIBLE head may fill the target without skipping within either category.
Task admission applies the same comparison to the active L1 Profile per Tile.
Profile-incompatible work is not a runtime candidate: the compiler must emit a
complete transition beforehand.

Group admission FIFO and registration/issue round robin are distinct
arbitration layers; neither guarantees global starvation freedom. A
`nest.await` fences later registration **within that Context**, and a
`nest.barrier` waits that Context's earlier actions, not all roots.
`note_completion` reclaims credits even when the action table is full.

Representative runnable comparisons: [Tile-SPMD matmul](../examples/workloads/matmul_2048x512x64_boa256x256x32.mlir),
[K-chunk double buffering](../examples/workloads/reduce_sum_ktiled_single_context.mlir),
[S0/S1 branch](../examples/scenarios/ready_action_branch.mlir),
[context-local split-K](../examples/workloads/matmul_splitk_multicontext_pipeline.mlir),
[shared](../examples/scenarios/l2_shared_weight.mlir) versus
[private weight](../examples/scenarios/l2_private_weight.mlir), and
[Profile transition](../examples/scenarios/profile_reconfiguration.mlir).
Use [`examples/run.sh`](../examples/run.sh) for actual bindings/configuration;
see [architecture §20](../design/ELENOR_Architecture_Design_v1.md) for the
full method/condition matrix.

## Source IR and mandatory resource contracts

The custom-assembly xDSL dialect has three ownership levels:

| Prefix    | Owner                       | Examples                                                                                 |
| --------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| `tile.*`  | one Task/Tile Program       | `tile.alloc`, `tile.load.async`, `tile.gather.global.async`, `tile.await`, `tile.signal` |
| `nest.*`  | one root Context/Tile Group | `nest.alloc`, `nest.dispatch.tasks.async`, `nest.dma.prefetch.async`, `nest.await`       |
| `nexus.*` | Host/CPU model              | `nexus.submit_context.async`, `nexus.await`, `nexus.return`                              |

Every `nest.context` has a mandatory L2-only contract:

```mlir
#nest.context_resources<
  l2_mode = 0,
  allowed_profiles = [0, 1, 2],
  logical_tasks = 1,
  l2_spm_bytes = 4096,
  requested_contexts_per_tile = 1
>
```

Every `tile.program` has a mandatory L1 contract:

```mlir
#tile.resources<
  allowed_profiles = [0, 1, 2],
  tile_l1_spm_bytes_per_context = 4096
>
```

`nest.dispatch.tasks.async` has a mandatory `l1_mode=N`. Context owns the L2
selector; the dispatch owns the requested L1 selector. `allowed_profiles` is
non-empty, unique, and must contain the Context's `l2_mode` or the dispatch's
requested `l1_mode`. Mode IDs are identifiers, not an ordering by capacity.
The compiler checks every advertised mode, every real L1/L2 combination used
by the call, per-bank striped layout/padding, Frame Slots, live event frontier,
Grid/control bounds, and the `R ×` child-Arena envelope. It never drops an
illegal allowed mode or silently lowers `requested_contexts_per_tile`.

The optional `l1_cache`/`l2_cache` member has exactly four fields:

```mlir
l1_cache = {
  required = true,
  access = "read",
  bypass = "forbidden",
  target_bytes = 8192
}
```

`access` is `"none"`, `"read"`, or `"read_write"`; `bypass` is `"allowed"`
or `"forbidden"`. These are capability declarations. `target_bytes` and
Gather's `cache_target_bytes` are shared-cache hints, not private allocations,
minimum-capacity quotas, or predictors of hit rate. Gather accepts no separate
minimum-cache property.

This is a complete legal source example; the 64-byte logical buffer is padded
to one 16-bank × 256-byte stripe round, so each declared Arena is 4096 bytes:

```mlir
builtin.module {
  tile.program @read_tile(
    %task : !nest.task,
    %buf : !nest.l2_buffer<1x32xbf16>
  ) resource_contract = #tile.resources<
    allowed_profiles = [0, 1, 2],
    tile_l1_spm_bytes_per_context = 4096
  > {
    %view = tile.subview %buf task = %task task_dim = 0
      offsets = [0, 0] sizes = [1, 32] strides = [1, 1]
      : !nest.l2_view<1x32xbf16>
    %scratch = tile.alloc shape = [32] dtype = "bf16" alignment = 256
      : !tile.l1_buffer<32xbf16>
    %loaded = tile.load.async %view into %scratch : !tile.event<"loaded">
    tile.await %loaded
    tile.signal input_released(%task)
    tile.free %scratch
    tile.return
  }

  nest.context @read_context(
    %input : !nest.global_memref<1x32xbf16>
  ) placement = 1 resource_contract = #nest.context_resources<
    l2_mode = 0,
    allowed_profiles = [0, 1, 2],
    logical_tasks = 1,
    l2_spm_bytes = 4096,
    requested_contexts_per_tile = 1
  > {
    %buf = nest.alloc slot = "input" role = "in"
      shape = [1, 32] dtype = "bf16" alignment = 256
      : !nest.l2_buffer<1x32xbf16>
    %src = nest.subview %input offsets = [0, 0] sizes = [1, 32]
      strides = [1, 1] : !nest.global_view<1x32xbf16>
    %pref = nest.dma.prefetch.async %src into %buf : !nest.event<"pref">
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range
    %grid, %inrel, %unused = nest.dispatch.tasks.async @read_tile
      l1_mode = 0 tasks(%tasks) globals() bindings(%buf) ins(%buf) outs()
      signal_policy {
        input_released = #nest.aggregate<all_tasks>
      } depends_on(%pref)
      : (!nest.event<"grid">, !nest.event<"inrel">, !nest.event<"">)
    nest.release %buf depends_on(%inrel, %pref)
    nest.await %grid
    nest.return
  }

  nexus.program @run(%input : !nest.global_memref<1x32xbf16>) {
    %done = nexus.submit_context.async @read_context(%input)
      : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
```

`logical_tasks` is the sum of all dispatch task counts, including repeated
ranges. The current mapping still requires each non-empty range count to equal
`popcount(placement)`. A zero-dispatch Context is legal with
`logical_tasks=0`; a zero-length `nest.task.range` is not.

## Arena, view, free, and lease semantics

The compiler lays each owner out as a deterministic striped `ArenaLayout`.
The physical bank stride is always the target bank size; the current Profile
only moves the user-SPM/cache boundary. Padding is reserved but is never
visible through a buffer view.

- One root invocation owns one L2 Arena (`RootInvocation`).
- One committed logical Task owns one L1 Arena (`TaskIdentity`).
- L2 `nest.alloc` and L1 `tile.alloc` create buffer views inside their
  already-reserved parent Arena at compiled bind points.
- `nest.release` and `tile.free` invalidate only the named view after its real
  events, pins, Frame references, and in-flight transactions are safe. They do
  **not** return parent Arena extents to the global pool and do not wake another
  owner for capacity.
- A Task Arena returns only when that Task safely retires (or cancellation is
  confirmed). A root Arena returns only after every Route/Task, required
  output, transfer, view, pin, and lease has retired.
- A zero-byte owner still has an explicit Arena/control/lease record even
  though it reserves no SRAM extent.

L1 can statically assign a later `tile.alloc` the same Slot/offset when the
compiler proves an earlier `tile.free` ended its lifetime; runtime merely
rebinds that frozen layout, never searches a free map. L2 is permanently
no-rebind within a root Arena: every local buffer keeps its own whole-stripe
padded span even after release and a Context barrier. If this raises the
high-water mark above source `l2_spm_bytes`, compilation raises the _executable_
reservation and rechecks every allowed Profile, without lowering R or
deleting modes. Freeing dead scratch does not require a Store; values needed
after Task retirement require the relevant Store completion.

`requested_contexts_per_tile` is the R lease bound for
`(parent binding, launch generation, tile)`. The lease is acquired only on
successful Task commit and is returned only at safe Task retirement or
confirmed cancellation. `tile.free`, `input_released`, `output_ready`, and a
cancel request do not return R early. A Grid cannot complete while any selected
Tile is pending admission, active, or retiring.

The Grid's `input_released` covers its selected Tasks' L2 input reads;
`output_ready` covers L1→L2 output writes; done waits all Tasks' safe
retirement. Neither direction signal implies HBM visibility or Arena
retirement. The current aggregate is only `#nest.aggregate<all_tasks>`.
`private` L2 is the default, `context-local` supports same-root leaf/partial/
combine without HBM scratch, and `readonly` uses publish/shared.ref/claim
across roots within the same Group/L2 epoch.

Temporary capacity, fragmentation, Slot, controller, and R-limit waits are
reported separately. A pure plan has no side effects; permanently impossible
per-bank/empty-pool requests fail instead of entering a wait queue.

## Profile Registry, transitions, maintenance, and reset

L1 and L2 Profiles are independent. For each call the compiler keeps the
current mode when it is in that owner's allowed set (COMPATIBLE when it differs
from the requested baseline), otherwise it selects the requested baseline and
emits the required transition. SAME/COMPATIBLE admission never changes a
generation or issues a Commit.

Every actual transition is explicit in the compiled executable. A
`ProfileReconfigDesc` contains the expected/target mode, Registry hash,
generation-bound ordinary-await proof, affected domain, full member list,
source reference, and the fixed sequence:

```text
ACQUIRE → CHECK_FRONTIER → CLOSE_ISSUE → DRAIN_REFERENCES
→ CLEAN_INVALIDATE → DRAIN_DOWNSTREAM → PREPARE → WAIT_READY_ACK
→ COMMIT → WAIT_COMMIT_ACK → OPEN_ISSUE → RELEASE
```

`ProfileController` is the unique L1/L2 writer shared by Device and Group.
The real member bus services one ready request/response per Tick. A new mode is
published and its issue gate opens only after every target member has matching
Prepare and Commit ACKs. L1-only reconfiguration rebuilds only the per-Tile L1
Arena/cache/MSHR generation; it does not reset or invalidate a live parent L2
Arena. L2-only reconfiguration leaves the L1 mode/generation unchanged but
must first close all work that can touch L2.

The compiler also emits range-qualified `MemoryMaintenanceDesc` controls when
HBM writes conflict with cached readers. Their fixed sequence blocks only the
affected ranges where supported, drains references, clean/invalidates, waits
for real downstream writeback, acknowledges, and reopens issue. A target with
only full-domain maintenance requires the correspondingly quiescent compiled
boundary. Runtime does not invent maintenance or turn an ordinary await into a
global barrier.

Startup initializes every member to each layer's `reset_mode`, generation 0,
and consumes actual ACKs. A warm launch must match the artifact's compiled
entry Profile. `Simulator.run` neither resets nor replans a mismatch; explicit
quiescent recovery is required. Faulted or cancelled commands keep their gates
closed until emitted member/DMA work is isolated. Generations are monotonic,
so late returns cannot write a newly reused SPM/cache generation.

## Fidelity and byte-proof boundary

`SimConfig(fidelity=...)` selects memory timing depth:

| Fidelity      | Physical allocation/addressing                   | Transfer timing                         |
| ------------- | ------------------------------------------------ | --------------------------------------- |
| `timing_only` | logical contracts only                           | one collapsed leg                       |
| `runtime`     | real HBM bindings and L1/L2 Arena/view lifetimes | one collapsed leg                       |
| `full_memory` | real bindings/Arenas plus bank segments          | HBM/Global DMA/NoC/L2/L1/Local DMA legs |

All fidelities execute the same compiled resource, Profile, maintenance,
generation, and gate contracts. Only `full_memory` with an injected
`ByteStore` proves byte movement. `ByteStore` is sparse, rejects uninitialized
reads, validates host seed coverage against actual bindings, captures source
bytes when the read leg completes, and commits destination bytes only when the
write leg completes:

```python
from pipeline_validator.memory.byte_store import ByteStore

store = ByteStore()
store.seed_hbm(0x100000, bytes(range(64)))
simulator = Simulator(hw, sim, byte_store=store)
result = simulator.run(loaded)
copied = store.read_hbm(0x200000, 64)
```

For byte-checked profiled Gather, each request additionally needs
`bind_profiled_source(binding_id, request_id, source_offset)`. Cache test
seeding is available through `seed_cache_line(...)`. Without `ByteStore`,
Gather remains a deterministic source-authored `L1_HIT`/`L2_HIT`/`HBM_MISS`
timing profile. Cache capacity never predicts a hit rate, and BOA/EVU/USE do
not compute tensor values.

## Hardware YAML schema 2 and default experimental modes

`pipeline_validator/hardware_config.yaml` is schema 2. `memory.target` and
`memory.profile_source` are typed subtrees; unknown/duplicate keys are
rejected. A partial user YAML may omit `schema_version` and inherits omitted
subtrees from the bundled target. If a layer's `memory.target` is explicitly
present, `system_reserved_spm_per_bank` is mandatory. A custom layer `modes`
mapping replaces that whole table rather than merging individual modes.

The bundled `simulator_experiment` modes are per-bank bytes:

| Layer | Mode | SPM bytes | Cache bytes | System-reserved SPM |
| ----- | ---: | --------: | ----------: | ------------------: |
| L1    |    0 |    65,536 |           0 |               2,048 |
| L1    |    1 |    57,344 |       8,192 |               2,048 |
| L1    |    2 |    49,152 |      16,384 |               2,048 |
| L2    |    0 |   524,288 |           0 |               4,096 |
| L2    |    1 |   458,752 |      65,536 |               4,096 |
| L2    |    2 |   393,216 |     131,072 |               4,096 |

SPM includes the system reservation; user capacity is
`banks × (spm_bytes_per_bank - system_reserved_spm_per_bank)`. SPM + Cache
must exactly conserve each physical bank, and SPM/Cache/reserved bytes must be
aligned to `lcm(partition_granule_bytes, cache_line_bytes)`. The active Profile
is the sole Cache-capacity source; lookup latency and MSHR counts remain
independent target parameters.

Both layers use `reset_mode=0`, `spm_mapping_id="striped_arena_v0"`,
`cache_org_id="profiled_lru_v0"`, `cache_write_policy="read_only"`, and
`maintenance_caps=[invalidate_range, clean_invalidate_all, bypass]` in the
bundled target. Tag/MSHR/ACK/control storage is separate from user SPM; it is
not subtracted from the Profile data-region bytes.

The default target has 16 banks per Tile L1 (1 MiB physical) and 16 banks in
the Group L2 (8 MiB physical), reset mode 0, 64-byte partition granule, and a
2,000,000-cycle Profile command timeout. These values are simulator modelling
choices, `由后续规格冻结`; they are not RTL/PPA commitments.

## CLI: source compilation and independent replay

```bash
# Source convenience path: compile, persist, load, and run.
python -m pipeline_validator -w pow \
  --input-binding Y=0x100000:524288:rw

# Compile source only. Writes review.json plus sibling executable/source/target dumps.
python -m pipeline_validator --ir-file path/to/workload.mlir \
  --input-binding Y=0x100000:131072:rw \
  --compile-only --compiled-output artifacts/review.json

# Replay the artifact in an independent process; no source or compiler import is needed.
python -m pipeline_validator --compiled-file artifacts/review.json \
  --hw-config artifacts/review.target.yaml \
  --input-binding Y=0x100000:131072:rw

# Override one existing mode only while compiling source.
python -m pipeline_validator --ir-file path/to/workload.mlir \
  --profile-bytes l1:1=53248:12288 --compile-only

# Print author source IR only; this does not compile or run.
python -m pipeline_validator --ir-file path/to/workload.mlir --print-ir
```

Other current controls remain orthogonal to the source/artifact mode:

| Option                                           | Meaning                                                                     |
| ------------------------------------------------ | --------------------------------------------------------------------------- |
| `-l`, `--list`                                   | list built-in workloads                                                     |
| `-w NAME` / `-a`                                 | compile/run one or all built-ins (`-a` cannot use one explicit output path) |
| `--hw-config PATH` / `--hw-override KEY=VALUE`   | select or override `HardwareConfig`                                         |
| `--sim-override KEY=VALUE`                       | override `SimConfig`, including nested `device.*`/`group.*` capacities      |
| `--group-policy s0/s1/s2`                        | select the runtime Group ready-action policy; S2 currently matches S1       |
| `--context-mode N`                               | exact Tile UCE contexts per Tile                                            |
| `--device-context-mode N`                        | CPU outstanding Group-launch limit                                          |
| `--max-cycles N`                                 | execution cycle cap                                                         |
| `--json` / `--report PATH`                       | choose report encoding/destination                                          |
| `--detailed`                                     | include the six verbose detail sections in the text report                  |
| `--trace-json`, `--trace-html`, `--memory-trace` | select trace outputs/detail                                                 |

`--compile-only`, `--compiled-output`, and `--profile-bytes` are source-mode
options. `--compiled-file` is mutually exclusive with source input,
`--compile-only`, `--compiled-output`, `--profile-bytes`, and `--print-ir`.
Without `--compiled-output`, artifacts use
`examples/artifacts/compiled/<artifact_hash>.json`. Existing files are reused
only when their contents match; the CLI refuses to overwrite different
content.

For `review.json`, the siblings are:

- `review.exec.txt`: the complete strict executable/package dump;
- `review.compiled.mlir.txt`: inspectable generic MLIR with compiler-inserted
  ordinary awaits, Profile commands, maintenance controls, instruction IDs,
  and source references;
- `review.target.yaml`: a complete `HardwareConfig` schema-2 snapshot.

The target YAML does **not** snapshot `SimConfig`. Compiled replay must also use
the same static simulation capacities used at compile time:
`context_count`, `device_context_count`, Device pending/completion capacity,
and Group active/pending/action/quota/scan/event/inflight/prefetch/store/
dispatch capacities. Supply those again with `--context-mode`,
`--device-context-mode`, and matching `--sim-override` values.
Trace/fidelity/max-cycles/timing knobs and Group scheduling policy are not in
the static target fingerprint.

## Profiling, dumps, and observability

`--trace-json` emits Perfetto/Chrome trace JSON; `--trace-html` emits the
standalone viewer. `--memory-trace` additionally records per-leg memory
traffic, capacity counters, and report memory peaks.

`--detailed` only affects the text report: without it the text output omits
the verbose `Configured/effective resources`, `Profile controller`,
`Arena pools`, `Group scheduler`, `CPU device controller`, and `CPU request
timing` sections. Report content is unchanged by `--detailed` — the JSON
report retains the detail fields, and trace content still depends on which
trace options (`--trace-json`, `--memory-trace`, ...) you enable.

Profile initialization completes before workload cycle zero. `Simulator.run`
excludes the pre-run `profile_initialize` / `profile_initialized` markers and
`INITIALIZE` member request/ACK events from the workload trace, so their
separate clock cannot appear to overlap workload DMA. Cycle-zero capacity
baselines, prior workload records in a reused Tracer, and all in-run Profile
reconfiguration events are retained.
With unchanged bindings, warm runs reuse the existing HBM handles. Their
`hbm_bind` records and allocated/free-byte samples remain in the accumulated
trace, together with the original L1/L2 pool and bank capacity baselines;
unchanged state does not require a new bind or duplicate counter sample.

JSON `ts` is in microseconds and maps directly to the workload cycles in
reports and `accepted_cycle` / `completion_cycle` arguments, with no startup
offset: `ts = cycle * hw.cycle_ns() / 1000`. Initialization is likewise
excluded from report workload cycles.

UCE context `ACCEPT`/`READY`/wait states with positive duration are complete
slices; same-cycle transitions are instant markers, never zero-duration
begin/end pairs. On Task termination, the current state closes and
`DONE:<program>` or `FAULT:<program>` is an instant marker on the same lane.
Use Task/root completion records for retirement timing; a terminal marker
does not reserve a UCE interval.

Profile/control observability includes:

- `profile_command` and every `profile_step`;
- `profile_member_request`/`profile_member_ack` with command, member,
  transaction, generation, stage, status, binding, and `source_ref`;
- ordinary `nexus.await`/`nest.await` as separate instructions rather than
  hidden drains;
- requested/resolved L1 modes and L1 generation on Task lease events;
- requested/resolved L2 modes in compiled `call_bindings`, plus active L2 mode
  and generation in the Profile snapshot.

Memory lifetime observability includes `arena_reserve`, `arena_retire`,
`buffer_view_invalidate`, `l2_extent_release`, `task_lease_acquire`, and
`task_lease_release`. Pool snapshots and post-mutation counters include
backing/claim/reference counts, live bytes, reservation, padding,
system-reserved, Cache, and free bytes. For L2, `live_view_bytes` counts
logical valid bytes once per live backing; `logical_live_view_bytes` sums
logical live views and can count aliases. `physical_live_backing_bytes` counts
unique padded backing units and excludes slack, while `arena_reserved_bytes`
includes backing units plus held slack. Readonly aliases share one immutable
L2 backing; reports retain origin-retired backing and claim rows without
counting aliases as additional physical reservation. Protocol-live L2 byte
totals are deduplicated by backing identity. Reports expose bounded
`profile`, `arenas`, and `task_leases` snapshots plus `compiled_artifact_hash`
and `registry_hash`.

Every executable instruction has an `instruction_id` and `SourceRef`
(`source_name`, symbol, body-op index, op name). Compiler-generated controls
also record `generated_by` and a reason. xDSL source locations do not invent
file line numbers when none exist.

Transfer legs are emitted at completion using real accept/complete cycles;
cancelled predictions are not fabricated. Chrome `s`/`t`/`f` flow events join
the legs of a transaction. Counters are change-only. In collapsed
`timing_only`/`runtime` modes a transaction has one leg and the flow reduces to
an `s`+`f` pair.

## Workloads and reports

Built-in workloads and `--ir-file` source follow the same compile/persist/
load/run boundary. `WorkloadInfo` is embedded in the artifact, so
`--compiled-file` reporting does not rebuild a source `ModuleOp`.

Reports retain engine/PMU fingerprints and profiled Gather request counters.
The label `deterministic_profiled_not_address_or_value_accurate` means the hit/
miss sequence comes from `tile.profiled.access`; it is not a measured cache hit
rate. Byte equality claims require the explicit `full_memory` + `ByteStore`
path described above.

## Running the tests

```bash
conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -v
```

The suite runs in parallel by default: `pytest.ini` at the repository root
sets `addopts = -n auto` (`pytest-xdist`, one worker process per CPU). Pass
`-n0` for a serial run or `-n N` to cap workers. Tests write only to their
own pytest `tmp_path`, so parallel workers are isolated.

## Files

```text
pipeline_validator/
├── compiler/
│   ├── __init__.py          # public compile_program only
│   ├── api.py               # compile pipeline and inspectable compiled-source dump
│   ├── lowering.py          # private source → immutable execution DTO lowering
│   ├── resources.py         # striped layouts, capabilities, static budgets
│   └── profile_pass.py      # Profile binding, awaits, maintenance/control
├── compiled_program.py      # immutable package, strict codec, executable dump
├── execution_verifier.py    # read-only package semantic verification
├── loader.py                # CompiledProgram + actual bindings → LoadedProgram
├── profiles.py              # Registry, contracts, layouts, control descriptors
├── config.py                # HardwareConfig / SimConfig and schema-2 YAML
├── hardware_config.yaml     # bundled target/Profile defaults
├── dialects/elenor.py       # xDSL custom-assembly dialect
├── workload_ir.py           # source parse / print / verify
├── execution_ir.py          # frozen compiler/Loader/runtime DTOs
├── memory/
│   ├── arena.py             # owner-scoped striped Arena pools and views
│   ├── profile_controller.py # unique writer, ACK protocol, maintenance/recovery
│   ├── byte_store.py        # sparse optional full-memory byte oracle
│   ├── cache.py             # profiled Cache and maintenance
│   └── transfer.py          # generation-aware transfer/cancel isolation
├── runtime/group_port.py    # bounded root/control message adapter
├── group_scheduler.py       # finite ready-action scheduler
├── tile_group.py            # root/Grid admission and retirement
├── tile.py                  # per-Tile Task admission/UCE
├── simulator.py             # LoadedProgram-only cycle driver
├── report.py                # report from immutable metadata + real snapshots
└── cli.py                   # source compile and compiled replay entry point
```
