# ELENOR Runtime Pipeline Validator

A **cycle-accurate functional simulator** of one ELENOR Tile Group
(1 Tile Group Sequencer + 4 Compute Tiles), built to validate the runtime
pipeline efficiency described in the `design/` architecture specs.

It models the full `Graph → Group Task → Tile-SPMD Tile Program roles → Engine`
control flow, the Stream Queue producer-consumer pipeline, and the
BOA / EVU / MFE / USE engine partition, then reports a PMU fingerprint
with pass/fail checks against the architecture's predicted bottlenecks.

## IR Dialect

The validator uses a **function-call style** xDSL dialect (see
`reference.mlir`). The module contains top-level named definitions:
`nest.context @name { ... }` and `tile.program @name { ... }`. The
context body dispatches tile programs by symbol reference
(`@prog_name`), producing SSA event values typed
`!nest.event<tag>` / `!tile.event<tag>`.

### Prefix hierarchy

| Prefix    | Level        | Examples                                                                                   |
| --------- | ------------ | ------------------------------------------------------------------------------------------ |
| `tile.*`  | Tile program | `tile.program`, `tile.load.async`, `tile.gather.global.async`, `tile.await`, `tile.signal` |
| `nest.*`  | Tile group   | `nest.context`, `nest.dispatch.tasks.async`, `nest.dma.prefetch.async`, `nest.await`       |
| `nexus.*` | Host / CPU   | `nexus.program`, `nexus.submit_context.async`, `nexus.await`, `nexus.return`               |

### Key IR design points (per `reference.mlir`)

**Placement** — `nest.context @name placement = M` declares the Tile
Group placement mask (`0xF` = all 4 placement slots in the group).
This is a **group-level** constraint: the CPU/IR does NOT specify
physical Tile IDs or Hardware Context IDs. The tile-local scheduler
maps logical tasks to physical tiles/contexts at runtime (reference.mlir
§27-33, §188-189). In this validator the mapping is 1:1 (logical task
i → tile i), so `placement = 0xF` with `task.range 0..4` dispatches
4 tasks across 4 tiles.

**Dispatch** — `nest.dispatch.tasks.async @prog` consumes a logical
task range, mandatory `globals(...)`, ins/outs L2 buffers, and an
optional `depends_on` event. `globals()` is always printed even when
empty. It returns THREE aggregated events: `grid_done` (all tasks
returned), `input_released` (all tasks completed their L2 read phase),
and `output_ready` (all tasks completed their L2 write phase). Its
always-printed `signal_policy { ... }` block declares each emitted phase
with `#nest.aggregate<all_tasks>`.

**Tile phase signals** — `tile.signal input_released(%task)` /
`tile.signal output_ready(%task)` drive per-grid phase aggregation; the
operand is tile-program block argument 0 (`!nest.task`). Each emission
is keyed by `(context launch generation, grid instance, phase, logical
task)`, and a phase result fires exactly once only after every expected
logical task in that grid has signalled. Physical tile masks and UCE
hardware-context ids are not aggregation identities.

**Buffers** — `nest.alloc` produces SSA values typed
`!nest.l2_buffer<slot>`; the slot name is the runtime L2 buffer id.
At context admission, every `l2_buffers` entry is planned and committed
as one atomic bundle on the L2 `BankedFreeExtentAllocator` (owner,
generation, alignment, bank segments). `input_released` aggregation
unpins only role=`"in"` consumer pins. `output_ready` makes output
visible in L2 but does not unpin role=`"out"`/`"inout"` consumers; their
`nest.release` is gated by the final store after all required
`output_ready` events. Release additionally validates its explicit event,
owner, generation, live handle, and pin state before reclaiming the
buffer. The event type tag doubles as the runtime event id shared by the
simulator and the trace.

**L2 admission wait (PR 3.5)** — a submit is accepted onto a device slot
even when its atomic L2 bundle transiently cannot fit: the context
enters `ADMISSION_WAIT` (slot reserved, no UCE/L1/L2/stream/DMA
resource held, no fault). Only a `nest.release` allocator final-free
wakes the strict-FIFO wait queue; the head is admitted in the same
cycle and issues its first group action the next cycle, so a later
context can overlap an earlier context's compute/store. Invalid or
empty-pool-impossible bundles fault immediately and never queue. The
submit result event still means full context completion.

**`depends_on(%e)`** expresses data dependencies directly on async ops
(dispatch, store, release), lowered to wait actions by the lowering.

### Print / parse

The IR is printed and parsed in **custom-assembly format** (not generic
xDSL). `--print-ir` outputs the custom assembly; `--ir-file` loads
custom-assembly IR from disk.

## Scope

| Aspect               | Modelled                                                             | Source spec                                     |
| -------------------- | -------------------------------------------------------------------- | ----------------------------------------------- |
| Tile Group           | 1 group, 4 tiles                                                     | `design/elenor_tile_group/`                     |
| Tile Group Sequencer | Group Task actions, role dispatch, DMA prefetch, barriers            | `design/elenor_tile_group_sequencer/`, arch §16 |
| Compute Tile         | UCE + BOA/EVU/MFE/USE + L1 SRAM bandwidth                            | `design/elenor_compute_tile/`                   |
| Tile UCE             | Tile Program ISA: launch/wait/signal                                 | arch §16.4, §17.6                               |
| Stream Queue         | credit invariant, backpressure, EOS, reset/drain, PMU                | `design/elenor_stream_queue/`                   |
| BOA                  | 4×OPA (16×16) MAC throughput, bandwidth ceiling                      | `design/elenor_boa/`                            |
| EVU                  | 32-lane vector FMA throughput                                        | `design/elenor_evu/`                            |
| MFE                  | load/store plus deterministic profiled Gather through Cache/MSHR/HBM | `design/elenor_mfe/`                            |
| USE                  | slower-clock state engine                                            | `design/elenor_use/`                            |
| PMU                  | unique stall attribution (one primary owner per cycle)               | arch §21.6                                      |

## Fidelity modes

`SimConfig(fidelity=...)` selects the memory model depth (default
`full_memory`):

| Fidelity      | Handles / addresses                 | Latency model                         |
| ------------- | ----------------------------------- | ------------------------------------- |
| `timing_only` | none (src/dst views are null)       | one collapsed bandwidth+launch leg    |
| `runtime`     | real L1/L2 allocation + HBM binding | one collapsed bandwidth+launch leg    |
| `full_memory` | real L1/L2 allocation + HBM binding | full per-leg route (HBM/NoC/DMA/bank) |

Gather never collapses to a synthetic single latency. In all fidelity
modes it keeps the same profiled
`L1 lookup → L2 lookup/MSHR → optional HBM/NoC refill → L1 fill →
ordered destination write` state machine; `timing_only` simply leaves
physical views null. Cache quota is metadata-only and does not consume
live L1/L2 scratchpad extents.

All three modes enforce the global-binding contract (missing / overlapping /
out-of-capacity / wrong-permission bindings fail at load time). In
`runtime`/`full_memory`, every allocation is an immutable
`AllocationHandle` with owner, generation and bank segments; capacity,
alignment, owner, generation and use-after-release errors raise
`MemoryInvariantError` and never produce a success event.

`full_memory` advances regular transfers leg-by-leg: prefetch walks
`HBM_READ → GLOBAL_DMA → NOC_RESPONSE → L2_WRITE`, tile load walks
`L2_READ → LOCAL_DMA → L1_WRITE`, with per-bank segment issuance.
Gather uses dedicated lookup/fill stages and reuses real HBM outstanding,
NoC credit, local DMA, and destination L1-bank resources. Snapshot
verification is available under
`memory.cache/mshr/transfers` plus `memory.hbm/l2/l1/noc`.

Hardware defaults follow the **Balanced-small** profile (arch §12.3):
64 tiles / 1 MB L1 per tile / 8 MB Group SRAM. The validator runs a
4-tile slice of that group.

## Setup (conda)

```bash
conda env create -f pipeline_validator/environment.yml
conda activate elenor-validator
```

## Run

````bash
# run the default pow workload (bind its global input Y)
python -m pipeline_validator -w pow --input-binding Y=0x100000:524288:rw

# list workloads
python -m pipeline_validator -l

# override a hardware param (e.g. faster clock)
python -m pipeline_validator -w pow --input-binding Y=0x100000:524288:rw \
  --hw-override clock_mhz=2000

# print the IR (custom assembly) without simulating
python -m pipeline_validator -w pow --print-ir

# load and run an external IR file (model IR that declares global inputs
# requires one --input-binding NAME=BASE:SIZE:PERM per input)
python -m pipeline_validator --ir-file path/to/workload.mlir --input-binding Y=0x100000:131072:rw

# JSON output
python -m pipeline_validator -w pow --input-binding Y=0x100000:524288:rw --json

# run with dual tile-UCE contexts and emit an HTML trace
python -m pipeline_validator -w pow --input-binding Y=0x100000:524288:rw \
  --context-mode 2 --trace-html ctx2.html

# organized examples include their required bindings and overrides
bash examples/run.sh list
bash examples/run.sh pow-dual-context
bash examples/run.sh gather --json

# edit/copy a custom IR and provide its bindings explicitly
bash examples/run.sh file path/to/workload.mlir \
  --input-binding Y=0x100000:131072:rw

# full example catalog and modification workflow: examples/README.md

### Profiling / Trace Visualization

The validator can emit **Perfetto / Chrome `chrome://tracing`-compatible**
trace files for visual Gantt-chart + counter inspection. By default,
`--trace-json` / `--trace-html` only enable the lightweight control-flow
trace (engine jobs, task/tile lifecycle, stream-queue counters). Pass
`--memory-trace` to add the PR 5 memory-detail lanes/counters/flows and
to populate `WorkloadReport.memory`.

```bash
# write a Perfetto-loadable control-flow trace.json
python -m pipeline_validator -w pow --input-binding Y=0x100000:524288:rw \
  --trace-json trace.json

# add PR 5 memory-detail lanes/counters/flows + report.memory
python -m pipeline_validator -w pow --input-binding Y=0x100000:524288:rw \
  --memory-trace --trace-json trace-memory.json --trace-html trace-memory.html
```

**Lane hierarchy** (PR 5): control-flow lanes are always present once
tracing is enabled. The memory-detail lanes/counters/flows below require
`--memory-trace`. Directional group-transfer summaries are always present;
their clickable `flow_id` handoff to per-leg slices only appears with
`--memory-trace`. Concurrent summaries take the first free visual slot in
their direction; `#{n}` identifies a display slot, not a hardware channel.
Within each `Tile{n}` process, the dataflow region has a fixed visual order:
`MFE_LD{i}` → `BOA` / `EVU` / `MFE` / `USE` → `MFE_ST{j}`.

| Process | Thread lane | Contents |
|---|---|---|
| Device | `Slot:{i}` | `context_submit` → `context_done` windows |
| TileGroup | `Task` | group task begin→end |
| TileGroup | `TileRole:{role_id}` | role dispatch→complete window |
| TileGroup | `Scheduler:L2` | admission wait/retry/first-action instants, `phase_aggregate`, `transfer_cancelled` |
| TileGroup | `HBM → L2 Input #{n}` | Prefetch summaries; bars are named `context / buffer` |
| TileGroup | `L2 → HBM Output #{n}` | Storeback summaries; bars are named `context / buffer` |
| TileGroup | `Memory:HBM` | `hbm_bind`/`hbm_unbind` instants, `hbm_allocated_bytes`, `hbm_free_bytes`, `hbm_outstanding`, `hbm_credits` |
| TileGroup | `Memory:L2 Read` | `l2_read`, `l2_cache_lookup` leg slices |
| TileGroup | `Memory:L2 Write` | `l2_write`, `l2_cache_fill` leg slices |
| TileGroup | `Memory:L2 State` | L2 capacity/cache/MSHR counters, `{l2,hbm}_alloc`/`{l2,hbm}_release` instants |
| TileGroup | `L2 Bank:{n}` | `l2_bank_allocated_bytes` |
| TileGroup | `Global DMA Ch:{n}` | per-channel leg slices + `busy` counter |
| TileGroup | `HBM Ch:{n}` | per-channel leg slices + `busy` counter |
| TileGroup | `NoC:VC{n}` | `noc_occupancy`, `noc_credit_available`, NoC leg slices |
| TileGroup | `StreamQ:{id}` | `occupancy`, `credit_available` |
| Tile{n} | `UCE CTX{i}` | UCE state slices (`ACCEPT`/`READY`/`WAIT_*`/`DONE`) |
| Tile{n} | `UCE` | `active_context_count`, `ready_context_count` |
| Tile{n} | `BOA`/`EVU`/`MFE`/`USE` | engine job slices |
| Tile{n} | `MFE_LD{i}` / `MFE_ST{j}` | MFE lane slices |
| Tile{n} | `Local DMA Load`/`Store` | tile-local DMA leg slices |
| Tile{n} | `Memory:L1 Read` | `l1_read`, `l1_cache_lookup` leg slices |
| Tile{n} | `Memory:L1 Write` | `l1_write`, `l1_cache_fill` leg slices |
| Tile{n} | `Memory:L1 State` | L1 capacity/cache counters, `l1_alloc`/`l1_release` instants |
| Tile{n} | `L1 Bank:{n}` | `l1_bank_allocated_bytes` |
| Tile{n} | `MSHR` | `l1_mshr_*` counters |
| Tile{n} | `Lifecycle` | `frame_prepare`/`frame_bind`/`frame_release` instants |

**Counter directory** (all change-only — a sample is emitted only when the
value changes, not every cycle):

- HBM: `hbm_allocated_bytes`, `hbm_free_bytes`, `hbm_outstanding`, `hbm_credits`.
- L2: `l2_allocated_bytes`, `l2_free_bytes`, `l2_largest_free_extent`,
  `l2_live_allocations`, `l2_pending_release`, `l2_bank_allocated_bytes`,
  `l2_cache_resident_bytes`, `l2_cache_resident_lines`, `l2_mshr_*`.
- L1 (per tile): the `l1_*` mirror of the L2 set above.
- NoC: `noc_occupancy`, `noc_credit_available` per VC.
- Channels: `busy` (0/1) per Global DMA / HBM channel.
- Stream queues: `occupancy`, `credit_available` per queue.
- UCE: `active_context_count`, `ready_context_count` per tile.

**Flows**: every memory transaction is a Chrome flow. Each transfer leg
emits an `X` slice named after its `TransferLegKind` (e.g. `hbm_read`,
`noc_response`, `l2_read`, `local_dma`); the first completed leg opens
the flow (`s`), intermediate legs step it (`t`), and the last leg closes
it (`f`). Each directional `context / buffer` summary carries the same
`flow_id`, so clicking it in the HTML viewer highlights all legs of that
transaction across lanes and draws connecting arrows.

**Sampling fidelity**: leg slices are emitted when a leg *completes*
(using its real accept/complete cycles), so cancelled legs never leave a
fabricated prediction. Counters are change-only. Collapsed-leg fidelities
(`timing_only`, `runtime`) produce one leg per transaction, so a flow
degrades to a single `s`+`f` pair. `Tracer.assert_well_formed()` (called
by the test suite) enforces: metadata coverage, no consecutive duplicate
counter samples, closed B/E pairs, one `s`/one `f` per flow, and the
required identity args on every leg slice.

**Report reconciliation**: with `--memory-trace`, `WorkloadReport.memory`
exposes `l2_peak_allocated_bytes`, `l1_peak_allocated_bytes` (per tile),
`hbm_outstanding_peak`, and `hbm_used_bytes` read from the group snapshot
(never reconstructed from the trace); tests compare these against the
trace counter maxima. Without `--memory-trace`, the report's `memory`
field is an empty dict.

## Tests

```bash
python -m pytest pipeline_validator/tests/ -v
```

## Workloads

| Workload | Roles            | Validates                                                  |
| -------- | ---------------- | ---------------------------------------------------------- |
| `pow`    | single (4 tiles) | EVU elementwise pow + MFE load/store + pipelined group DMA |

Each workload declares an **expected PMU fingerprint** (e.g. EVU-active,
low stream stall). The report checks the measured fingerprint against
these expectations and prints `PASS` / `FAIL`.

Profiled Gather reports `gather_requests`, `gather_l1_hits`,
`gather_l2_hits`, `gather_hbm_misses`, `gather_mshr_merges`,
`gather_mshr_stalls`, `gather_reorder_wait_cycles`, and `gather_bytes`.
Reports automatically check request conservation and zero MSHR/transfer/
allocation leakage. `gather_fidelity` is explicitly
`deterministic_profiled_not_address_or_value_accurate`; these counters are
not measured cache hit rates from real indices.

## Files

```
pipeline_validator/
├── __init__.py          # public API
├── config.py            # HardwareConfig / WorkloadConfig / SimConfig
├── hardware_config.yaml # HardwareConfig 默认值（分组 YAML 单一事实来源）
├── dialects/            # xDSL `elenor` dialect (function-call style: tile.*/nest.* ops)
├── workload_ir.py       # parse / print / verify / load custom-assembly IR
├── workload_builders.py # direct xDSL workload builders (pow + identity)
├── execution_ir.py      # private execution DTOs for the hot path
├── ir_lowering.py       # xDSL -> execution DTO lowering (1:1 walk of IR body)
├── stream_queue.py      # StreamQueue (credit, backpressure, EOS, PMU)
├── engines.py           # BOA/EVU/MFE/USE timing models
├── pmu.py               # PMU counters + unique stall attribution
├── tile.py              # ComputeTile + TileUCE controller
├── tile_group_sequencer.py  # TileGroupSequencer controller
├── tile_group.py        # TileGroup (sequencer + 4 tiles + phase aggregation)
├── simulator.py         # cycle-accurate driver
├── workloads.py         # PowWorkload
├── report.py            # PMU fingerprint + pass/fail checks
├── cli.py               # CLI entry point
├── tests/               # pytest suite
└── environment.yml      # conda env spec
```
````
