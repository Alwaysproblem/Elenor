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

| Prefix    | Level        | Examples                                                                             |
| --------- | ------------ | ------------------------------------------------------------------------------------ |
| `tile.*`  | Tile program | `tile.program`, `tile.load.async`, `tile.pow.async`, `tile.await`, `tile.signal`     |
| `nest.*`  | Tile group   | `nest.context`, `nest.dispatch.tasks.async`, `nest.dma.prefetch.async`, `nest.await` |
| `nexus.*` | Host / CPU   | (deferred — not yet implemented)                                                     |

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
task range, ins/outs L2 buffers, and a `depends_on` event. It returns
THREE aggregated events: `grid_done` (all tasks returned),
`input_released` (all tasks completed their L2 read phase), and
`output_ready` (all tasks completed their L2 write phase).

**Tile phase signals** — `tile.signal input_released` /
`tile.signal output_ready` drive the dispatch phase events: when every
dispatched task signals a phase, the corresponding dispatch result
fires. The phase event is aggregated across the placement mask, not
across logical task IDs — it fires when every physical tile in the
placement has signalled.

**Buffers** — `nest.alloc` produces SSA values typed
`!nest.l2_buffer<slot>`; the slot name is the L2 slot id used by the
group DMA latency model. `nest.release` frees the slot (in
`full_memory` fidelity). The event type tag doubles as the runtime
event id shared by the simulator and the trace.

**`depends_on(%e)`** expresses data dependencies directly on async ops
(dispatch, store, release), lowered to wait actions by the lowering.

### Print / parse

The IR is printed and parsed in **custom-assembly format** (not generic
xDSL). `--print-ir` outputs the custom assembly; `--ir-file` loads
custom-assembly IR from disk.

## Scope

| Aspect               | Modelled                                                  | Source spec                                     |
| -------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| Tile Group           | 1 group, 4 tiles                                          | `design/elenor_tile_group/`                     |
| Tile Group Sequencer | Group Task actions, role dispatch, DMA prefetch, barriers | `design/elenor_tile_group_sequencer/`, arch §16 |
| Compute Tile         | UCE + BOA/EVU/MFE/USE + L1 SRAM bandwidth                 | `design/elenor_compute_tile/`                   |
| Tile UCE             | Tile Program ISA: launch/wait/signal                      | arch §16.4, §17.6                               |
| Stream Queue         | credit invariant, backpressure, EOS, reset/drain, PMU     | `design/elenor_stream_queue/`                   |
| BOA                  | 4×OPA (16×16) MAC throughput, bandwidth ceiling           | `design/elenor_boa/`                            |
| EVU                  | 32-lane vector FMA throughput                             | `design/elenor_evu/`                            |
| MFE                  | bandwidth-bound stream shaping                            | `design/elenor_mfe/`                            |
| USE                  | slower-clock state engine                                 | `design/elenor_use/`                            |
| PMU                  | unique stall attribution (one primary owner per cycle)    | arch §21.6                                      |

Hardware defaults follow the **Balanced-small** profile (arch §12.3):
64 tiles / 1 MB L1 per tile / 8 MB Group SRAM. The validator runs a
4-tile slice of that group.

## Setup (conda)

```bash
conda env create -f pipeline_validator/environment.yml
conda activate elenor-validator
```

## Run

```bash
# run the default pow workload
python -m pipeline_validator -w pow

# list workloads
python -m pipeline_validator -l

# override a hardware param (e.g. faster clock)
python -m pipeline_validator -w pow --hw-override clock_mhz=2000

# print the IR (custom assembly) without simulating
python -m pipeline_validator -w pow --print-ir

# load and run an external IR file
python -m pipeline_validator --ir-file path/to/workload.mlir

# JSON output
python -m pipeline_validator -w pow --json

# run with dual tile-UCE contexts and emit an HTML trace
python -m pipeline_validator -w pow --context-mode 2 --trace-html ctx2.html
```

### Profiling / Trace Visualization

The validator can emit **Perfetto / Chrome `chrome://tracing`-compatible**
trace files for visual Gantt-chart inspection of every engine job, stream
queue occupancy, and task/tile lifecycle event.

```bash
# write a Perfetto-loadable trace.json (load at perfetto.dev or chrome://tracing)
python -m pipeline_validator -w pow --trace-json trace.json

# write a standalone trace.html (open in any browser, no server needed)
python -m pipeline_validator -w pow --trace-html trace.html
```

**Trace contents:**

- **Slices** (Gantt bars): every BOA/EVU/MFE/USE engine job with op name,
  ops/bytes, event_id, tile_id. Each tile gets its own track
  (Tile0/Tile1/Tile2/Tile3) with sub-tracks per engine (BOA/EVU/MFE/USE).
  TileGroup runtime windows: `TileGroup/Task` (task begin→end),
  `TileGroup/TileRole` (role dispatch→complete), `TileGroup/Global DMA`
  (HBM↔L2 prefetch/store), `TileGroup/Collective` (reduce/broadcast).
  Tile L2↔L1 traffic is MFE load/store on each tile track.
- **Instant markers**: `tile_done`, `tile_signal`, `tile_role_dispatch`,
  `tile_role_complete`, `group_task_done`, `dma_complete`, `collective_complete`.
- **Multi-context trace details**: with `--context-mode 2`, tile tracks expose
  `UCE CTX0` / `UCE CTX1` lanes, `ctx_switch` instants, and
  `active_context_count` / `ready_context_count` counters.

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

## Files

```
pipeline_validator/
├── __init__.py          # public API
├── config.py            # HardwareConfig / WorkloadConfig / SimConfig
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
