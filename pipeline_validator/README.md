# ELENOR Runtime Pipeline Validator

A **cycle-accurate functional simulator** of one ELENOR Tile Group
(1 Region Sequencer + 4 Compute Tiles), built to validate the runtime
pipeline efficiency described in the `design/` architecture specs.

It models the full `Graph → Region → Tile → Engine` control flow, the
Stream Queue producer-consumer pipeline, and the BOA / EVU / MFE / USE
engine partition, then reports a PMU fingerprint with pass/fail checks
against the architecture's predicted bottlenecks.

## Scope

| Aspect           | Modelled                                                   | Source spec                                 |
| ---------------- | ---------------------------------------------------------- | ------------------------------------------- |
| Tile Group       | 1 group, 4 tiles                                           | `design/elenor_tile_group/`                 |
| Region Sequencer | Region Program ISA, stage dispatch, DMA prefetch, barriers | `design/elenor_region_sequencer/`, arch §16 |
| Compute Tile     | UCE + BOA/EVU/MFE/USE + L1 SRAM bandwidth                  | `design/elenor_compute_tile/`               |
| Tile UCE         | Tile Program ISA: launch/wait/stream/branch                | arch §16.4, §17.6                           |
| Stream Queue     | credit invariant, backpressure, EOS, reset/drain, PMU      | `design/elenor_stream_queue/`               |
| BOA              | 4×OPA (16×16) MAC throughput, bandwidth ceiling            | `design/elenor_boa/`                        |
| EVU              | 32-lane vector FMA throughput                              | `design/elenor_evu/`                        |
| MFE              | bandwidth-bound stream shaping                             | `design/elenor_mfe/`                        |
| USE              | slower-clock state engine                                  | `design/elenor_use/`                        |
| PMU              | unique stall attribution (one primary owner per cycle)     | arch §21.6                                  |

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
# run the default matmul workload
python -m pipeline_validator -w matmul

# run all workloads and write a report
python -m pipeline_validator --all --report report.txt

# list workloads
python -m pipeline_validator -l

# override a hardware param (e.g. faster clock)
python -m pipeline_validator -w attention --hw-override clock_mhz=2000

# JSON output
python -m pipeline_validator --all --json
```

### Profiling / Trace Visualization

The validator can emit **Perfetto / Chrome `chrome://tracing`-compatible**
trace files for visual Gantt-chart inspection of every engine job, stream
queue occupancy, and region/tile lifecycle event.

```bash
# write a Perfetto-loadable trace.json (load at perfetto.dev or chrome://tracing)
python -m pipeline_validator -w paged_attention --trace-json trace.json

# write a standalone trace.html (open in any browser, no server needed)
python -m pipeline_validator -w paged_attention --trace-html trace.html

# both at once, for all workloads (auto-suffixes _<workload>.json/.html)
python -m pipeline_validator --all --trace-json all.json --trace-html all.html
```

**Trace contents:**

- **Slices** (Gantt bars): every BOA/EVU/MFE/USE engine job with op name,
  ops/bytes, event_id, tile_id. Each tile gets its own track
  (Tile0/Tile1/Tile2/Tile3) with sub-tracks per engine (BOA/EVU/MFE/USE).
  TileGroup runtime windows: `TileGroup/Region` (region begin→end),
  `TileGroup/Stage` (dispatch→stage complete), `TileGroup/Global DMA`
  (HBM↔L2 prefetch/store), `TileGroup/Collective` (reduce/broadcast).
  Tile L2↔L1 traffic is MFE load/store on each tile track.
- **Counters** (line graphs): Stream Queue occupancy and credit_available
  sampled per cycle — only for workloads with streams (attention, moe).
- **Instant markers**: `tile_done`, `stage_dispatch`, `stage_complete`,
  `region_done`, `dma_complete`, `collective_complete`.

To view:

- **`trace.json`**: drag into [Perfetto UI](https://ui.perfetto.dev) or
  open `chrome://tracing` in Chrome/Edge and "Load file".
- **`trace.html`**: open directly in any browser — a self-contained
  Gantt chart with hover tooltips.

## Tests

```bash
python -m pytest pipeline_validator/tests/ -v
```

## Workloads

| Workload          | Stages                                  | Validates                                                                      |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------------------ |
| `matmul`          | single (4 tiles)                        | BOA peak compute + MFE/DMA load overlap                                        |
| `tiled_matmul`    | single (4 tiles)                        | K-dimension tiling + double-buffer MFE/BOA pipeline overlap                    |
| `conv_relu`       | single (4 tiles)                        | fused BOA conv (im2col) + EVU relu epilogue                                    |
| `paged_attention` | single (4 tiles)                        | full paged-attention pipeline: MFE page-stream → BOA QK → EVU softmax → BOA PV |
| `attention`       | 2 (QK → softmax+AV via Stream Queue S0) | stream pipeline, credit backpressure, BOA/EVU overlap                          |
| `moe`             | 2 (MFE segment-stream → BOA expert MLP) | MFE segment stream + BOA batch utilization                                     |

Each workload declares an **expected PMU fingerprint** (e.g. BOA-bound,
low stream stall). The report checks the measured fingerprint against
these expectations and prints `PASS` / `FAIL`.

## Files

```
pipeline_validator/
├── __init__.py          # public API
├── config.py            # HardwareConfig / WorkloadConfig / SimConfig
├── ir.py                # Tile/Region Program IR + builders (mirrors arch §16-17)
├── stream_queue.py      # StreamQueue (credit, backpressure, EOS, PMU)
├── engines.py           # BOA/EVU/MFE/USE timing models
├── pmu.py               # PMU counters + unique stall attribution
├── tile.py              # ComputeTile + TileUCE controller
├── region.py            # RegionSequencer controller
├── tile_group.py        # TileGroup (region seq + 4 tiles + streams)
├── simulator.py         # cycle-accurate driver
├── workloads.py         # Matmul / Attention / MoE workloads
├── report.py            # PMU fingerprint + pass/fail checks
├── cli.py               # CLI entry point
├── tests/__init__.py    # pytest suite
└── environment.yml      # conda env spec
```
