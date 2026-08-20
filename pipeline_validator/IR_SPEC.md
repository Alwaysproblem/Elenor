# ELENOR Pipeline Validator — IR Specification

This document specifies the function-call style xDSL dialect used by the
pipeline validator. The design follows `reference.mlir` at
`/reference.mlir`.

## 1. Module Structure

A valid module is one of two shapes:

**Legacy** (exactly one `nest.context`, no `nexus.program`):

```mlir
builtin.module {
  tile.program @pow_4k_tile { ... }    // tile program definition
  nest.context @pow_task placement = 15 { ... }  // tile group context
}
```

**Model** (exactly one `nexus.program` + one or more `nest.context`):

```mlir
builtin.module {
  tile.program @pow_4k_tile { ... }
  nest.context @pow_task placement = 15 { ... }
  nexus.program @run_pow (%Y : !nest.global_memref<4x128x128xbf16>) { ... }
}
```

### 1.1 `nest.context @name placement = M context = N { ... }`

Defines one Tile Group context. The `placement` property is the Tile Group
placement mask (integer bitmask). This is a **group-level** constraint: the
CPU/IR does NOT specify physical Tile IDs or Hardware Context IDs, except
that `nest.context` and `nest.dispatch.tasks.async` may carry an optional
`context = N` (see §3.5 for dispatch semantics).

- **Semantics**: The placement mask selects which placement slots in the
  Tile Group participate in dispatches. The tile-local scheduler maps
  logical tasks to physical tiles/contexts at runtime (reference.mlir
  §27-33, §188-189).
- **Device slot pin**: `nest.context` may carry `context = N` to pin
  this context to **device execution slot N** when submitted via
  `nexus.submit_context.async` (mirrors the UCE context pin of
  `nest.dispatch.tasks.async` one level up). Omitted = first available
  slot; occupied slot = submission waits (backpressure, PMU
  `device_submit_wait`). Legal range `0..device_context_count-1`;
  out-of-range rejected at model/task load. In a legacy single-context
  module the pin selects the (only) slot and must be 0.
- **Validator mapping**: In this validator the mapping is 1:1 (logical
  task i → tile i), so `placement = 0xF` (4 bits set) with `task.range
0..4` dispatches 4 tasks across 4 tiles.
- **Verifier**: placement must be non-zero; `context = N` (if present)
  must be >= 0 and < `device_context_count` (upper bound checked at
  model/task load).

### 1.2 `tile.program @name { ... }`

Defines one tile program. The body contains tile-level async engine ops,
`tile.await`, `tile.signal`, and `tile.return`. The program is referenced
by `nest.dispatch.tasks.async` via its symbol name.

### 1.3 `nexus.program @name (%a : !nest.global_memref<...>) { ... }`

Model entry point. A model-mode module contains exactly one
`nexus.program`. The body is a linear device-level program consisting of
`nexus.submit_context.async`, `nexus.await`, and `nexus.return` (ending
with return). Entry block args are declarative (V1: parsed/printed only;
body ops must not reference them — no memref plumbing yet).

**Device slot scheduling**: The device has `device_context_count`
execution slots that all share the **same physical TileGroup** (the
base `num_tiles` tiles are shared, not duplicated). When
`nexus.submit_context.async @ctx` is reached, the context is assigned to
a slot: if `@ctx`'s `nest.context` carries `context = N`, it is pinned to
slot N (must be free); otherwise the first free slot is used. If no slot
is free, the submission waits (backpressure, PMU `device_submit_wait`).
Each slot runs its task concurrently on the shared tiles via UCE
context switching: unpinned dispatch bindings auto-assign UCE context
`slot_index`, requiring `context_count >= device_context_count`. When
the context finishes, the slot is released and a `!nexus.event` fires;
`nexus.await` blocks the device PC until the awaited event has fired.

## 2. SSA Types

### 2.1 `!nest.event<tag>`

Group-level async event. The `tag` (a string literal) is the runtime
event id used by the simulator and the trace. Produced by:
`nest.dma.prefetch.async`, `nest.dma.store.async`, `nest.dispatch.tasks.async`,
`nest.collective.async`.

### 2.2 `!tile.event<tag>`

Tile-level async event. Same semantics as `!nest.event` but scoped to a
single tile. Produced by: `tile.load.async`, `tile.store.async`,
`tile.pow.async`, `tile.evu.async`, `tile.boa.async`.

### 2.3 `!nest.l2_buffer<slot>`

Context-owned L2 buffer. The `slot` is the L2 slot id used by the group
DMA latency model (capacity allocation in `full_memory` fidelity).
Produced by `nest.alloc`.

### 2.4 `!nest.task_range`

Logical task domain. Produced by `nest.task.range`. Task IDs are logical
IDs, NOT physical Tile IDs or Hardware Context IDs (reference.mlir §170-171).

### 2.5 `!nexus.event<"tag">`

Device-level async event. The `tag` is the runtime event id shared by
the device scheduler and the trace. Produced by
`nexus.submit_context.async`; consumed by `nexus.await`.

### 2.6 `!nest.global_memref<DxDx...xdtype>`

Declarative global memref type, e.g. `!nest.global_memref<4x128x128xbf16>`.
V1: parse/print only — no runtime data movement. Used as a
declaration-only entry-block argument type for `nexus.program`.

## 3. nest.\* Context-Body Ops

### 3.1 `nest.alloc`

```mlir
%buf = nest.alloc slot = "l2_buf" bytes = 131072 : !nest.l2_buffer<"l2_buf">
```

Allocates a context-owned L2 buffer. No runtime action (the L2 slot is
allocated lazily by the DMA latency model in `full_memory` fidelity).

### 3.2 `nest.task.range`

```mlir
%tasks = nest.task.range from = 0 to = 4 : !nest.task_range
```

Declares a logical task domain `[from, to)`. No runtime action
(informational: the task count is validated against the placement
popcount by the verifier, which requires a 1:1 mapping in this
validator).

### 3.3 `nest.dma.prefetch.async`

```mlir
%ev = nest.dma.prefetch.async %buf bytes = 131072 : !nest.event<"ev_dma_in">
```

HBM → L2 prefetch into the context-owned buffer. Produces one event.

### 3.4 `nest.dma.store.async`

```mlir
%ev = nest.dma.store.async %buf bytes = 131072 depends_on(%out) : !nest.event<"ev_store">
```

L2 → HBM final store, gated on the dispatch `output_ready` event via
`depends_on`. Produces one event.

### 3.5 `nest.dispatch.tasks.async`

```mlir
%grid, %inrel, %out = nest.dispatch.tasks.async @pow_4k_tile context = 1
    tasks(%t) ins(%buf) outs(%buf) depends_on(%pref)
    : (!nest.event<"grid">, !nest.event<"inrel">, !nest.event<"out">)
```

Function-call dispatch (reference.mlir §194-239). The tile program is
referenced by symbol. Placement comes from the enclosing `nest.context`,
not from this op.

Returns THREE aggregated events:

- `grid_done` — all logical tasks returned (`tile.return`).
- `input_released` — all tasks completed their L2 read phase
  (`tile.signal input_released`). Aggregated across the placement mask:
  fires when every physical tile in the placement has signalled.
- `output_ready` — all tasks completed their L2 write phase
  (`tile.signal output_ready`). Same aggregation.

`depends_on` is optional (omitted if the dispatch has no dependency).

Optional `context = N` pins every task of this dispatch to the
tile-local UCE context index `N` — the same index on every tile in the
placement, not a physical tile id. Omitted = first available context.
When the pinned context is occupied the dispatch waits for it to be
released (`dispatch_wait` stall), reusing the existing backpressure
path — no new fault mode. Legal range is `0..context_count-1`; an
out-of-range pin is rejected at task load (not at IR verify) to avoid
a silent deadlock to the cycle cap.

### 3.6 `nest.collective.async`

```mlir
%ev = nest.collective.async "reduce" bytes = 65536 mask = 15 : !nest.event<"ev_col">
```

Collective engine op (reduce/broadcast/multicast). Produces one event.

### 3.7 `nest.release`

```mlir
nest.release %buf depends_on(%store_ev)
```

Reclaims the context-owned L2 buffer. In `full_memory` fidelity, this
frees the L2 slot (`L2SRAM.free_slot`). `depends_on` is required (the
buffer can only be reclaimed after the DMA has finished reading it).

### 3.8 `nest.await`

```mlir
nest.await %grid, %store
```

Waits for one or more nest events. Lowered to one `WAIT_EVENT` action
per operand (or `WAITALL` if multiple).

### 3.9 `nest.barrier`

```mlir
nest.barrier
```

Group barrier. Zero-cycle, all tiles must reach before proceeding.

### 3.10 `nest.return`

```mlir
nest.return
```

Context completion. Signals the context's `completion_event` (default
`"context_done"`). This is the CPU-visible context completion:
`context_done` covers both `grid_done` (all tasks returned) and the
final store reaching HBM (reference.mlir §284-289).

## 4. tile.\* Program-Body Ops

### 4.1 `tile.load.async`

```mlir
%ev = tile.load.async bytes = 32768 : !tile.event<"e_load">
```

MFE L2 → L1 load. Produces one tile event.

### 4.2 `tile.store.async`

```mlir
%ev = tile.store.async bytes = 32768 : !tile.event<"e_store">
```

MFE L1 → L2 store. Produces one tile event.

### 4.3 `tile.pow.async`

```mlir
%ev = tile.pow.async bytes = 32768 exponent = 2 ops = 65536 : !tile.event<"e_pow">
```

EVU elementwise pow. `bytes` is the chunk size, `exponent` is the power,
`ops` is the total op count (feeds the EVU latency model).

### 4.4 `tile.evu.async`

```mlir
%ev = tile.evu.async "relu" ops = 16 : !tile.event<"e_evu">
```

Generic EVU op (softmax, norm, relu, etc.). `op_name` is the engine op,
`ops` is the total op count.

### 4.5 `tile.boa.async`

```mlir
%ev = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"e_mm">
%ev = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 accumulate : !tile.event<"e_mm">
```

BOA dense compute op. `accumulate` is optional (default false; when
present, the matmul result accumulates into the existing L1 buffer
rather than overwriting).

### 4.6 `tile.await`

```mlir
tile.await %ev1, %ev2
```

Waits for one or more tile events. Suspends only the current Hardware
Context (reference.mlir §375-376). Lowered to `WAIT` (1 operand) or
`WAITALL` (2+ operands).

### 4.7 `tile.signal`

```mlir
tile.signal input_released
tile.signal output_ready
```

Phase signal (reference.mlir §378-379, §414-415). Drives the dispatch
phase events:

- `input_released` — this task will not read its L2 input subview again.
- `output_ready` — this task's output is now visible in L2.

When every dispatched task of one dispatch has signalled a phase, the
corresponding dispatch result event fires. The aggregation is across
the placement mask (physical tiles), not across logical task IDs.

### 4.8 `tile.return`

```mlir
tile.return
```

Tile program completion. Contributes to `grid_done` (reference.mlir
§417-418).

## 5. Verification Rules

### 5.1 Module-level

- **Legacy mode**: exactly one `nest.context`, no `nexus.program`.
- **Model mode**: exactly one `nexus.program` + one or more
  `nest.context` ops.
- Zero or more `tile.program` ops (both modes).
- No other top-level ops.
- Tile program symbol names must be unique.
- `nest.context` symbol names must be unique.

- `context = N` on `nest.context` (if present) must be >= 0 and <
  `device_context_count` (upper bound checked at model/task load).
- All event tags (from `!nest.event<tag>` results) must be unique within
  the context body. Empty tags (for unused phase events) are skipped.
- `nest.dispatch.tasks.async`:
  - `@prog` must reference a defined `tile.program`.
  - Task range count must equal `popcount(placement)` (1:1 mapping).
  - `depends_on` operands must be events defined earlier in the body.
  - `context = N` (if present) must be >= 0; the upper bound is the
    simulator's `context_count` (checked at task load, not at IR verify).
- `nest.await` operands must be events defined earlier.
- `nest.dma.store.async` / `nest.release` `depends_on` operands must be
  events defined earlier.

### 5.3 Tile program body

- All event tags (from `!tile.event<tag>` results) must be unique within
  the program body.
- `tile.await` operands must be events defined earlier.
- `tile.signal` phase must be `input_released` or `output_ready`.

### 5.4 `nexus.program` body

- `nexus.submit_context.async` `@ctx` must reference a defined
  `nest.context`.
- Event tags (from `!nexus.event<"tag">` results) must be non-empty and
  unique within the program body.
- `nexus.await` operands must be events defined earlier in the body
  (by a prior `nexus.submit_context.async`).
- The body must end with `nexus.return`.
- No other ops are allowed in the body.

## 6. Lowering (IR → Runtime)

The lowering (`ir_lowering.py`) is a direct 1:1 walk of the IR body,
producing `ExecTileGroupTask` DTOs consumed by the cycle-accurate
simulator. The event type tag is used directly as the runtime event id,
so the trace (engine jobs, event ids, PMU counters) corresponds exactly
to the IR ops.

### 6.1 Context body → ExecGroupAction list

| IR op                       | ExecGroupAction                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------- |
| `nest.alloc`                | (no action; L2 slot allocated lazily by DMA latency model)                                |
| `nest.task.range`           | (no action; validated by verifier)                                                        |
| `nest.dma.prefetch.async`   | `DMA_PREFETCH` args=(desc_id, slot, bytes)                                                |
| `nest.dma.store.async`      | `WAIT_EVENT` per depends_on; then `DMA_STORE` args=(desc_id, slot, bytes)                 |
| `nest.dispatch.tasks.async` | `WAIT_EVENT` per depends_on; then `DISPATCH_ROLE` args=(role_id, inrel_tag, outready_tag) |
| `nest.collective.async`     | `COLLECTIVE_RUN` args=(name, op, bytes, mask)                                             |
| `nest.release`              | `WAIT_EVENT` per depends_on; then `RELEASE_L2` args=(slot)                                |
| `nest.await`                | `WAIT_EVENT` per operand                                                                  |
| `nest.barrier`              | `BARRIER_GROUP`                                                                           |
| `nest.return`               | `SIGNAL_EVENT` args=(completion_event)                                                    |

### 6.2 Tile program body → ExecTileInst list

| IR op              | ExecTileInst                         |
| ------------------ | ------------------------------------ |
| `tile.load.async`  | `LAUNCH_MFE` (MFE "load")            |
| `tile.store.async` | `LAUNCH_MFE` (MFE "store")           |
| `tile.pow.async`   | `LAUNCH_EVU` (EVU "pow")             |
| `tile.evu.async`   | `LAUNCH_EVU` (EVU op_name)           |
| `tile.boa.async`   | `LAUNCH_BOA` (BOA op_name)           |
| `tile.await`       | `WAIT` (1 operand) or `WAITALL` (2+) |
| `tile.signal`      | `SIGNAL_PHASE` args=(phase_name)     |
| `tile.return`      | `RET`                                |

### 6.3 Role binding

Each unique `(program, placement_mask)` pair gets an auto-assigned
`role_id` (starting from 0). Re-dispatching the same program with the
same mask reuses the existing binding.

### 6.4 Model lowering (`nexus.*` → `ExecDeviceOp`)

In model mode, `lower_model_ir` produces an `ExecModel` containing:

- `tasks`: `nest.context` ops lowered to `ExecTileGroupTask` (same as
  §6.1) keyed by context symbol name.
- `context_pins`: per-context device slot pin (from `nest.context`
  `context = N`, or `None`).
- `body`: `nexus.submit_context.async` → `ExecDeviceOp("submit", ...)`,
  `nexus.await` → `ExecDeviceOp("await", ...)` per operand,
  `nexus.return` → `ExecDeviceOp("return")`.

The device PC loop (`Simulator._run_model`) walks `body` linearly:
submit assigns a slot (pin or first-free), await blocks until the event
fires, return completes. Slots share ONE `TileGroup` instance; each
submit deep-clones its `ExecTileGroupTask` and namespaces event/stream
IDs with a monotonic launch ID (`s{slot}l{launch}_`), then registers a
fresh `TileGroupSequencer` advanced in lockstep each cycle. Completing
sequencers are pruned; when all slots drain and `return` was reached,
the model completes.

## 7. Runtime: Phase Signal Aggregation

When a tile executes `tile.signal <phase>`, the UCE calls
`_phase_signal_callback(role_event_id, phase, tile_id)`.

`TileGroup._on_phase_signal`:

1. Looks up the dispatch's `phase_event_ids` map for the phase event id.
2. Adds `tile_id` to the phase event's done-tile set.
3. When the done count reaches `popcount(placement_mask)`, calls
   `sequencer.notify_event(phase_event_id)`.

This fires the `input_released` or `output_ready` dispatch result event,
which resolves any `nest.await` or `depends_on` waiting on it.

## 8. Runtime: L2 Buffer Lifecycle

- `nest.alloc` — no runtime action; L2 slot is allocated lazily by
  `L2SRAM.alloc_slot` when the DMA latency model runs (prefetch or store).
- `alloc_slot` is **idempotent**: if a slot with the same name already
  exists (e.g. store re-addressing a prefetch slot), it is returned
  as-is — no double accounting.
- `nest.release` — `RELEASE_L2` action calls `L2SRAM.free_slot(slot)`,
  decrementing the used capacity. This enables buffer reuse in
  `full_memory` fidelity.
- `L2SRAM` capacity fault: if `alloc_slot` returns `None` (capacity
  exhausted), the sequencer faults and terminates the task.
