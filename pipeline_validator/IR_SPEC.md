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
  tile.program @pow_4k_tile (%task : !nest.task, %l2_buf : !nest.l2_buffer<4x128x128xbf16>) { ... }
  nest.context @pow_task (%Y : !nest.global_memref<4x128x128xbf16>) placement = 15 { ... }
  nexus.program @run_pow (%Y0 : !nest.global_memref<4x128x128xbf16>, %Y1 : !nest.global_memref<4x128x128xbf16>) { ... }
}
```

### 1.1 `nest.context @name (%Y : !nest.global_memref<...>) placement = M context = N { ... }`

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

### 1.2 `tile.program @name (%task : !nest.task, %l2_buf : !nest.l2_buffer<...>) { ... }`

Defines one tile program. The body contains tile-level async engine ops,
`tile.await`, `tile.signal`, and `tile.return`. The program is referenced
by `nest.dispatch.tasks.async` via its symbol name.

The entry block declares the program's data formals: the **first** formal
must be `!nest.task` (the logical task handle), followed by zero or more
`!nest.l2_buffer<...>` formals. Each L2 formal is bound positionally to a
dispatch actual (`ins`/`outs`); the body subviews it (`tile.subview`) and
moves data with explicit `src`/`dst` (`tile.load.async`/`tile.store.async`).

### 1.3 `nexus.program @name (%a : !nest.global_memref<...>) { ... }`

Model entry point. A model-mode module contains exactly one
`nexus.program`. The body is a linear device-level program consisting of
`nexus.submit_context.async`, `nexus.await`, and `nexus.return` (ending
with return). Entry block args are named global inputs (each must carry a
non-empty SSA name, e.g. `%Y0`). They flow as real SSA values:
`nexus.submit_context.async @ctx(%Y0)` binds each arg to a `nest.context`
formal by position; the context body subviews the formal and moves it
with explicit `src`/`dst` transfer ops. Bytes are derived from
view/buffer shapes, not from a `bytes` property.

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

**L2 admission wait (PR 3.5)**: accepting a submit reserves the device
slot but does not guarantee immediate L2 capacity. If the context's L2
bundle (every `nest.alloc` slot as one atomic allocation) transiently
cannot fit the live free map, the context enters `ADMISSION_WAIT`: the
slot stays busy but no UCE context, L1 frame, stream queue, L2 handle or
DMA/engine work is held, and no fault is recorded. A legal
`nest.release` whose allocator final-free makes capacity available
wakes the strict-FIFO wait queue in the same cycle; the admitted context
issues its first group action the next cycle. Invalid bundles
(size/alignment) and bundles that can never fit an empty L2 fault
immediately and never queue. The submit result event still means the
**full context completion**, not admission acceptance; two submits with
no intermediate `nexus.await` may therefore be ACTIVE and
ADMISSION_WAIT concurrently. PMU: `l2_admission_wait`,
`l2_admission_retry`, `l2_admission_wakeup`, `l2_admission_wait_cycles`,
`l2_admission_queue_peak`, `l2_admission_permanent_fault`.

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

### 2.3 `!nest.l2_buffer<DxDx...xdtype>`

Context-owned L2 buffer, shape-typed: e.g.
`!nest.l2_buffer<4x128x128xbf16>`. Produced by `nest.alloc`. The L2 slot
id (used by the group DMA latency model and the L2 allocator) lives on
the defining `nest.alloc`'s `slot` attribute, not in the type.

### 2.4 `!nest.task`

Logical task handle. Appears as the **first** `tile.program` formal; a
`tile.subview` may bind it via `task = %task` to offset its view by the
logical task id along one dimension.

### 2.5 `!nest.global_view<DxDx...xdtype>`

Logical view of a global memref, e.g.
`!nest.global_view<4x128x128xbf16>`. Produced by `nest.subview`; consumed
as the `src`/`dst` of `nest.dma.prefetch.async`/`nest.dma.store.async`.

### 2.6 `!nest.l2_view<DxDx...xdtype>`

Logical per-task view of an L2 buffer, e.g.
`!nest.l2_view<1x128x128xbf16>`. Produced by `tile.subview`; consumed as
the L2 side of `tile.load.async`/`tile.store.async`.

### 2.7 `!tile.l1_buffer<DxDx...xdtype>`

Tile-local L1 buffer, shape-typed: e.g.
`!tile.l1_buffer<128x128xbf16>`. Produced by `tile.alloc`; consumed as the
L1 side of `tile.load.async`/`tile.store.async`.

### 2.8 `!nest.task_range`

Logical task domain. Produced by `nest.task.range`. Task IDs are logical
IDs, NOT physical Tile IDs or Hardware Context IDs (reference.mlir §170-171).

### 2.9 `!nexus.event<"tag">`

Device-level async event. The `tag` is the runtime event id shared by
the device scheduler and the trace. Produced by
`nexus.submit_context.async`; consumed by `nexus.await`.

### 2.10 `!nest.global_memref<DxDx...xdtype>`

Host-visible global input, shape-typed: e.g.
`!nest.global_memref<4x128x128xbf16>`. Appears as a `nexus.program` or
`nest.context` block-arg formal; consumed by `nest.subview` to produce a
`!nest.global_view` for explicit prefetch/store.

## 3. nest.\* Context-Body Ops

### 3.1 `nest.alloc`

```mlir
%buf = nest.alloc slot = "l2_buf" role = "inout"
    shape = [4, 128, 128] dtype = "bf16" alignment = 256
    : !nest.l2_buffer<4x128x128xbf16>
```

Allocates a context-owned L2 buffer. `slot` is the L2 object id (used by
the DMA latency model and L2 allocator); `role` is `in`/`out`/`inout`;
`shape`/`dtype` must match the result type; `alignment` is optional. No
runtime action (the L2 slot is allocated lazily by the DMA latency model
in `full_memory` fidelity).

### 3.2 `nest.subview`

```mlir
%src = nest.subview %Y offsets = [0, 0, 0] sizes = [4, 128, 128] strides = [1, 1, 1]
    : !nest.global_view<4x128x128xbf16>
```

Creates a logical view of a global memref formal. V1: `src` must be a
context block-arg formal (no view chains); `strides` must be all-1;
`offsets[d] + sizes[d] <= parent_dims[d]` for every dim. Produces a
`!nest.global_view` consumed by prefetch/store.

### 3.3 `nest.task.range`

```mlir
%tasks = nest.task.range from = 0 to = 4 : !nest.task_range
```

Declares a logical task domain `[from, to)`. No runtime action
(informational: the task count is validated against the placement
popcount by the verifier, which requires a 1:1 mapping in this
validator).

### 3.4 `nest.dma.prefetch.async`

```mlir
%ev = nest.dma.prefetch.async %src into %l2_buf : !nest.event<"ev_dma_in">
```

HBM → L2 prefetch from a `!nest.global_view` (`%src`) into a
`!nest.l2_buffer` (`%l2_buf`). The byte count is `prod(sizes) * dtype_size`
derived from the view/buffer shapes. Produces one event.

### 3.5 `nest.dma.store.async`

```mlir
%ev = nest.dma.store.async %l2_buf into %src depends_on(%out) : !nest.event<"ev_store">
```

L2 → HBM final store from the `!nest.l2_buffer` into the
`!nest.global_view`, gated on the dispatch `output_ready` event via
`depends_on`. Produces one event.

### 3.6 `nest.dispatch.tasks.async`

```mlir
%grid, %inrel, %out = nest.dispatch.tasks.async @pow_4k_tile context = 1
    tasks(%t) ins(%buf) outs(%buf)
    signal_policy {
      input_released = #nest.aggregate<all_tasks>,
      output_ready = #nest.aggregate<all_tasks>
    }
    depends_on(%pref)
    : (!nest.event<"grid">, !nest.event<"inrel">, !nest.event<"out">)
```

Function-call dispatch (reference.mlir §194-239). The tile program is
referenced by symbol. `ins`/`outs` are the L2 buffer actuals bound
positionally to the tile program's L2 formals (after the leading
`!nest.task` formal); each must be a whole `!nest.l2_buffer` whose
shape/dtype exactly match the corresponding formal. Placement comes from
the enclosing `nest.context`, not from this op.

`signal_policy { ... }` is always printed, possibly as
`signal_policy {}`. Each entry declares how one program phase aggregates
with `#nest.aggregate<all_tasks>` (the only V1 mode). A policy entry
requires the matching non-empty phase result tag; an omitted entry
requires that result tag to be empty.

- `grid_done` — all logical tasks returned (`tile.return`).
- `input_released` — every expected logical task emitted
  `tile.signal input_released(%task)`.
- `output_ready` — every expected logical task emitted
  `tile.signal output_ready(%task)`.

Phase signals are isolated by `(context launch generation, grid instance,
phase, logical task)`. `GridInstanceId` contains the context name, device
slot, launch generation, and dispatch ordinal; physical tile and Hardware
Context are location details, never phase-aggregation identities.

`depends_on` is optional (omitted if the dispatch has no dependency).

Optional `context = N` pins every task of this dispatch to the
tile-local UCE context index `N` — the same index on every tile in the
placement, not a physical tile id. Omitted = first available context.
When the pinned context is occupied the dispatch waits for it to be
released (`dispatch_wait` stall), reusing the existing backpressure
path — no new fault mode. Legal range is `0..context_count-1`; an
out-of-range pin is rejected at task load (not at IR verify) to avoid
a silent deadlock to the cycle cap.

### 3.7 `nest.collective.async`

```mlir
%ev = nest.collective.async "reduce" bytes = 65536 mask = 15 : !nest.event<"ev_col">
```

Collective engine op (reduce/broadcast/multicast). Produces one event.

### 3.8 `nest.release`

```mlir
nest.release %buf depends_on(%store_ev)
```

Reclaims the context-owned L2 buffer. `depends_on` is required, and
release legality is determined by the buffer role from `nest.alloc`:

- role=`"in"` must depend on exactly the `input_released` results of
  every dispatch that consumes the buffer through `ins`.
- role=`"out"` and role=`"inout"` must depend only on the final
  `nest.dma.store.async` result for that buffer. That final store must
  itself `depends_on` every producing dispatch's `output_ready` result.

Every `nest.alloc` must have exactly one legal `nest.release` in the
same context before `nest.return`. At runtime, release additionally
checks the explicit events plus the allocation's owner, generation, live
state, and consumer pins; it is not a best-effort free.

### 3.9 `nest.await`

```mlir
nest.await %grid, %store
```

Waits for one or more nest events. Lowered to one `WAIT_EVENT` action
per operand (or `WAITALL` if multiple).

### 3.10 `nest.barrier`

```mlir
nest.barrier
```

Group barrier. Zero-cycle, all tiles must reach before proceeding.

### 3.11 `nest.return`

```mlir
nest.return
```

Context completion. Signals the context's `completion_event` (default
`"context_done"`). This is the CPU-visible context completion:
`context_done` covers both `grid_done` (all tasks returned) and the
final store reaching HBM (reference.mlir §284-289).

## 4. tile.\* Program-Body Ops

### 4.1 `tile.subview`

```mlir
%l2_tile = tile.subview %l2_buf task = %task task_dim = 0
    offsets = [0, 0, 0] sizes = [1, 128, 128] strides = [1, 1, 1]
    : !nest.l2_view<1x128x128xbf16>
```

Creates a logical per-task view of an L2 buffer formal. V1: `src` must be
a `tile.program` L2 formal (no view chains); `strides` must be all-1.
When `task = %task` + `task_dim = d` are present (they must appear
together), the effective offset along dimension `d` is
`offsets[d] + logical_task_id`. The result dims must equal `sizes` and its
dtype must match the source.

### 4.2 `tile.alloc`

```mlir
%l1 = tile.alloc shape = [128, 128] dtype = "bf16" alignment = 256
    : !tile.l1_buffer<128x128xbf16>
```

Allocates a tile-local L1 buffer. `shape`/`dtype` must match the result
type; `alignment` is optional. Consumed by `tile.load.async` /
`tile.store.async` as the L1 side.

### 4.3 `tile.load.async`

```mlir
%ev = tile.load.async %l2_tile into %l1 : !tile.event<"e_load">
```

MFE L2 → L1 load from a `!nest.l2_view` into a `!tile.l1_buffer`. The
byte count is derived from the view/buffer shapes. Produces one tile event.

### 4.4 `tile.store.async`

```mlir
%ev = tile.store.async %l1 into %l2_tile : !tile.event<"e_store">
```

MFE L1 → L2 store from a `!tile.l1_buffer` into a `!nest.l2_view`.
Produces one tile event.

### 4.5 `tile.pow.async`

```mlir
%ev = tile.pow.async bytes = 32768 exponent = 2 pow_ops = 65536 : !tile.event<"e_pow">
```

EVU elementwise pow. `bytes` is the chunk size, `exponent` is the power,
`pow_ops` is the total op count (feeds the EVU latency model).

### 4.6 `tile.evu.async`

```mlir
%ev = tile.evu.async "relu" ops = 16 : !tile.event<"e_evu">
```

Generic EVU op (softmax, norm, relu, etc.). `op_name` is the engine op,
`ops` is the total op count.

### 4.7 `tile.boa.async`

```mlir
%ev = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 : !tile.event<"e_mm">
%ev = tile.boa.async "matmul" m = 128 n = 128 k = 64 ops = 2097152 accumulate : !tile.event<"e_mm">
```

BOA dense compute op. `accumulate` is optional (default false; when
present, the matmul result accumulates into the existing L1 buffer
rather than overwriting).

### 4.8 `tile.await`

```mlir
tile.await %ev1, %ev2
```

Waits for one or more tile events. Suspends only the current Hardware
Context (reference.mlir §375-376). Lowered to `WAIT` (1 operand) or
`WAITALL` (2+ operands).

### 4.9 `tile.signal`

```mlir
tile.signal input_released(%task)
tile.signal output_ready(%task)
```

Phase signal (reference.mlir §378-379, §414-415). Its required operand
must be block argument 0 of the enclosing `tile.program` (the
`!nest.task` formal), so the runtime binds each emission to a logical
task:

- `input_released` — this task will not read its L2 input subview again.
- `output_ready` — this task's output is now visible in L2.

The dispatch's `signal_policy` selects the declared phases. For each
phase, the event fires exactly once only after every expected logical task
in that `GridInstanceId` has signalled; duplicate task/phase signals are
ignored. The physical placement mask does not aggregate phase signals.

### 4.10 `tile.return`

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

### 5.2 Inputs, bindings, and views (PR 1 memory contract)

- **`nexus.program` inputs**: every block arg must be
  `!nest.global_memref` and must carry a non-empty SSA name (used as the
  input-binding key).
- **`nest.context` formals**: every block arg must be
  `!nest.global_memref`.
- **submit ↔ context signature**: `nexus.submit_context.async @ctx` must
  pass exactly as many actuals as `@ctx` declares formals, and each
  actual's dims+dtype must equal the corresponding formal's.
- **`tile.program` formals**: at least one formal; the first must be
  `!nest.task`; the rest must be `!nest.l2_buffer`.
- **dispatch ↔ tile.program binding**: `ins` and `outs` each bind,
  positionally and independently, to the program's L2 formals (count and
  dims+dtype must match); dispatch actuals must be whole `!nest.l2_buffer`
  values.
- **View bounds (`nest.subview` / `tile.subview`)**: every dim requires
  `offset >= 0`, `size >= 1`, `offset + size <= parent_dim`; view byte
  count must not overflow int64. `tile.subview` bounds against a task
  dimension are checked at the dispatch checkpoint with the maximum task
  id of the dispatch's task range.
- **Strides**: V1 requires all-1 strides on both subview ops.
- **Transfer byte equality**: prefetch/store/load/store require
  `prod(src dims) * dtype_size == prod(dst dims) * dtype_size`.
- **Root-object constraint (no view chains)**: `nest.subview` `src` must
  be a context global formal; `tile.subview` `src` must be a
  `tile.program` L2 formal.
- **Input bindings** (simulator load time, not IR verify): every program
  input needs a same-name binding; unknown bindings, undersized bindings,
  overlapping IOVA ranges, and ranges past HBM capacity are rejected with
  a `ValueError`.

### 5.3 Context body

- `context = N` on `nest.context` (if present) must be >= 0 and <
  `device_context_count` (upper bound checked at model/task load).
- All event tags (from `!nest.event<tag>` results) must be unique within
  the context body. Empty tags (for unused phase events) are skipped.
- `nest.dispatch.tasks.async`:
  - `@prog` must reference a defined `tile.program`.
  - Task range count must equal `popcount(placement)` (1:1 mapping).
  - `depends_on` operands must be events defined earlier in the body.
  - Its `signal_policy` key set must exactly match the referenced
    program's emitted `tile.signal` phase set; each mode must be
    `all_tasks`. A declared phase requires a non-empty matching result
    tag, and an undeclared phase requires an empty tag.
  - `context = N` (if present) must be >= 0; the upper bound is the
    simulator's `context_count` (checked at task load, not at IR verify).
- `nest.dma.store.async` / `nest.release` `depends_on` operands must be
  events defined earlier. Every `nest.alloc` has exactly one release
  before `nest.return`: an `"in"` release's dependency SSA set is exactly
  all matching consumer `input_released` events; an `"out"`/`"inout"`
  release depends only on the final store, which waits on all matching
  producer `output_ready` events.
- `nest.await` operands must be events defined earlier.

### 5.4 Tile program body

- All event tags (from `!tile.event<tag>` results) must be unique within
  the program body.
- `tile.await` operands must be events defined earlier.
- `tile.signal` phase must be `input_released` or `output_ready`, and
  its sole operand must be block argument 0 (the program's `!nest.task`
  formal).

### 5.5 `nexus.program` body

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

| IR op                       | ExecGroupAction                                                               |
| --------------------------- | ----------------------------------------------------------------------------- |
| `nest.alloc`                | (no action; records `ExecL2Buffer`; L2 bundle admitted at context start)      |
| `nest.subview`              | (no action; records `ExecMemoryView`)                                         |
| `nest.task.range`           | (no action; records `ExecTaskDomain`, attached to dispatch role bindings)     |
| `nest.dma.prefetch.async`   | `DMA_PREFETCH` args=(desc_id, ExecTransfer)                                   |
| `nest.dma.store.async`      | `WAIT_EVENT` per depends_on; then `DMA_STORE` args=(desc_id, ExecTransfer)    |
| `nest.dispatch.tasks.async` | `WAIT_EVENT` per depends_on; then `DISPATCH_ROLE` args=(ExecDispatchRequest,) |
| `nest.collective.async`     | `COLLECTIVE_RUN` args=(name, op, bytes, mask)                                 |
| `nest.release`              | `WAIT_EVENT` per depends_on; then `RELEASE_L2` args=(ExecReleaseRequest,)     |
| `nest.await`                | `WAIT_EVENT` per operand                                                      |
| `nest.barrier`              | `BARRIER_GROUP`                                                               |
| `nest.return`               | `SIGNAL_EVENT` args=(completion_event)                                        |

`ExecTransfer` carries explicit `src`/`dst` `ExecMemoryView`s and the
byte count; the sequencer reads `transfer.dst.base`/`transfer.bytes` for
prefetch and `transfer.src.base`/`transfer.bytes` for store. `global_inputs`,
`l2_buffers`, `task_domain`, and per-binding `actuals` are recorded on the
`ExecTileGroupTask`/`ExecTileRoleBinding` DTOs.

`ExecDispatchRequest` preserves the role id, source-order dispatch ordinal,
per-phase `ExecSignalPolicy`, and phase event ids; its action `dst` remains
the `grid_done` event. `ExecReleaseRequest` preserves the verified buffer
slot/role, matching consumer dispatch ordinals, and explicit dependency
event ids. These structured DTOs are consumed directly by the runtime; it
does not recover identity or release dependencies by parsing event strings.

### 6.2 Tile program body → ExecTileInst list

| IR op              | ExecTileInst                                        |
| ------------------ | --------------------------------------------------- |
| `tile.alloc`       | (no action; records `ExecL1Buffer`)                 |
| `tile.subview`     | (no action; records `ExecMemoryView`)               |
| `tile.load.async`  | `LAUNCH_MFE` (MFE "load", transfer on the desc)     |
| `tile.store.async` | `LAUNCH_MFE` (MFE "store", transfer on the desc)    |
| `tile.pow.async`   | `LAUNCH_EVU` (EVU "pow")                            |
| `tile.evu.async`   | `LAUNCH_EVU` (EVU op_name)                          |
| `tile.boa.async`   | `LAUNCH_BOA` (BOA op_name)                          |
| `tile.await`       | `WAIT` (1 operand) or `WAITALL` (2+)                |
| `tile.signal`      | `SIGNAL_PHASE` args=(phase_name, task_formal_index) |
| `tile.return`      | `RET`                                               |

MFE load/store descriptors carry an `ExecTransfer` (src/dst views +
bytes); the tile reads `desc.transfer.bytes` for the latency model.

### 6.3 Role binding

Each unique `(program, placement_mask, context identity, task domain,
actuals)` binding gets an auto-assigned `role_id` (starting from 0).
Device slot is deliberately not part of this static role-binding identity.
Each source dispatch still receives its own source-order `dispatch_ordinal`
inside `ExecDispatchRequest`, which becomes part of `GridInstanceId` at
runtime.

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

When a tile executes `tile.signal <phase>(%task)`, the UCE resolves the
lowered task-formal index against its current `TaskIdentity` and calls
`_on_phase_signal(PhaseSignal(task, phase), cycle)`.

`TileGroup` keeps one signal state per `GridInstanceId`, with the expected
logical task-id set, declared `ExecSignalPolicy`, phase event ids, and
already-seen task ids. A signal is processed as follows:

1. A retired/non-live launch generation is stale and ignored.
2. A live but unknown grid, task, or phase is invalid and faults the
   owning sequencer.
3. A duplicate `(grid, phase, task_id)` is ignored.
4. A first valid signal is recorded; when its seen task ids exactly equal
   the grid's expected logical-task set, `notify_event` fires that phase
   event exactly once.

Thus `input_released` and `output_ready` resolve only the matching grid's
waiters; physical tile id and UCE hardware-context id do not participate in
the aggregation key.

## 8. Runtime: L2 Buffer Lifecycle

- `nest.alloc` - no runtime action at issue time; at context admission
  every `l2_buffers` entry is planned and committed as one atomic
  bundle on the L2 `BankedFreeExtentAllocator` (owner
  `ContextBufferOwner`, launch generation, alignment, bank segments).
  A typed `AdmissionFailureKind` classifies a failed plan:
  `INVALID_REQUEST` (size/alignment never legal) and
  `PERMANENT_CAPACITY` (cannot fit even an empty pool) fault the
  sequencer before any DMA starts; `TEMPORARY_CAPACITY` (legally
  placeable but not under the current live free map) enters the
  strict-FIFO admission wait queue instead. A failed plan never
  mutates the free map, pool version, counters or peak.
- `nest.release` - `RELEASE_L2` first requires every explicit dependency
  event to be complete, then validates the buffer role, owner, generation,
  live handle, and required pin state before calling `request_release`.
  `input_released` aggregation unpins only role=`"in"` consumers.
  `output_ready` only makes L2 output visible; role=`"out"`/`"inout"`
  pins remain until their final-store-gated release. An unsatisfied or
  inconsistent release is an invariant fault, never a silent success.
  Only an allocator **final-free** (request_release returning with no
  remaining pins) marks a capacity change: signal aggregates and unpins
  alone never wake the admission queue.
- `L2SRAM` capacity fault: a permanent/invalid `AdmissionFailure`
  faults the sequencer with `L2 capacity fault during context
admission` and no completion event is produced; a transient miss
  never writes the fault ring. The strict-FIFO wait queue retries only
  on a release final-free; the head is admitted in that same cycle and
  issues its first group action the next cycle. Reset/fault cleanup
  cancels waiting tickets (they own no allocation) and accumulates
  `l2_admission_wait_cycles = terminal - enqueue` exactly once.
