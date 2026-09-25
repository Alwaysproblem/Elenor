# ELENOR Pipeline Validator — Source and Executable IR Specification

This document specifies the function-call-style xDSL source dialect, its
explicit compiler boundary, and the runtime contracts carried by the immutable
executable. Source IR is never executed directly.

## 1. Module Structure and Execution Boundary

A valid source module has one of two shapes.

**Standalone** contains exactly one `nest.context` and no `nexus.program`.
This complete zero-dispatch module is executable after explicit compilation
and loading:

```mlir
builtin.module {
  nest.context @empty placement = 1
    resource_contract = #nest.context_resources<
      l2_mode = 0,
      allowed_profiles = [0, 1, 2],
      logical_tasks = 0,
      l2_spm_bytes = 0,
      requested_contexts_per_tile = 1
    > {
    nest.return
  }
}
```

**Model** contains exactly one `nexus.program` and one or more
`nest.context` definitions. This complete module is also executable after
explicit compilation and loading:

```mlir
builtin.module {
  nest.context @empty(%arena : !nest.global_memref<32xbf16>) placement = 1
    resource_contract = #nest.context_resources<
      l2_mode = 0,
      allowed_profiles = [0, 1, 2],
      logical_tasks = 0,
      l2_spm_bytes = 0,
      requested_contexts_per_tile = 1
    > {
    nest.return
  }
  nexus.program @run(%arena : !nest.global_memref<32xbf16>) {
    %done = nexus.submit_context.async @empty(%arena)
      : !nexus.event<"done">
    nexus.await %done
    nexus.return
  }
}
```

### 1.1 `nest.context`

Grammar:

```text
nest.context @name(%global-formals...)
  placement = M [context = N] [completion = "event"]
  resource_contract = #nest.context_resources<...> { ... }
```

`placement` is a nonzero Tile mask. The current validator maps logical Task
$i$ to selected Tile $i$, so every non-empty dispatch range count must equal
`popcount(placement)`. Optional `context=N` is a model-mode Group execution
slot affinity bounded by `group.active_context_capacity`; it is not a Tile ID
or Tile UCE context. The standalone driver uses Group slot 0.

The mandatory resource contract is L2-only:

```text
#nest.context_resources<
  l2_mode = N,
  allowed_profiles = [N, ...],
  logical_tasks = N,
  l2_spm_bytes = N,
  requested_contexts_per_tile = R
  [, l2_cache = {required = bool, access = "none"|"read"|"read_write",
                 bypass = "allowed"|"forbidden", target_bytes = N}]
>
```

`l2_mode` must belong to the non-empty, duplicate-free
`allowed_profiles`. `logical_tasks` is the sum of all dispatch range counts.
`l2_spm_bytes` is the whole root-Arena reservation, including layout padding;
zero is legal when there are no L2 buffers. `requested_contexts_per_tile` is a
positive R lease bound and may not exceed `SimConfig.context_count`.

### 1.2 `tile.program`

Grammar:

```text
tile.program @name(%task : !nest.task, %globals..., %l2-buffers...)
  resource_contract = #tile.resources<...> { ... }
```

The first formal is `!nest.task`; zero or more
`!nest.global_view<...>` formals follow, then zero or more
`!nest.l2_buffer<...>` formals. The mandatory contract is:

```text
#tile.resources<
  allowed_profiles = [N, ...],
  tile_l1_spm_bytes_per_context = N
  [, l1_cache = {required = bool, access = "none"|"read"|"read_write",
                 bypass = "allowed"|"forbidden", target_bytes = N}]
  [, l2_cache = {required = bool, access = "none"|"read"|"read_write",
                 bypass = "allowed"|"forbidden", target_bytes = N}]
>
```

The allowed L1 set is non-empty and duplicate-free.
`tile_l1_spm_bytes_per_context` is the whole per-Task L1 Arena reservation,
including striped padding. Cache declarations are hard capabilities;
`target_bytes` is a shared-cache hint, not a private allocation or minimum
quota.

### 1.3 `nexus.program`

The model entry is linear: `nexus.submit_context.async`, `nexus.await`, and
one terminal `nexus.return`. Entry block arguments are named
`!nest.global_memref` inputs. Submit actuals bind Context formals positionally,
while external HBM bindings are matched by entry-argument name at load time.

The following is a **Device-body fragment**, not a standalone module:

```mlir
%a = nexus.submit_context.async @producer(%input, %middle)
  : !nexus.event<"a">
%c = nexus.submit_context.async @consumer(%middle, %out) depends_on(%a)
  : !nexus.event<"c">
%b = nexus.submit_context.async @independent(%other)
  : !nexus.event<"b">
nexus.await %b, %c
```

Submit dependencies and compiler-derived cross-root RAW/WAR/WAW hazards are
preserved in the executable. WAIT_DEPS consumes CPU pending metadata but no
Group slot, Arena, event table, UCE, or engine resource. Completion means the
entire root is retired, not merely submitted or at its final PC.

### 1.4 Source → immutable executable → runtime

The formal API is:

```python
from pipeline_validator.compiler import compile_program
from pipeline_validator import load_program, Simulator

compiled = compile_program(module, hw, sim, source_name="input.mlir")
loaded = load_program(compiled, hw, sim, actual_bindings=bindings)
result = Simulator(hw, sim).run(loaded)
```

`compile_program(ModuleOp, HardwareConfig, SimConfig, ...)` is the sole public
source compiler entry. It returns a deeply immutable `CompiledProgram`.
`load_program` verifies and binds that artifact without importing the compiler.
`Simulator.run` accepts only `LoadedProgram`; source `ModuleOp`, private
execution DTOs, and unloaded `CompiledProgram` are rejected.

The compiler specializes each submit call from a clean Context template,
derives static dependencies/effects/layouts, binds L1/L2 Profiles, inserts
ordinary awaits plus explicit configuration/maintenance controls, assigns
content identities, and seals a strict artifact. The Loader is read-only: it
never recompiles source, mutates the graph, inserts a missing edge, or performs
a runtime Profile choice.

## 2. SSA Types

### 2.1 `!nest.event<tag>`

Group-level async event. The `tag` (a string literal) is the runtime
event id used by the simulator and the trace. Produced by:
`nest.dma.prefetch.async`, `nest.dma.store.async`, `nest.dispatch.tasks.async`,
`nest.collective.async`.

### 2.2 `!tile.event<tag>`

Tile-level async event. Same semantics as `!nest.event` but scoped to a
single tile. Produced by: `tile.load.async`, `tile.store.async`,
`tile.gather.global.async`, `tile.pow.async`, `tile.evu.async`,
`tile.boa.async`.

### 2.3 `!nest.l2_buffer<DxDx...xdtype>`

Context-owned L2 buffer, shape-typed: e.g.
`!nest.l2_buffer<4x128x128xbf16>`. Produced by `nest.alloc`. The defining
operation's `slot` is the compiled buffer/layout identity; it is not encoded
in the type.

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
`!tile.l1_buffer<128x128xbf16>`. Produced by `tile.alloc`; consumed by
`tile.load.async`/`tile.store.async`, Gather indices/destination, and `tile.free`.

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

Unless a block is explicitly called a complete module, the MLIR snippets in
§§3–4 are operation/body fragments. They assume legal enclosing
`resource_contract` attributes, symbol definitions, SSA producers, and (for
dispatch) a mandatory `l1_mode`; they are not standalone executable files.

### 3.1 `nest.alloc`

```mlir
%buf = nest.alloc slot = "l2_buf" role = "inout"
    shape = [4, 128, 128] dtype = "bf16" alignment = 256
    : !nest.l2_buffer<4x128x128xbf16>
```

Declares a Context-owned L2 buffer. `slot` is the compiled buffer/view id;
`role` is `in`/`out`/`inout`; `shape`/`dtype` must match the result type;
`alignment` is optional. The compiler places every declaration in the
Context's striped L2 `ArenaLayout`. Root admission atomically reserves the
whole Arena, including padding. A generated `BIND_L2_VIEW` action binds this
buffer's valid-byte segments at its compiled lifetime start; the source op is
not a standalone dynamic allocation.

`sharing` is optional and defaults to `private`; its only other value is
`readonly`. A readonly allocation is local producer storage until it is
published once with `nest.publish`; publication seals the initialized bytes and
creates no additional allocation. A consumer obtains its typed readonly import
with `nexus.shared.ref` on a producer submit result and passes that reference as
a context actual. Imported views are aliases, not entries in the consumer's
`ArenaLayout` or private L2 reservation.

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
Optional `depends_on(...)` names earlier Group SSA events and remains on
the action descriptor. Lowering also preserves overlapping global-range
and L2 read/write hazards; independent input prefetches remain independent.

### 3.5 `nest.dma.store.async`

```mlir
%ev = nest.dma.store.async %l2_buf into %src depends_on(%out) : !nest.event<"ev_store">
```

L2 → HBM store from the `!nest.l2_buffer` into the `!nest.global_view`.
Every Store explicitly depends on all previously defined dispatches that
actually write this buffer, through their `output_ready` results. The last
Store must cover every writer in the context. Pure readers do not add
`output_ready` prerequisites; unrelated explicit control dependencies remain
legal. Produces one completion event. Stores to overlapping global ranges
retain WAW order; independent destination ranges are not implicitly serialized.

### 3.6 `nest.dispatch.tasks.async`

```mlir
%grid, %inrel, %out = nest.dispatch.tasks.async @pow_4k_tile
    l1_mode = 0 context = 1
    tasks(%t) globals() bindings(%buf) ins(%buf) outs(%buf)
    signal_policy {
      input_released = #nest.aggregate<all_tasks>,
      output_ready = #nest.aggregate<all_tasks>
    }
    depends_on(%pref)
    : (!nest.event<"grid">, !nest.event<"inrel">, !nest.event<"out">)
```

The tile program is referenced by symbol. All four groups `globals(...)`,
`bindings(...)`, `ins(...)`, and `outs(...)` are mandatory and print even when
empty. `globals` binds global-view formals; `bindings` is the sole positional
binding for all L2 formals. Each count and shape/dtype must match exactly.
Each L2 actual must be a `nest.alloc` result in this context body.
Placement comes from the enclosing `nest.context`, not this op.

`l1_mode` is mandatory and must belong to the referenced Tile Program's
`allowed_profiles`. It is the dispatch baseline request, not a mutable runtime
override. The compiler may resolve the call to the already-active compatible
L1 mode; the requested and resolved modes are both frozen in the executable.

`ins` and `outs` must each be duplicate-free subsets of `bindings`, exactly
equal to the program's actual read/write effects mapped to actual buffers.
Reads come from `tile.load.async` sources and writes from `tile.store.async`
destinations, through current-program `tile.subview` operations. Global
Gather accesses do not contribute L2 effects. Extra or missing declarations
are rejected; effect order is irrelevant.

Aliases in `bindings` are legal: effects merge by actual SSA identity, with
one pin per task and actual buffer. An unused L2 formal remains bound but
contributes no effect, pin, or required phase. Lowering canonicalizes effect
tuples in first-occurrence binding order; it never infers positions from
read/write direction.

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
When the pinned context is occupied, that Tile waits with `WAIT_SLOT`;
other Tiles and eligible Routes may continue. The legal range is
`0..SimConfig.context_count-1`; compilation and loading reject an
out-of-range static pin instead of allowing a runtime deadlock.

### 3.7 `nest.collective.async`

```mlir
%ev = nest.collective.async "reduce" bytes = 65536 mask = 15 : !nest.event<"ev_col">
```

Collective engine op (reduce/broadcast/multicast). Produces one event.

### 3.8 `nest.release`

```mlir
nest.release %buf depends_on(%reader_inrel, %prefetch_ev, %store_ev)
```

Invalidates the Context-owned L2 view after its required closure. Local
`nest.alloc` defaults to `sharing="private"`; an allocation may opt into
`sharing="readonly"` only when it is exported through exactly one
`nest.publish`:

```mlir
%published = nest.publish %weight depends_on(%prefetched) : !nest.event<"published">
%shared = nexus.shared.ref %producer_done slot = "W" : !nest.l2_buffer<64x64xbf16>
```

`nest.publish` seals the initialized backing after the producer's last access.
Its direct dependencies cover every prefetch and Store completion, reader
`input_released` and writer `output_ready` for that buffer; publication
requires full-view initialization. A full-buffer prefetch establishes this
by destination coverage and equal byte count even if its HBM source view has
a different shape. A published backing is immutable. Producer release revokes
only its own view, not any declared reader claim. `nexus.shared.ref` identifies
one producer submit instance and
export slot; it creates no physical allocation. Imported `!nest.l2_buffer`
formals are read-only, follow all HBM global formals, and must match each
submit actual's shape and dtype exactly.

For a local private allocation, the dependency SSA set is exactly
`R(buffer) ∪ P(buffer) ∪ S(buffer)`:

- `R`: `input_released` of every distinct dispatch that actually reads it.
- `P`: every prefetch completion into this allocation.
- `S`: every HBM Store completion from this allocation, not just the last.

An exported readonly allocation additionally depends on its publish event.
An imported readonly view requires its `input_released` and every Store
completion that reads it as a source. No duplicate dependencies,
`grid_done` substitutions, or reader `output_ready` substitutions are allowed.
Examples list readers by dispatch ordinal, then prefetches and Stores in
source order. Verification treats dependencies as a set; lowering preserves
author order and appends inferred bind/hazard dependencies without duplicates.
An input buffer with no asynchronous use may have an empty dependency set.
Private role `"in"` forbids Tile writes; private `"out"`/`"inout"` still require
a real Tile writer and at least one HBM Store. A published readonly
`"out"`/`"inout"` export may instead feed
readers directly without an intermediate HBM Store.

Every local allocation has exactly one release before `nest.return`; no
binding, prefetch, Store, or other buffer use may appear after it. Runtime
preflights events, owner, role, Arena/allocation/Profile generations,
reader/writer phases, remaining pins, and in-flight transactions before
mutation. Failed preflight cannot partially sweep writers. Every imported
formal is released exactly once; use after release, duplicate release,
wrong producer/slot, or a writable use is rejected.

A successful L2 release is a **permanent forfeiture** for this invocation:
the owner view is invalidated and the owner loses all access rights. A shared
physical backing remains allocated while any producer ownership, `DECLARED`
or `BOUND` reader claim, borrower view, pin, or accepted transfer remains.
Claims are keyed by consumer submit binding and local formal and transition
`DECLARED → BOUND → RELEASED`; only fault/reset cleanup may cancel a claim.
Once producer ownership is revoked, all claims are terminal, all views are
released, and pins/in-flight transactions have drained safely, the unique
backing finalizer returns its complete stripe-rounded padded span (not just
valid bytes) to the same L2 free map. The pool version moves only at that
physical final-free. `nest.barrier` never re-grants a forfeited owner's
rights or re-binds a released L2 address. Same-profile root admission may be
woken by that real final-free; a different L2 profile still waits for the
full root completion frontier. The L1 Task Arena contract is unchanged:
`tile.free`/view release never returns L1 extents, and only Task Arena
retirement does.

### 3.9 `nest.await`

```mlir
nest.await %grid, %store
```

An explicit frontend submission fence for the named events. Earlier queued
actions keep executing while the frontend waits; ordinary descriptor
dependencies do not create this fence.

### 3.10 `nest.barrier`

```mlir
nest.barrier
```

A context-local full-prefix completion fence. It waits for every earlier
registered/inflight action, including local release, before allowing later
registration. It is not a zero-cycle no-op, cross-context barrier, or an
oversubscribed Tile task barrier.

### 3.11 `nest.return`

```mlir
nest.return
```

Closes context submission and signals `completion_event` only after its
prerequisites complete. CPU completion additionally requires queued/inflight
actions, grids and phase events, final HBM store and resource cleanup to drain.
Reaching the final PC alone never denotes completion.

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

Declares a Tile-local L1 buffer. `shape`/`dtype` must match the result type;
`alignment` is optional. The compiler places all declarations in the Task's
striped `ArenaLayout`, with proven non-overlapping lifetimes eligible for
static Slot/offset reuse. Task admission reserves the whole Arena atomically.
When the Tile PC reaches `tile.alloc`, `ALLOC_L1` binds that buffer's
valid-byte view to its compiled Slot. This is not a dynamic pool allocation.

#### 4.2.1 `tile.free`

```mlir
%loaded = tile.load.async %view into %scratch : !tile.event<"loaded">
tile.await %loaded
// All uses of scratch must be complete before this instruction.
tile.free %scratch
```

Synchronous one-operand invalidation of a current-program `tile.alloc` view.
There is no result event, type suffix, implicit wait, or `depends_on` group.
Free consumes a normal UCE instruction issue but does **not** return the
parent Task Arena's extents or R lease. Thus another owner cannot gain
capacity merely because one local view was freed.

Before free, every preceding load destination, Store source, or Gather
indices/destination access to that allocation must have been awaited by SSA
event identity. Unrelated asynchronous memory operations do not block it.
BOA/EVU/Pow currently have opaque timing descriptors without L1 operands, so
all preceding such compute events must also have been awaited. Later explicit
load/store/Gather use, double free, forward/foreign references, and L2/global
operands are rejected. Freeing an unused local allocation is legal.

Runtime preflights owner, Tile/UCE/Task/launch identity, Arena/allocation/
Profile generations, pins, active/shadow Frame binding, queued/active engine
accesses, and unfinished transfers. It invalidates only the named Frame Slot
view; other Slots and the Frame generation remain. Timing-only still enforces
the logical lifetime. The whole Task Arena and R lease return only after safe
Task retirement or confirmed cancellation; unfreed views are invalidated
during that retirement.

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

### 4.8 `tile.gather.global.async` / `tile.profiled.access`

```mlir
%done = tile.gather.global.async %table
    indices(%indices_l1) into %gather_dst
    result_bytes = 128 cache_target_bytes = 65536 l1_mshr_hint = 16 {
  tile.profiled.access id = "r0" outcome = "L1_HIT"
      bytes = 64 line = "line0"
  tile.profiled.access id = "r1" outcome = "HBM_MISS"
      bytes = 64 line = "line42" merge = "line42"
} : !tile.event<"gather_done">
```

Deterministic profiled Gather. Operands are exactly one global-view
formal source, one L1 indices allocation, and a different L1 destination
allocation. The profile region is single-block, has no terminator, and
contains only `tile.profiled.access`; it has no control flow, event, or
side-effect op. Access properties print in fixed
`id/outcome/bytes/line/merge` order. Outcomes are `L1_HIT`, `L2_HIT`, or
`HBM_MISS`; `line` and `merge` are opaque profile identities, never
addresses or bank selectors.

All requests issue lookup legs concurrently. Responses may complete out
of order, but destination L1 writes follow profile ordinal order.
`gather_done` fires exactly once after the final destination write.
`HBM_MISS` uses per-tile L1 MSHR plus shared TileGroup L2 MSHR; one
non-empty merge group has one leader and waiter completions. The result
exists only in the explicit L1 destination; Gather creates no L2 output
and no StreamQueue token.

`cache_target_bytes` is a non-negative performance hint for the shared Cache;
it is not a minimum or a private quota and is not added across Contexts.
Required access/bypass capability belongs in the Tile/Context resource
contracts. Cache sizes do not infer or rewrite source-authored outcomes, and
Gather accepts no separate minimum-cache property.

### 4.9 `tile.await`

```mlir
tile.await %ev1, %ev2
```

Waits for one or more tile events. Suspends only the current Hardware
Context (reference.mlir §375-376). Lowered to `WAIT` (1 operand) or
`WAITALL` (2+ operands).

### 4.10 `tile.signal`

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

Each phase occurs at most once in the straight-line program. A real L2
reader/writer must emit its corresponding phase. Before input release,
every preceding L2 load completion must have been awaited by SSA identity,
and no L2 load may follow it; output readiness seals stores symmetrically.
Waiting on compute or another event cannot substitute for transfer completion.
The phases may occur in either order. Explicit empty phases are allowed;
their policy/tag declarations still match the emitted phase set exactly.

The dispatch's `signal_policy` selects the declared phases. For each phase,
the event fires exactly once only after every expected logical Task in that
`GridInstanceId` has signalled. A duplicate signal in a live Grid is a
protocol fault; a signal from a retired generation is stale and ignored.
Physical Tile and UCE context IDs are not aggregation identities.

### 4.11 `tile.return`

```mlir
tile.return
```

Tile program completion. Contributes to `grid_done`. Public Tile Programs
are single-block, straight-line bodies with exactly one terminal return;
memory accesses or signals after an early return are rejected. This does not
restrict the private execution IR's branch/stream instructions.

## 5. Verification and Compilation Rules

### 5.1 Source/module rules

- **Standalone**: exactly one `nest.context`, no `nexus.program`.
- **Model**: exactly one `nexus.program` plus one or more `nest.context`
  definitions.
- Zero or more uniquely named `tile.program` definitions; no other top-level
  operations.
- Every Context and Tile Program has its mandatory typed resource contract.
- Source verification is necessary but not sufficient for execution: static
  layout/Profile/control analysis occurs in `compile_program`, and target/
  binding/executable verification occurs again in `load_program`.

### 5.2 Inputs, bindings, views, and Gather

- Every `nexus.program` and `nest.context` block argument is a shape-typed
  `!nest.global_memref`; Device entry arguments have non-empty names.
- Submit arity and dims/dtype exactly match the referenced Context.
- A Tile Program starts with `!nest.task`, followed by a contiguous
  global-view prefix and then a contiguous L2-buffer suffix.
- Dispatch `globals` and `bindings` positionally cover all corresponding
  formals. `ins`/`outs` are exact duplicate-free actual read/write sets;
  aliases merge effects and unused formals remain bound without pins.
- Every view dimension has non-negative offset, positive size, and stays
  inside its backing shape. V1 strides are all 1; view chains are unsupported.
  Transfer endpoints have identical derived byte counts.
- Gather source is a current-program global formal. Indices and destination
  are distinct live `tile.alloc` results. Its profile is non-empty and contains
  only `tile.profiled.access`; request IDs are unique, byte counts are
  positive/in-range, and their sum equals positive `result_bytes`.
  `cache_target_bytes >= 0` and `l1_mshr_hint > 0`; no separate minimum-cache
  field is accepted. Outcomes are exactly `L1_HIT`, `L2_HIT`, or `HBM_MISS`.
  A non-empty merge group is HBM-miss-only and all members have the same
  non-empty line token and byte size.
- Actual HBM bindings are Loader inputs, not source verification data. The
  Loader rejects missing/unknown names, insufficient ranges or permissions,
  forbidden alias/overlap, ranges beyond target HBM, and violations of the
  compiled binding guards.

### 5.3 Context body, phases, and release

- `context=N` is a non-negative Group-slot affinity and is checked against the
  static Group capacity.
- `logical_tasks` equals the sum of all dispatch range counts. Every non-empty
  range count equals `popcount(placement)`; a range itself cannot be empty.
- Every dispatch references a defined Tile Program, supplies mandatory
  `l1_mode`, and requests a mode in that program's `allowed_profiles`.
  `context=N`, when present, is below `SimConfig.context_count`.
- Group event tags are unique, except empty result tags for omitted phase
  results. Dependencies refer to earlier SSA events owned by this Context.
- `signal_policy` exactly matches the Tile Program's emitted phase set and
  uses only `#nest.aggregate<all_tasks>`. A declared phase has a non-empty
  result tag; an omitted phase has an empty result tag.
- Each Store depends on every earlier real writer's `output_ready`; the last
  Store covers all writers. Each allocation has exactly one release before
  return, with exactly `R(buffer) ∪ P(buffer) ∪ S(buffer)` (§3.8). Input-role
  Tile writes are forbidden; `out`/`inout` requires a real writer and HBM
  Store. No use follows release.
- An ordinary `nest.await` references earlier events and remains a local
  frontend fence. It is not rewritten into a global barrier.

### 5.4 Tile Program body and lifetimes

- Tile event tags are unique and every `tile.await` references earlier
  same-program events.
- Each real L2 read/write direction emits exactly one corresponding
  `tile.signal`; the signal operand is block argument 0. All prior transfers
  in that direction are awaited, and no later transfer in that direction is
  allowed.
- L1 load/store/Gather operands are earlier live current-program
  `tile.alloc` results. `tile.free` obeys §4.2.1 and no explicit use follows it.
- Exactly one terminal `tile.return` is required. Public source Tile Programs
  remain single-block and straight-line.

### 5.5 Resource, Profile, and finite-control proof

- Context `allowed_profiles` is non-empty/unique and contains `l2_mode`;
  Tile `allowed_profiles` is non-empty/unique and contains every referencing
  dispatch's `l1_mode`. Every mode exists in the embedded Registry.
- The compiler derives deterministic all-bank striped L1/L2 layouts. Declared
  bytes must cover logical bytes plus padding and align to the Profile quantum.
  Every advertised mode is checked per bank and against the empty pool.
- `requested_contexts_per_tile` is positive, no larger than the physical Tile
  context count, and its `R × max(child per-bank Arena)` envelope fits every
  allowed L1 Profile. It is never reduced automatically.
- Cache capability declarations cannot understate actual access. Required hits
  need nonzero Cache; disabled Cache paths need target and contract bypass
  support. `read_write` requires a write-back target. `target_bytes` hints do
  not reserve capacity.
- Frame Slot demand, maximum live Grid Routes, event live frontier (not all
  historical events), engine/control queues, program SRAM, and finite Device/
  Group capacities are checked against the same `HardwareConfig`/`SimConfig`
  used to compile.
- The compiler validates every actual permitted L1/L2 combination. Independent
  allowed sets do not imply their Cartesian product is legal at runtime.

### 5.6 Device body

- Submit symbols/signatures are valid; Device event tags are non-empty and
  unique.
- Await and `depends_on` operands are unique earlier Nexus SSA results.
- Overlapping cross-root global ranges with a writer receive a transitive
  dependency or prior await. Each call site is specialized from an unmodified
  source template, so aliases from one call cannot pollute another.
- The body ends in `nexus.return` and contains no other source operations.

## 6. Compiler and Immutable Executable

### 6.1 `CompiledProgram`

The public compiler call is:

```python
compile_program(
    module: ModuleOp,
    hw: HardwareConfig,
    sim: SimConfig,
    *,
    binding_assumptions: Mapping[str, GlobalBinding] | None = None,
    source_name: str = "<memory>",
    workload_info: WorkloadInfo | None = None,
) -> CompiledProgram
```

The result has compiled schema 2 and compiler ABI `v1`. It contains canonical
`source_ir`/`source_hash`, embedded `ProfileRegistry`/`registry_hash`, static
`target_hash`, `artifact_hash`, `"standalone"|"model"` entry kind, immutable
entry/prefix, call bindings, explicit relocations, instruction source map,
dependency proofs, entry/exit Profile states, static effects/resource budgets,
binding guards, shared readonly-import claims, and `WorkloadInfo`.

All executable dataclasses are frozen; sequences are tuples and mappings are
deep immutable. Program IDs are positive deterministic first-appearance IDs.
Program/content/artifact/target/Registry identities are canonical SHA-256
digests; the 256-bit program hash is encoded as a 64-hex-digit JSON string.
Dynamic launch IDs, physical addresses, Slot assignment, and runtime
generations are not baked into content identity.

Serialization is strict JSON with an explicit type/opcode/enum allowlist:

```python
text = serialize_compiled_program(compiled)
compiled_again = parse_compiled_program(text)
loaded = load_program(compiled_again, hw, sim, actual_bindings=bindings)
```

Unknown/missing fields, unknown schema/ABI, duplicate keys, non-finite numbers,
invalid enums or integer domains, identity mismatches, and artifact-hash
mismatches are rejected. Parsing does not use pickle, `eval`, dynamic import,
or source recompilation.

### 6.2 Source references, dumps, and relocations

Every executable Device, Group, and Tile instruction has a stable
`instruction_id` and `SourceRef(source_name, symbol, body_op_index, op_name)`.
Generated instructions additionally carry `generated_by` and a reason. xDSL
does not provide source line numbers here, so the compiler never invents them.

`dump_executable_ir(program)` is a complete strict package/executable dump.
The CLI also writes a generic `.compiled.mlir.txt` view where generated
ordinary awaits, `profile.reconfig`, and `memory.maintenance` appear as actual
operations with source references—not comments or runtime inference.

Launch instantiation applies only declared relocations (event namespaces,
queues, actual HBM bindings). It uses immutable replacement objects and never
writes back into the shared template or guesses relocations from event names.

### 6.3 Source operation mapping

| Source operation                       | Immutable executable operation                                |
| -------------------------------------- | ------------------------------------------------------------- |
| `nest.alloc` / `nest.publish`          | `BIND_L2_VIEW` / `PUBLISH_L2` with internal completion events |
| `nexus.shared.ref`                     | submit-time claim metadata; imported formal `BIND_L2_IMPORT`  |
| `nest.subview`                         | immutable `ExecMemoryView`                                    |
| `nest.task.range`                      | immutable `ExecTaskDomain`                                    |
| `nest.dma.prefetch.async`              | `DMA_PREFETCH` + `ExecTransfer`                               |
| `nest.dma.store.async`                 | `DMA_STORE` + `ExecTransfer`                                  |
| `nest.dispatch.tasks.async`            | `DISPATCH_ROLE` + `ExecDispatchRequest`                       |
| `nest.collective.async`                | `COLLECTIVE_RUN`                                              |
| `nest.release`                         | `RELEASE_L2` + structured `ExecReleaseRequest`                |
| `nest.await`                           | one `WAIT_EVENT` per operand                                  |
| `nest.barrier`                         | `BARRIER_GROUP`                                               |
| `nest.return`                          | `SIGNAL_EVENT`                                                |
| `tile.alloc`                           | `ALLOC_L1` with compiled buffer/layout index                  |
| `tile.free`                            | `FREE_L1`                                                     |
| `tile.load.async` / `tile.store.async` | `LAUNCH_MFE` + `ExecTransfer`                                 |
| `tile.gather.global.async`             | `LAUNCH_GATHER` + immutable `ExecGatherDesc`                  |
| `tile.pow.async` / `tile.evu.async`    | `LAUNCH_EVU`                                                  |
| `tile.boa.async`                       | `LAUNCH_BOA`                                                  |
| `tile.await`                           | `WAIT` or `WAITALL` over the original Tile events             |
| `tile.signal`                          | `SIGNAL_PHASE`                                                |
| `tile.return`                          | `RET`                                                         |

WAIT/WAITALL mechanically preserve the named events. Release dependencies,
access effects, dispatch phases, requested/resolved modes, and binding identity
are structured fields; runtime does not reconstruct them from names.

### 6.4 Profile binding and generated synchronization

The compiler starts from target reset modes. For each Context it keeps the
current L2 mode when allowed, otherwise selects the declared `l2_mode`.
For each dispatch it keeps the current L1 mode when allowed, otherwise selects
the dispatch's requested `l1_mode`. A kept non-baseline mode is COMPATIBLE;
equal requested/current is SAME. The resolved cross-layer combination is
frozen in `CallBinding.permitted_profiles`.

An internal L1 change requires a finite prior Grid frontier. The compiler
preserves source `nest.await` and inserts missing ordinary waits for every
still-live `grid_done`, followed by explicit L1 configuration. A root
containing an internal L1 transition becomes L1-exclusive at Device level:
the compiler drains other L1 producers before it and prevents a new producer
until it retires. L2 transitions occur only at Device/root boundaries after
all prior L2 roots retire. When both layers change, control order is L2 then
L1.

For an HBM writer followed by an overlapping cached reader, the compiler emits
range-qualified maintenance after the writer dependency. A write-back target
also emits the required pre-clean so an older dirty line cannot overwrite a
new producer. If the target lacks range invalidation, the compiler uses a
full-domain range only where the source control flow supplies a legal
quiescent boundary; otherwise compilation fails.

The Loader independently derives effects from descriptors, views, and roles.
Deleting a required dependency, ordinary await, configuration step, or
maintenance command remains invalid even if an attacker recomputes the
artifact hash. Extra legal dependencies are allowed; the Loader does not
require textual equality with one optimizer output.

## 7. Runtime Scheduling, Signals, and Admission

### 7.1 Phase aggregation

A signal is keyed by `(root launch generation, GridInstanceId, phase,
logical Task)`. A valid first signal is recorded; when the seen Task IDs equal
the expected set, its aggregate event fires exactly once. A duplicate in a
live Grid faults the owner without recounting. A retired-generation signal is
stale and ignored. Physical Tile/UCE context IDs are not part of the
aggregation identity.

`input_released` may remove that Task's pure-read L2 pin only after the Tile
Program has awaited its last real load. `output_ready` similarly proves the
last real L2 Store. Neither phase means the Task, Grid, or parent Arena has
retired.

### 7.2 Root admission

`GroupPortAdapter.try_submit` accepts only bounded pending metadata
(`group.context_pending_capacity`). It does not pre-occupy a Group hardware
slot, L2 Arena, Event Table reservation, UCE context, or engine queue.

For each active L2 Profile, ready roots have independent SAME and COMPATIBLE
FIFO heads. SAME is attempted first; if its head cannot fully commit, the
COMPATIBLE head may fill, but no request may bypass the head of its own
category. Full root admission atomically commits a vacant/pinned Group slot,
private L2 Arena, imported readonly views/claims, event/control budget, and
launch state. A shared import aliases the producer's backing and adds no
physical reservation. Lowering adds the producer-submit completion dependency
to each consuming submit, so reader admission occurs only after producer
completion and publish; runtime adds no shared-readiness wait queue. Failed
admission leaves arenas, claims, views, and pool version unchanged. Dormant
claim-manifest entries are metadata, not live capacity or profile-quiescence
resources. Empty-pool impossible requests are permanent errors, not waiters.

### 7.3 Grid Route and per-Tile Task admission

`DISPATCH_ROLE` registers one bounded Grid Route and consumes a dispatch credit
until the complete Grid retires. It does not gang-commit Tile resources. Each
Tile may commit at most one candidate per Tick, comparing only its SAME and
COMPATIBLE FIFO heads. A blocked Tile does not roll back already committed
Tasks on other Tiles.

One Task commit atomically covers its L1 Arena, UCE pin/Slot, Frame/control
state, parent L2 pins, and R lease. Observable wait classes are
`WAIT_CAPACITY`, `WAIT_FRAGMENTATION`, `WAIT_SLOT`,
`WAIT_CONTROL_RESOURCE`, and `WAIT_CONTEXT_LIMIT`. Total free bytes do not
erase a stripe-fragmentation failure. A Grid cannot complete while any
selected Task is pending admission, active, or retiring.

The R ledger key is `(parent binding_id, launch_generation, tile_id)`.
`requested_contexts_per_tile` bounds committed Tasks across all Grids of that
parent. The lease is returned only at safe Task retirement or confirmed
cancellation—not at local free, input release, output readiness, or cancel
request.

## 8. Arena and View Lifecycle

`ArenaPool` owns one physical L1 Tile pool or the Group L2 pool.
`RootInvocation` owns an L2 Arena; `TaskIdentity` owns an L1 Arena. Planning is
pure and records the pool/Profile versions. Commit atomically consumes exact
per-bank extents and mints an `ArenaHandle` containing the whole reservation,
including padding.

`bind_view` creates an `AllocationHandle` whose segments contain only logical
valid bytes and carry both allocation and Profile generations. Local
`invalidate_view` returns `False` while pins/in-flight users remain and
finishes later; it never changes the free map by itself. Repeated, stale,
wrong-owner, or out-of-generation operations are invariant failures.

**L2 physical extent and readonly-sharing lifecycle.** Each local L2
allocation has an independent, non-overlapping stripe-rounded padded span; the
compiler never re-binds a released L2 region. The independent executable
verifier rejects layouts whose padded spans overlap or whose spans plus slack
do not conserve every bank of the reservation. The Arena reservation is
committed once as exact per-bank units: one unit set per physical backing
(including tail padding) plus owner-arena slack. Aliased reader views never
increase physical capacity. Each run has a finite claim manifest keyed by
producer submit binding/slot and consumer submit binding/local slot; each
materialized claim transitions `DECLARED → BOUND → RELEASED`, with
`CANCELLED` reserved for fault/reset cleanup.

`nest.publish` seals a fully initialized readonly export after its final
producer access. Producer release revokes only the producer view; unsubmitted
and capacity-waiting reader claims keep the published bytes. Reader admission
creates its logical view without changing the free map or pool version. The
origin Arena may retire while a backing remains held by readers, so snapshots
report origin-retired backings separately rather than counting them as live
root Arenas. `arena_reserved_bytes`/`allocated_bytes` count live padded
backings plus held Arena slack. `physical_live_backing_bytes` counts each live
backing's padded units once and excludes slack. `live_view_bytes` counts each
live backing's logical bytes once, so it remains nonzero while a published
backing is retained by a DECLARED claim. Aliases do not increase it.
`logical_live_view_bytes` separately sums live view bytes and may count
aliases. Per bank, `allocated_bytes + free_bytes == user_spm_per_bank`.

The backing is final-freed exactly once only after producer ownership is
revoked, every claim is `RELEASED` or safely `CANCELLED`, all producer and
borrower views are released, and pins/in-flight accepted transfers are zero.
The finalizer returns the full padded units and notifies the ordinary
release-driven capacity retry; it never fabricates free space on closure
failure. Claims, backings, views, pins, and transfers are run-scoped and cannot
cross Group/L2 Profile epochs.

For a capacity-blocked same-profile FIFO head, an optimistic first-fit proof
retains only materialized live backings with claims owned by the head, a later
same-class request, or a submit that cannot precede the head's completion
(including a device `await` frontier). If the head still cannot fit per bank,
admission faults as `PERMANENT_CAPACITY`; otherwise it waits for capacity or
aligned contiguous space without skipping a same-class head. Borrowing and
non-final alias release do not advance the physical pool version.

**L1 keeps the original contract.** A Task Arena retires after all views,
Frame state, accesses, transactions, and the Task terminal event are safe; L1
`tile.free`/view release never returns extents to the free map. L1-only
reconfiguration uses only L1 pools and does not wait for or mutate shared L2
backings. A root L2 Arena retires after every Route/Task, required HBM output,
remaining local view, pin, transfer, and lease closes; only then can
`context_done` publish. No runtime optimization invents overlap at either
level.

## 9. Profile Controller, Maintenance, Cancellation, and Recovery

`ProfileController` is the unique writer for both layers. Device and Group
submit typed descriptors to the same instance; a busy controller returns
backpressure. Runtime command identity is
`(run_generation, owner_launch_generation, static_command_id)`, so a warm run
cannot reuse an old await or ACK.

A `ProfileReconfigDesc` executes exactly:

```text
ACQUIRE → CHECK_FRONTIER → CLOSE_ISSUE → DRAIN_REFERENCES
→ CLEAN_INVALIDATE → DRAIN_DOWNSTREAM → PREPARE → WAIT_READY_ACK
→ COMMIT → WAIT_COMMIT_ACK → OPEN_ISSUE → RELEASE
```

The frontier step consumes the compiled same-run ordinary-await proof.
Issue closes only after the proof succeeds; old completion traffic continues
to progress. Every real member `(level, pool_id, bank_id)` maintains active
and shadow mode/generation. The controller publishes the new mode and
increments only that layer's generation after every matching Prepare and
Commit ACK. Missing, duplicate, stale-generation, unknown-member, or failed
ACKs cannot open the gate.

An L2 Profile change additionally requires the complete compiled root
completion frontier for the old L2 epoch and a quiescent L2 backing/claim/view
ledger. A retired origin Arena is insufficient while a shared backing,
materialized pending/active claim, alias pin, or accepted transfer remains.
Publish, `input_released`, and an extent-free event are not substitutes for
the frontier. Source compilation and independent loading reject readers
assigned to a different L2 Profile epoch instead of waiting for a future
submit. An L1-only Profile change checks only its L1 domain and leaves shared
L2 backing state untouched.

A `MemoryMaintenanceDesc` executes exactly:

```text
ACQUIRE → WAIT_DEPENDENCIES → BLOCK_RANGE_ISSUE
→ DRAIN_RANGE_REFERENCES → CLEAN_INVALIDATE → DRAIN_DOWNSTREAM
→ ACK → UNBLOCK_RANGE_ISSUE → RELEASE
```

Range maintenance blocks only overlapping new requests. Existing/non-
overlapping traffic progresses. Dirty clean creates real downstream L1→L2 or
L2→HBM transactions and waits for them before ACK. Non-oracle Gather cache
entries retain allocation-qualified conservative source-view provenance;
`ByteStore` mode binds each request to its actual source byte range.

Initialization installs each layer's target `reset_mode`, generation 0, and
waits for all member ACKs before opening issue. `Simulator.run` requires the
actual modes to match the artifact's compiled entry state and never performs
an implicit reset or replan. Explicit recovery requires quiescent/isolated
domains, re-acknowledges every member, and advances generations monotonically.

Cancellation is not deletion. Unissued work may be withdrawn synchronously;
an accepted DMA/refill/member request remains `CANCEL_REQUESTED` until its real
completion or explicit isolation confirmation. Credits, addresses, pins,
Arenas, and gates are not reused early. Late old-generation returns are
isolated/faulted and cannot write a new SPM/cache generation.

## 10. Configuration and Target Binding

Hardware YAML uses `schema_version: 2`. `memory.target` and
`memory.profile_source` are typed subtrees. Duplicate/unknown keys, unknown
versions, incomplete explicit layer targets, illegal geometry, and duplicate
effective modes are rejected. Partial override YAML may omit its version and
inherits omitted subtrees; a custom layer `modes` table replaces the entire
table.

The bundled per-bank experimental modes are:

| Layer | Mode |    SPM |  Cache | System-reserved SPM |
| ----- | ---: | -----: | -----: | ------------------: |
| L1    |    0 |  65536 |      0 |                2048 |
| L1    |    1 |  57344 |   8192 |                2048 |
| L1    |    2 |  49152 |  16384 |                2048 |
| L2    |    0 | 524288 |      0 |                4096 |
| L2    |    1 | 458752 |  65536 |                4096 |
| L2    |    2 | 393216 | 131072 |                4096 |

For each layer/mode, SPM + Cache equals the fixed physical bank size and SPM,
Cache, and reservation align to
`lcm(partition_granule_bytes, cache_line_bytes)`. SPM includes the system
reservation. The active Profile is the only Cache-capacity source; lookup
latency and MSHR counts remain independent target controls. The bundled values
and 2,000,000-cycle Profile-command timeout are simulator experiments,
`由后续规格冻结`.

The compiled target fingerprint covers topology/storage and static simulation
feasibility: `context_count`, `device_context_count`, Device pending/completion
capacity, and Group active/pending/action/quota/scan/event/inflight/prefetch/
store/dispatch capacities. It excludes runtime-only trace, fidelity,
max-cycles, timing knobs, seed, and scheduler policy.

## 11. Fidelity and Byte Semantics

All fidelities enforce source/executable contracts, Profile/maintenance
control, gates, generations, and static budgets.

- `timing_only`: no physical allocation handles; transfers collapse to one
  timing leg.
- `runtime`: real HBM bindings and L1/L2 Arena/view ownership, but one
  collapsed transfer leg.
- `full_memory`: real bank segments and staged HBM/DMA/NoC/L2/L1 routes.

Only `full_memory` with an injected `ByteStore` proves byte movement. The
sparse store rejects uninitialized reads, validates seeded HBM coverage against
actual bindings, captures source data at read completion, and writes the
destination only at write completion. Byte-checked Gather additionally
requires `bind_profiled_source(binding_id, request_id, source_offset)`.

Without the oracle, Gather outcomes remain explicit source-authored timing
profiles. Cache capacity never predicts a hit rate. BOA/EVU/USE remain timing
models and do not establish tensor numerical correctness.

## 12. CLI Artifacts, Replay, and Observability

Source mode explicitly compiles and persists before loading/running.
`--compile-only` stops after persistence; `--compiled-output PATH` chooses the
JSON artifact; `--profile-bytes LEVEL:MODE=SPM:CACHE` changes an existing mode
only during source compilation. `--compiled-file PATH` independently parses,
loads, and runs without source or compiler import and is mutually exclusive
with those source options and `--print-ir`.

For `name.json`, the CLI writes `name.exec.txt`,
`name.compiled.mlir.txt`, and `name.target.yaml`. The target snapshot contains
the complete `HardwareConfig` only. It does not store `SimConfig`; replay must
repeat the static simulation capacities listed in §10 with matching
`--context-mode`, `--device-context-mode`, and `--sim-override` values.

Trace includes `profile_command`, each `profile_step`, member request/ACK,
ordinary awaits, source references, Arena reserve/retire, view invalidation,
extent final-free, Task lease acquire/release, requested/resolved L1 mode and
generation, and change-only counters sourced from post-mutation ArenaPool
snapshots. Pool counters distinguish unique `live_backings`,
`pending_shared_claims`, `active_shared_references`, `live_view_bytes` (logical
bytes per live backing, deduplicated), and `logical_live_view_bytes` (the sum
of live logical alias views). `physical_live_backing_bytes` counts padded
backing units only and excludes slack; current `arena_reserved_bytes` includes
backings plus held Arena slack. Padding, system-reserved, Cache, and free
bytes are reported separately.
Aliased views count once toward physical reservation; per-bank
`allocated_bytes + free_bytes == user_spm_capacity_per_bank`. Reports include
backing/claim rows (including origin-retired backings) and protocol-live L2
byte totals deduplicated by backing identity. A zero-leak check covers
backings, claims, views, pins, and in-flight references. Bounded
`profile`/`arenas`/`task_leases` snapshots include `compiled_artifact_hash` and
`registry_hash`; `call_bindings` records requested/resolved L2 modes while
Profile snapshots expose active modes/generations.
