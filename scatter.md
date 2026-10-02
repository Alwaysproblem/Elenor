# Scatter 协议当前状态汇总（Scatter Assign / ScatterReduce Add）

- 日期：2026-10-02
- 范围：Scatter 类不规则写算子的 IR 语义、ADR-0001 destination claim 协议、形式化模型覆盖、周期模型覆盖与已测数据
- 不在范围：Gather、Transpose、GEMM 等其它算子；Gather 协议见其自身章节
- 证据标记沿用 AGENTS.md 7.2：`[EXPLICIT]` 文档/代码明确声明，`[INFERRED]` 合理推断，`[HYPOTHESIS]` 待验证，`[UNKNOWN]` 证据不足

---

## 1. 结论摘要

1. Scatter 在本仓库被拆成**两个语义不同的算子**：[EXPLICIT]
   - `Scatter Assign`（`ir_design/scatter.ir`）：非原子 `masked_writeback`，必须走 ADR-0001 的**运行时 destination claim 排他协议**。
   - `ScatterReduce Add`（`ir_design/scatterreduce.ir`）：允许重复 index，靠 **L1/L2 reduction cache 合并 + 最终 `atomic_reduce_add`** 解决跨 Context 冲突，**不使用 exclusive lease**。
2. 两个算子的**正确性机制不同且不可互换**：Assign 用排他 lease，Reduce 用原子 commit。任一方降级为另一种都会丢数据。[EXPLICIT] `docs/architecture/adr-0001-scatter-destination-claim.md:130,157-172`
3. 形式化模型已对 ADR-0001 做了**独立穷举验证**：[EXPLICIT] `models/formal/resource_contract_model.py:350-894`
   991 个可达状态、9 个终态、0 个 unsafe 状态、0 个无终态路径的非终态；12/12 formal 回归 PASS（本次在 `nest-cycle-model` 环境实测复现）。
4. 周期模型**没有**建模 claim/lease 协议本身，只建模了 scatter 的延迟抖动与一个 output-region 排他锁工作负载。[EXPLICIT] `models/cycle_model/nest_sim.py:106-107,1560-1566,1626-1639`
   因此 **cycle-model 的 scatter 性能数据不能用于评估 ADR-0001 的性能代价**（ADR-0001 自己的 Performance Impact 一节同样标注 `[UNKNOWN]`，见 `adr-0001-scatter-destination-claim.md:174-180`）。
5. 存在一处**跨文档不一致**：背景文档仍建议 Scatter V1 走 write-through / bypass L1 Cache，而当前 IR 把 L1 write-combine cache 定为 `admission = required` 正确性资源。详见 §8.1。

---

## 2. 语义定义

### 2.1 Scatter Assign

```text
Y[indices[i], feature_base : feature_base + 32] = Values[i, :]
```

[EXPLICIT] `ir_design/scatter.ir:1-8`

Context 属性：

| 属性              | 值                             | 来源               |
| ----------------- | ------------------------------ | ------------------ |
| `kind`            | `scatter`                      | `scatter.ir:48`    |
| `update`          | `assign`                       | `scatter.ir:49`    |
| `cardinality`     | `fixed`                        | `scatter.ir:50`    |
| `address`         | `runtime_index`                | `scatter.ir:51`    |
| `conflict`        | `unique`                       | `scatter.ir:52`    |
| `side_effect`     | `write_only`                   | `scatter.ir:53`    |
| `conflict_domain` | `context_group`                | `scatter.ir:54`    |
| `cache_contract`  | `l2 = required, l1 = required` | `scatter.ir:57-60` |

关键点：`conflict = unique` 的作用域只是**同一个 `context_id` 的 context group**（`scatter.ir:54`，`docs/architecture/ir_operator_rules.md:82-83`）。NEST 允许多个 L2 Context 并发执行，单 Context 内的 unique 约束**不能**阻止两个 Context 的非原子 masked writeback 互相覆盖。[EXPLICIT] `adr-0001-scatter-destination-claim.md:16-19`

### 2.2 ScatterReduce Add

```text
Y[indices[i], f : f + 32] += Values[i, f : f + 32]
```

[EXPLICIT] `ir_design/scatterreduce.ir:1-8`

| 属性                                              | 值                         | 来源                      |
| ------------------------------------------------- | -------------------------- | ------------------------- |
| `kind`                                            | `scatter_reduce`           | `scatterreduce.ir:48`     |
| `reduction`                                       | `add`                      | `scatterreduce.ir:49`     |
| `conflict`                                        | `merge`（重复 index 合法） | `scatterreduce.ir:52,137` |
| `side_effect`                                     | `read_modify_write`        | `scatterreduce.ir:53`     |
| `conflict_domain`                                 | `device`                   | `scatterreduce.ir:54`     |
| `input_type` / `accumulator_type` / `output_type` | `bf16` / `f32` / `f32`     | `scatterreduce.ir:57-64`  |
| `reproducibility`                                 | `relaxed`                  | `scatterreduce.ir:64`     |

---

## 3. 资源合同

### 3.1 Tile 私有资源（两算子相同）

[EXPLICIT] `scatter.ir:10-27`，`scatterreduce.ir:10-27`

```text
scratchpad_bytes = 4352B
  indices  256B   (64 × i32)
  values  4096B   (64 × 32 × bf16)
```

### 3.2 L2 Context 拥有的输入 buffer

[EXPLICIT] `scatter.ir:73-97`

| Buffer        | shape            | type | 对齐 | 字节   |
| ------------- | ---------------- | ---- | ---- | ------ |
| `%l2_indices` | `[2, 64]`        | i32  | 64B  | 512B   |
| `%l2_values`  | `[2, 2, 64, 32]` | bf16 | 256B | 16384B |

两算子的 L2 prefetch 总量一致：512B + 16KiB = 16896B（`scatter.ir:41` 的 `prefetch_scratchpad_bytes`）。`output_scratchpad_bytes = 0B`——Scatter 的输出不落 L2 output scratchpad，直接经 cache 层级写回 global。[EXPLICIT] `scatter.ir:39-42`；规则见 `docs/architecture/ir_operator_rules.md:149-151`

### 3.3 L2 write-combine cache（Assign）

[EXPLICIT] `scatter.ir:106-140`

```text
level            = l2
scope            = context
role             = scatter_write_combine
data_capacity    = 256 KiB
line_bytes       = 64
associativity    = 4
banks            = 8
write_combine_entries = 32
writeback_entries     = 16
write_policy     = write_back
partial_line     = masked_write
sharing          = all_tasks_in_context
admission        = required
```

### 3.4 L2 reduction cache（Reduce）

[EXPLICIT] `scatterreduce.ir:111-148`

```text
data_capacity      = 256 KiB
line_bytes         = 128          （比 Assign 大：f32 累加器）
associativity      = 4
banks              = 8
conflict_entries   = 64
reduction_entries  = 32
writeback_entries  = 16
storage_mode       = delta        （只存增量，不在分配时读 Y 原值）
duplicate_policy   = merge
commit_policy      = atomic_reduce_add
admission          = required
```

### 3.5 Distributed L1 cache

|                                            | Assign                      | Reduce                      |
| ------------------------------------------ | --------------------------- | --------------------------- |
| scope                                      | `dispatch`                  | `dispatch`                  |
| sharing                                    | `hardware_contexts_on_tile` | `hardware_contexts_on_tile` |
| data capacity / tile                       | 16 KiB                      | 32 KiB                      |
| line_bytes                                 | 64                          | 128                         |
| associativity                              | 2                           | 2                           |
| write_combine / reduction entries per tile | 16                          | 16                          |
| eviction entries per tile                  | 8                           | 8                           |
| admission                                  | `required`                  | `required`                  |

[EXPLICIT] `scatter.ir:149-171`（Assign）、`scatterreduce.ir:157-190`（Reduce）

不规则写资源**不复用** Gather 的 `l1_cache_hint` / `l1_mshr_hint`，且共享 cache 容量**不计入** `scratchpad_bytes`，避免同一物理资源被重复预约。[EXPLICIT] `docs/architecture/ir_operator_rules.md:68-81`

`writeback_entries` / `eviction_entries` 是**保留 drain path 的正确性资源**，普通 update 不得消耗其最后一份保留容量。[EXPLICIT] `docs/architecture/ir_operator_rules.md:99-100`；`background/spmd_ai_inference_accelerator_design_proposal.md:763`

---

## 4. 执行流程

### 4.1 Scatter Assign 成功路径

[EXPLICIT] `adr-0001-scatter-destination-claim.md:51-69`，IR 实现 `scatter.ir:73-412`

```text
 1. 创建 L2 Context #N
 2. nest.alloc %l2_indices / %l2_values              scatter.ir:73,86
 3. nest.cache.reserve %Y (L2 write-combine)           scatter.ir:106
 4. nest.cache.reserve.distributed %Y (L1 per Tile)    scatter.ir:149
 5. nest.prefetch.async indices / values（相互独立）    scatter.ir:177,193
 6. %indices_ready 后扫描完整 indices，验证 duplicate/OOB   scatter.ir:234-240
 7. 根据完整 feature range 生成 destination cache-line set  scatter.ir:225-229
 8. nest.destination.claim.async 原子获取 device-scope exclusive lease  scatter.ir:220-274
 9. %line_claim_ready 后 dispatch                     scatter.ir:346-349
10. Tile 复制 inputs，tile.signal indices/values_released  scatter.ir:547-548
11. %updates_issued 后 L1→L2 flush（overwrite_unique）    scatter.ir:371-380
12. 释放 distributed L1 cache                           scatter.ir:397-399
13. L2→global masked_writeback                          scatter.ir:386-395
14. %store_done 后释放 L2 cache 与 destination lease      scatter.ir:401-406
15. nest.await %grid_done, %store_done → context.return  scatter.ir:410-412
```

Claim 粒度是 **cache line**，`access = exclusive_write`，`overlap = serialize`；line set 不重叠时允许并发，重叠时后到 context 在 `nest.destination.claim.async` 处等待。[EXPLICIT] `scatter.ir:228-233`；`adr-0001-scatter-destination-claim.md:60-61`

lease owner 是 **L2 `context_id`**，不是 Tile ID 或 Tile hardware-context ID。[EXPLICIT] `adr-0001-scatter-destination-claim.md:47,123`

`nest.context %ctx#N` / `nest.dispatch.* context_id(N)` / `nest.destination.claim.async context_id(N)` 三者必须使用同一数值 ID。[EXPLICIT] `adr-0001-scatter-destination-claim.md:39-45`

### 4.2 Claim 失败路径（`duplicate=reject` 或 `out_of_bounds=trap`）

[EXPLICIT] `adr-0001-scatter-destination-claim.md:76-89`，IR 实现 `scatter.ir:241-271`

```text
1. nest.await %indices_ready, %values_ready     （values prefetch 可能在飞，不允许提前释放）
2. nest.destination.cancel %partial_lease      （幂等，空 partial lease 也安全）
3. nest.cache.cancel.distributed %l1... {require_clean = true}
4. nest.cache.cancel %l2...          {require_clean = true}
5. nest.release %l2_indices / %l2_values
6. nest.context.abort %claim_error
7. success-only dispatch continuation 被抑制
```

失败 region 运行时 dispatch 尚未开始，因此两级 cache 必须没有 dirty entry；`require_clean` 失败是**协议错误**，不能静默丢弃更新。[EXPLICIT] `adr-0001-scatter-destination-claim.md:88-89`

`%line_claim_ready` 只在成功时产生；`on_failure` 终止 context，不能留下永久等待的 dispatch。[EXPLICIT] `docs/architecture/ir_operator_rules.md:95-96`

### 4.3 ScatterReduce Add 流程

[EXPLICIT] `scatterreduce.ir:71-334`

与 Assign 的差异只有四处：

1. 无 `nest.destination.claim.async`，dispatch 只依赖 `%indices_ready, %values_ready`（`scatterreduce.ir:271-273`）。
2. L1 flush 用 `merge_policy = reduce_add`（`scatterreduce.ir:296-298`）。
3. L2 flush 用 `commit_policy = atomic_reduce_add`，`accumulator_type = f32, destination_type = f32`（`scatterreduce.ir:313-320`）。
4. 释放链：L1 依赖 `%l1_merge_done`，L2 cache 依赖 `%store_done`；无 lease 释放（`scatterreduce.ir:323-329`）。

`atomic_reduce_add` 是**正确性要求**，不是优化：若最终 global flush 用普通 writeback，两个 L2 Context 对同一 destination 的 delta 会丢失。[EXPLICIT] `docs/architecture/ir_operator_rules.md:391-392`；`background/spmd_ai_inference_accelerator_design_proposal.md:1405-1408`

### 4.4 Tile Program

[EXPLICIT] `scatter.ir:415-605`，`scatterreduce.ir:336-535`

```text
%task_id      = tile.task.id %task                 （逻辑坐标，与物理 Tile ID 无关）
%update_tile  = task_id /u feature_tiles           scatter.ir:462-464
%feature_tile = task_id %u feature_tiles           scatter.ir:466-468
%feature_offset = %feature_base + %feature_tile*32  scatter.ir:470-476
%l1_scatter_cache = tile.cache.local %set          （同 Tile 所有 Hardware Context 共享同一 handle）
tile.load.async L2 subview → %l1_indices / %l1_values（context_private = true）
tile.signal indices_released / values_released     （只挂起本 Hardware Context）
tile.scatter.async ... / tile.scatter_reduce.async ...
tile.signal updates_issued
tile.return
```

- Assign：`update = assign`、`conflict_policy = unique`、`duplicate_policy = reject`、`out_of_bounds = trap`、`ordering = unordered`、`cache_policy = write_combine`、`result_cardinality = 64`。[EXPLICIT] `scatter.ir:564-598`
- Reduce：`reduction = add`、`conflict_policy = reduce`、`duplicate_policy = merge`、`out_of_bounds = trap`、`cache_policy = reduce_and_accumulate`。[EXPLICIT] `scatterreduce.ir:486-529`
- `tile.scatter.async` 的完成事件只表示 values 已被**不规则 cache 层级接受**，不表示已写回 global。[EXPLICIT] `scatter.ir:560-561`
- 逻辑 task 域为 `nest.task.range 0 to 4`，即 4 个逻辑 task；`program_image = shared`、`tile_scheduler = multi_context`、`max_active_contexts_per_tile = 4`。[EXPLICIT] `scatter.ir:280-282,320-332`

### 4.5 完成事件映射

[EXPLICIT] `docs/architecture/ir_operator_rules.md:394-402`；`background/spmd_ai_inference_accelerator_design_proposal.md:653-663`

| IR 事件                      | 标准阶段                       | 语义                                                                                       |
| ---------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------ |
| `indices_released`           | `INPUT_RELEASED`（按输入拆分） | 所有 Tile Context 已复制 `%l2_indices`                                                     |
| `values_released`            | `INPUT_RELEASED`（按输入拆分） | 所有 Tile Context 已复制 `%l2_values`                                                      |
| `updates_issued`             | Scatter 专用中间事件           | 所有 update 已进入 L1/L2 cache 层级，**尚未 global visible**                               |
| `grid_done`                  | `TASK_DONE`                    | 所有 Tile Program 已退役，可回收 Tile 执行状态                                             |
| `store_done`                 | `OUTPUT_READY`                 | 最终 L2 flush 完成，global destination 已可见                                              |
| `claim_error` / `on_failure` | `CONTEXT_ABORT`                | 等待 in-flight prefetch 并回收全部 pre-dispatch reservation；**不产生 success completion** |

不变量 8：`updates_issued` 不能替代 `OUTPUT_READY`。[EXPLICIT] `adr-0001-scatter-destination-claim.md:129`

---

## 5. Buffer 与资源生命周期

[EXPLICIT] `adr-0001-scatter-destination-claim.md:99-106`

| 资源                   | Owner         | Allocate/Reserve      | Ready/Use                | 正常释放            | 异常回收                                       | Reuse 条件          |
| ---------------------- | ------------- | --------------------- | ------------------------ | ------------------- | ---------------------------------------------- | ------------------- |
| `%l2_indices`          | L2 Context    | `nest.alloc`          | `%indices_ready`         | `%indices_released` | 等 `%indices_ready`+`%values_ready` 后 release | release 完成        |
| `%l2_values`           | L2 Context    | `nest.alloc`          | `%values_ready`          | `%values_released`  | 同上                                           | release 完成        |
| L1 distributed cache   | Dispatch      | `reserve.distributed` | dispatch updates         | `%l1_flush_done`    | pre-dispatch failure 时 clean-cancel           | release/cancel 完成 |
| L2 write-combine cache | L2 Context    | `nest.cache.reserve`  | L1 flush / global flush  | `%store_done`       | pre-dispatch failure 时 clean-cancel           | release/cancel 完成 |
| destination lease      | L2 Context #N | claim success/partial | dispatch 至 global flush | `%store_done`       | cancel optional partial lease                  | release/cancel 完成 |
| Tile local buffers     | Tile Context  | dispatch admission    | Tile Program             | `tile.return`       | dispatch 未发生时不分配                        | `TASK_DONE`         |

释放顺序硬约束：destination lease 与 L2 cache **必须同时持有到 `%store_done`**，不得在 `%grid_done`、`%updates_issued` 或 L1 flush 完成时提前释放。[EXPLICIT] `adr-0001-scatter-destination-claim.md:93-95`

---

## 6. ADR-0001 协议要点

来源：`docs/architecture/adr-0001-scatter-destination-claim.md`（Accepted for design IR；缩小形式化模型已实现；parser/lowering/RTL/SVA pending）

### 6.1 九条不变量

[EXPLICIT] `adr-0001-scatter-destination-claim.md:120-130`

1. dispatch 前必须持有覆盖完整 wave 的 destination lease。
2. 外层 context、dispatch、claim 的 `context_id` 必须一致。
3. 同一 destination cache line 同时至多有一个 Scatter Assign lease owner。
4. claim failure 不能启动任何 Tile Program。
5. claim failure 时 L1/L2 write-combine cache 必须 clean。
6. 所有在途 prefetch 完成前不得释放其目标 L2 buffer。
7. 成功路径 destination lease 不得早于 `%store_done` 释放。
8. `updates_issued` 不能替代 `OUTPUT_READY`。
9. ScatterReduce 不使用 exclusive lease；最终 global commit 必须是 atomic reduce。

### 6.2 需要的 SVA

[EXPLICIT] `adr-0001-scatter-destination-claim.md:223-228`

```text
claim_held(context) before dispatch(context)
exclusive_owner_count(Y_line) <= 1
claim_release(context) -> store_done(context) or claim_failed(context)
claim_failed(context) -> eventually all_pre_dispatch_resources_free(context)
```

这些 SVA 目前**尚未实现**（状态字段见 `adr-0001-scatter-destination-claim.md:3`）。

### 6.3 Context 状态字段

[EXPLICIT] `adr-0001-scatter-destination-claim.md:110-118`

| 字段                  | 位宽                                         | 更新时机                              | Flush/异常              |
| --------------------- | -------------------------------------------- | ------------------------------------- | ----------------------- |
| `claim_state`         | 2 bits（IDLE/CLAIMING/HELD/FAILED）          | prefetch ready、claim result、release | abort/release 回到 IDLE |
| `partial_lease_valid` | 1 bit                                        | claim acquisition/cancel              | cancel 后清零           |
| `lease_handle`        | `[UNKNOWN]`，由 lease table 容量决定         | claim success/cancel/release          | abort 时幂等 cancel     |
| line-set descriptor   | `[UNKNOWN]`，可用精确 list / 区间 / 哈希     | indices validation 后生成             | release/cancel 回收     |
| `claim_error`         | `[UNKNOWN]`，至少区分 duplicate/OOB/protocol | validation/cancel                     | context abort 后回收    |

`context_id` 位宽 `[INFERRED] ceil(log2(L2 context slots))`。[EXPLICIT] `adr-0001-scatter-destination-claim.md:118`

该机制**不增加 Tile Program image，不改变 Matrix/Vector datapath**。[EXPLICIT] `adr-0001-scatter-destination-claim.md:189`

### 6.4 被拒绝的替代方案

[EXPLICIT] `adr-0001-scatter-destination-claim.md:132-172`

| 方案                                                  | 处置                                             | 理由                                                                      |
| ----------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| A. 声明所有并发 Scatter Assign 输入全局不相交         | 仅在有可验证 disjointness certificate 时作为优化 | runtime indices 无法静态证明；错误声明会静默丢数据                        |
| B. 对同一 `%Y` 的所有 Scatter Assign Context 全局串行 | **rollback / fallback**，非默认路径              | 不相交 destination 也无法并发                                             |
| C. 静态 output / cache-line 分区                      | 未来优化                                         | runtime-index 算子不能普遍证明                                            |
| D. 改为原子写                                         | **拒绝**                                         | assign 的跨 Context 顺序语义仍需定义；逐元素原子增加 memory system 复杂度 |
| E. 在 cache reserve 时猜测 line set                   | **拒绝**                                         | indices 尚未 ready，没有事实依据                                          |

Rollback 方案：禁用 `nest.destination.claim.async` lowering，Host/Runtime 对同一 `%Y` 使用单一 dependency token，同时最多 admission 一个 Scatter Assign Context，删除 lease table / line-set comparator，不改 ScatterReduce atomic path。[EXPLICIT] `adr-0001-scatter-destination-claim.md:250-262`

---

## 7. 验证覆盖现状

### 7.1 形式化模型（已实现）

[EXPLICIT] `models/formal/resource_contract_model.py:350-894`；测试 `models/formal/test_resource_contract_model.py:113-236`

该模型与通用 dispatch/drain 资源合同模型**分离**，因为 Scatter 在 dispatch 前就要预约 input/cache，再做 runtime line-set claim，通用模型从 dispatch admission 出发无法表达 claim overlap 与 pre-dispatch unwind。[EXPLICIT] `resource_contract_model.py:350-355`

缩小规模常量：

| 常量                  | 值  | 含义                  |
| --------------------- | --- | --------------------- |
| `GROUP_COUNT`         | 2   | 2 个 L2 Context       |
| `TILE_COUNT`          | 2   | 每 Context 2 个 Tile  |
| `SCATTER_LINE_COUNT`  | 2   | 2 个 destination line |
| `SCATTER_INPUT_COUNT` | 2   | 2 份 L2 input         |

`ScatterClaimState` 跟踪字段（`resource_contract_model.py:392-411`）：

```text
phases, indices_ready, values_ready,
input_reserved, event_reserved, context_reserved, drain_reserved,
dma_reserved, mshr_reserved, noc_reserved,
l1_cache_reserved (per tile), l2_cache_reserved,
line_sets, line_owner, partial_lease, claim_error,
local_reserved (per tile), dirty_cache
```

Phase 机（`resource_contract_model.py:359-381,577-798`）：

```text
PREFETCHING → VALIDATING → WAIT_CLAIM → CLAIMED → DISPATCHED
→ INPUT_RELEASED → UPDATES_ISSUED → L1_FLUSHED → STORE_DONE → COMPLETE
                              ↘ CLAIM_FAILED → ABORTED
```

本次实测（`conda run -n nest-cycle-model python -m pytest models/formal/test_resource_contract_model.py -q`）：

| 指标               | 值        |
| ------------------ | --------- |
| formal 回归        | 12 passed |
| 可达状态           | 991       |
| 终态               | 9         |
| unsafe 状态        | 0         |
| 无终态路径的非终态 | 0         |

阶段分布（跨状态统计）：`CLAIM_FAILED` 528、`WAIT_CLAIM` 264、`CLAIMED` 208、`ABORTED` 132、`PREFETCHING` 132、`VALIDATING` 132、`DISPATCHED`/`INPUT_RELEASED`/`UPDATES_ISSUED`/`L1_FLUSHED`/`STORE_DONE` 各 104、`COMPLETE` 66。

不变量检查内容（`resource_contract_model.py:456-574`）包括：line owner 必须在 lease phase 内且 line set 匹配；每 Context 最多持有一条 line；`indices_ready` 与 `PREFETCHING` 双向一致；claim 路径不得在 `values_ready` 前完成；input / L1 cache / L2 cache / event / context / drain / DMA / MSHR / NoC 预约状态与 phase 严格对应；`local_reserved` 只在 `DISPATCHED`/`INPUT_RELEASED` 存在；`dirty_cache` 只在 `UPDATES_ISSUED`/`L1_FLUSHED` 为真；失败/abort 后不得有 local reservation 或 dirty cache；终态不得残留任何预约或 partial lease。

针对性测试（`test_resource_contract_model.py:113-236`）：

| 测试                                                          | 覆盖的不变量                                     |
| ------------------------------------------------------------- | ------------------------------------------------ |
| `test_scatter_claim_states_are_safe_and_can_reach_terminal`   | 全状态安全 + 可达终态                            |
| `test_nonoverlapping_scatter_claims_dispatch_concurrently`    | 不相交 claim 并发（`line_sets ∈ {(0,1),(1,0)}`） |
| `test_overlapping_scatter_claims_serialize`                   | 相交 claim 停在 `WAIT_CLAIM`                     |
| `test_claim_failure_waits_for_values_then_reclaims_resources` | 延迟 values prefetch + 全资源回收                |
| `test_partial_lease_is_cancelled_on_claim_abort`              | partial lease 幂等取消                           |
| `test_store_done_holds_lease_until_complete`                  | 不变量 7（lease 持有到 `store_done`）            |

### 7.2 尚未覆盖的形式化项

[EXPLICIT] `adr-0001-scatter-destination-claim.md:238-244`

- 第 7、9 项（最终资源回收、claim 最终可 admission）目前只检查"每个非终态至少存在一条到终态的路径"，**不是 all-fair-traces liveness proof**。
- 第 8 项（dirty clean-cancel 必须报错）仍需 RTL/SVA。
- 第 10 项（ScatterReduce 不受 exclusive claim 影响）仍需协议组合验证。
- 每 Context 的缩小 line set 只含一条 line；**多 line 原子 claim 是待扩展边界**。

### 7.3 周期模型

[EXPLICIT] `models/cycle_model/nest_sim.py`

已建模的 scatter 相关机制：

| 机制                 | 位置                                          | 说明                                                                                              |
| -------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| scatter 冲突延迟抖动 | `nest_sim.py:106-107, 840-842`                | `scatter_conflict_modulus` / `scatter_conflict_penalty_cycles`，作用于 `kind="store"` 的 latency  |
| output-region 排他锁 | `nest_sim.py:1560-1566, 1626-1639, 1791-1792` | store 期间锁住 `output_region_id`，同 region 其它 group 的 store 被阻塞；`handle_store_done` 解锁 |
| hazard 计数          | `nest_sim.py:1693-1703`                       | `output_hazard_retry_count`，统计被 region lock 阻塞的 `STORE_READY` task                         |
| MPMD comparator 降级 | `nest_sim.py:3021-3029`                       | `output_hazard_modulus > 0` 时 `fixed_role_split` 回退为 `hazard_unsupported_fallback`            |

**未建模**：ADR-0001 的 claim / lease / abort / unwind、line-set 构造与 overlap 比较、`updates_issued` 中间事件、masked writeback、`atomic_reduce_add`。

两个 scatter 工作负载（`nest_sim.py:462-507`）：

|                          | `scatter_segment_reduce`                                                                            | `scatter_add_hazard`                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| task_groups              | 10                                                                                                  | 8（每组 2 个 ContextSpec，数值相同）                                                              |
| ContextSpec              | `waves=8, prefetch=44, load=8, matrix=4, vector_reduce=12, local_store=10, drain=48, dma=1, mshr=2` | `waves=4, prefetch=32, load=6, matrix=8, vector_reduce=6, local_store=6, drain=30, dma=1, mshr=1` |
| hbm_jitter               | 10                                                                                                  | 0                                                                                                 |
| bank_conflict_jitter     | 2                                                                                                   | 3                                                                                                 |
| noc_backpressure_jitter  | 4                                                                                                   | 4                                                                                                 |
| scatter_conflict_modulus | 4                                                                                                   | 2                                                                                                 |
| scatter_conflict_penalty | 32                                                                                                  | 24                                                                                                |
| dma_completion_jitter    | 8                                                                                                   | 0                                                                                                 |
| `output_region_groups`   | 无（默认每 group 独立 region，不触发锁）                                                            | `((0,1,4,5), (2,3,6,7))`                                                                          |
| `output_hazard_modulus`  | 0                                                                                                   | 1                                                                                                 |

`scatter_add_hazard` 显式把 8 个 group 分成 2 个共享 output region 的集合，强制制造同 destination 并发写的排他压力——这是 cycle model 里最接近 claim 语义的机制，但它是**region 粒度串行化**，不是 cache-line 粒度 lease。[EXPLICIT] `nest_sim.py:502-506,1563-1565`

### 7.4 验证计划中的待办场景

[EXPLICIT] `docs/verification/cycle_simulator_v2_verification_plan.md:157-163`

- **S8 Scatter Assign Success/Conflict/Abort**：不相交 lease 并发；相交 lease 串行；duplicate/OOB 等待在途 values prefetch 后 clean unwind。
- **S9 ScatterReduce**：重复 index 合并，L1→L2 reduce，最终 atomic reduce commit；**不得申请 exclusive assign lease**。

对应 work package WP11（`docs/architecture/cycle_simulator_v2_work_packages.md:276-290`）的 owner paths 尚未落地——`models/cycle_model_v2/` 目录当前不存在。

---

## 8. 文档间不一致与缺口

### 8.1 L1 Cache 策略冲突

| 来源                                                                         | 主张                                                                                                                                                                   |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `background/nest_dual_mode_l1_multicontext_scheduler_design.md:310-347`      | Gather V1 用 L1 **read-only** cache；Scatter V1 走 **write-through / bypass L1 → L2**；Scatter-Reduce 走 L2/Reduction/Atomic Path；Cache Bank 释放只需 invalidate tags |
| `ir_design/scatter.ir:149-171` + `docs/architecture/ir_operator_rules.md:79` | Scatter 的 L1 write-combine cache 是 `admission = required` 的**正确性资源**                                                                                           |
| `background/spmd_ai_inference_accelerator_design_proposal.md:755-763`        | 明确写入 Tile-shared L1 write-combine / reduction cache，并把 writeback/eviction entry 定为正确性资源                                                                  |

后者是更新的、且已被 IR 与 ADR 采用的设计。背景文档的 "Scatter V1 write-through/bypass L1" 属于被取代的早期建议，未同步更新。

### 8.2 cycle model 与 formal model 无桥接

[EXPLICIT] `README.md:169-180`

formal model 与 cycle simulator 相互独立，无 cross-model refinement。cycle model 的 group 状态枚举甚至**没有显式 `TASK_DONE` 标签**（从 `OUTPUT_READY` 直接进入 `DRAINING`），因此 ADR-0001 的 `updates_issued` 中间事件在 cycle model 中完全没有对应物。

### 8.3 claim 性能代价无数据

[EXPLICIT] `adr-0001-scatter-destination-claim.md:174-180`

claim 增加 indices 扫描、line-set 构造与 admission 比较延迟；claim 可与 values prefetch 重叠；只有 line set 重叠的 context 被串行。这些均为 `[INFERRED]`，**无 cycle-model 数据**，不能报告 speedup/slowdown。失败路径等待 in-flight prefetch 是为保证生命周期正确，不引入提前取消 DMA 的新协议。

---

## 9. 本次实测数据

模拟器：合成确定性周期模型，**不是硬件测量**。硬件向量：16 physical tiles、L2 in/out buffer 4/4、tile context slot 4/tile、L1/accumulator slot 4/4、DMA channel 4、MSHR 16、Event 8、Drain/NoC credit 4/4。[EXPLICIT] `README.md:184-195`

复现命令（原始 JSON 落在 `/tmp`，未提交仓库；如需归档请改 `--output` 指向
`experiments/results/`）：

```bash
conda run -n nest-cycle-model python models/cycle_model/nest_sim.py \
  --format json --workload scatter_segment_reduce \
  --revision $(git rev-parse HEAD) --output /tmp/ssr.json
conda run -n nest-cycle-model python models/cycle_model/nest_sim.py \
  --format json --workload scatter_add_hazard --output /tmp/sah.json
```

本次实测 revision `62e5585a871a`，workload fingerprint `8b7aa7229ea4…`（两个
workload 共享同一 fingerprint 定义域，后者命令未显式传 `--revision`，其 JSON 内
`revision` 字段为默认值 `unknown`，不影响 makespan——模拟器为确定性模型，
revision 仅为记录字段）。

形式化侧复现：

```bash
conda run -n nest-cycle-model python -m pytest \
  models/formal/test_resource_contract_model.py -q   # 12 passed
```

### 9.1 `scatter_segment_reduce`（makespan / B0c 归一化加速比）

| 配置                              | makespan (cycles) | 相对 B0c   | steady-state interval | single-request latency | execution mode   |
| --------------------------------- | ----------------- | ---------- | --------------------- | ---------------------- | ---------------- |
| B0a Static SPMD                   | 36803             | 0.976×     | 3680.8                | 3676                   | spmd             |
| B0b + Tile Double Buffer          | 21335             | 1.684×     | 2130.1                | 2164                   | spmd             |
| B0c Strong Static + L2 Pipeline   | 35928             | 1.000×     | 3583.6                | 3676                   | spmd             |
| B1 L2-MC Only                     | 36368             | 0.988×     | 3633.1                | 3676                   | spmd             |
| B2 Tile-MC Only                   | 13050             | 2.753×     | 1304.6                | 1309                   | spmd             |
| B3a Nested-MC Conservative Quota  | 11044             | 3.253×     | 820.0                 | 3676                   | spmd             |
| B3b Nested-MC Aggressive Quota    | 11044             | 3.253×     | 820.0                 | 3676                   | spmd             |
| Fixed Two-Program MPMD Comparator | 22592             | 1.590×     | 2244.0                | 2396                   | fixed_role_split |
| **Proposed NEST**                 | **11044**         | **3.253×** | **820.0**             | 3676                   | spmd             |

Proposed steady-state interval 相对 B0c：3583.6 / 820.0 = **4.37×**。
两资源下界效率（评审报告口径，`NEST_simulation_review_round2.md:578-592`）：**83.53%**。该表覆盖的是 7 个 workload（当时 `scatter_add_hazard` 尚未纳入），scatter 是其中**第二低**（仅高于 `gemm_gelu` 的 82.20%），说明仍有非 matrix/DMA 资源在限制它（`vector_reduce=12`、`local_store=10`、`drain=48` 三个非零项）。
B3a 与 B3b 与 Proposed 完全相同，**未产生区分**。

### 9.2 `scatter_add_hazard`（makespan / B0c 归一化加速比）

| 配置                              | makespan | 相对 B0c   | `output_hazard_retry_count` | execution mode                  |
| --------------------------------- | -------- | ---------- | --------------------------- | ------------------------------- |
| B0a Static SPMD                   | 14389    | 0.969×     | 1376                        | spmd                            |
| B0b + Tile Double Buffer          | 10661    | 1.308×     | 3444                        | spmd                            |
| B0c Strong Static + L2 Pipeline   | 13946    | 1.000×     | 1386                        | spmd                            |
| B1 L2-MC Only                     | 14165    | 0.985×     | 1380                        | spmd                            |
| B2 Tile-MC Only                   | 10661    | 1.308×     | 3312                        | spmd                            |
| B3a Nested-MC Conservative Quota  | 5183     | 2.691×     | 5724                        | spmd                            |
| B3b Nested-MC Aggressive Quota    | 5183     | 2.691×     | 5724                        | spmd                            |
| Fixed Two-Program MPMD Comparator | 5183     | 2.691×     | 5724                        | **hazard_unsupported_fallback** |
| **Proposed NEST**                 | **5183** | **2.691×** | 5724                        | spmd                            |

全部配置 `tile_store_count = 512`（8 groups × 16 tiles × 4 waves，逻辑工作量一致）。
Proposed DMA busy 14329 / total 5183 cycles，peak DMA slots 4（打满），peak MSHR 4，`dma_completion_reorder_count = 1000`，`invariant_failures = []`。
MPMD comparator 行因 `output_hazard_modulus > 0` 回退为 SPMD，其 2.691× **不是 role-split 结果**，不得计入 MPMD 几何平均。

### 9.3 数据解读限制

- `scatter_segment_reduce` 未启用 hazard region（B0c 35928 远大于 `scatter_add_hazard` 的 13946，主要因 vector_reduce/drain 参数更高），因此该行**不测任何排他协议代价**。
- 两个 workload 的加速比差异主要来自工作负载参数，不构成"Scatter 协议更好/更差"的证据。
- `output_hazard_retry_count` 在 Proposed 下反而**更高**（5724 vs B0c 1386）：Proposed 并发度更高，撞上 region lock 的机会更多，但总 makespan 更短。该计数是**碰撞次数**，不是性能损失度量，不能单独作为负面证据。
- 全部数字为模型输出，禁止作为硬件结论。

---

## 10. 尚未解决的问题

1. **claim 性能代价未知**。ADR-0001 自己标注 `[UNKNOWN]`，cycle model 无 claim 建模。需要 cycle_model_v2 的 WP11 才能给出数字。
2. **多 line 原子 claim 未验证**。形式化模型每 Context 只有一条 line，line-set descriptor 表示法也未定（`[UNKNOWN]`）。
3. **liveness 未证明**。第 7、9 项只有"存在一条到终态的路径"，不是 all-fair-traces liveness。
4. **ScatterReduce 与 exclusive claim 的协议组合验证缺失**（ADR 第 10 项）。
5. **dirty clean-cancel 的错误报告路径未实现**（ADR 第 8 项，需 RTL/SVA）。
6. **`claim_error` 编码宽度未定**，至少需区分 duplicate / OOB / protocol error。
7. **parser / lowering / RTL / SVA 全部 pending**（ADR 状态行 `[3]`）。当前 `scatter.ir` / `scatterreduce.ir` 是设计稿，无 parser 消费。
8. **cycle model 的 `updates_issued` 中间事件缺失**，无法表达"lease 持有到 `store_done`"的时序代价。
9. **背景文档 §8.1 的 L1 策略冲突未清理**。

---

## 11. 下一步最小行动

1. 在 `models/cycle_model_v2/` 建立 WP11 骨架：`components/l2/destination_claim.py` + `lowering/scatter.py`，先把 claim 延迟（indices 扫描 + line-set 构造 + overlap 比较）作为可配置 latency 注入，**不改**协议语义。
2. 用 `scatter_add_hazard` 的 region 集合结构做 line-set 化：从 2 个 region 推广到每 context 多条 line，替换 region 粒度锁。
3. 给 `S8`/`S9` 写集成测试，断言与 `resource_contract_model.py:456-574` 的 invariant 一一对应（WP11 验收条件已写明 "invariant mapping 文档一致"）。
4. 把 `updates_issued` 作为独立计数事件加入 cycle model metrics，使 lease 持有时长可测。
5. 同步更新 `background/nest_dual_mode_l1_multicontext_scheduler_design.md:310-347`，标注 L1 策略已被 `scatter.ir` 取代。

---

## 12. 引用文件索引

| 文件                                                          | 角色                                                    |
| ------------------------------------------------------------- | ------------------------------------------------------- |
| `ir_design/scatter.ir`（606 行）                              | Scatter Assign 设计 IR                                  |
| `ir_design/scatterreduce.ir`（536 行）                        | ScatterReduce Add 设计 IR                               |
| `docs/architecture/adr-0001-scatter-destination-claim.md`     | 排他 claim 协议（Accepted）                             |
| `docs/architecture/ir_operator_rules.md`                      | 算子 IR 规则（A5 / F2 / H4 / H5 / 事件映射）            |
| `background/spmd_ai_inference_accelerator_design_proposal.md` | §9.4 / §10.3 / §13 设计推导（部分行号与当前文件不匹配） |
| `docs/verification/cycle_simulator_v2_verification_plan.md`   | S8 / S9 场景定义                                        |
| `docs/architecture/cycle_simulator_v2_work_packages.md`       | WP11 交付清单                                           |
| `models/formal/resource_contract_model.py:350-894`            | Scatter claim 缩小形式化模型                            |
| `models/formal/test_resource_contract_model.py:113-236`       | Scatter claim 针对性测试                                |
| `models/cycle_model/nest_sim.py`                              | scatter 延迟抖动 + output-region 排他锁                 |
| `NEST_simulation_review_round2.md`                            | 评审对 scatter 行效率与 MPMD 比较的解读                 |
