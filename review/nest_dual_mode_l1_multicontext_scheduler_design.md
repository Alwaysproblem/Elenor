# NEST / HSTI：Dual-Mode L1 SRAM 与 Multi-Context 在线资源调度设计

> **状态**：当前对话阶段推荐设计
> **目标**：解决 `MatMul / regular context` 与 `Gather / irregular context` 在同一 Tile 上采用 multi-context 并发时，对 L1 SRAM 容量、Bank 模式、带宽和 Context admission 的冲突问题。
> **核心原则**：编译器负责“每个 Context 需要什么资源”，运行时/L2 负责“当前哪些 Context 能同时 resident”。

---

# 1. 问题定义

当前 Tile 采用 **task-level multi-context**：

- 一个 Context 对应一个 Tile-level task/program。
- 每个 Tile 最多同时 resident 固定数量的 Context，例如：

```text
MAX_RESIDENT_CONTEXTS = 4
```

- Context 不是按 4 个一组批量执行，而是一个 **sliding resident set**：

```text
t0:
[C0][C1][C2][C3]

C1 完成：

[C0][  ][C2][C3]
      │
      └── C4 立即补入

=> [C0][C4][C2][C3]

之后 C3 完成：

=> [C0][C4][C2][C5]
```

因此，运行时实际同时 resident 的 Context 组合由：

- cache miss
- L2/HBM latency
- DMA latency
- Matrix/Vector engine contention
- 其他动态 stall

共同决定。

## 1.1 关键矛盾

假设 Tile 有：

```text
L1 = 16 Banks
```

先到一个 MatMul：

```text
Context M0:
16 Banks 全部作为 Scratchpad
```

后面 Gather 到达：

```text
Context G0:
需要若干 Cache Banks
```

即使每个 Bank 都具有 SPM/Cache 两种能力，只要 16 个 Bank 中仍然存在 MatMul 的 live data：

```text
SPM live
```

就不能直接：

```text
SPM -> CACHE
```

否则需要：

- live data migration
- spill
- reload
- address remapping
- Context state repair

这会严重增加硬件复杂度。

因此：

> **“Bank 支持双模式”只能解决 Bank capability 问题，不能解决 live allocation 冲突。**

---

# 2. 当前推荐的总体架构

当前建议收敛为：

```text
                  Compiler
                     │
                     │ Context Resource Contract
                     ▼
               L2 Ready Queue
                     │
             Bounded Lookahead
                     │
                     ▼
          Online Admission / Packing
                     │
                     ▼
               Tile Context Slots
                     │
         ┌───────────┴───────────┐
         │                       │
   L1 SPM Allocation        L1 Cache Allocation
         │                       │
         └───────────┬───────────┘
                     ▼
             Dual-Mode L1 Banks
```

架构职责拆分：

| 层级           | 职责                                                                |
| -------------- | ------------------------------------------------------------------- |
| Compiler       | 分析单个 Context 的 live-set、带宽、资源需求和可选 schedule variant |
| L2 Scheduler   | 根据当前 resident set + ready queue 做在线 admission                |
| Tile Admission | 实际分配 Context slot、SPM bank mask、Cache bank mask               |
| Tile Hardware  | 执行 Context；处理 backpressure / scoreboard / engine scheduling    |
| L1 Bank        | 根据 mode 作为 Scratchpad 或 Irregular Cache Data Bank              |

---

# 3. L1 SRAM：所有 Bank 均支持 SPM / Cache 双模式

## 3.1 物理组织

例如：

```text
L1 = 128 KB
16 Banks × 8 KB
```

每一个 Bank：

```text
Bank0   [ SPM | CACHE ]
Bank1   [ SPM | CACHE ]
Bank2   [ SPM | CACHE ]
...
Bank15  [ SPM | CACHE ]
```

推荐称为：

> **Dual-Purpose L1 Bank**
> 或
> **Reconfigurable L1 Bank**

而不是“两套 SRAM”。

物理上：

```text
                 Bank N
             ┌───────────────┐
             │ Data SRAM     │
             │ 8 KB          │
             └───────▲───────┘
                     │
                Address MUX
              ┌──────┴──────┐
              │             │
        SPM Address      Cache Index
              │             │
        SPM Controller   Cache Frontend
```

## 3.2 Bank 状态

建议：

```cpp
enum class BankMode {
    Free,
    Scratchpad,
    Cache,
};
```

每个 Bank 至少维护：

```cpp
struct BankState {
    BankMode mode;

    // Scratchpad owner
    uint8_t owner_context;

    // Optional runtime state
    bool reclaim_pending;
};
```

---

# 4. 必须冻结的 Bank Mode Invariant

## 4.1 禁止 live SPM 直接切 Cache

状态机：

```text
            ┌───────────────┐
            │               │
            ▼               │
FREE ─────► SPM ─────► FREE
 │
 │
 └────────► CACHE ───► FREE
```

禁止：

```text
SPM(live) ─────► CACHE
```

Cache 与 SPM 之间必须经过：

```text
FREE
```

也就是说：

> **不支持 live-bank migration。**

理由：

- 避免 SRAM 数据搬迁
- 避免地址更新
- 避免 Context replay
- 避免复杂 eviction state
- 避免硬件 scheduler 与 compiler memory planner 耦合

---

# 5. Gather Cache 建议定义成性能资源，而不是 correctness 资源

这是 multi-context 能否保持弹性的关键。

建议 V1：

```text
Gather
  │
  ├── L1 Cache available
  │        │
  │        ▼
  │      L1 Cache
  │        │ miss
  │        ▼
  │       L2
  │
  └── L1 Cache unavailable
           │
           ▼
         L2 directly
```

因此：

```text
cache_min = 0
```

是合法的。

即：

> Gather 没有 L1 Cache 仍然能够正确执行，只是 L2/HBM traffic 和 latency 更高。

这样可避免：

```text
MatMul 占满全部 SPM
     ↓
Gather 彻底无法 admission
```

变成：

```text
MatMul 占满 SPM
     ↓
Gather 先 bypass L1 Cache
     ↓
后续 Bank 释放后再为新的 irregular Context 分配 Cache
```

---

# 6. Gather Cache V1 推荐 Read-Only

第一版建议：

```text
Gather:
L1 read cache
```

不做：

```text
write-back cache
dirty line
scatter write caching
scatter-reduce atomic cache
```

因此 Cache Bank 释放时：

```text
invalidate tags
```

即可，不需要 writeback。

Scatter / Scatter-Reduce V1 可以：

```text
Scatter
   │
   ├── write-through
   └── bypass L1 Cache -> L2

Scatter-Reduce
   │
   └── L2 / Reduction / Atomic Path
```

---

# 7. Cache Controller 不应在每 Bank 完整复制

虽然所有 Bank 都具备 Cache capability，但不要设计成：

```text
Bank0:
  MSHR
  refill
  replacement
  miss queue

Bank1:
  MSHR
  refill
  replacement
  miss queue

...
```

建议：

```text
                 Irregular Requests
                        │
                        ▼
             ┌────────────────────┐
             │ Shared Cache Front │
             │                    │
             │ Shared MSHR        │
             │ Miss Queue         │
             │ Refill Engine      │
             │ L2 Interface       │
             └─────────┬──────────┘
                       │
                 CACHE_BANK_MASK
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
     Bank0           Bank1           Bank15
   Tag + Data      Tag + Data       Tag + Data
```

每 Bank 只需要：

- mode bit
- tag storage
- valid bits
- cache address mux
- selected-bank tag compare

Tile 级共享：

- MSHR
- miss queue
- refill logic
- cache scheduler
- L2 request logic

---

# 8. 为什么“Compiler 静态 Mix Window”不适合当前执行模型

最初一种思路是：

```text
Epoch 0:
MatMul + MatMul + Gather + Vector

Epoch 1:
MatMul + Gather + Gather + Vector
```

然后提前决定：

```text
SPM / Cache = 80 / 20
```

但是实际 Context 完成顺序不是编译器可完全预测的：

```text
resident:
M0 M1 G0 V0

可能：
G0 先完成
也可能：
M0 先完成
也可能：
V0 先完成
```

尤其 irregular Context 受：

```text
cache hit
L2 miss
HBM latency
memory contention
```

影响明显。

因此：

> **Compiler 不应该生成精确的 resident timeline。**

---

# 9. 推荐边界：Compiler Resource Contract + Runtime Admission

## 9.1 Compiler 负责单个 Context

Compiler 能可靠分析：

### Scratchpad

```text
peak live bytes
buffer lifetime
bank coloring
prefetch footprint
double buffer footprint
```

### Bandwidth

```text
L1 read BW
L1 write BW
DMA BW
Matrix operand BW
```

### Engine pressure

```text
Matrix pressure
Vector pressure
Memory pressure
```

### Context execution variant

例如 MatMul 可以有：

```text
FAST
BALANCED
COMPACT
```

---

# 10. Context Resource Contract

建议每个 Context 携带：

```cpp
struct ContextResourceProfile {
    // SPM
    uint8_t spm_min_banks;
    uint8_t spm_target_banks;
    uint8_t spm_max_banks;

    // Cache
    uint8_t cache_min_banks;
    uint8_t cache_target_banks;
    uint8_t cache_max_banks;

    // Bandwidth
    uint16_t l1_read_bw;
    uint16_t l1_write_bw;

    // Engine pressure: normalized or quantized
    uint8_t matrix_pressure;
    uint8_t vector_pressure;
    uint8_t memory_pressure;

    // Performance estimation
    uint32_t expected_cycles;

    // Type
    ContextClass context_class;
};
```

Context class：

```cpp
enum class ContextClass {
    MatrixHeavy,
    VectorHeavy,
    MemoryRegular,
    MemoryIrregular,
    Mixed,
};
```

---

# 11. Compiler 如何计算 SPM Bank Requirement

首先做 live-range analysis：

```text
           time →

A0        █████
A1              █████
B0        █████
B1              █████
OUT          ████
```

计算：

```text
peak_live_bytes
```

例如：

```text
peak_live_bytes = 56 KB
bank_size       = 8 KB
```

则容量至少：

```text
ceil(56 / 8) = 7 Banks
```

但是不能只看容量。

还需要计算：

```text
bandwidth_required_banks
```

例如：

```text
A read = 256 B/cycle
B read = 256 B/cycle

Bank BW = 64 B/cycle
```

那么：

```text
required = (256 + 256) / 64
         = 8 Banks
```

因此：

```text
SPM_bank_requirement =
    max(
        capacity_requirement,
        bandwidth_requirement,
        bank_coloring_requirement
    )
```

示例：

```text
capacity = 7
bandwidth = 8
coloring = 6

=> min SPM banks = 8
```

---

# 12. Gather Cache Requirement 不能仅靠 Live Analysis

SPM：

```text
buffer address
size
lifetime
```

编译器通常知道。

但 Gather Cache 的工作集取决于：

```text
runtime indices
reuse distance
source tensor locality
runtime access distribution
cache hit rate
```

因此建议：

```text
cache_min    = 0
cache_target = estimated value
cache_max    = performance saturation value
```

例如：

```text
Gather Profile:

SPM:
min = 2 banks

Cache:
min    = 0
target = 4
max    = 8
```

Cache target 可来自：

- static pattern analysis
- index range analysis
- offline profiling
- microbenchmark
- PGO

---

# 13. 每个 Context 应允许多个 Resource Variant

例如 MatMul：

| Variant  | SPM Banks | Estimated Cycles | 用途               |
| -------- | --------: | ---------------: | ------------------ |
| FAST     |         6 |              600 | Tile 较空          |
| BALANCED |         4 |              720 | 普通 multi-context |
| COMPACT  |         3 |              900 | 高 occupancy       |

IR 可以表示成：

```mlir
nest.resource_profile @matmul {
  variant "fast" {
    spm_banks = 6
    read_bw = 384
    expected_cycles = 600
  }

  variant "balanced" {
    spm_banks = 4
    read_bw = 256
    expected_cycles = 720
  }

  variant "compact" {
    spm_banks = 3
    read_bw = 192
    expected_cycles = 900
  }
}
```

关键价值：

```text
Tile 很空
   ↓
FAST

Tile 已经有 Gather / Vector Context
   ↓
BALANCED / COMPACT
```

因此不需要永久固定：

```text
MatMul <= 80%
```

---

# 14. Runtime 模型：Sliding Resident Set

假设：

```text
MAX_RESIDENT_CONTEXTS = 4
```

scheduler 是 event-driven：

```text
Context completion
       │
       ▼
release slot / SPM / cache pressure
       │
       ▼
scan ready queue
       │
       ▼
select candidate combination
       │
       ▼
admit
```

伪代码：

```cpp
void on_context_complete(ContextId done) {
    release_resources(done);

    auto plan = build_next_admission_plan();

    for (auto &candidate : plan) {
        allocate_resources(candidate);
        launch(candidate);
    }
}
```

---

# 15. 当前最推荐的 Admission 算法：Bounded Lookahead Packing

由于：

```text
resident contexts <= 4
```

搜索空间非常小。

不需要运行时 ILP。

建议：

```text
READY_LOOKAHEAD = 8
```

每次 Context 完成后，仅考察 Ready Queue 前 8 个 Context。

如果：

```text
free slots = 2
```

则枚举：

```text
(C4,C5)
(C4,C6)
(C4,C7)
...
(C10,C11)
```

再枚举其 resource variant。

---

# 16. Feasibility Check

对于一个 candidate resident set：

```python
def feasible(plan):
    if len(plan) > MAX_CONTEXT:
        return False

    spm_banks = sum(x.spm_banks for x in plan)

    cache_banks = estimate_shared_cache_requirement(plan)

    if spm_banks + cache_banks > TOTAL_BANKS:
        return False

    if sum(x.read_bw for x in plan) > MAX_L1_READ_BW:
        return False

    if sum(x.write_bw for x in plan) > MAX_L1_WRITE_BW:
        return False

    return True
```

需要注意：

> Cache 推荐 Tile-shared，因此多个 Gather Context 不应简单把 `cache_banks` 相加。

可以近似：

```text
shared_cache_target =
    max(cache_target of active irregular contexts)
```

或者根据 profiling 使用更精细函数：

```text
shared_cache_target =
    f(irregular_context_set)
```

---

# 17. Scheduler Score

不应该只追求：

```text
resident context 数量最大
```

因为：

```text
Gather + Gather + Gather + Gather
```

可能把 memory subsystem 打满。

而：

```text
MatMul + Gather + Vector
```

可能资源互补。

初始 score：

```python
def score(plan):
    return (
        throughput_score(plan)
        + complementarity_score(plan)
        + aging_score(plan)
        - l1_bandwidth_penalty(plan)
        - memory_pressure_penalty(plan)
        - engine_conflict_penalty(plan)
    )
```

V1 可以进一步简化成 lookup table。

---

# 18. Context Compatibility Matrix

Compiler 给出：

```text
MATRIX_HEAVY
VECTOR_HEAVY
MEMORY_IRREGULAR
```

硬件/L2 使用：

| Resident | New Context | Compatibility |
| -------- | ----------- | ------------: |
| Matrix   | Matrix      |           0.3 |
| Matrix   | Gather      |           0.9 |
| Matrix   | Vector      |           0.8 |
| Gather   | Gather      |           0.2 |
| Gather   | Vector      |           0.6 |
| Vector   | Matrix      |           0.8 |

目标：

> 让 active Context 尽量使用不同瓶颈资源。

---

# 19. 必须解决 Head-of-Line Blocking

假设：

```text
Ready Queue:

G0 need 4 banks
M3 need 2 banks
V0 need 1 bank
```

当前：

```text
free banks = 3
free slot  = 1
```

Strict FIFO 会：

```text
G0 不 fit
  ↓
整个 slot 空闲
```

这是错误的。

推荐：

> **Bounded backfill**

即：

```text
G0:
temporarily skip

M3:
admit
```

同时为 G0 增加：

```text
aging / starvation score
```

避免长期饿死。

---

# 20. 为什么不是复杂 RCPSP / ILP

从理论模型看，这个问题接近：

```text
Resource-Constrained Scheduling
+
Multidimensional Packing
+
Backfilling
```

但是当前硬件约束：

```text
resident <= 4
lookahead <= 8
variant <= 3
```

因此可以直接做小规模枚举：

```text
O(subsets × variants)
```

规模非常有限。

没有必要：

- runtime ILP
- MILP
- 全图重调度
- 大型 heuristic optimizer

---

# 21. 一个具体 Admission 示例

Tile：

```text
16 Banks
4 Context Slots
```

Ready Queue：

```text
M0
M1
M2
G0
M3
G1
```

MatMul：

```text
FAST    = 5 SPM Banks
COMPACT = 3 SPM Banks
```

Gather：

```text
SPM = 2 Banks
Cache target = 2 Banks
```

## Naive Scheduler

```text
M0 FAST = 5
M1 FAST = 5
M2 FAST = 5

total = 15
```

只剩：

```text
1 Bank
```

导致：

```text
G0 无法获得 target resource
```

## Lookahead Scheduler

选择：

```text
M0 FAST    = 5
M1 COMPACT = 3
M2 COMPACT = 3
G0 SPM     = 2
Cache      = 2
```

合计：

```text
5 + 3 + 3 + 2 + 2 = 15
```

最终：

```text
[M0][M1][M2][G0]
```

4 个 Context 同时 resident。

后续谁先完成并不重要。

Context 完成后：

```text
release
  ↓
重新 packing
```

---

# 22. 80/20 如何保留

不建议硬件固定：

```text
80% SPM
20% Cache
```

但是可以保留成 **Admission Policy Baseline**：

```text
if irregular context exists in lookahead:
    regular_spm_soft_ceiling ≈ 75~87.5%
else:
    regular_spm_soft_ceiling = 100%
```

例如 16 Banks：

```text
100 / 0
14 / 2
12 / 4
10 / 6
8  / 8
```

几档即可。

重点：

> 80/20 是 runtime scheduling policy，不是物理硬件限制。

---

# 23. Compiler 不应该做跨 Context Wall-Clock Live Range

不推荐：

```text
M0:
cycle 0~500

G0:
cycle 200~700
```

因为实际 overlap 受运行时影响。

Compiler 应该只分析：

```text
Context 内部：
  peak live SPM
  phase
  resource release point
  bandwidth
  execution variants
```

Runtime 决定：

```text
Context 之间：
  谁和谁同时 resident
```

这是最重要的软件/硬件边界之一。

---

# 24. 可选优化：Context 内 Resource Release Point

V2 可加入：

```text
Context M0:

Phase0:
6 Banks

Phase1:
4 Banks

Phase2:
2 Banks
```

IR：

```mlir
%a = tile.alloc ...
%b = tile.alloc ...

...

tile.release %a
tile.signal resource_release

...

tile.release %b
```

于是：

```text
M0 还未完成
```

但已经释放：

```text
2 Banks
```

Scheduler 可立即 admission 新 Context：

```text
M0:
[6 banks][4 banks][2 banks]
             │
             └──── G0 开始
```

这比“等整个 Context 结束”更灵活。

但建议：

> V1 先只做 Context completion release。
> V2 再加入 intra-context resource release event。

---

# 25. L1 Bandwidth 必须是一等资源

不能只做：

```text
sum(SPM bytes) <= L1 capacity
```

还必须：

```text
sum(read BW)  <= L1 read capacity
sum(write BW) <= L1 write capacity
```

例如：

```text
MatMul:
A read 256 B/cycle
B read 256 B/cycle

Gather:
Cache read 128 B/cycle
```

如果 bank capacity 虽然够：

```text
SPM + Cache <= 16 banks
```

但：

```text
read bandwidth > physical bank BW
```

则仍然不应该同时 admission，或者需要使用更 compact 的 schedule variant。

---

# 26. Prefetch 也进入 Resource Contract

对于 MatMul：

```text
A prefetch
B prefetch
MMA operand read
output write
```

都是 L1 bank BW consumer。

因此 Context profile 至少应该区分：

```cpp
struct L1BandwidthProfile {
    uint16_t compute_read_bw;
    uint16_t compute_write_bw;

    uint16_t prefetch_write_bw;
    uint16_t store_read_bw;
};
```

否则会出现：

```text
容量 admission PASS
```

但实际：

```text
MMA + Prefetch
```

把 L1 Bank port 打满。

---

# 27. 推荐的 IR 形态

## 27.1 L2 Context

```mlir
nest.context @gemm_ctx
    attributes {
      class = #nest.context_class<matrix_heavy>
    }
    resources {
      profiles = [
        #nest.resource_profile<
          name = "fast",
          spm_banks = 6,
          l1_read_bw = 384,
          l1_write_bw = 128,
          estimated_cycles = 600
        >,

        #nest.resource_profile<
          name = "balanced",
          spm_banks = 4,
          l1_read_bw = 256,
          l1_write_bw = 96,
          estimated_cycles = 720
        >
      ]
    }
{
    ...
}
```

Gather：

```mlir
nest.context @gather_ctx
    attributes {
      class = #nest.context_class<memory_irregular>
    }
    resources {
      spm_min_banks = 2,

      cache_min_banks = 0,
      cache_target_banks = 4,
      cache_max_banks = 8,

      cache_bypass = true
    }
{
    ...
}
```

---

# 28. Tile Admission Result

L2 / Tile resource manager 最终产生：

```text
Context M0
  profile       = balanced
  SPM_BANK_MASK = 0x000F

Context M1
  profile       = compact
  SPM_BANK_MASK = 0x0070

Tile Shared Cache
  CACHE_BANK_MASK = 0x0F00

Free
  mask = 0xF080
```

Bank mode：

```text
BANK_MODE[0..3]  = SPM
BANK_MODE[4..6]  = SPM
BANK_MODE[8..11] = CACHE
```

---

# 29. Cache Bank Mapping 的两个实现选项

## 方案 A：Bank Mask 改变时 invalidate

```text
CACHE_BANK_MASK:
0011
  ↓
1111
```

执行：

```text
invalidate cache tags
reconfigure mapping
```

因为 V1 cache read-only：

```text
no dirty writeback
```

优点：

- 最简单
- mapping 自由
- RTL 清晰

缺点：

- mode change 后 cold miss

### V1 推荐

**推荐采用。**

---

## 方案 B：固定地址到物理 Bank Hash

```text
bank = address_hash(address)
```

如果该 Bank 不是 CACHE：

```text
bypass -> L2
```

优点：

- 扩 Cache Bank 不需要 remap
- 不需要全 cache invalidate

缺点：

- cache capacity utilization 依赖 hash
- 部分地址天然 bypass

适合后续优化。

---

# 30. FIFO 当前建议

当前不要把 SRAM FIFO mode 和这套 SPM/Cache 双模式绑死。

V1：

```text
L1 Bank:
SPM / Cache

Datapath:
tiny elastic FIFO
```

例如：

```text
MMA
 │
 ▼
2~8 entry Elastic FIFO
 │
 ▼
Vector
```

用途：

- short backpressure
- producer-consumer decoupling
- fragment buffering

暂不加入：

```text
L1 SRAM-backed FIFO mode
```

是否需要，后续通过 profiling 决定。

---

# 31. MatMul -> Act 的中间数据处理

不应该：

```text
MatMul
  │
  ▼
[整个 output tensor]
  │
 FIFO
  │
  ▼
Act
```

应该：

```text
MatMul

 F0 ───────► Act(F0)
 F1 ───────► Act(F1)
 F2 ───────► Act(F2)

        ↑
   small elastic FIFO
```

FIFO 只用于：

```text
burst mismatch
short stall
rate jitter
```

如果长期：

```text
producer_bw > consumer_bw
```

任何有限 FIFO 最终都会满。

此时应该：

- throttle producer
- multi-context switch
- 增加 consumer throughput
- fuse activation
- compiler 改 storage strategy

而不是无限扩大 FIFO。

---

# 32. Scheduler V1 推荐算法

完整伪代码：

```python
MAX_RESIDENT = 4
LOOKAHEAD = 8


def on_context_complete(done):
    release(done)

    while resident_count() < MAX_RESIDENT:
        plan = find_best_admission_plan()

        if not plan:
            break

        for ctx, profile in plan:
            allocate(ctx, profile)
            launch(ctx)


def find_best_admission_plan():
    free_slots = MAX_RESIDENT - resident_count()

    candidates = ready_queue[:LOOKAHEAD]

    best_plan = None
    best_score = float("-inf")

    for subset in enumerate_subsets(
        candidates,
        max_size=free_slots
    ):
        for profile_selection in enumerate_profiles(subset):

            trial = resident + profile_selection

            if not feasible(trial):
                continue

            s = score(trial)

            if s > best_score:
                best_score = s
                best_plan = profile_selection

    return best_plan
```

---

# 33. Feasibility 条件

建议至少检查：

```text
1. Context slots
2. SPM capacity
3. Cache bank availability
4. L1 read bandwidth
5. L1 write bandwidth
6. DMA/prefetch bandwidth
7. Matrix engine pressure
8. Vector engine pressure
```

形式：

```python
def feasible(plan):
    if len(plan) > MAX_RESIDENT:
        return False

    if total_spm_banks(plan) + shared_cache_banks(plan) > 16:
        return False

    if total_l1_read_bw(plan) > L1_READ_LIMIT:
        return False

    if total_l1_write_bw(plan) > L1_WRITE_LIMIT:
        return False

    if total_dma_bw(plan) > DMA_LIMIT:
        return False

    return True
```

---

# 34. Scheduler Score V1

V1 不必复杂。

可以：

```python
score = (
      10.0 * predicted_engine_utilization
    +  5.0 * resource_complementarity
    +  2.0 * aging
    -  8.0 * l1_bw_overcommit_risk
    -  6.0 * memory_pressure
    -  4.0 * same_engine_contention
)
```

实际硬件可量化为整数 lookup table，不需要浮点。

---

# 35. 硬件 Scheduler 不需要做复杂搜索

推荐：

```text
Compiler / L2 software or firmware:
  bounded packing

Tile hardware:
  simple admission enforcement
  Context ready scoreboard
  engine arbitration
```

如果 L2 Controller 是可编程控制器，可以将 bounded lookahead packing 放在 L2 firmware。

Tile 只收到：

```text
admit context X
profile = balanced
SPM mask = ...
Cache mask = ...
```

这样 Tile RTL 不需要实现组合搜索。

---

# 36. 当前不建议做的设计

## 36.1 不做固定永久 80/20

不建议：

```text
80% 永远 SPM
20% 永远 Cache
```

因为 pure MatMul 时浪费 Cache Banks。

---

## 36.2 不做 first-come-first-served Bank Allocation

不建议：

```text
第一个 MatMul 来
  ↓
拿 16 Banks
  ↓
后面的 Gather 自己想办法
```

应该经过 admission。

---

## 36.3 不做 live SPM migration

不建议：

```text
MatMul live SPM
  ↓
强行搬走
  ↓
切 Cache
```

---

## 36.4 不让 Compiler 预测精确完成顺序

不建议：

```text
compiler:
M0 600 cycles
G0 500 cycles
所以 G0 必定先结束
```

这个假设对于 irregular workload 不可靠。

---

## 36.5 不做全动态 SRAM FIFO spill

不建议：

```text
elastic FIFO full
  ↓
自动 spill SRAM FIFO
  ↓
SRAM FIFO full
  ↓
自动 spill SPM
```

会引入复杂 ordering / replay。

---

# 37. V1 Architecture Freeze 建议

## L1 SRAM

```text
16 Banks
Every Bank:
SPM / Cache dual-capable
```

### Cache

```text
Tile-shared
Read-only
Bypassable
Shared MSHR
```

### FIFO

```text
tiny elastic FIFO only
SRAM-backed FIFO TBD
```

---

## Multi-Context

```text
Max resident = 4

Sliding resident set

Completion-driven admission
```

---

## Compiler

生成：

```text
Context Resource Contract

- SPM min / target / max
- Cache min / target / max
- L1 bandwidth
- engine pressure
- schedule variants
- expected cycles
```

---

## L2 Scheduler

使用：

```text
bounded lookahead = 8

small enumeration
+
backfill
+
aging
```

---

## Tile

负责：

```text
actual bank allocation
context slot allocation
bank mode programming
scoreboard
engine scheduling
```

---

# 38. 建议的性能实验

需要把之前只看 Context 数量的模拟扩展成资源-aware。

至少增加以下指标。

## 38.1 Admission

```text
average_resident_contexts

admission_blocked_cycles

admission_blocked_by_spm

admission_blocked_by_cache

admission_blocked_by_l1_bw

head_of_line_block_cycles
```

---

## 38.2 SRAM

```text
spm_bank_utilization

cache_bank_utilization

free_bank_ratio

bank_mode_switch_count

bank_fragmentation

peak_live_spm_bytes
```

---

## 38.3 Cache

```text
l1_irregular_cache_hit_rate

l1_cache_bypass_rate

l2_traffic_due_to_bypass

mshr_occupancy

memory_stall_cycles
```

---

## 38.4 Context Profile

```text
profile_fast_selected
profile_balanced_selected
profile_compact_selected
```

---

## 38.5 Engine

```text
matrix_utilization
vector_utilization
dma_utilization

matrix_memory_overlap
matrix_vector_overlap
```

---

# 39. 推荐对比实验

至少比较：

| Config | Admission                                        |
| ------ | ------------------------------------------------ |
| A      | Strict FIFO + fixed resource                     |
| B      | FIFO + backfill                                  |
| C      | Bounded lookahead + fixed profile                |
| D      | Bounded lookahead + multi-profile                |
| E      | Bounded lookahead + multi-profile + Cache bypass |
| F      | E + intra-context release point                  |

建议 SPM/Cache 探索：

```text
16 / 0
14 / 2
12 / 4
10 / 6
8  / 8
```

但这些是 runtime plan，不是硬件固定比例。

---

# 40. 需要重点测试的 Workload Combination

除了单 Context benchmark，还应加入混合队列：

```text
MatMul
MatMul
MatMul
Gather
MatMul
Gather
Vector
MatMul
```

以及：

```text
Gather
Gather
MatMul
Vector
Gather
MatMul
```

重点观察：

> Scheduler 是否能够避免前三个 MatMul 使用最激进的 SPM profile，从而让后续 irregular Context 长时间 admission blocked。

---

# 41. 一个重要的 Compiler 优化：Bank-Friendly Lifetime Packing

普通 memory allocator 只关注：

```text
minimize bytes
```

这里应该再加入：

```text
minimize bank fragmentation
maximize whole-bank release
```

例如：

错误：

```text
Bank0:
long-lived + short-lived

Bank1:
long-lived + short-lived

Bank2:
short-lived
```

可能导致很多 Bank 长时间无法释放。

推荐：

```text
Bank0:
long-lived

Bank1:
long-lived

Bank2:
short-lived

Bank3:
short-lived
```

短生命周期 buffer 结束后：

```text
Bank2 / Bank3
whole-bank FREE
```

可以立即：

```text
SPM -> Cache
```

因此后续 compiler memory planner 可以加入：

> **Whole-Bank Release Cost**

---

# 42. 中长期扩展方向

## 42.1 Phase-Aware Resource Contract

Context 可以声明：

```text
Phase0:
SPM 6 Banks

Phase1:
SPM 4 Banks

Phase2:
SPM 2 Banks
```

运行时可在 Context 尚未结束时 admission 新 Context。

---

## 42.2 Profile-Guided Cache Sizing

对 Gather：

```text
Cache Banks -> Hit Rate
```

建立 offline profile：

```text
2 Banks -> 32%
4 Banks -> 55%
6 Banks -> 67%
8 Banks -> 71%
```

Compiler/L2 scheduler 根据边际收益决定是否继续增加 Cache。

---

## 42.3 Learned / Adaptive Scheduler

不建议 V1 使用。

长期可考虑根据 runtime counters：

```text
cache miss rate
matrix occupancy
memory stall
context latency
```

在线更新：

```text
compatibility score
resource profile score
```

但必须保留 deterministic fallback。

---

# 43. 当前设计的核心结论

当前最适合的设计不是：

```text
Compiler 预先决定固定 Mix Window
```

也不是：

```text
Hardware 完全动态、first-come-first-served
```

而是：

```text
             Compile Time
                 │
                 ▼
     Context Resource Contract
                 │
                 ▼
          Runtime Ready Queue
                 │
                 ▼
 Completion-Driven Bounded Packing
                 │
                 ▼
        Sliding Resident Set
                 │
                 ▼
         Dual-Mode L1 Banks
```

可以总结成一句：

> **Compiler 负责提供资源边界与可选 schedule，Runtime 负责解决真实完成时序。**

---

# 44. 当前推荐冻结边界

建议当前阶段冻结：

1. **所有 L1 Bank 支持 SPM / Cache 双模式。**
2. **同一 Bank 同一时刻只能属于一种 mode。**
3. **live SPM Bank 不允许直接切 Cache。**
4. **Gather Cache V1 read-only、Tile-shared、可 bypass。**
5. **SRAM-backed FIFO 暂不纳入 L1 Bank mode。**
6. **最大 resident Context = 4。**
7. **Context completion 触发新的 admission。**
8. **Compiler 只分析 Context 内资源，不预测跨 Context 精确 wall-clock overlap。**
9. **Compiler 为 Context 生成多个 resource/schedule profile。**
10. **L2 使用 bounded lookahead + backfill + aging 做在线 packing。**
11. **SPM 容量与 L1 bandwidth 同时作为 admission resource。**
12. **禁止 live-bank migration。**
13. **80/20 仅作为 mixed workload baseline policy，不写死到 RTL/ISA。**
14. **后续通过 simulation 决定是否需要 intra-context resource release 与 SRAM FIFO mode。**

---

# 45. 建议下一步

建议下一阶段直接做一个 **resource-aware multi-context scheduler simulator**。

第一版参数：

```text
L1:
16 Banks

Resident:
4 Contexts

Ready Lookahead:
8

Profiles:
2~3 / Context

Cache:
0~8 Banks
Read-only
Bypassable

Scheduler:
Bounded Enumeration
+ Backfill
+ Aging
```

先不要继续增加硬件结构。

先回答三个问题：

1. **Bounded lookahead admission 是否显著降低 bank admission stall？**
2. **MatMul compact profile 的单 Context 性能损失，是否能被 multi-context throughput 收益抵消？**
3. **Gather L1 Cache 在 2/4/6/8 Bank 下的性能收益是否足以证明 dual-mode Bank 的面积成本？**

如果这三个结果成立，再进入 RTL 级别的：

```text
Bank mode mux
Tag RAM
Shared MSHR
Cache frontend
Admission interface
```

设计会更稳妥。

---

## 最终架构一句话

> **NEST Tile 使用全 Bank SPM/Cache 双用途 L1 SRAM；Compiler 为每个 Context 提供资源合同与多个执行 profile；L2 在 Context 完成事件上采用 bounded lookahead 的在线多资源 packing 维持最多 4 个 sliding resident contexts，从而在不预测动态完成顺序、不迁移 live SRAM 数据的前提下，实现 MatMul、Gather 等 regular/irregular Context 的高效共驻与资源复用。**
