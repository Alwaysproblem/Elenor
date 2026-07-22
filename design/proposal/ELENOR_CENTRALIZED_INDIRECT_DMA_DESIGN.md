# ELENOR 集中式 Indirect DMA：Dynamic Gather / Scatter 设计方案

> 文档状态：Design Proposal / Agent Implementation Guide
> 目标版本：ELENOR V1 设计冻结候选
> 更新日期：2026-07-13
> 适用对象：Architecture、Compiler、Runtime、Simulator、RTL、Verification Agent

---

## 1. 执行摘要

本文在 ELENOR 原始内存架构上增加受限但实用的 Dynamic Gather / Scatter 能力。

原始架构保持不变：

```text
CPU
  → Command Queue
  → Tile Group PC / Sequencer
  → Group Global DMA：HBM ↔ L2
  → Tile MFE：L2 ↔ L1
  → BOA / EVU
```

本方案的核心决策如下：

1. **Tile 不直接访问 HBM**，不增加 GPU 式 demand-load、cache miss、MSHR 或 per-Tile global LSU。
2. **L2 继续作为显式管理的 Tile Group SRAM**，不是硬件 cache，不采用 80%/20% cache 分区。
3. 在现有 Group Global DMA 中增加 **Indirect DMA Engine（IDE）**：
   - `GATHER`：从 HBM 的动态地址集合读取，按线性布局写入 L2；
   - `SCATTER`：从 L2 读取线性 values，写入 HBM 的动态地址集合。
4. CPU 只提交高层 descriptor 和 DAG，不逐 index 计算、读取或下发 DMA。
5. 动态 index/count 由 Tile 产生时，通过 event 直接唤醒预先入队的 Group DMA command，**不经过 CPU round-trip**。
6. V1 基线 MUST 支持 Gather 和 Unique Scatter；V1 条件功能包括：
   - 单 Group、单 descriptor 内的 Ordered Overwrite Scatter；
   - 通过 destination ownership + 本地化归约实现的 Scatter Add/Max/Min。
7. V1 不支持：
   - 任意跨 Tile Group 的无序 atomic update；
   - Tile 直接发起 HBM load/store；
   - 任意 pointer chasing；
   - 完整 GPU 式 global atomic subsystem。

一句话定义本方案：

> **由 CPU/编译器描述任务，由 Tile Group 集中执行动态间接访存，由显式 L2 slot 和 event 保持可预测性。**

---

## 2. 基线架构与术语

### 2.1 内存层级

| 名称 | 作用域        | 角色                          | 管理方式                      |
| ---- | ------------- | ----------------------------- | ----------------------------- |
| L1   | Tile          | Tile 本地 operand/output SRAM | 编译器分配，UCE/MFE 执行      |
| L2   | Tile Group    | 多 Tile 共享 SRAM             | 编译器/runtime 显式 slot 分配 |
| HBM  | Device Global | 模型、输入输出和大型 tensor   | Group Global DMA 访问         |

本文中的 L2 不是 tag-based cache。每个 indirect buffer 都必须绑定一个显式 `GroupL2Ref`（概念上是 group-level allocation/ref，而不是 Tile L1 Slot Frame），至少包含：

- 地址和容量；
- owner；
- 读写权限；
- lifetime；
- producer `EventRef`；
- consumer/release `EventRef`。

### 2.2 控制层级

| 组件                 | 职责                                                  |
| -------------------- | ----------------------------------------------------- |
| Host CPU / Runtime   | 提交 DAG、descriptor、资源配置和异常慢路径            |
| Tile Group Sequencer | 等待 event、启动 DMA、维护 bounded loop、触发后继任务 |
| Group Global DMA     | HBM↔L2 线性、strided 和 indirect 搬运                 |
| Tile UCE             | 执行 Tile Program、控制 L1 级数据和计算引擎           |
| Tile MFE             | L2↔L1、layout transform、window generation            |
| BOA / EVU            | dense contraction 与 vector compute                   |

### 2.3 规范性术语

- **MUST**：V1 正确性所必需。
- **SHOULD**：推荐实现，除非有明确理由偏离。
- **MAY**：可选优化。
- **Unique Scatter**：有效 index 映射出的目标区间互不重叠。
- **Ordered Overwrite**：重复目标按逻辑 sequence 顺序处理，最后一个 update 生效。
- **Ownership**：一个目标地址在给定执行 epoch 中只允许一个 Tile Group 更新。

---

## 3. 目标与非目标

### 3.1 目标

1. 支持 `Gather`、`GatherElements`、受限 `GatherND`。
2. 支持 `Scatter`、`ScatterElements`、受限 `ScatterND`。
3. 支持 embedding lookup、TopK→Gather、routing table lookup 等动态 index 场景。
4. 支持无冲突 Scatter 和可静态/运行时分区的 Scatter Reduce。
5. 保持所有 global memory traffic 由 Tile Group 层统一控制。
6. 允许 Gather/Scatter 与 Tile compute 通过双缓冲重叠。
7. 保证 queue backpressure、L2 容量不足和 HBM stall 不导致循环等待。
8. 为 compiler/runtime/simulator/RTL 提供一致的语义契约。

### 3.2 非目标

1. 不追求 GPU load/store 指令级通用性。
2. 不支持任意链式 pointer chasing：

   ```cpp
   p1 = table0[index];
   p2 = table1[p1];
   value = table2[p2];
   ```

3. 不在 V1 中提供跨 Group FP32/BF16/FP16 global atomic 完整语义。
4. 不保证任意重复 Scatter 的确定性；确定性必须由 descriptor 明确请求并满足约束。
5. 不允许 CPU 逐元素读取设备产生的 index 并重新提交 DMA。

---

## 4. 算子覆盖与支持边界

| 算子模式                            | V1 状态           | Lowering 策略                                  |
| ----------------------------------- | ----------------- | ---------------------------------------------- |
| `Gather(axis=0)` / Embedding lookup | 支持              | 一个 index 对应一个连续 block                  |
| `GatherElements`                    | 支持受限          | 线性化为 scalar/block gather                   |
| `GatherND`                          | 支持受限          | 编译器预计算线性 index；复杂坐标由 EVU 生成    |
| Unique `ScatterElements`            | 支持              | 直接 Group Scatter DMA                         |
| Unique `ScatterND`                  | 支持受限          | 线性 index + block scatter                     |
| Ordered overwrite                   | 单 Group 条件支持 | sequence-preserving conflict handling          |
| `scatter_add/max/min`               | 条件支持          | ownership partition + local reduce + writeback |
| Histogram                           | 条件支持          | 按 bin owner 分区；热点需 combine              |
| Embedding backward                  | 条件支持          | owner routing + reduction，不直接随机 atomic   |
| 跨 Group 任意 atomic scatter        | 不支持            | 软件 log/merge fallback 或未来 V2 atomic       |
| Pointer-chasing gather/scatter      | 不支持            | CPU/可编程控制核慢路径或图重写                 |

V1 的功能判定不是“能否表达一个地址”，而是同时满足：

- 地址可由 `base + index × stride` 表达；
- 每个 index 对应一个连续 block；
- count 有编译期上界；
- L2 slot 和 scratch 能在启动前预留；
- Scatter 的跨 Group 冲突可被证明不存在或被 ownership 消除。

---

## 5. 总体硬件结构

在现有 Group Global DMA 内增加 Indirect DMA Engine，不新增独立的 HBM master。

```text
                       Tile Group
+----------------------------------------------------------------+
| Group Sequencer                                                |
|   - descriptor fetch/validate                                  |
|   - wait event / completion event                              |
|   - bounded chunk loop                                         |
|               |                                                |
|               v                                                |
| Group Global DMA                                               |
|   +----------------------+      +----------------------------+ |
|   | Linear/Strided DMA   |      | Indirect DMA Engine        | |
|   +----------------------+      | - Index Fetch              | |
|                                 | - AGU                      | |
| L2 Explicit SRAM <------------> | - Gather Reorder           | |
|   indices / values / result     | - Scatter Combine          | |
|   scratch / count               | - Request/Write Queue      | |
|                                 +-------------+--------------+ |
+------------------------------------------------|---------------+
                                                 v
                                      Device Memory Controller
                                                 |
                                                HBM
```

### 5.1 Indirect DMA 子模块

| 子模块                  | Gather                       | Scatter                                |
| ----------------------- | ---------------------------- | -------------------------------------- |
| Index Fetch             | 从 L2 读取 index             | 从 L2 读取 index                       |
| Linear Data Reader      | 不需要                       | 从 L2 读取 values                      |
| AGU                     | 生成 HBM source address      | 生成 HBM destination address           |
| Line Grouper            | 合并相同 HBM read line       | 合并相同 HBM write line                |
| Reorder Table           | 将乱序返回写到正确 L2 offset | 通常不需要                             |
| Conflict/Combine Buffer | 去重相同 read line           | 检测重复 target，执行 overwrite/reduce |
| Outstanding Table       | 追踪 HBM read                | 追踪 HBM write acknowledgement         |
| Completion Unit         | L2 可见后完成                | HBM 可见性满足后完成                   |

### 5.2 建议的初始可参数化配置

以下是 cycle simulator 的默认起点，不是最终物理参数：

| 参数                      |            建议默认值 | 说明                          |
| ------------------------- | --------------------: | ----------------------------- |
| Index decode rate         |         4 index/cycle | 后续通过带宽仿真调整          |
| Index FIFO                |            32 entries | 吸收 L2 bank stall            |
| Gather reorder entries    |                    32 | 与 outstanding read 对齐      |
| Scatter combine entries   |                    16 | set-associative/hash 组织     |
| HBM outstanding reads     |                    32 | Gather latency hiding         |
| HBM outstanding writes    |                    32 | Scatter latency hiding        |
| Maximum block bytes/index |            4096 bytes | 大于该值由多个 sub-block 执行 |
| L2 DMA data width         | 与现有 Group DMA 一致 | 避免新增宽数据通路            |

所有 FIFO、outstanding table 和 combine buffer MUST 使用 credit/backpressure，不能假设 HBM 固定延迟。

---

## 6. 统一地址语义

### 6.1 Gather

对第 `i` 个 index：

```text
src_i = indexed_base + indices[i] * indexed_stride
dst_i = linear_base  + i          * linear_stride
copy block_bytes from src_i(HBM) to dst_i(L2)
```

这既支持 scalar gather，也支持 embedding row gather：

```text
indices = [7, 2, 10]
block_bytes = embedding_dim * sizeof(dtype)

HBM row 7  ──→ L2 row 0
HBM row 2  ──→ L2 row 1
HBM row 10 ──→ L2 row 2
```

### 6.2 Scatter

对第 `i` 个 index：

```text
src_i = linear_base  + i          * linear_stride
dst_i = indexed_base + indices[i] * indexed_stride
copy/reduce block_bytes from src_i(L2) to dst_i(HBM)
```

### 6.3 越界规则

Descriptor MUST 提供 `indexed_extent`，用于验证：

```text
0 <= index
index * indexed_stride + block_bytes <= indexed_extent
```

V1 支持三种策略：

| 策略           | Gather           | Scatter             |
| -------------- | ---------------- | ------------------- |
| `TRAP`         | command fault    | command fault       |
| `ZERO_OR_DROP` | Gather 写零      | Scatter 丢弃 update |
| `CLAMP`        | 可选，不推荐默认 | 不支持              |

`TRAP` MUST 是安全/调试默认值；部署版本可由 compiler 明确选择 `ZERO_OR_DROP`。

---

## 7. Descriptor ABI 草案

### 7.1 设计原则

1. Descriptor 大小固定为 128 bytes，按 64 bytes 对齐。
2. Descriptor 一经 `VALID`，CPU/producer 不得修改。
3. 变长 count 通过 L2 scalar slot 提供，并受 `max_count` 限制。
4. V1 indices MUST 位于 L2；CPU 已知的 indices 先通过 Linear DMA 预取到 L2。
5. Descriptor 地址与 tensor extent 必须由 runtime 做设备地址空间验证。
6. 分块执行使用 Group Sequencer 每次 launch 生成的内部 execution context（`start_offset`、`chunk_count`、`slot_selector`、chunk-local event/release identity）；不得回写或改写已 `VALID` 的 base descriptor。

### 7.2 C ABI

```cpp
#include <cstddef>
#include <cstdint>

enum class IndirectOpcode : std::uint8_t {
  kGather = 0,
  kScatter = 1,
};

enum class ReductionOp : std::uint8_t {
  kNone = 0,
  kAdd = 1,
  kMax = 2,
  kMin = 3,
};

enum class ConflictPolicy : std::uint8_t {
  kUniqueGuaranteed = 0,
  kOrderedLast = 1,
  kLocalCombine = 2,
  kOwnerExclusive = 3,
};

enum IndirectFlags : std::uint16_t {
  kCountFromL2       = 1u << 0,
  kBoundsCheck       = 1u << 1,
  kDropOrZeroOnOob   = 1u << 2,
  kSignedIndex       = 1u << 3,
  kDeterministic     = 1u << 4,
  kEnableDescCrc     = 1u << 5,
};

struct EventRef {
  std::uint32_t event_id;
  std::uint32_t sequence;
};

struct alignas(64) IndirectDmaDesc {
  // 0x00
  std::uint8_t opcode;
  std::uint8_t version;
  std::uint16_t flags;
  std::uint8_t index_width;       // 4 or 8 bytes
  std::uint8_t element_size;      // 1, 2, 4, 8 or 16 bytes
  std::uint8_t reduction_op;
  std::uint8_t conflict_policy;

  // 0x08
  std::uint32_t count_imm;
  std::uint32_t max_count;

  // 0x10
  std::uint64_t indexed_base;     // HBM source for gather, HBM dst for scatter
  std::uint64_t linear_base;      // L2 dst for gather, L2 source for scatter
  std::uint64_t indices_l2_addr;
  std::uint64_t dynamic_count_l2_addr;
  std::uint64_t scratch_l2_addr;
  std::uint64_t sequence_l2_addr; // optional; zero means implicit i-order

  // 0x40
  std::uint32_t block_bytes;
  std::uint32_t indexed_stride;
  std::uint32_t linear_stride;
  std::uint32_t scratch_size;

  // 0x50
  EventRef wait_ref;              // shared Runtime ABI tuple; {0,0} means no wait
  EventRef completion_ref;        // terminal descriptor completion identity
  std::uint32_t owner_mask;
  std::uint16_t owner_shift;
  std::uint16_t reserved0;

  // 0x68
  std::uint64_t indexed_extent;   // HBM-side bounds
  std::uint64_t user_tag;

  // 0x78
  std::uint32_t descriptor_crc32;
  std::uint32_t reserved1;
};

static_assert(sizeof(IndirectDmaDesc) == 128);
static_assert(alignof(IndirectDmaDesc) == 64);
```

### 7.3 ABI 校验规则

Group Sequencer/IDE 在接受 descriptor 前 MUST 检查：

- `version` 是否支持；
- opcode、index width、element size、reduction op 合法；
- `block_bytes > 0` 且为 `element_size` 的整数倍；
- `linear_stride >= block_bytes`；
- `count <= max_count`；
- L2 indices/linear/scratch 区间不越界（依据绑定的 `GroupL2Ref` 做 range/permission/producer/release 检查）；
- HBM tensor extent 不越界；
- Scatter reduce 是否具有合法 ownership contract；
- requested scratch 是否已预留；
- `wait_ref` / `completion_ref` 的 `event_id + sequence` 与共享 Runtime ABI 一致且未过期；
- 启用时 descriptor CRC 正确。

---

## 8. Event、可见性与动态执行

### 8.1 Event 语义

Indirect DMA 不再定义私有的 “裸 event id + 独立 epoch token” 格式，而是直接复用共享 Runtime ABI 的二元组：

```cpp
struct EventRef {
  std::uint32_t event_id;
  std::uint32_t sequence;
};
```

IDE 必须先 acquire descriptor `wait_ref`，再读取 `dynamic_count_l2_addr` / `indices_l2_addr` 并生成 chunk execution context；终止时只允许写回 descriptor `completion_ref` 这一处 terminal completion identity。wait side 把 `sequence` 解释为 `expected_sequence`；completion/signal side 写回同一 `event_id + sequence` 身份，避免 stale wakeup。

### 8.2 Gather completion

`gather_done` 只能在以下条件全部满足后触发：

1. 所有 HBM read 已返回或 fault；
2. 所有返回数据已经写入目标 L2 slot；
3. L2 write 对后续 Tile MFE 可见；
4. reorder/outstanding entry 已释放；
5. descriptor 状态已经原子地写为 DONE 或 FAULT。

### 8.3 Scatter completion

`scatter_done` 只能在以下条件全部满足后触发：

1. 所有 values/indices 已从 L2 消费；
2. combine/write queue 已 drain；
3. 所有 HBM write 已获得目标 memory scope 所需的 acknowledgement；
4. 后续 device command 能观察到这些写入；
5. descriptor 状态已经写为 DONE 或 FAULT。

CPU 读取最终结果前，runtime MUST 再执行 device-to-host visibility fence；`scatter_done` 本身不自动等价于 host coherence。

### 8.4 动态 index/count 不经过 CPU

推荐 DAG：

```text
Tile ProduceIndices
    │ writes indices/count into L2
    │ release → e_indices_ready
    v
Group Indirect DMA / IDE (pre-enqueued; acquire descriptor wait_ref = e_indices_ready)
    │ read count/indices from L2，create chunk contexts，execute gather/scatter
    │ signal descriptor completion_ref = e_indirect_done
    v
Tile ConsumeResult or next Group command
```

伪 IR：

```mlir
%idx_ready = elenor.launch_tile @produce_indices(
    %input,
    indices_out = %indices_l2,
    count_out = %count_l2
)

%gather_done = elenor.group.indirect_dma {
    opcode = #elenor<gather>,
    indexed_base = %global_source,
    linear_base = %gather_result_l2,
    indices = %indices_l2,
    count = %count_l2,
    max_count = 4096,
    block_bytes = 128,
    conflict = #elenor<unique>,
    wait_ref = %idx_ready,
    completion_ref = #elenor.event<gather_done>
}

elenor.launch_tile @consume_gather(
    %gather_result_l2
) await(%gather_done)
```

CPU 在图启动前即可把三段 command 全部入队。

---

## 9. Gather 微架构与状态机

### 9.1 状态机

```text
IDLE
 → FETCH_DESC
 → WAIT_EVENT
 → VALIDATE_AND_RESERVE
 → READ_DYNAMIC_COUNT
 → FETCH_INDICES
 → GENERATE_AND_GROUP_REQUESTS
 → ISSUE_HBM_READS
 → REORDER_AND_WRITE_L2
 → DRAIN
 → COMPLETE | FAULT
```

### 9.2 请求合并

对于 scalar 或小 block gather，多个 index 可能位于同一个 HBM line：

```text
index addresses: 0x1004, 0x1008, 0x100C, 0x3000

HBM line 0x1000: one read, extract three elements
HBM line 0x3000: one read, extract one element
```

每个合并项必须记录多个 consumer：

```cpp
struct GatherConsumer {
  std::uint32_t logical_seq;
  std::uint32_t dst_l2_offset;
  std::uint16_t line_offset;
  std::uint16_t bytes;
};
```

相同 index 可以共享 HBM read，但必须复制到各自的 L2 destination。

### 9.3 大 block

当 `block_bytes` 大于 HBM line 时，IDE 将一个 index 展开为多个连续 sub-request：

```text
index request
  → sub-block 0
  → sub-block 1
  → ...
```

完成计数以 `(index, sub-block)` 为单位，只有一个 index 的全部 sub-block 写入 L2 后，该 index 才完成。

---

## 10. Scatter 微架构与冲突语义

### 10.1 Unique Scatter

`kUniqueGuaranteed` 是 compiler/runtime contract：有效目标区间不得重叠。

Release 硬件 MAY 不做完整重复检测，但 simulator 和 debug mode MUST 检测并报告：

```text
ERR_UNIQUE_CONTRACT_VIOLATION
```

Unique Scatter 可以自由重排 write，以获得更好的 line combining 和 HBM bank parallelism。

### 10.2 Ordered Overwrite

语义：

```cpp
for (i = 0; i < count; ++i)
  dst[index[i]] = value[i];
```

相同目标出现多次时，逻辑 sequence 最大的 value 最终生效。

单 descriptor 内可以在 combine buffer 中保留：

```text
(target_address, largest_sequence, value)
```

注意：

- 该保证只覆盖单个 Tile Group、单个 descriptor；
- 多 Group 写相同 target 在 V1 中非法；
- `kDeterministic` 要求禁止破坏 last-writer 语义的跨 entry 重排；
- 对部分 cache line overwrite，必须正确维护 byte-enable。

### 10.3 Scatter Combine Buffer

建议 entry：

```cpp
struct ScatterCombineEntry {
  std::uint64_t line_address;
  ByteMask valid_bytes;
  DataLine data;
  SequenceMeta sequence;
  bool dirty;
};
```

主要功能：

- 合并同一 HBM line 上的多个小写；
- 对 Ordered Overwrite 选择最大 sequence；
- 对 Local Combine 执行 add/max/min；
- buffer 满时选择可安全 flush 的 entry；
- 对同一 target 的未完成 write 建立 hazard，防止错误穿越。

### 10.4 写放大

完全随机的 4-byte Scatter 即使功能正确，也可能产生严重写放大。需要记录：

```text
write_amplification = external_bytes_written / useful_value_bytes
```

如果 memory system 以 64-byte line 处理，而每个 line 只有一次 4-byte update，理论有效比例只有：

```text
4 / 64 = 6.25%
```

IDE 不能假设下游天然支持任意 byte-enable：

- 如果 Memory Controller 支持 masked/partial write，IDE 可以直接提交带 byte mask 的 transaction；
- 如果不支持，IDE 必须执行受保护的 read-merge-write，或者只允许完整 transaction/block 写；
- read-merge-write 期间同一 line 必须由唯一 owner 持有，否则仍然会丢失并发更新；
- Runtime 必须根据 memory-controller capability 拒绝不支持的 descriptor，不能静默扩大写入范围。

这不是单纯增加 DMA 带宽就能解决的问题。Compiler SHOULD 在以下场景启用 sort/bin/log-reduce：

- `block_bytes <= 16`；
- count 很大；
- index 局部性差；
- 重复率高或热点明显。

阈值必须由 simulator/profile 决定，不能在初版编译器中永久硬编码。

---

## 11. Scatter Reduce 的正确实现（V1 条件功能）

### 11.1 为什么 Group 内 combine 不充分

以下两个 Group 同时执行：

```text
Group 0: dst[7] += 3
Group 1: dst[7] += 5
```

即使每个 Group 内部正确 combine，最终仍可能丢失更新。因此：

> **Local Combine 是带宽优化，不是跨 Group atomic 正确性的替代品。**

### 11.2 推荐方案：Destination Ownership

定义：

```text
owner(index) = partition(index) or hash(index) % num_owner_groups
```

执行分为三阶段：

1. **Partition/Route**：将 `(index, value, sequence)` 发给目标 owner Group；
2. **Local Reduce**：owner Group 对相同 index 执行 add/max/min；
3. **Commit**：由唯一 owner 写回目标 HBM 区域。

```text
Producer Groups
      │  partition by destination owner
      v
Owner Group update logs / L2 bins
      │  local sort/hash reduce
      v
Unique reduced records
      │
      v
Group Scatter DMA or linear writeback → HBM
```

### 11.3 两种 commit 路径

#### 路径 A：目标初始为 reduction identity

例如输出预先清零，owner 得到每个 index 的完整 update 集合：

```text
reduce all updates → one unique value per index → Unique Scatter
```

#### 路径 B：目标包含旧值

优先使用显式分块：

```text
Linear DMA: owner destination block HBM → L2
EVU: apply all owned updates in L2
Linear DMA: L2 block → HBM
```

若目标块不能放入 L2，按 destination range 分 chunk。V1 不通过随机 HBM read-modify-write 冒充 atomic。

### 11.4 浮点确定性

FP add 不满足严格结合律。若模型要求 bitwise deterministic：

- 编译器必须生成固定 owner；
- update 必须按固定 sequence 排序；
- reduction tree/order 必须固定；
- 禁止不同运行之间改变 Group 数量或 partition 策略。

确定性模式会牺牲吞吐量，应通过 op attribute 显式开启。

---

## 12. 多 Tile、多 Group 内存一致性规则

### 12.1 V1 合法性规则

一个 Scatter descriptor 合法，当且仅当满足以下至少一项：

1. 编译器证明目标区间与其他并发 descriptor 不重叠；
2. 所有可能冲突的 update 路由到同一个 owner Group；
3. descriptors 之间存在完成 event/barrier，保证串行；
4. 操作被降低为软件 log/merge，不直接更新最终目标。

否则 runtime MUST 拒绝启动，而不是产生未定义数据。

### 12.2 Alias 信息

Host/Group IR 应携带：

```mlir
#elenor.mem_effect<
  reads  = [%indices_l2, %values_l2],
  writes = [%global_output],
  target_range = [%base, %extent],
  ownership = #elenor<group_exclusive>
>
```

对于无法静态确定的 index，`target_range` 至少给出 conservative tensor extent。

### 12.3 与并发 Linear DMA 的关系

同一 HBM target range 上：

- Scatter write 与 Linear DMA read/write 不得无序并发；
- Group Sequencer 必须通过 range scoreboard 或 compiler event 保证顺序；
- 不同 range 可并发；
- 只判断 descriptor base 不足以证明无 alias，必须使用 extent/range。

---

## 13. L2 Slot 与流水线设计

### 13.1 显式 slot

一个动态间接访存 pipeline 至少包括：

```text
index_slot[2]
value_or_result_slot[2]
scratch_slot[2]       // only when combine/partition needs it
count_slot[2]
```

这些 slot 由 compiler/runtime 显式分配，没有 cache eviction，也不存在 demand path 抢占问题。

### 13.2 Gather 双缓冲

```text
时间 ─────────────────────────────────────────────>

Group DMA : Gather chunk 0 | Gather chunk 1 | Gather chunk 2
Tile      :                | Compute chunk 0| Compute chunk 1
L2 slot   :      A         |       B         |       A
```

事件关系：

```text
gather_done[k]     → tile_compute[k]
tile_consumed[k]   → slot[k % 2] reusable
```

IDE 在目标 slot 获得 release event 前不得覆盖该 slot。

### 13.3 Scatter 双缓冲

```text
时间 ─────────────────────────────────────────────>

Tile      : Produce chunk 0 | Produce chunk 1 | Produce chunk 2
Group DMA :                 | Scatter chunk 0 | Scatter chunk 1
L2 slot   :        A        |        B        |        A
```

事件关系：

```text
tile_produced[k] → scatter[k]
scatter_done[k]  → slot[k % 2] reusable
```

### 13.4 Dynamic count 与 bounded chunk loop

对于运行时 count，Group Sequencer 只负责 issue `dma.indirect`；真正的顺序是：

1. IDE acquire descriptor `wait_ref`；
2. IDE 读取 `0 <= count <= max_count`；
3. IDE 在本地创建 chunk execution context；
4. IDE 完成所有 chunk 后 signal descriptor `completion_ref`。

```cpp
const auto count = ide_read_count_after_wait_ref(desc);
for (std::uint32_t offset = 0; offset < count; offset += chunk_capacity) {
  const auto chunk_id = offset / chunk_capacity;
  const auto n = min(chunk_capacity, count - offset);
  wait(slot_reusable[chunk_id % 2]);
  ide_launch_chunk(/*base_desc=*/desc,
                   /*start_offset=*/offset,
                   /*chunk_count=*/n,
                   /*slot_selector=*/chunk_id % 2,
                   /*chunk_id=*/chunk_id);
}
signal(desc.completion_ref);
```

这里的 `start_offset`、`chunk_count`、`slot_selector` 和 chunk-local wait/release identity 是 IDE/TGS 内部 execution context，不是通过改写已 `VALID` descriptor 来实现。slot/release event 一律按 `chunk_id % 2` 选择，CPU 不参与每个 chunk。

---

## 14. 资源预留与无死锁要求

### 14.1 接受命令前的原子预留

IDE 将 command 从 `WAIT_EVENT` 转为 `RUNNING` 前，MUST 同时获得：

- descriptor state entry；
- 最小 index FIFO credit；
- 最小 outstanding entry；
- gather reorder 或 scatter combine entry 的最低保证；
- 目标/源 L2 slot 权限；
- scratch slot；
- completion event slot。

如果不能同时获得，保持在 `WAIT_RESOURCE`，不得部分持有资源后再等待另一项无限期资源。

### 14.2 Progress 规则

1. 已接受的 HBM response MUST 比新 request 优先获得返回通路。
2. Scatter dirty combine entry MUST 有独立的 drain credit。
3. Completion/error writeback MUST 有保留 credit，不能被普通 DMA 填满。
4. HBM backpressure 时，L2 reader 必须能够停止，不能覆盖 FIFO。
5. Fault command 必须 drain 或 cancel 所有已发请求，再触发唯一一次 fault completion。
6. Command queue 仲裁 SHOULD 使用 age-based 或 bounded round-robin，防止 indirect DMA 饥饿。

### 14.3 建议验证的不变量

```text
reserved_entries <= physical_entries
issued_requests - retired_requests == outstanding_count
each accepted descriptor completes_or_faults exactly once
no L2 slot is written while owned by an uncompleted consumer
scatter_done implies write_queue_empty
gather_done implies reorder_table_empty
```

---

## 15. Compiler 设计

### 15.1 IR 层次

建议保持三层语义：

1. **Graph/Tensor IR**：`gather`、`scatter`、`scatter_reduce`；
2. **ELENOR Host/Group IR**：event、ownership、L2 slot、indirect DMA；
3. **Descriptor ABI**：固定 128-byte command。

不要在高层 IR 中暴露 HBM transaction、combine buffer entry 等微架构细节。

### 15.2 Lowering 决策树

```text
Is op gather?
  ├─ yes → affine linearize indices → Group Gather DMA
  └─ no → scatter
           ├─ proven unique → Unique Scatter DMA
           ├─ ordered overwrite and single owner → Ordered Scatter DMA
           ├─ associative reduce + ownership possible
           │    → partition → local reduce → unique commit
           └─ otherwise → log/merge or CPU/firmware fallback
```

### 15.3 必须生成的分析结果

- `index_width`；
- `block_bytes`；
- `indexed_stride` 和 `linear_stride`；
- `max_count`；
- conservative target range；
- uniqueness proof 或 conflict policy；
- destination ownership；
- deterministic requirement；
- L2 index/value/result/scratch slot；
- wait/completion/release event；
- fallback reason。

### 15.4 Unique proof

编译器可在以下情况证明 unique：

- index 是单调无重复范围；
- index 来自 partitioned tile id 映射；
- index 是 permutation；
- destination range 按 Tile/Group 静态切分；
- 上游算子提供 `unique_indices` semantic attribute。

无法证明时不得默认 unique。Profile 显示“通常没有重复”不是正确性证明。

### 15.5 成本模型

建议初始模型：

```text
direct_cost = command_overhead
            + unique_cache_lines * line_transfer_cost
            + conflict_stall

partition_cost = route_bytes / noc_bandwidth
               + local_reduce_cost
               + commit_cost

fallback_cost = log_write + sort_or_hash_reduce + final_commit
```

成本模型输入来自 simulator/profile，而不是凭经验固定阈值。

---

## 16. Runtime 与 Command Queue 设计

### 16.1 Runtime 职责

- 验证 tensor device address 和 extent；
- 分配 L2 slot、scratch 和 event；
- 初始化 descriptor；
- 提交完整 DAG；
- 选择运行时 shape/count 对应的 variant；
- 处理 fault/timeout；
- 最终 host visibility fence；
- 记录 PMU/profile 数据。

### 16.2 CPU 不应承担的工作

- 不读取 Tile 产生的每个 index；
- 不为每个 index 生成一个 DMA descriptor；
- 不在 chunk 之间反复 doorbell；
- 不解决大规模重复 index 冲突；
- 不参与正常路径的 completion polling loop。

### 16.3 状态与错误码

```cpp
enum class IndirectDmaStatus : std::uint32_t {
  kSuccess = 0,
  kInvalidDescriptor,
  kUnsupportedMode,
  kCountOverflow,
  kIndexOutOfBounds,
  kL2RangeFault,
  kHbmAccessFault,
  kEventEpochMismatch,
  kUniqueContractViolation,
  kOwnershipViolation,
  kTimeout,
  kInternalInvariantFailure,
};
```

错误记录至少包含：

- descriptor address；
- user tag；
- failing logical index/sequence；
- calculated address；
- status code；
- Group/engine id；
- timestamp。

---

## 17. Profiling 与 PMU

IDE MUST 提供以下计数器：

### 17.1 通用

- descriptor count；
- useful bytes；
- external HBM bytes；
- total cycles；
- active cycles；
- resource-wait cycles；
- HBM-backpressure cycles；
- L2-bank-stall cycles；
- max/average outstanding；
- fault count。

### 17.2 Gather

- index count；
- unique HBM line count；
- line merge ratio；
- duplicate index/read reuse count；
- reorder occupancy/high-watermark；
- result write cycles。

### 17.3 Scatter

- target line count；
- combine hit/miss；
- duplicate target count；
- dirty flush count；
- conflict stall cycles；
- partial-line write count；
- write amplification；
- ownership routed bytes；
- local reduction count。

建议派生指标：

```text
useful_bandwidth = useful_bytes / elapsed_time
line_merge_ratio = logical_requests / physical_line_requests
combine_hit_rate = combine_hits / scatter_updates
write_amplification = external_write_bytes / useful_value_bytes
```

---

## 18. Verification 计划

### 18.1 Functional reference model

首先实现与硬件无关的 reference：

```cpp
for (std::uint32_t i = 0; i < count; ++i) {
  const auto index = read_index(i);
  if (!in_bounds(index)) {
    handle_oob(i);
    continue;
  }

  if (opcode == Gather) {
    copy_block(hbm[index], l2[i]);
  } else {
    apply_scatter(policy, hbm[index], l2_value[i], sequence(i));
  }
}
```

Reference model MUST 支持 deterministic 模式，作为 cycle model 和 RTL 的 golden result。

### 18.2 功能测试矩阵

| 维度         | 测试值                                          |
| ------------ | ----------------------------------------------- |
| Count        | 0、1、FIFO-1、FIFO、FIFO+1、max_count           |
| Index width  | 32-bit、64-bit                                  |
| Element size | 1、2、4、8、16 bytes                            |
| Block bytes  | 1、16、64、128、4096、跨 line 非整齐值          |
| Index 分布   | 连续、随机、逆序、重复、Zipf/hotspot            |
| Bounds       | first、last、负数、越界、最大合法值             |
| Conflict     | none、pair duplicate、all same、跨 line overlap |
| HBM          | 固定延迟、乱序返回、长 stall、fault             |
| L2           | bank conflict、slot full、consumer 延迟释放     |
| Event        | 正常、旧 epoch、producer fault、timeout         |

### 18.3 Scatter 专项

1. Unique contract 正确和违规。
2. Ordered overwrite：相同 target 的 last writer 必须稳定。
3. block overlap：不同 index 产生部分重叠 block 时必须拒绝或正确排序。
4. byte-enable：同 line 不同小块合并不能污染其他 byte。
5. 多 Group overlap：无 ownership 时 runtime 必须拒绝。
6. FP deterministic reduction：固定顺序与非固定顺序结果分别验证。

### 18.4 Deadlock/forward-progress

必须构造：

- outstanding table 满；
- combine buffer 全 dirty；
- HBM write credit 为零后恢复；
- gather response 返回但 L2 bank 长时间阻塞；
- completion queue 几乎满；
- Linear DMA 与 Indirect DMA 同时饱和；
- producer 等 consumer、consumer 等 slot 的错误 DAG。

期望：硬件持续 backpressure 或报告非法 DAG，不得静默卡死。

### 18.5 性能 workload

至少仿真：

1. Embedding gather：不同 row width、uniform/Zipf token。
2. TopK→Gather：动态 count 和 event pipeline。
3. Random scalar scatter：观察最坏写放大。
4. Unique block scatter：测试 line combining。
5. Embedding gradient：owner partition + local reduce。
6. Histogram：热点冲突与 combine buffer sensitivity。

---

## 19. 实施阶段与退出条件

### Phase 0：语义冻结

交付：

- 本文中的 op semantics；
- 128-byte descriptor ABI；
- event visibility；
- V1 支持矩阵；
- error code。

退出条件：Compiler、Runtime、Simulator、Hardware Agent 对同一输入产生一致的地址和完成语义。

### Phase 1：Functional POC

交付：

- C++ reference model；
- descriptor encoder/decoder；
- Gather + Unique Scatter；
- 单元测试和随机 differential test。

退出条件：至少 100k randomized cases 与 reference 一致，越界和错误路径覆盖。

### Phase 2：Cycle model

交付：

- FIFO/credit/outstanding/HBM latency 模型；
- Gather reorder；
- Scatter combine；
- PMU；
- 双缓冲 event pipeline。

退出条件：无 credit invariant 失败；在所有 backpressure 测试中完成或显式 fault。

### Phase 3：Compiler/Runtime integration

交付：

- MLIR/Host IR op；
- lowering decision tree；
- L2 slot/event 分配；
- command queue ABI；
- fallback path。

退出条件：TopK→Gather 和 Unique Scatter 端到端执行，CPU 不参与动态中间步骤。

### Phase 4：Scatter Reduce

交付：

- ownership analysis；
- partition/route；
- local reduce；
- unique commit；
- deterministic variant。

退出条件：多 Group embedding gradient/histogram 正确，无 global atomic 依赖。

### Phase 5：RTL readiness

交付：

- 寄存器和接口规范；
- 状态机；
- queue sizing sweep；
- assertion list；
- fault injection plan；
- PPA 风险评估。

退出条件：参数选择由 workload 数据支持，未解决问题有明确 cutline。

---

## 20. Agent 工作包

以下任务可以交给不同 Agent，但必须以 Phase 0 的语义为共同输入。

### Agent A：Architecture / ABI

任务：

- 审核 descriptor 字段和 offset；
- 输出 bit-level ABI 表；
- 定义 event、fault、visibility；
- 定义 DMA 与 L2/HBM 接口信号。

完成标准：

- C/C++ `static_assert` 通过；
- 所有字段有 producer/consumer；
- 没有未定义的 completion 语义。

### Agent B：Functional Simulator

任务：

- 实现 golden reference；
- 支持 Gather、Unique/Ordered Scatter；
- 实现 OOB 和 deterministic；
- 建立 randomized differential test。

完成标准：

- 测试矩阵完整；
- 每个错误码都有触发用例；
- 输出可复现 seed。

### Agent C：Cycle Simulator / Performance

任务：

- 建模 index FIFO、reorder、combine、outstanding 和 HBM；
- 实现 credit/backpressure；
- 输出 PMU；
- 扫描 queue size 和 workload 分布。

完成标准：

- invariants 全部在线检查；
- 输出 useful bandwidth、merge ratio、write amplification；
- 给出推荐硬件参数和敏感性曲线。

### Agent D：Compiler

任务：

- 定义 Group IR op/type/attribute；
- 实现 index linearization；
- 实现 uniqueness、alias、ownership 分析接口；
- 生成 descriptor 和 fallback reason；
- 集成 L2 slot/event lifetime。

完成标准：

- 覆盖算子矩阵；
- 无 proof 时不会错误选择 Unique；
- 能生成 TopK→Gather 的预入队 DAG。

### Agent E：Runtime

任务：

- 实现 descriptor builder/validator；
- 管理 L2 slot、scratch、event epoch；
- 提交 command DAG；
- 处理 fault、timeout、host fence；
- 暴露 profile API。

完成标准：

- 动态 index 正常路径无 CPU round-trip；
- 非法 ownership/range 在启动前失败；
- descriptor ABI 与 simulator 字节级一致。

### Agent F：RTL / Verification

任务：

- 实现或细化 IDE microarchitecture；
- 建立 formal/SVA invariants；
- 验证乱序 HBM response 和 backpressure；
- fault injection；
- 输出 PPA/timing 风险。

完成标准：

- 每个 accepted command exactly-once complete/fault；
- 无 FIFO overflow/underflow；
- no-use-after-release；
- completion visibility 正确。

### 依赖顺序

```text
Agent A: Semantics/ABI
       ├──→ Agent B: Functional Model
       ├──→ Agent D: Compiler
       └──→ Agent E: Runtime

Agent B
       └──→ Agent C: Cycle Model

Agent A + B + C
       └──→ Agent F: RTL/Verification

Agent C profile
       └──→ Agent D cost model + Agent F sizing
```

任何 Agent 若需要修改 op semantics 或 descriptor ABI，必须先回到 Phase 0 更新本文，不能在各自实现中私自扩展。

---

## 21. 开放问题

以下问题在 RTL 冻结前必须用数据回答：

1. Index decode rate 需要 2、4 还是 8 index/cycle？
2. Gather reorder 32 entries 是否足够隐藏目标 HBM latency？
3. Scatter combine buffer 是 full-associative、set-associative 还是 hash bank？
4. `block_bytes` 最大值是否需要超过 4096？
5. Ordered Scatter 是否值得进入 V1，还是只保留 Unique？
6. Owner routing 使用 NoC message、HBM log 还是 Group collective scratch？
7. L2 bank 数量和 IDE 访问模式是否造成 Tile MFE 饥饿？
8. Descriptor CRC、watchdog 和 error record 的 FuSa cutline 是什么？
9. HBM partial write 的真实 transaction/write amplification 模型是什么？
10. 哪些 framework Scatter 语义要求确定性，哪些允许未指定顺序？

这些问题必须通过 cycle simulator、workload trace 和 memory controller 模型解决，不能只靠架构直觉。

---

## 22. 最终 V1 Cutline

### V1 必做

- 统一 128-byte Indirect DMA descriptor；
- indices 位于 L2；
- HBM→L2 Gather；
- L2→HBM Unique Scatter；
- dynamic count + max bound；
- event wait/completion + epoch；
- bounds/range checking；
- Gather reorder；
- Scatter line combining；
- credit/backpressure；
- PMU 与 fault record；
- compiler/runtime legality check。

### V1 条件纳入

- 单 Group Ordered Overwrite；
- Group 内 add/max/min combine；
- owner partition + local reduction；
- deterministic mode。

### V1 明确不做

- Tile→HBM demand-load/store；
- L2 cache 化和 80%/20% 分区；
- 跨 Group任意 global atomic；
- arbitrary pointer chasing；
- CPU 逐 index 调度；
- 未证明 ownership 的并发 Scatter。

---

## 23. 设计结论

该方案保留了 ELENOR 原始架构最重要的优势：global memory traffic 可预测、L2 生命周期可静态管理、Tile 硬件简单、编译器可以进行跨 kernel pipeline。

Gather/Scatter 的灵活性由 Group Indirect DMA 提供，而不是通过把每个 Tile 改造成 GPU SM 来获得。对于真正动态的 index，CPU 仍然可以在图级别负责调度，但不进入逐元素关键路径。

V1 最重要的正确性边界是：

> **Gather 可以乱序读取后重排；Scatter 只有在目标唯一、顺序明确或 ownership 已建立时才能直接提交。Group 内 combine 只优化带宽，不能替代跨 Group 原子一致性。**

这应作为后续 Compiler、Runtime、Simulator 和 RTL 实现的共同契约。
