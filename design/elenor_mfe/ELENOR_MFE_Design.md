# MFE 设计文档

## 1. 定位、目标和 First Silicon cutline

MFE（Memory Flow Engine）是 ELENOR 的 memory-flow engine，负责把动态、不规则、分段、分页的数据流规整成 BOA、EVU、USE 和 SRAM 可以消费的 stream。MFE 是数据相关动态内存访问的 owner：page table walk、block table decode、physical address generation、page gather、KV prefetch、layout transform、reorder、segment gather 和 stream fill。Tile UCE 负责 program control 和 engine orchestration；MFE 不执行 Tile Program PC，不解释高层 graph。

MFE 不是通用 memory processor。First Silicon V1 必须控制复杂度，优先覆盖：

- Page Stream：Paged Attention、KV cache、long context。
- Segment Stream：embedding bag、ragged tensor、MoE token group、recommender/GNN 的基础 gather + local reduce。

Architecture V1 可预留 Sparse Block Stream 和 Persistent Memory Stream；First Silicon V1 不应被这些后续能力阻塞。

| 能力                     | First Silicon V1                                             | 后续能力                                                   |
| ------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| Page Stream              | page walk、KV prefetch、reorder、stream fill、double buffer  | residency predictor、deeper prefetch、跨 group page policy |
| Segment Stream           | offsets decode、segment gather、local reduce、ordered output | unordered scatter、atomic update、cross-tile reduce        |
| Sparse Block Stream      | 预留 flags 和 metadata shape                                 | V2 block sparse attention / sparse matmul                  |
| Persistent Memory Stream | 预留 state/page binding 字段                                 | V3 agent memory / recurrent memory                         |
| PMU                      | stall、prefetch hit/miss、stream occupancy、backpressure     | trace sampling 和 PMU feedback scheduler                   |

## 2. 职责、非职责和 ownership

MFE 负责：

- 解析 page table、block table、offsets、indices、sparse metadata 的 V1 子集。
- 生成 physical address 或 L2/L1 stream address。
- 对 page/segment 数据执行 prefetch、coalesce、reorder、double buffering。
- 将 stream 写入 Tile L1 stream slot / metadata slot，或通过 stream queue 交给 BOA/EVU/USE。
- 维护 Page Stream 和 Segment Stream 的 EOS/error token。
- 对 invalid page、out-of-bound index、timeout、stream overflow、descriptor fault 产生 fault record。
- 提供 PMU：prefetch hit/miss、stream stall、reorder occupancy、credit empty/full、fault count。
- 负责 window generator 相关工作以确保 BOA 可以计算高效的算 CONV 算子并且确保：
  - window stream 顺序与 BOA GEMM A 矩阵 layout 对齐
  - padding 区域填 0 或按 descriptor policy 处理
  - stream credit 足够，不让 BOA 长时间 operand stall
  - 对 KH/KW/IC 做 bank-aware packing
  - Line Buffer / Row Buffer 复用 KH 行输入，减少重复 SRAM read
  - Window Stream FIFO 将 window 连续化，喂给 BOA A buffer
  - Bank-aware Packing 让 BOA 读取 A/B tile 时少 bank conflict
- zero fill padding 区域，或根据 descriptor policy 处理 padding token。
- DMA 控制器

功能性支持：

- 1D / 2D / 3D DMA
- block load/store
- ping-pong buffer
- basic stride
- basic padding fill
- event/token
- conv window generator
- pooling window generator
- layout transform
- page gather
- scatter assist
- stream-to-BOA direct feed

MFE 不负责：

- 任意图遍历和通用 pointer chasing。
- Tile Program fetch/decode、branch、event wait；这些属于 Tile UCE。
- BOA dense compute 或 EVU elementwise。
- USE state lifecycle、checkpoint/restore。
- 跨 tile unordered atomic update 的全局一致性。
- 全芯片 memory policy、IOMMU 管理和 runtime scheduling。

ownership 规则：

| 对象                                     | owner                                    | MFE 行为                                 |
| ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| page table base / block table descriptor | runtime / compiler                       | MFE 读取并 walk，fault 时记录 page id    |
| page list / segment offset patch         | MFE                                      | 数据相关动态地址由 MFE 管理              |
| Tile Program control                     | Tile UCE                                 | MFE 只响应 launch/wait/cancel            |
| metadata/page-list slot                  | MFE writer，UCE/USE reader               | 写入 owner 必须唯一                      |
| stream buffer                            | MFE producer，BOA/EVU/USE consumer       | credit/EOS/error 明确                    |
| segment reduce partial owner             | descriptor 声明为 MFE、EVU 或 Collective | 不能隐式共享                             |
| duplicate index                          | descriptor mode                          | 未声明行为必须 fault 或 ordered fallback |

## 3. 微架构和状态机

### 3.1 Pipeline

```text
Descriptor Fetch
    |
Descriptor Validate
    |
Metadata Decode
    |
Page / Segment Walk
    |
Address Generation
    |
Prefetch / Request Issue
    |
Reorder / Coalesce
    |
Stream Buffer Fill
    |
Consumer Handoff / Commit
```

内部模块：

| 模块                  | 说明                                                                  | 关键点                          |
| --------------------- | --------------------------------------------------------------------- | ------------------------------- |
| Launch Frontend       | 接收 Tile UCE launch、读取 descriptor                                 | command id、event id、mode      |
| Descriptor Validator  | 检查 ABI、mode、slot、bounds、reserved bits                           | 输出 fault record               |
| Metadata Decoder      | 解析 page table、offsets、indices、block metadata                     | 格式由后续规格冻结              |
| Page Walker           | page id -> physical page / L2 address                                 | invalid page fail-fast          |
| Segment Walker        | offsets/indices -> segment item stream                                | duplicate policy descriptor 化  |
| Address Generator     | base + stride + index + segment + page offset                         | boundary check                  |
| Prefetch Queue        | 提前发起 KV page、embedding row、expert weight 请求                   | hit/miss 统计                   |
| Request Tracker       | 管理 outstanding request、timeout、cancel                             | tag、age、fault                 |
| Async LD/ST Queues    | 分离 load / store 接受与完成；为 V1.x window overlap 提供 credit 边界 | queue credit、epoch、visibility |
| Reorder Buffer        | 合并乱序返回，恢复 logical order                                      | page/segment order policy       |
| Coalescer             | 合并相邻地址，提高 burst efficiency                                   | 不改变可见顺序                  |
| Stream Buffer         | ping-pong buffer，写入 L1 stream slot                                 | credit、EOS/error               |
| Store Visibility Unit | store payload 对 L1/L2 consumer 可见后产生 completion identity        | event sequence、fence           |
| Commit Unit           | event、fault、PMU snapshot                                            | done/fault 原子提交             |

### 3.2 总状态机

```text
IDLE
  |
  v
DESC_FETCH
  |
  v
VALIDATE
  | invalid
  +---------> FAULT_COMMIT -> IDLE
  |
  v
METADATA_WALK
  |
  v
PREFETCH
  |
  v
STREAM
  |
  v
DRAIN_REORDER
  |
  v
COMMIT
  |
  v
IDLE
```

`STREAM` 状态可细分为 request issue、response collect、reorder、stream buffer write。`DRAIN_REORDER` 确保所有已发请求要么完成、要么 timeout/cancel 并产生确定 fault。fault path 禁止继续写新的 consumer-visible stream data，但可 drain 已安全完成且不会破坏顺序的 response，由 fault policy 冻结。

### 3.3 Page Stream 状态机

```text
PAGE_INIT
  |
  v
READ_BLOCK_TABLE
  |
  v
WALK_PAGE
  | invalid
  +-------> PAGE_FAULT
  |
  v
ISSUE_KV_PREFETCH
  |
  v
COLLECT_PAGE
  |
  v
REORDER_BY_LOGICAL_TOKEN
  |
  v
FILL_K_STREAM / FILL_V_STREAM
  |
  v
PAGE_EOS
```

Page Stream 必须保持 logical token order，即使 physical page 返回乱序。Page size、head_dim、prefetch depth、stream queue depth、L1/L2 footprint 和 MFE bandwidth 是 canonical paged attention case 必须冻结的值；当前未冻结项写作由后续规格冻结。

### 3.4 Segment Stream 状态机

```text
SEG_INIT
  |
  v
READ_OFFSETS
  |
  v
READ_INDICES
  |
  v
GENERATE_ITEM_ADDR
  |
  v
GATHER_ITEM
  |
  v
LOCAL_REDUCE_OPTIONAL
  |
  v
ORDERED_OUTPUT
  |
  v
SEG_EOS
```

Segment Stream V1 推荐：

```text
Segment gather + local reduce + ordered output
```

V1 不默认支持跨 tile unordered atomic update。需要全局一致性的 scatter/update，优先通过 runtime 分桶、group-level reduce 或后续 atomic path 解决。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Page Stream descriptor

```c
typedef struct {
    uint64_t page_table_addr;
    uint64_t q_addr;
    uint64_t k_stream_addr;
    uint64_t v_stream_addr;

    uint32_t batch;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t seq_len;
    uint32_t flags;
} elenor_mfe_page_stream_desc_t;
```

建议 slot-based 扩展：

```c
typedef struct {
    uint16_t mode;
    uint16_t flags;
    uint16_t page_table_slot;
    uint16_t block_table_slot;
    uint16_t k_stream_slot;
    uint16_t v_stream_slot;
    uint16_t metadata_slot;
    uint16_t event_policy;

    uint32_t batch;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t page_size;
    uint32_t seq_len;
    uint32_t prefetch_depth;

    uint32_t page_table_format;
    uint32_t layout_transform;
    uint32_t reorder_policy;
    uint32_t reserved;
} mfe_page_desc_v0_t;
```

字段约束：

- `page_size` 不能为 0，必须与 page table format 对齐。
- `head_dim` 必须匹配 BOA/EVU 后续 descriptor 的 expected layout。
- `prefetch_depth` 大于 stream buffer 容量时必须 validation fault。
- invalid page、page permission fault、out-of-range token 必须产生 deterministic fault。
- `layout_transform` 只能选择 V1 支持的 KV packed layout；其他值 fault。

### 4.2 Segment Stream descriptor

```c
typedef enum {
    MFE_SEG_GATHER_ONLY = 0,
    MFE_SEG_GATHER_REDUCE_LOCAL = 1,
    MFE_SEG_SCATTER_ORDERED = 2,
    MFE_SEG_SCATTER_ATOMIC_ADD = 3
} mfe_segment_mode_t;
```

```c
typedef struct {
    uint64_t table_base;
    uint64_t indices_addr;
    uint64_t offsets_addr;
    uint64_t output_addr;

    uint32_t batch;
    uint32_t feature_dim;
    uint32_t reduce_op;
    uint32_t dtype;
    uint32_t segment_mode;
    uint32_t flags;
} elenor_mfe_segment_desc_t;
```

slot-based 扩展：

```c
typedef struct {
    uint16_t mode;
    uint16_t flags;
    uint16_t table_slot;
    uint16_t indices_slot;
    uint16_t offsets_slot;
    uint16_t output_slot;
    uint16_t workspace_slot;
    uint16_t duplicate_policy;

    uint32_t batch;
    uint32_t feature_dim;
    uint32_t reduce_op;
    uint32_t dtype;
    uint32_t max_indices;
    uint32_t max_segment_len;

    uint32_t output_order_policy;
    uint32_t consistency_scope;
    uint32_t reserved;
} mfe_segment_desc_v0_t;
```

### 4.3 Consistency boundaries

MFE 的一致性边界必须 descriptor 化：

| 边界              | V1 行为                                                 | 后续能力                                |
| ----------------- | ------------------------------------------------------- | --------------------------------------- |
| Page Stream order | logical token order；乱序返回由 reorder buffer 恢复     | 更复杂 page residency policy            |
| Page fault        | first fault record + EOS/error token；不 silent skip    | fault recovery policy 由后续规格冻结    |
| Segment gather    | indices 顺序或 segment order 输出，由 descriptor 声明   | coalesced reorder 但可见顺序不变        |
| Local reduce      | partial owner 为 MFE 时只在 tile-local scope 内有效     | group/global reduce 交给 Collective     |
| Duplicate index   | `duplicate_policy` 明确 first/last/sum/fault/ordered    | atomic add 由后续规格冻结               |
| Scatter           | V1 仅 ordered scatter 可选；unordered atomic 不默认支持 | V2 atomic path                          |
| Cross-tile update | 不在 MFE V1 内解决                                      | runtime 分桶、group reduce、atomic path |

### 4.4 协议

Tile UCE launch：

```text
launch.mfe desc_slot, event_id, event_sequence
wait event_id, expected_sequence
```

Stream handoff：

```text
mfe_stream_valid
mfe_stream_ready
mfe_stream_data
mfe_stream_meta
mfe_stream_eos
mfe_stream_error
mfe_stream_credit
```

协议要求：

- `EOS` 表示 logical stream 完整结束。
- `error` token 表示该 stream 不再产生正常数据，consumer 必须停止使用后续 partial。
- credit empty 的 primary owner 是 stream backpressure；若源自 consumer 不读，secondary tag 可指向 BOA/EVU/USE。
- stream buffer overflow 必须 fault，不能覆盖未消费数据。
- UCE V1.x 可以在 P0 store 未完成时发起 P1 load，条件是 Slot Frame / UCE hazard table 证明二者不访问同一 active buffer。
- MFE store completion event 表示 store visibility，而不是仅表示请求被接受；event 必须匹配 `event_id + sequence`。
- Async LD/ST queue credit 是 MFE 的 backpressure 边界；queue full 时 UCE window admission 或 `launch.mfe` 必须 stall，不得丢弃请求。

### 4.5 状态寄存器和 PMU

| 寄存器                        | 说明                                                                                              |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| `MFE_STATUS`                  | idle、busy、fault、drain、stream_blocked                                                          |
| `MFE_CMD_ID`                  | 当前 command id                                                                                   |
| `MFE_FAULT_CODE`              | invalid descriptor、invalid page、address fault、timeout、stream overflow、duplicate policy fault |
| `MFE_FAULT_INDEX`             | page id、segment id 或 index id                                                                   |
| `MFE_PMU_ACTIVE`              | active cycles                                                                                     |
| `MFE_PMU_STALL`               | primary stall cycles                                                                              |
| `MFE_PMU_PREFETCH_HIT`        | prefetch hit count                                                                                |
| `MFE_PMU_PREFETCH_MISS`       | prefetch miss count                                                                               |
| `MFE_PMU_REORDER_OCC`         | reorder occupancy high-watermark                                                                  |
| `MFE_PMU_STREAM_CREDIT_EMPTY` | consumer backpressure                                                                             |
| `MFE_PMU_STREAM_CREDIT_FULL`  | producer blocked / buffer full                                                                    |
| `MFE_PMU_ASYNC_QUEUE_OCC`     | async LD/ST queue occupancy high-watermark                                                        |
| `MFE_PMU_STORE_VIS_WAIT`      | store visibility event 等待周期                                                                   |

## 5. 数据流、控制流和时序路径

### 5.1 Page Stream 数据流

```text
Runtime / Compiler KV metadata
        |
        v
MFE page walk / block table decode
        |
        v
KV physical address generation
        |
        v
Prefetch K/V pages from L2/HBM path
        |
        v
Reorder by logical token
        |
        v
Fill K/V stream slots in L1
        |
        +--> BOA QK
        +--> EVU softmax after QK
        +--> BOA AV
```

Paged attention 的关键性能条件：

```text
T_prefetch <= T_qk
```

若 MFE 预取时间不超过 BOA QK 时间，BOA 可隐藏大部分 KV memory latency；否则 BOA `operand_stall` 与 MFE `prefetch miss / stream stall` 应同时上升。

### 5.2 Segment Stream 数据流

```text
offsets / indices
        |
        v
Segment Walker
        |
        v
Address Generator
        |
        v
Gather embedding / token / expert item
        |
        v
Optional local reduce
        |
        v
Ordered output stream
        |
        +--> EVU combine / reduce
        +--> BOA expert GEMM input
        +--> Collective combine
```

MoE 中，MFE 主要用于 token grouping、expert batching 和 expert weight/token stream。单个 expert GEMM 由 BOA 执行；dispatch、load balance 和 combine 需要 MFE、EVU、USE 和 Collective 协同。

### 5.3 Banking 和 SRAM 交互

- MFE stream buffer 使用 ping-pong buffer。
- MFE 对 metadata/page-list slot 可写，UCE/USE 可读，但写 owner 必须唯一。
- MFE burst write/read 应尽量 coalesce，峰值 bandwidth 由 SRAM profile 冻结。
- MFE 与 BOA 并发时，K/V stream slot 不应和 BOA accumulator hot bank 冲突。
- MFE 与 EVU 并发时，index/mask/vector buffer bank hint 应由 compiler memory planner 统一规划。
- Program / Descriptor / Event region 不应与 MFE stream hot path 固定共享同一组 bank。
- UCE sliding window profile 只消费预先规划的 ping-pong / multi-buffer stream/operand slot；MFE 不承担 tile-local dynamic allocator。

### 5.4 关键时序路径

| 路径                                  | 风险                         | 建议                                           |
| ------------------------------------- | ---------------------------- | ---------------------------------------------- |
| metadata decode -> address generation | 格式复杂导致组合路径长       | 格式 profile 化，decode 后寄存                 |
| reorder buffer tag compare            | outstanding 增大后 fanout 高 | CAM 深度由 PPA exploration 冻结，分 bank ROB   |
| coalescer merge                       | 地址排序复杂                 | V1 只合并相邻 burst，不做全排序                |
| stream credit -> request throttle     | 反压路径长                   | credit registered，允许小 skid buffer          |
| fault detection -> cancel outstanding | 控制扇出大                   | fault epoch + drain 状态，不单拍广播到所有请求 |

### 5.5 工作负载映射示例

| 工作负载              | MFE 映射                                                              | 协同模块                                                             | 关键检查                                                     |
| --------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------ |
| Paged Attention       | Page Stream walk、K/V prefetch、logical reorder、K/V stream fill      | BOA QK/AV，EVU softmax，Tile UCE event chain                         | invalid page、EOS/error token、`T_prefetch <= T_qk` PMU 指纹 |
| Long-context KV cache | page table decode、residency hint、prefetch depth 控制                | runtime 管 KV metadata，MFE 管数据相关动态地址                       | page_size、head_dim、layout transform 一致                   |
| MoE token dispatch    | Segment Stream offsets/indices decode、token grouping、ordered output | BOA expert GEMM，EVU/Collective combine，USE 可辅助 routing metadata | duplicate index 和 reduce owner 由 descriptor 明确           |
| Embedding bag         | Segment gather、local sum/max reduce、coalesced burst                 | EVU 可做后续 normalize 或 activation                                 | segment boundary、all-empty segment identity 明确            |
| GNN aggregation       | Segment gather + local reduce，跨 tile aggregate 交给 Collective      | runtime 分桶避免 unordered atomic                                    | V1 不假定跨 tile atomic consistency                          |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置

| 参数                 | First Silicon V1 建议                            | 冻结方式                   |
| -------------------- | ------------------------------------------------ | -------------------------- |
| stream buffer        | K/V 或 segment ping-pong                         | 由 SRAM profile 冻结       |
| prefetch depth       | 支持隐藏 canonical QK latency                    | 由后续规格冻结             |
| reorder depth        | 覆盖 page return jitter                          | 由 PPA exploration 冻结    |
| outstanding requests | 匹配 L2/HBM path                                 | 由 NoC/memory profile 冻结 |
| Page Stream format   | page table + block table V1                      | 由后续规格冻结             |
| Segment mode         | gather only、gather local reduce、ordered output | atomic add 后续冻结        |

### 6.2 性能模型

```text
MFE_bw_eff = useful_stream_bytes / total_cycles
T_prefetch = Bytes_kv / MFE_bw_eff
stream_eff = useful_stream_bytes / (useful_stream_bytes + padding_bytes + replay_bytes)
```

Paged attention latency：

```text
T_total = T_page_walk + T_prefetch + T_qk + T_softmax + T_av + T_writeback - T_overlap
```

若 UCE V1.x 启用 sliding window：

```text
T_overlap = min(T_store_visibility_current, T_load_prepare_next)
```

该 overlap 只有在 buffer independence、async queue credit 和 event sequence 均成立时有效；否则应计入 `uce_slot_hazard_stall`、`mfe_stream_credit_full` 或 `mfe_store_visibility_wait`，不能把 unsafe overlap 记入性能收益。

Segment Stream：

```text
T_segment = T_offsets + T_indices + T_gather + T_local_reduce + T_output
coalesce_eff = burst_bytes / requested_bytes
```

MoE imbalance 对 BOA 利用率影响：

```text
imbalance = max(tokens_per_expert) / avg(tokens_per_expert)
U_boa = 1 / imbalance
```

MFE 的目标是通过 token sorting、expert batching 和 coalescing 降低实际 imbalance 和随机访问成本。

### 6.3 PMU

必需 counter：

- `mfe_active_cycles`。
- `mfe_stall_cycles`。
- `mfe_prefetch_hit` / `mfe_prefetch_miss`。
- `mfe_page_walk_cycles`。
- `mfe_segment_walk_cycles`。
- `mfe_reorder_occupancy_high`。
- `mfe_stream_buffer_occupancy_high`。
- `mfe_stream_credit_empty` / `mfe_stream_credit_full`。
- `mfe_timeout_count`。
- `mfe_fault_count_by_type`。
- `mfe_coalesced_burst_bytes` / `mfe_requested_bytes`。

PMU 唯一归因：当 MFE 因 consumer 不读而阻塞时，MFE primary stall 是 `stream_credit_full`；consumer 侧如果同时等待 event，不应重复计入同一 stall cycle 的 primary owner。

## 7. RTL/软件实现建议

### 7.1 RTL

- Page Stream 和 Segment Stream 共享 request/reorder/stream buffer，但 metadata decoder 和 walker 分开，避免 mode-specific 复杂度污染主路径。
- descriptor validator 必须在任何 stream write 前完成。
- request tracker 使用 epoch；cancel/fault 后旧 response 不得写入新 command 的 stream buffer。
- Async LD/ST queue 可接受不同 window entry 的独立 request；store visibility event 必须在 payload 对声明 consumer 可见后产生，不能在 request enqueue 时提前 DONE。
- reorder buffer 恢复 logical order；coalescer 不得改变可见 stream order。
- stream buffer 支持 EOS/error token，consumer 必须能区分正常结束和 fault 结束。
- timeout 计数器按 request age 或 command age 实现，阈值由后续规格冻结。
- Sparse/Persistent mode 未实现时 descriptor 必须 fault，不允许 silent no-op。
- clock gating：metadata idle、prefetch queue empty、stream buffer full wait、reorder idle 分域。

### 7.2 软件和 compiler

- compiler/runtime 生成 page table base、block table descriptor、segment offsets/indices 和 slot binding。
- runtime 管理全局 KV cache metadata；MFE 管理数据相关动态地址和 stream fill。
- Tile UCE 控制 launch MFE/BOA/EVU/USE、wait event、处理 EOS/error branch。
- Paged attention lowering 必须串联：MFE K/V page stream -> BOA QK -> EVU scale/mask/softmax -> BOA AV -> MFE/DMA store。
- Segment Stream lowering 必须声明 duplicate policy、reduce owner、output order 和 consistency scope。
- 对 V1 不支持的 unordered scatter/atomic，compiler 应选择 runtime 分桶、ordered fallback 或 group collective combine。

## 8. 验证、bring-up 和验收标准

### 8.1 RTL/SVA 重点

- invalid descriptor 在任何 stream write 前 fault。
- page walk invalid page 产生 fault record，包含 command id、tile id、page id。
- reorder buffer 输出 logical order，不因 response 乱序改变 stream order。
- stream buffer full 时不覆盖未消费 entry。
- EOS/error token 只能出现一次，且在最后一个可见 data token 之后。
- fault/cancel epoch 后旧 response 不写入新 command。
- Segment duplicate index 行为与 descriptor mode 一致。
- local reduce partial owner 唯一；MFE/EVU/Collective 不双写同一 output。
- PMU active、prefetch wait、stream credit stall、NoC backpressure primary 互斥。
- Async LD/ST queue：queue full backpressure、store visibility event、event sequence mismatch、fault epoch 后旧 response 丢弃均需覆盖。

### 8.2 Bring-up 顺序

1. MFE descriptor validation、event/fault path。
2. Page Stream：单 page、连续 page、乱序 response reorder。
3. Page Stream fault：invalid page、timeout、stream overflow。
4. Paged attention command trace：MFE K/V -> BOA QK -> EVU softmax -> BOA AV。
5. 观察 `T_prefetch <= T_qk` case 中 BOA stall 下降。
6. Segment Stream：offsets decode、indices gather、ordered output。
7. Segment local reduce：sum/max golden。
8. duplicate index policy：fault/ordered/local reduce 模式分别验证。
9. MoE dispatch：8/16 expert routing imbalance benchmark。

### 8.3 验收标准

- Page Stream 可在 reorder、timeout、invalid page 上产生确定行为。
- Paged Attention end-to-end 与 golden 对齐。
- MFE prefetch hit/miss、stream stall、BOA operand stall 的 PMU 指纹符合预期。
- UCE V1.x sliding window trace 中，P0 store 与 P1 load overlap 只有在 slot 独立时出现；store visibility event 和 PMU wait 指纹可解释。
- Segment gather + local reduce 的 duplicate index 行为由 descriptor mode 明确定义并通过 golden。
- MoE combine 结果与 golden 对齐，BOA utilization 可由 imbalance model 解释。
- 未实现的 Sparse/Persistent mode fail-fast，并记录 invalid mode fault。

## 9. 风险、取舍和后续细化方向

| 风险                                   | 影响                                 | 缓解                                                     |
| -------------------------------------- | ------------------------------------ | -------------------------------------------------------- |
| MFE overdesign 成通用 memory processor | 验证和时序失控                       | V1 只做 Page + Segment，Sparse/Persistent 后移           |
| page metadata 格式不稳定               | descriptor ABI 反复破坏              | profile 化 page table format，reserved 字段清零验证      |
| unordered scatter/atomic 一致性复杂    | 数据错误难复现                       | V1 不默认支持，runtime 分桶或 group reduce               |
| reorder buffer 过大                    | 面积和时序高                         | canonical paged attention case 冻结 depth                |
| MFE/BOA/EVU SRAM 冲突                  | stream stall、BOA operand stall 上升 | bank-aware slot planning 和 PMU 归因                     |
| async store visibility 过早 DONE       | 后续 window 读到旧数据或未写完数据   | event sequence + visibility point SVA + UCE hazard stall |
| fault 后旧 response 污染新 command     | 高可靠性风险                         | request epoch、drain、SVA 覆盖                           |

后续需要冻结：Page Stream binary format、Segment Stream duplicate policy、prefetch/reorder depth、stream token ABI、timeout policy、SRAM/NoC bandwidth profile、PMU counter 编号和 Paged Attention canonical case。
