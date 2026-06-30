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
- zero-copy buffer aliasing
- view descriptor system

MFE 不负责：

- 任意图遍历和通用 pointer chasing。
- Tile Program fetch/decode、branch、event wait；这些属于 Tile UCE。
- BOA dense compute 或 EVU elementwise。
- USE state lifecycle、checkpoint/restore。
- 跨 tile unordered atomic update 的全局一致性。
- 全芯片 memory policy、IOMMU 管理和 runtime scheduling。

ownership 规则：

| 对象                                     | owner                                        | MFE 行为                                                           |
| ---------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------ |
| page table base / block table descriptor | runtime / compiler                           | MFE 读取并 walk，fault 时记录 page id                              |
| page list / segment offset patch         | MFE                                          | 数据相关动态地址由 MFE 管理                                        |
| Tile Program control                     | Tile UCE                                     | MFE 只响应 launch/wait/cancel                                      |
| metadata/page-list slot                  | MFE writer，UCE/USE reader                   | 写入 owner 必须唯一                                                |
| stream buffer                            | MFE producer，BOA/EVU/USE consumer           | credit/EOS/error 明确                                              |
| segment reduce partial owner             | descriptor 声明为 MFE、EVU 或 Collective     | 不能隐式共享                                                       |
| duplicate index                          | descriptor mode                              | 未声明行为必须 fault 或 ordered fallback                           |
| TensorView / view binding                | compiler / runtime 定义，MFE 消费            | 复用共享 TensorView 语义解释 zero-copy view，不单独发明平行 schema |
| alias lifecycle / release                | producer owner + runtime / Tile UCE contract | backing store 在所有 alias view release / barrier 完成前不得回收   |

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

| 模块                 | 说明                                                | 关键点                         |
| -------------------- | --------------------------------------------------- | ------------------------------ |
| Launch Frontend      | 接收 Tile UCE launch、读取 descriptor               | command id、event id、mode     |
| Descriptor Validator | 检查 ABI、mode、slot、bounds、reserved bits         | 输出 fault record              |
| Metadata Decoder     | 解析 page table、offsets、indices、block metadata   | 格式由后续规格冻结             |
| Page Walker          | page id -> physical page / L2 address               | invalid page fail-fast         |
| Segment Walker       | offsets/indices -> segment item stream              | duplicate policy descriptor 化 |
| Address Generator    | base + stride + index + segment + page offset       | boundary check                 |
| Prefetch Queue       | 提前发起 KV page、embedding row、expert weight 请求 | hit/miss 统计                  |
| Request Tracker      | 管理 outstanding request、timeout、cancel           | tag、age、fault                |
| Reorder Buffer       | 合并乱序返回，恢复 logical order                    | page/segment order policy      |
| Coalescer            | 合并相邻地址，提高 burst efficiency                 | 不改变可见顺序                 |
| Stream Buffer        | ping-pong buffer，写入 L1 stream slot               | credit、EOS/error              |
| Commit Unit          | event、fault、PMU snapshot                          | done/fault 原子提交            |

#### 3.1.1 Queue 架构和 ingress 分层

MFE 的 queue 架构应保持 **external command ingress** 与 **internal data/event pipeline** 分层，而不是把所有功能都做成独立 queue：

- Tile UCE 到 MFE 至少区分 load/store 两类 command ingress（可实现为 `LD_CMD_Q` / `ST_CMD_Q` 或等价 launch class）；精确深度、仲裁和是否共享物理 storage 由后续规格冻结。
- MFE 内部保留 `RD_REQ`、`RD_RESP`、`WR_REQ` 和 event/commit path 的最小 queue 组合；精确 entries / beats 预算由 SRAM profile、NoC profile 和 PPA exploration 冻结。
- 现有 **Prefetch Queue** 继续作为 MFE 内部预取/请求跟踪路径的一部分存在；它不是 UCE→MFE 的 external command ingress 替代物。
- 只有当可见顺序、burst 拼接或 layout reorder 确实需要时，才引入小型 skid buffer 或 reorder buffer；默认 V1 不做 full-tile write-data staging FIFO。

不推荐的 V1 默认结构：

- WindowGen queue。
- LayoutTransform queue。
- per-feature command queue。
- full-tile WR data FIFO。

#### 3.1.2 Load/Store 路径

推荐的数据路径形态：

```text
Load:
  Tile UCE
    -> load command ingress
    -> Window / Address Generation
    -> RD request path
    -> DMA Read
    -> RD response path
    -> optional layout transform
    -> L1 stream / metadata slot

Store:
  Tile UCE
    -> store command ingress
    -> L1 read
    -> optional layout transform
    -> WR request path
    -> DMA Write
    -> L2 / HBM
    -> event / commit path
```

这个拆分的目的不是把 MFE 变成两个独立 engine，而是在同一个 MFE 中保持 **load/store orchestration、data movement 和 completion/event** 三条路径的边界清晰。

#### 3.1.3 Window Generator 和 Layout Transform 规则

- Window Generator 是 streaming address-generation stage，直接喂给 read-request path；V1 不建议把它设计成独立 command queue。
- Layout Transform 在 streaming case 走 `DMA -> XFORM -> SRAM` 或带小 skid buffer 的路径；只有当可见顺序必须恢复时才引入 reorder buffer。
- Layout Transform 不应拥有独立 command queue；它属于 load/store pipeline 内部 stage。
- 对 CONV / pooling 的 window 生成，复用现有 window generator / line-buffer 逻辑即可；不要再额外复制一套 feature queue。

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

#### Zero-copy buffer aliasing 和 view descriptor system

MFE 的 view descriptor system 复用共享 `elenor_tensor_view_t` 语义，而不是定义一套完全独立的 MFE view ABI。MFE descriptor 通过 slot/frame 先解析 backing storage，再通过 view binding 解释逻辑视图。

概念性 view binding 可写成：

```c
typedef struct {
    uint16_t src_view_slot;
    uint16_t dst_view_slot;
    uint16_t view_op;
    uint16_t alias_policy;
} mfe_view_binding_v0_t;
```

这里的 `src_view_slot` / `dst_view_slot` 指向共享 `TensorView` 条目或 slot-based 等价对象；exact field layout、slot 编码和 ABI 归属由后续规格冻结。V1 先冻结以下语义：

1. zero-copy aliasing 只表示逻辑 view 共享 backing store，不隐含新的 buffer allocation。
2. subview、slice、stride/extent 改写、只读 layout reinterpret 可保持 zero-copy。
3. 需要真实 reorder、pack/unpack materialization 或可见顺序恢复时，MFE 必须退回 bufferized path。
4. writable alias 仍受 Slot Frame policy 约束：V1 不允许同时存活的重叠 writable alias；只读 alias 或显式 release/barrier 分隔的 phase-disjoint handoff 才允许。
5. backing store 在所有 alias view release / barrier 完成前不得回收。

### 4.3 Consistency boundaries

MFE 的一致性边界必须 descriptor 化：

| 边界              | V1 行为                                                               | 后续能力                                        |
| ----------------- | --------------------------------------------------------------------- | ----------------------------------------------- |
| Page Stream order | logical token order；乱序返回由 reorder buffer 恢复                   | 更复杂 page residency policy                    |
| Page fault        | first fault record + EOS/error token；不 silent skip                  | fault recovery policy 由后续规格冻结            |
| Segment gather    | indices 顺序或 segment order 输出，由 descriptor 声明                 | coalesced reorder 但可见顺序不变                |
| Local reduce      | partial owner 为 MFE 时只在 tile-local scope 内有效                   | group/global reduce 交给 Collective             |
| Duplicate index   | `duplicate_policy` 明确 first/last/sum/fault/ordered                  | atomic add 由后续规格冻结                       |
| Scatter           | V1 仅 ordered scatter 可选；unordered atomic 不默认支持               | V2 atomic path                                  |
| Cross-tile update | 不在 MFE V1 内解决                                                    | runtime 分桶、group reduce、atomic path         |
| View alias        | 复用共享 TensorView 语义；只读 alias 或 phase-disjoint handoff 才允许 | 更细粒度 alias policy / writable alias contract |

### 4.4 协议

Tile UCE launch：

```text
launch.mfe desc_slot, event_id
wait event_id
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

#### Store visibility model

MFE 的 store completion 不应只暴露一个扁平 “store done” 语义。V1 至少区分三层 **概念性可见性阶段**；具体 event 名称、编码和 ABI 归属由后续共享规格冻结：

| 概念阶段       | 含义                                                                 |
| -------------- | -------------------------------------------------------------------- |
| store accepted | MFE 已接受 store command，buffer lifecycle 可按显式 release 规则推进 |
| L2 visible     | 数据已提交到 L2 / group-visible region，可参与后续 barrier 统计      |
| global visible | 数据已对更大系统范围可见（例如 host / system agent 需要的路径）      |

规则：

- Tile / L1 不默认因为 store 尚未达到 L2/global visible 而阻塞；是否等待由 descriptor、event dependency 或上层 runtime contract 明确声明。
- 对当前 **multi-level matmul + gather** 映射，gather 的 source of truth 不是 “DMA accepted” 或单独 ready event，而是 **L2 visible arrival + 上层显式 L2 barrier complete**。
- exact event name / code / ABI field 由后续共享规格冻结，本节只冻结语义层次，不伪造二进制编码。

#### L2 barrier、可选 scoreboarding 和 gather sync

MFE commit/event path 可以维护一个概念化的 bookkeeping 结构，例如：

```text
ready_table[region][tile][version]
```

但在当前 V1 `matmul -> L2 barrier -> gather` 路径中，它只是 barrier manager / tile-group / MFE 的内部实现选择，不是对外冻结的 gather 触发 ABI。V1 先冻结以下规则：

1. producer matmul/store 先把结果写到 L2，并在达到 **L2 visible** 时向 barrier scope 报到。
2. gather / dependent consumer 只在显式 **L2 barrier complete** 后启动。
3. partial matmul / split accumulation store 不单独释放 gather；它们只贡献 barrier arrival。
4. 若后续版本引入 tile-level replay 或更细粒度 ready event，再由共享规格单独冻结。

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
- zero-copy alias view 不自动分配新的 stream buffer；只有当 descriptor 要求 materialize/reorder 时才占用额外 SRAM workspace。

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

补充约束：

- queue budget 默认保持最小分层：load/store command ingress + request/response/event path；精确容量由后续规格冻结，不在本规格中伪造 `2~4` / `4~8` 数值。
- flow control 默认采用 **buffer reservation + outstanding tracking + registered backpressure**，而不是为每个功能堆深 FIFO。
- 只有当 reorder、visibility isolation 或跨时钟边界真的需要时，才引入额外 staging buffer。

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
- reorder buffer 恢复 logical order；coalescer 不得改变可见 stream order。
- stream buffer 支持 EOS/error token，consumer 必须能区分正常结束和 fault 结束。
- timeout 计数器按 request age 或 command age 实现，阈值由后续规格冻结。
- Sparse/Persistent mode 未实现时 descriptor 必须 fault，不允许 silent no-op。
- clock gating：metadata idle、prefetch queue empty、stream buffer full wait、reorder idle 分域。
- command ingress 与 data/event path 分离：load/store command ingress、request/response path、event/commit path 不应混成一个巨型 feature queue。
- Window Generator 作为 sequencer 内 streaming stage 实现；Layout Transform 默认走 pipeline stage + skid buffer，不单独建 command queue。
- event/commit path 应覆盖 accepted、L2-visible、fault/timeout 以及 barrier-arrival 一类语义提交；gather 释放点由上层 barrier-complete 决定，literal event 名称由后续共享规格冻结。
- 默认避免 full-tile WR data FIFO；只有当 store visibility isolation 或 reorder 需求证明必须时才引入。

### 7.2 软件和 compiler

- compiler/runtime 生成 page table base、block table descriptor、segment offsets/indices 和 slot binding。
- runtime 管理全局 KV cache metadata；MFE 管理数据相关动态地址和 stream fill。
- Tile UCE 控制 launch MFE/BOA/EVU/USE、wait event、处理 EOS/error branch。
- Paged attention lowering 必须串联：MFE K/V page stream -> BOA QK -> EVU scale/mask/softmax -> BOA AV -> MFE/DMA store。
- Segment Stream lowering 必须声明 duplicate policy、reduce owner、output order 和 consistency scope。
- 对 V1 不支持的 unordered scatter/atomic，compiler 应选择 runtime 分桶、ordered fallback 或 group collective combine。
- 对 zero-copy aliasing path，compiler/runtime 负责生成共享 TensorView 语义的 view binding，并通过显式 release/barrier 管理 backing store 生命周期。

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
- store accepted / L2-visible / global-visible 三层语义单调推进，不允许 consumer 在 L2-visible 之前观察到可见数据。
- 对当前 `matmul -> L2 barrier -> gather` 路径，barrier complete 之前不得启动 gather；partial matmul store 不得错误释放 gather。
- TensorView / view binding 的 zero-copy alias 不得绕过 Slot Frame writable alias 规则；backing store release 必须晚于所有 alias consumer 完成。

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
- Segment gather + local reduce 的 duplicate index 行为由 descriptor mode 明确定义并通过 golden。
- MoE combine 结果与 golden 对齐，BOA utilization 可由 imbalance model 解释。
- 未实现的 Sparse/Persistent mode fail-fast，并记录 invalid mode fault。

## 9. 风险、取舍和后续细化方向

| 风险                                   | 影响                                 | 缓解                                                |
| -------------------------------------- | ------------------------------------ | --------------------------------------------------- |
| MFE overdesign 成通用 memory processor | 验证和时序失控                       | V1 只做 Page + Segment，Sparse/Persistent 后移      |
| page metadata 格式不稳定               | descriptor ABI 反复破坏              | profile 化 page table format，reserved 字段清零验证 |
| unordered scatter/atomic 一致性复杂    | 数据错误难复现                       | V1 不默认支持，runtime 分桶或 group reduce          |
| reorder buffer 过大                    | 面积和时序高                         | canonical paged attention case 冻结 depth           |
| MFE/BOA/EVU SRAM 冲突                  | stream stall、BOA operand stall 上升 | bank-aware slot planning 和 PMU 归因                |
| fault 后旧 response 污染新 command     | 高可靠性风险                         | request epoch、drain、SVA 覆盖                      |

后续需要冻结：Page Stream binary format、Segment Stream duplicate policy、prefetch/reorder depth、stream token ABI、timeout policy、SRAM/NoC bandwidth profile、PMU counter 编号和 Paged Attention canonical case。
