# ELENOR Stream Queue 设计文档

## 1. 定位、目标和 First Silicon cutline

Stream Queue 是 ELENOR TileGroupTask 中 role 间的硬件 producer-consumer contract。它连接 Tile Group Sequencer、Tile UCE、MFE、DMA、BOA/EVU/USE 后续 role 和 event fabric，使 block 粒度数据可以在 role 间流动，而不是等待上游 role 全量完成。

Stream Queue 不计算数据，不解释高层 graph，不替代 event/barrier。它只管理 token、payload 引用、credit、backpressure、EOS、error propagation 和 reset/drain。硬件执行的是 Tile Program 中的 stream 指令与 stream descriptor。

First Silicon cutline：

| 能力        | First Silicon V1                                                                  | V1.x / V2 保留                                        |
| ----------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 队列类型    | single-producer single-consumer、multi-producer single-consumer 的 all-EOS policy | broadcast multi-consumer、refcount token、动态 fanout |
| token       | 固定 header + payload pointer + fault index                                       | inline small payload、variable metadata extension     |
| credit      | per-queue credit pool、leak detection、reset reclaim                              | QoS credit partition、priority refill                 |
| EOS         | per-producer EOS bitmap、all-EOS close                                            | early-consumer-close、selective EOS                   |
| error       | error token 携带 fault record index 并传播到 group task/tile completion           | retryable error、partial replay                       |
| reset/drain | tile/group/device reset 下 token 和 credit 状态确定                               | preemption/resume                                     |

## 2. 职责、非职责和 ownership

### 2.1 ownership matrix

| 对象                                     | owner                           | 说明                                                       |
| ---------------------------------------- | ------------------------------- | ---------------------------------------------------------- |
| stream descriptor 静态字段               | Compiler                        | queue 深度、token size、producer/consumer mask、EOS policy |
| context id、queue base、fault table base | Runtime / firmware              | load package 或 bind context 时 patch                      |
| queue init / reset / drain               | Tile Group Sequencer            | group task 进入、退出、fault recovery 时执行               |
| token push/pop/release                   | Tile UCE 或 engine wrapper      | Tile Program 指令或 engine completion 触发                 |
| payload buffer                           | Slot Frame / L2 allocator owner | Stream Queue 只持有引用，不拥有数据本体                    |
| error token fault index                  | 产生错误的 DMA/MFE/UCE/engine   | Stream Queue 只转发和记录 first fault                      |
| PMU attribution                          | Stream Queue Engine + PMU       | occupancy、credit empty/full、stall owner                  |

### 2.2 非职责

- 不做 payload copy；payload 由 DMA、MFE 或 engine 写入 L1/L2 slot。
- 不保证全局 cache coherency；producer 写 payload 后必须通过 event/fence 形成可见性。
- 不做 graph scheduling；Tile Group Sequencer 选择 role，Stream Queue 只表达 backpressure。
- 不隐式吞掉错误；error token 必须可被 consumer、Tile Group Sequencer 和 fault record 观测。

## 3. 微架构和状态机

### 3.1 队列结构

每个 queue 至少包含：

```text
Queue Control Block
├── descriptor shadow
├── head / tail / occupancy
├── credit_available
├── producer_eos_bitmap
├── consumer_release_bitmap 或 release_count
├── first_fault_record_index
├── token SRAM / register FIFO
├── payload visibility fence state
└── PMU counters
```

Token FIFO 可以实现为 Group SRAM 小 FIFO、专用 SRAM macro 或寄存器 FIFO。First Silicon 推荐小深度专用 FIFO 加 overflow 禁止；深度由后续规格冻结。

### 3.2 producer token 生命周期状态机

```text
P_IDLE
  -> P_ACQUIRE_CREDIT
  -> P_FILL_PAYLOAD
  -> P_PAYLOAD_FENCE
  -> P_PUSH_TOKEN
  -> P_WAIT_ACCEPT
  -> P_IDLE

EOS path:
P_IDLE -> P_PUSH_EOS -> P_EOS_SENT

ERROR path:
任意状态 -> P_PUSH_ERROR -> P_FAULTED
```

语义：

- `ACQUIRE_CREDIT` 成功后 credit 减一；producer 获得一个可写 token slot。
- producer 在 push 前必须完成 payload write 或声明 payload 为 zero-byte control token。
- `PUSH_TOKEN` 对同一 producer 的 `sequence_id` 单调递增。
- EOS token 不消耗 payload buffer；是否消耗 FIFO credit 由 descriptor flag 冻结，First Silicon 建议消耗 credit 以保持实现简单。
- error token 优先于普通 token 可见，但不得重排同一 producer 已成功 push 的早期 token。

### 3.3 consumer 状态机

```text
C_IDLE
  -> C_WAIT_TOKEN
  -> C_POP_TOKEN
  -> C_CHECK_FLAGS
  -> C_CONSUME_PAYLOAD
  -> C_RELEASE_TOKEN
  -> C_IDLE

EOS path:
C_CHECK_FLAGS(EOS) -> C_MARK_EOS -> C_IDLE 或 C_DONE

ERROR path:
C_CHECK_FLAGS(ERROR) -> C_RECORD_FAULT -> C_PROPAGATE_ERROR -> C_FAULTED
```

语义：

- queue empty 时 consumer stall，PMU 记入 `stream_credit_empty_or_queue_empty` 的 stream wait 类。
- `RELEASE_TOKEN` 后 credit 加一；release 只能发生一次。
- multi-consumer 保留为 V1.x；First Silicon 若需要广播，使用独立 queue，而不是 refcount token。

### 3.4 queue reset/drain 状态机

```text
RUNNING
  -> DRAIN_REQUESTED
  -> STOP_ACCEPT_NEW_TOKEN
  -> WAIT_CONSUMER_RELEASE 或 CANCEL_TOKENS
  -> RECONCILE_CREDIT
  -> WRITE_DRAIN_EVENT
  -> RESET_CLEAN
  -> RUNNING

FAULTED
  -> STOP_ACCEPT_NEW_TOKEN
  -> PUSH_OR_LATCH_ERROR
  -> RECONCILE_CREDIT
  -> RESET_CLEAN
```

规则：

- drain 请求后禁止新普通 token；是否允许 error token 由 fault path 控制，First Silicon 允许 error token bypass 到 VC0/fault fabric。
- 已 pop 未 release 的 token 必须由 owner release 或由 reset domain cancel。
- reset 完成时 `credit_available == depth`，`occupancy == 0`，EOS bitmap 清零，fault latch 只保留到 fault record 中。

## 4. 接口、descriptor、寄存器和协议

### 4.1 binary descriptor v0

```c
typedef enum {
    ELENOR_STREAM_TOKEN_VALID = 1u << 0,
    ELENOR_STREAM_TOKEN_EOS   = 1u << 1,
    ELENOR_STREAM_TOKEN_ERROR = 1u << 2,
    ELENOR_STREAM_TOKEN_FENCE = 1u << 3,
} elenor_stream_token_flags_t;

typedef enum {
    ELENOR_STREAM_Q_SPSC       = 0,
    ELENOR_STREAM_Q_MPSC       = 1,
    ELENOR_STREAM_Q_BROADCAST  = 2,
} elenor_stream_queue_kind_t;

typedef enum {
    ELENOR_STREAM_EOS_SINGLE_PRODUCER = 0,
    ELENOR_STREAM_EOS_ALL_PRODUCERS   = 1,
    ELENOR_STREAM_EOS_PER_PRODUCER    = 2,
} elenor_stream_eos_policy_t;

typedef struct {
    uint16_t abi_version;
    uint16_t queue_kind;
    uint16_t eos_policy;
    uint16_t token_stride;
    uint32_t queue_id;
    uint32_t depth;
    uint32_t producer_mask;
    uint32_t consumer_mask;
    uint32_t payload_slot_id;
    uint32_t token_region_base;
    uint32_t token_region_bytes;
    uint32_t flags;
    uint32_t pmu_stream_id;
} elenor_stream_queue_desc_v0_t;

typedef struct {
    uint32_t token_id;
    uint32_t payload_addr;
    uint32_t payload_bytes;
    uint32_t flags;
    uint32_t producer_id;
    uint32_t sequence_id;
    uint32_t fault_record_index;
    uint32_t user_metadata;
} elenor_stream_token_v0_t;
```

约束：

- `depth` 必须大于 0 且不超过硬件 queue capacity；实际最大值由后续规格冻结。
- `token_stride` 必须覆盖 `elenor_stream_token_v0_t`，且满足 alignment。
- `producer_mask` 和 `consumer_mask` 必须非空；First Silicon 禁止 broadcast kind，除非 descriptor 显式打开并通过 RTL 验证。
- `payload_addr/payload_bytes` 必须落在 descriptor 指定 slot 或 L2 region 内。
- error token 必须设置 `fault_record_index`；普通 token 的该字段忽略但必须为 0。

### 4.2 Tile Program 指令语义草案

```asm
STREAM_INIT      qdesc
STREAM_ACQUIRE   token, qid
STREAM_PUSH      qid, token
STREAM_POP       token, qid
STREAM_RELEASE   qid, token
STREAM_PUSH_EOS  qid, producer_id
STREAM_PUSH_ERR  qid, fault_record_index
STREAM_DRAIN     qid
STREAM_RESET     qid
```

指令由 Tile UCE 或 Tile Group Sequencer 执行。UCE 拥有 Tile Program PC、branch、engine launch 和 descriptor patch；USE 只在 state/recurrence 路径需要时消费或产生状态 token，不拥有主控制流。

### 4.3 寄存器和可观测状态

| 寄存器                      | 说明                                              |
| --------------------------- | ------------------------------------------------- |
| `SQ_CTRL[q]`                | enable、drain、reset、fault_latch_clear           |
| `SQ_STATUS[q]`              | running、empty、full、draining、faulted、eos_seen |
| `SQ_HEAD_TAIL[q]`           | head、tail snapshot                               |
| `SQ_CREDIT[q]`              | available、leased、inflight                       |
| `SQ_EOS_BITMAP[q]`          | producer EOS bitmap                               |
| `SQ_FIRST_FAULT[q]`         | first fault record index                          |
| `SQ_OCC_CYCLES[q]`          | occupancy weighted cycles                         |
| `SQ_CREDIT_EMPTY_CYCLES[q]` | producer acquire stall                            |
| `SQ_QUEUE_EMPTY_CYCLES[q]`  | consumer pop stall                                |
| `SQ_RESET_SEQ[q]`           | reset/drain generation counter                    |

## 5. 数据流、控制流和时序路径

### 5.1 Group task role pipeline 数据流

```text
Tile Group Sequencer:
  init.stream S0
  dispatch.role role_id=0 (producer tiles)
  dispatch.role role_id=1 (consumer tiles)

Producer Tile UCE:
  acquire credit
  launch DMA/MFE/BOA/EVU to fill payload
  fence payload visibility
  push token

Consumer Tile UCE:
  pop token
  branch EOS/error
  launch DMA/BOA/EVU/USE using payload reference
  release token
```

### 5.2 ordering 规则

- 同一 producer 的 token 必须按 `sequence_id` 保序可见。
- 不同 producer 的 token 可 interleave；consumer 不得推断跨 producer 全序。
- MPSC all-EOS：所有 producer 都 push EOS 后，queue 才对 consumer 报告全局 EOS；consumer 仍可观察 per-producer EOS bitmap。
- error token 不丢弃已可见的早期 valid token；group task policy 可选择 stop affected queue 并 drain。
- payload write 对 consumer 可见需要 producer fence；Stream Queue 只排序 token，不排序 payload memory。
- reset generation 变化后，旧 token handle 无效；release 旧 generation token 必须 fault 或被忽略并记录。

### 5.3 credit 和 backpressure

Credit invariant：

```text
credit_available + tokens_in_fifo + tokens_popped_not_released == depth
```

- queue full：producer acquire 或 push stall，PMU 记 `stream_queue_full`。
- queue empty：consumer pop stall，PMU 记 `stream_queue_empty`。
- credit leak：reset/drain 或 periodic check 发现 invariant 不成立，必须产生 stream credit fault。
- 环形依赖：compiler/runtime 不能构造所有 role 同时等待下游 credit 的不可打破循环；硬件需提供 watchdog counter 辅助定位。

### 5.4 EOS/error/reset 传播

| token       | 行为                                                                          | completion 影响                     |
| ----------- | ----------------------------------------------------------------------------- | ----------------------------------- |
| VALID       | consumer 处理 payload，release 后 credit 回收                                 | 不直接完成 group task               |
| EOS         | consumer 标记 producer done；all-EOS 时 role done                             | 可触发 downstream EOS               |
| ERROR       | consumer 停止普通处理，记录 fault，向 downstream 推 error 或 group task fault | group task/tile completion 带 fault |
| VALID+FENCE | consumer 在 pop 后保证 payload fence 已成立                                   | 用于 DMA/MFE payload 可见性         |

reset domain：

- tile reset：取消该 tile 持有的 popped token，回收 credit，向 queue owner 记录 canceled producer/consumer。
- group reset：停止 group 内所有 queue，drain/cancel token，清空 queue SRAM。
- device reset：所有 queue 进入 reset clean；driver/firmware 重建 descriptor shadow。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 容量假设

| 配置     | queue 数量 / group | 默认 depth |      token SRAM 估算 | 说明                               |
| -------- | -----------------: | ---------: | -------------------: | ---------------------------------- |
| Edge     |            8 到 16 |     2 到 4 | 由 SRAM profile 冻结 | 小 batch pipeline                  |
| Balanced |           32 到 64 |     3 到 8 | 由 SRAM profile 冻结 | paged attention / MoE role overlap |
| High End |          64 到 128 |    4 到 16 | 由 SRAM profile 冻结 | 多 pipeline、多 context            |

First Silicon 可从 depth=3 的 canonical trace 开始，因为源文档 group task 示例使用 `depth=3`；产品值由后续规格冻结。

### 6.2 性能模型

```text
T_role_pipeline = max(T_producer, T_consumer, T_memory) + T_queue_overhead
T_queue_overhead = T_acquire + T_push + T_pop + T_release + T_fence
```

吞吐限制：

```text
Throughput_stream = min(
  producer_rate,
  consumer_rate,
  queue_depth / round_trip_release_latency,
  payload_memory_bw / payload_bytes
)
```

### 6.3 PMU hooks

必需 counter：

- `sq_push_count[q]`
- `sq_pop_count[q]`
- `sq_eos_count[q][producer]`
- `sq_error_count[q]`
- `sq_occupancy_cycles[q]`
- `sq_full_stall_cycles[q]`
- `sq_empty_stall_cycles[q]`
- `sq_credit_leak_fault_count[q]`
- `sq_reset_drain_cycles[q]`
- `sq_max_occupancy[q]`
- `sq_sequence_gap_count[q]`

PMU 归因：producer 等 credit 记为 stream_credit；consumer 等 token 记为 engine_wait_operand 或 stream_credit，按 primary stall hierarchy 冻结；NoC 造成 push/pop response 延迟时 secondary tag 为 NoC VC。

## 7. RTL/软件实现建议

- Queue descriptor shadow 在 group task init 时一次加载，运行中只允许状态寄存器变化。
- Token FIFO 使用 ready/valid 接口；valid token 在 ready 前必须稳定。
- head/tail/credit 使用同一时钟域；跨 clock domain 需要 async FIFO 和 CDC signoff。
- producer_id、consumer_id 从 Tile UCE / Tile Group Sequencer 的硬件 ID 派生，避免软件伪造越权 producer。
- queue SRAM 写入 token header 时同时写 ECC/parity；uncorrectable error 转 error token 或 group fault。
- firmware reset path 先置 `STOP_ACCEPT_NEW_TOKEN`，再等待 popped token release，最后 reconcile credit。
- Compiler lowering 应为每个 stream queue 生成 producer/consumer mask 和 EOS policy；多 consumer 初版使用多个 queue 显式表达。
- Runtime package 中的 stream descriptor 必须参与 ABI version check。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA / formal checks

- Credit conservation invariant 恒成立，reset/drain 结束后 credit 等于 depth。
- FIFO order：同一 producer `sequence_id` 递增，pop 顺序不逆转。
- No double release：同一 token handle release 两次必须 fault。
- No silent drop：RUNNING 状态下 accepted token 最终被 pop 或在 reset/drain 中被 canceled 并记录。
- EOS all-producer：all-EOS policy 下缺少任一 producer EOS 不得 role done。
- Error propagation：error token accepted 后，fault record index 必须出现在 queue status 或 downstream error token。
- Backpressure stability：full 时 producer stall；empty 时 consumer stall；stall 期间状态不破坏。
- Reset generation：旧 generation token 不得污染新 queue run。

### 8.2 测试矩阵

| 测试                  | 目的                         | 验收                                           |
| --------------------- | ---------------------------- | ---------------------------------------------- |
| SPSC depth sweep      | credit、order、release       | push/pop 序列与 golden 一致                    |
| MPSC interleave       | per-producer sequence 和 EOS | all-EOS 行为确定                               |
| full/empty stress     | backpressure                 | stall counter 与激励对齐                       |
| error token injection | fault propagation            | group task completion 带 fault record          |
| reset while popped    | credit reclaim               | reset 后 occupancy=0、credit=depth             |
| NoC congestion        | VC2 bulk 下 stream/event     | 无 token drop，PMU 有 congestion secondary tag |
| payload fence         | memory visibility            | consumer 不读到未提交 payload                  |

Bring-up 顺序：先 SPSC FIFO formal，再 RTL random push/pop，再接 Tile UCE stream 指令，再接 group task role pipeline，最后接 MFE Page Stream EOS/error。

### 8.3 跨模块 contract checklist

- Binary struct / protocol：stream queue descriptor、token、Tile Program stream 指令和状态寄存器均有 v0 草案。
- State machine：producer、consumer、queue drain/reset 三类状态机必须在 RTL 中一一可见。
- Capacity / bandwidth / area：queue 数量、depth、token SRAM、payload SRAM 占用和 round-trip release latency 分离建模；未冻结数值由后续规格冻结或由 SRAM profile 冻结。
- NoC VC behavior：token control、error token 和 event/fault 上报走 VC0 或等价高优先级控制路径；payload bulk data 走 VC2；VC2 拥塞不得阻止 EOS/error/reset 可见。
- Credit / EOS / error / reset：credit invariant、per-producer EOS bitmap、all-EOS policy、error token fault index、reset generation 和 credit reconcile 是 First Silicon 必需语义。
- Patch ownership：stream descriptor 静态字段由 Compiler，context/base/residency 由 Runtime/firmware，queue init/drain 由 Tile Group Sequencer，token 操作由 Tile UCE 或 engine wrapper。
- Ordering / coherency：token 保序不等于 payload coherency；producer payload fence 必须先于 push，consumer release 才回收 credit。
- SVA / formal：FIFO order、credit conservation、EOS all-producer、error propagation、reset generation invalidation 必须覆盖。
- PMU / error hooks：occupancy、full/empty stall、credit leak、EOS/error count 和 reset drain cycles 必须进入 PMU。

## 9. 风险、取舍和后续细化方向

| 风险                          | 影响                           | 缓解                                                          |
| ----------------------------- | ------------------------------ | ------------------------------------------------------------- |
| credit leak                   | queue 永久 full 或 credit 虚增 | invariant counter、reset reconcile、formal proof              |
| EOS 语义含混                  | role 过早结束或无法结束        | descriptor 中显式 EOS policy 和 producer mask                 |
| multi-consumer 过早实现       | refcount/reset 验证爆炸        | First Silicon 使用独立 queue 表达广播                         |
| error token 被普通 token 淹没 | fault 延迟不可控               | error latch + VC0 fault fabric + group task policy stop queue |
| payload/token coherency 混淆  | consumer 读旧数据              | payload fence、event ordering、SVA 可见性检查                 |
| PMU 归因重复                  | 性能调优误判                   | primary stall owner + secondary debug tag                     |

后续需要冻结 queue 数量、depth、token SRAM 实现、broadcast/refcount 是否进入 V1.x、EOS token 是否消耗 credit、watchdog 周期、NoC push/pop packet format、每类 stream 的 canonical trace。
