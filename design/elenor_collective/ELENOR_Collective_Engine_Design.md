# Collective Engine 设计文档

## 1. 定位、目标和 First Silicon cutline

Collective Engine 是 Tile Group 内负责 tile 间数据合并、广播和同步归约的专用 engine。它位于 Tile Group data/control boundary：Tile Group Sequencer 发起 collective command，Compute Tile 通过 Tile DMA/stream/event 提供 partial data，Collective Engine 在 Group Shared SRAM/L2 或专用 reduction datapath 中完成 reduce/broadcast/multicast，并通过 Barrier/Event Engine signal completion。

典型用途：

- Attention partial score merge。
- large GEMM split-K partial sum merge。
- MoE expert output combine。
- multi-tile normalization statistics。
- group 内权重、activation 或 metadata broadcast/multicast。

First Silicon V1 cutline：

| 能力        | First Silicon V1                                               | V1.x / V2 reserved                             |
| ----------- | -------------------------------------------------------------- | ---------------------------------------------- |
| Reduce op   | add、max、min，INT32/BF16/FP32 accumulator mode 由后续规格冻结 | custom op、atomic update、complex combine      |
| Broadcast   | group 内 L2 -> tile 或 tile -> group fanout descriptor         | cross-group broadcast、DMA multicast 合并      |
| All-reduce  | group 内可选，固定 participant mask                            | global all-reduce、hierarchical reduce-scatter |
| Data source | L2 partial result buffer、tile output stream token             | direct engine-to-engine bypass                 |
| Sync        | completion event、error event、participant barrier             | preemptible collective、QoS scheduling         |
| PMU         | active/stall/input wait/output backpressure/NoC VC             | trace sampling、fine-grained numeric debug     |

Collective Engine 不替代 BOA 内 reduce tree，也不替代 EVU local reduce。它处理跨 tile 或 group-level partial merge；tile-local vector/norm/softmax 仍由 EVU 或 tile-local engine 完成。

## 2. 职责、非职责和 ownership

Collective Engine owns：

- group 内 reduce add/max/min、broadcast、multicast、可选 all-reduce 的执行。
- participant mask、input readiness、output writeback 和 completion event。
- collective command descriptor validation 的硬件部分：range、dtype、op、participant、shape、stride。
- L2 partial result buffer 的 read/write arbitration request。
- collective VC 或 group internal reduction tree 的 flow control。
- collective PMU counter 和 numeric/protocol fault record。

Collective Engine 不负责：

- Tile-local BOA/EVU/MFE/USE compute。
- Tile UCE 的 tile program control、descriptor patch 和 L2 到 L1 DMA。
- Group DMA 的 HBM 到 L2 prefetch/storeback；Collective 只消费或产生 L2 buffer，storeback 仍由 Group DMA 或后续 stage 处理。
- Stream Queue token lifecycle；Collective 可消费 token metadata 或 signal event，但不拥有 queue credit。
- 高层 graph 的 expert routing、attention partition 或 schedule policy；compiler/runtime/Tile Group Sequencer 提供 descriptor。

Ownership 规则：

1. Collective input buffer 的 producer 必须唯一：BOA/EVU/Tile DMA/MFE 中只能有一个 owner 写入同一 partial slot。
2. Collective output buffer 的 writer 是 Collective Engine，除非 descriptor 声明 in-place reduce 且满足无读写冲突规则。
3. Barrier/Event Engine owns event status；Collective 只产生 completion/error request。
4. Group DMA 与 Collective 共享 L2 时通过 SRAM arbiter；Collective 不直接发 HBM transaction。
5. PMU primary stall owner 对 collective cycle 归因到 input wait、SRAM bank、NoC VC、output backpressure 或 active，不能与 Group DMA 重复计数。

## 3. 微架构和状态机

### 3.1 内部模块

```text
Collective Engine
├── Command / Descriptor Fetch
├── Descriptor Validator
├── Participant Scoreboard
├── Input Stream/Event Wait Unit
├── L2 Read Address Generator
├── Reduction Datapath
├── Broadcast / Multicast Fanout Unit
├── L2 Writeback Unit
├── Event Completion Unit
├── Fault / Timeout Unit
└── PMU Event Encoder
```

实现拓扑可以是集中式 reducer、tree reducer 或 hierarchical group reducer；Topology、lane 数、每 cycle 元素数和 pipeline stage 由后续规格冻结。First Silicon V1 推荐优先实现集中式或小树形 group reducer，降低验证复杂度。

### 3.2 Collective command 状态机

```text
RESET
  -> IDLE
  -> ACCEPT_CMD
  -> VALIDATE_DESC
  -> WAIT_INPUTS
  -> READ_PARTIALS
  -> REDUCE_OR_FANOUT
  -> WRITE_OUTPUTS
  -> SIGNAL_DONE
  -> IDLE

fault/timeout:
  -> FAULT_RECORD
  -> POISON_OUTPUT_OR_SUPPRESS
  -> SIGNAL_ERROR
  -> IDLE 或 RESET_REQUIRED
```

状态说明：

- `ACCEPT_CMD`：从 Tile Group Sequencer 或 command queue 接收 collective descriptor id。
- `VALIDATE_DESC`：检查 op、dtype、shape、participant_mask、L2 range、alignment、output policy。
- `WAIT_INPUTS`：等待 tile done event、stream token 或 explicit participant ready bit。
- `READ_PARTIALS`：按 descriptor stride 从 L2 partial buffer 读取输入。
- `REDUCE_OR_FANOUT`：执行 reduce add/max/min 或 broadcast/multicast fanout。
- `WRITE_OUTPUTS`：写回 L2 output buffer 或为 tile consumers 生成 output token/event。
- `SIGNAL_DONE`：向 Barrier/Event Engine signal DONE，PMU epoch 记录元素数和 stall。

### 3.3 Role pipeline semantics

Collective 可以作为 group task 的一个 role：

```text
Role0: tile BOA/EVU produce partial -> stream S_partial
Role1: collective.reduce consumes S_partial -> stream S_reduced
Role2: tile EVU/Group DMA consumes S_reduced -> storeback or next compute
```

语义要求：

- Collective role 可按 block 粒度工作；不要求整个 tensor 所有 block 到齐。
- 对同一 `collective_id + block_id + sequence_id`，participant mask 内所有 required input 到达后才能 reduce。
- 如果 descriptor 允许 partial participant，mask 和 neutral element 必须显式给出；默认不允许静默缺 participant。
- Collective 输出可以是 L2 buffer、stream token 或 broadcast fanout event；具体 mode 由 descriptor 指定。
- Backpressure 来自 output queue full、L2 write port busy、NoC VC3 busy 或 downstream event wait。
- EOS/error token 必须透传：输入任一 required participant ERROR，默认输出 ERROR 并 signal collective error event；可恢复 policy 由后续规格冻结。

### 3.4 Reset、drain 和 error 行为

Drain mode：

- graceful drain：停止 accept 新 collective，完成已 accepted command，signal DONE/ERROR，释放 input token。
- forced drain：timeout/reset 时取消未开始 read 的 command；已读但未写完整的 output 需标记 invalid 或 poison。
- group reset：清 participant scoreboard、input ready bits、in-flight read/write、event pending request、PMU active epoch。

Error 行为：

| Error               | 触发                                     | 行为                                                                       |
| ------------------- | ---------------------------------------- | -------------------------------------------------------------------------- |
| descriptor fault    | op/dtype/range/alignment/mask 非法       | 不读写 payload，signal ERROR                                               |
| participant timeout | required tile/event/token 未到           | fault record + ERROR/TIMEOUT                                               |
| input error token   | 上游 role error                          | propagate ERROR，携带 fault_record_slot                                    |
| SRAM fault          | L2 ECC/range/permission fault            | abort command，signal ERROR                                                |
| numeric fault       | overflow/NaN policy violation            | 按 descriptor numeric mode signal ERROR 或 saturate，policy 由后续规格冻结 |
| protocol fault      | duplicate participant、sequence mismatch | signal ERROR，PMU protocol_fault++                                         |

## 4. 接口、descriptor、寄存器和协议

### 4.1 Collective descriptor

```c
typedef enum {
    ELENOR_COLL_REDUCE_ADD = 0,
    ELENOR_COLL_REDUCE_MAX = 1,
    ELENOR_COLL_REDUCE_MIN = 2,
    ELENOR_COLL_BROADCAST  = 3,
    ELENOR_COLL_MULTICAST  = 4,
    ELENOR_COLL_ALL_REDUCE = 5
} elenor_collective_op_t;

typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t op;
    uint16_t dtype;

    uint32_t collective_id;
    uint32_t context_id;
    uint32_t task_id;
    uint32_t role_id;

    uint32_t participant_mask;
    uint32_t element_count;
    uint32_t input_stride_bytes;
    uint32_t output_stride_bytes;

    uint32_t input_l2_base;
    uint32_t output_l2_base;
    uint32_t scratch_l2_base;
    uint32_t scratch_bytes;

    uint32_t wait_event_base;
    uint16_t wait_event_count;
    uint16_t signal_event;

    uint32_t input_stream_id;
    uint32_t output_stream_id;
    uint32_t timeout_cycles;
    uint32_t flags;
} elenor_collective_desc_t;
```

字段语义：

- `participant_mask` 表示必须参与的 tile 或 subgroup。
- `input_l2_base/output_l2_base` 是 Group Shared SRAM/L2 offset，不是 HBM physical address。
- `input_stream_id/output_stream_id` 可为 disabled value，具体编码由后续规格冻结。
- `scratch_l2_base` 用于 tree reduce intermediate 或 all-reduce staging；是否必需由 op/topology 决定。
- dtype、rounding、saturation、NaN/Inf、overflow policy 由后续规格冻结。

### 4.2 Command examples

TileGroupTask 发起 split-K reduce：

```asm
    dispatch.role role_id=0, tile_mask=0xff, program=tile_kernel_splitk, out=s_partial
    wait.stream    s_partial, condition=all_participants_ready

    collective.run desc=coll_splitk_reduce -> ev_coll0
    wait.event     ev_coll0

    dispatch.role role_id=1, tile_mask=0x0f, program=tile_kernel_epilogue, in=s_reduced
```

MoE combine：

```asm
    dispatch.role role_id=0, tile_mask=0xff, program=tile_kernel_expert_mlp, out=s_expert_partial
    wait.stream    s_expert_partial, condition=block_ready

    collective.run desc=coll_moe_combine -> ev_combine
    wait.event     ev_combine

    dma.store      desc=moe_output_store, src=l2_moe_out -> ev_store
    wait.event     ev_store
```

Broadcast weights or metadata：

```asm
    dma.prefetch    desc=weight_block, dst=l2_weight -> ev_w
    wait.event      ev_w
    collective.run  desc=coll_broadcast_weight -> ev_bcast
    wait.event      ev_bcast
    dispatch.role  role_id=0, tile_mask=0xff, program=tile_kernel_gemm, in=s_weight_ready
```

### 4.3 Event/stream/barrier protocol

- Input readiness 可来自 wait_event、stream token 或 participant scoreboard。descriptor 必须选择一种 primary readiness mode，避免同一 input 被重复计数。
- Collective completion 通过 Event Engine 写 DONE/ERROR/TIMEOUT/RESET。
- 若输出到 stream，Collective 作为 producer 必须 acquire credit 后再 push token；若 credit 不足，stall owner 为 stream credit。
- 若 output broadcast 到多个 tile，fanout policy 是 broadcast token、per-consumer token 还是 event-only，由 descriptor flags 指定。
- group barrier 可用于 collective 前后同步，但 collective 本身不能假设 barrier 已经发生；它必须验证 required inputs。

### 4.4 CSR 建议

| CSR                      | 描述                                                |
| ------------------------ | --------------------------------------------------- |
| `COLL_CONTROL`           | enable、soft_reset、drain                           |
| `COLL_STATUS`            | idle、running、waiting_input、writing_output、error |
| `COLL_ACTIVE_DESC`       | collective_id、op、role_id                          |
| `COLL_PARTICIPANT_READY` | ready mask snapshot                                 |
| `COLL_FAULT_CODE`        | descriptor/protocol/SRAM/numeric fault              |
| `COLL_TIMEOUT`           | active command timeout                              |
| `COLL_PMU_SELECT/READ`   | PMU counter select/read                             |

CSR 地址、位宽和 clear-on-read policy 由后续规格冻结。

## 5. 数据流、控制流和时序路径

### 5.1 Reduce data flow

```text
Tile BOA/EVU/MFE output
  -> Tile DMA store or stream payload
  -> Group Shared SRAM / L2 partial result buffer
  -> Collective L2 read
  -> Reduction datapath
  -> L2 output buffer / output stream token
  -> next tile role or Group DMA storeback
```

Group DMA relationship：

- Group DMA 负责把输入 weights/activation/KV 或最终 output 在 HBM 与 L2 之间移动。
- Collective 只读写 L2 partial/result buffer；不直接发起 HBM transaction。
- Tile Group Sequencer 可在 Collective 完成后 issue Group DMA storeback。
- Group DMA 与 Collective 同时访问 L2 时，SRAM arbiter 必须按 bank/port 规则仲裁，PMU 记录冲突 owner。

### 5.2 Broadcast/multicast data flow

```text
L2 source buffer or tile producer
  -> Collective/Broadcast fanout
  -> output stream tokens or L2 per-tile slots
  -> Tile DMA L2 -> L1
  -> tile program consumes data
```

Broadcast 适合 weights、scale/mask metadata、小型 normalization statistics。大块权重广播是否使用 Collective、Multicast Unit 或 Group DMA multicast 由后续规格冻结；First Silicon V1 可用 L2 per-tile slot + tile DMA 方式实现，降低 NoC fanout 风险。

### 5.3 Control flow

```text
Tile Group Sequencer collective.run
  -> Collective Descriptor Validator
  -> Input Wait / Participant Scoreboard
  -> L2 read/reduce/write
  -> Event completion
  -> Tile Group Sequencer wait.event retires
```

控制面使用 command/event VC，collective data path 使用 collective VC 或 group internal tree。event/barrier 不得被 large reduce traffic 阻塞。

### 5.4 Timing paths

关键路径：

- participant ready mask reduction。
- L2 read response mux -> reduction datapath input。
- reduction tree carry/compare path。
- output writeback arbitration -> stream token push。
- event completion -> Tile Group Sequencer wakeup。

建议 reduction datapath pipeline 化，participant scoreboard 和 L2 address generation registered，避免 fan-in mask 和 arithmetic tree 落在同一 cycle。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 SRAM/L2 assumptions

Collective 使用 Group Shared SRAM/L2 中的 partial、scratch、output 区。容量和 bank placement 必须由 region resource descriptor 明确：

| Buffer          | 用途                                 | 约束                                              |
| --------------- | ------------------------------------ | ------------------------------------------------- |
| input partial   | per tile partial result              | participant slot 不重叠，alignment 由后续规格冻结 |
| scratch         | tree intermediate/all-reduce staging | 不与 input/output alias，除非 descriptor 允许     |
| output          | reduce/broadcast result              | writer owner 为 Collective                        |
| stream metadata | output token/status                  | 由 Stream Queue owns credit/head/tail             |

Banking 要求：

- input partial slots 应按 tile_id 分散到 bank，降低同 cycle read conflict。
- output buffer 与 Group DMA storeback buffer 避免同 bank 热点。
- event/status 区不应与 high bandwidth reduce payload 共 bank。
- 具体 bank 数、port 数、ECC、read latency 由 SRAM profile 冻结。

### 6.2 性能模型

Reduce latency：

```text
T_collective_reduce =
  T_wait_inputs
+ ceil(Bytes_input / BW_l2_read_eff)
+ T_reduce_datapath
+ ceil(Bytes_output / BW_l2_write_eff)
+ T_event
```

Broadcast latency：

```text
T_broadcast =
  T_wait_source
+ ceil(Bytes_source * fanout_factor / BW_fanout_eff)
+ T_token_or_event
```

关键约束：

```text
BW_l2_read_eff = BW_l2_read_peak * (1 - bank_conflict_rate)
BW_l2_write_eff = BW_l2_write_peak * (1 - output_conflict_rate)
```

所有 peak bandwidth、datapath lanes、fanout_factor model 由 SRAM profile 或 PPA exploration 冻结。

### 6.3 PMU counters

必需 counter：

- `coll_active_cycles`。
- `coll_cmd_count` by op。
- `coll_elements_processed`。
- `coll_bytes_read_l2`、`coll_bytes_written_l2`。
- `coll_wait_input_cycles`。
- `coll_wait_stream_credit_cycles`。
- `coll_sram_bank_conflict_cycles`。
- `coll_noc_vc_backpressure_cycles`。
- `coll_output_backpressure_cycles`。
- `coll_reduce_datapath_busy_cycles`。
- `coll_broadcast_fanout_cycles`。
- `coll_timeout_count`。
- `coll_error_count` by fault_code。
- `coll_participant_mismatch_count`。

PMU fingerprint examples：

- split-K GEMM：BOA active 高，Collective active 在 GEMM block 后出现，`coll_wait_input_cycles` 不应长期高于 BOA compute latency。
- MoE combine：routing imbalance 时 `coll_wait_input_cycles` 和 participant mismatch/debug counter 上升。
- Attention partial merge：`coll_sram_bank_conflict_cycles` 过高表示 partial layout/bank placement 不合理。

## 7. RTL/软件实现建议

RTL：

- Descriptor Validator 在任何 L2 read/write 前完成全部 range、alignment、op、dtype、participant 检查。
- Participant Scoreboard 使用 `collective_id, block_id, sequence_id, epoch` 标识输入，防止旧 token 混入新 command。
- Reduction datapath 与 L2 interface decouple：read FIFO、compute pipeline、write FIFO 分离，便于 timing closure。
- For add/max/min 使用统一 lane pipeline；numeric mode 通过 descriptor 控制，但 unsupported mode 必须 fault，不能静默降级。
- Output stream push 必须先 acquire credit；credit 不足时 hold output buffer valid 并计 PMU stall。
- Reset kill 清空 scoreboard/FIFO，并向 Event Engine 写 RESET 或 TIMEOUT，不留下 pending done。
- L2 writeback 必须支持 byte/element count exactness，tail element 行为由 descriptor dtype/element_count 决定。

Software/compiler：

- compiler 应显式决定 partial result owner、participant mask、L2 layout、bank placement hint。
- split-K、attention、MoE combine 的 collective descriptor 由 kernel library 模板生成，避免 runtime 拼接复杂语义。
- runtime 只 patch IOVA/L2 frame binding、event id、shape element_count，不改变 op 语义。
- 如果 collective output 需要 storeback，Tile Group Sequencer 在 completion event 后 issue Group DMA storeback。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA/formal verification points

- Descriptor safety：invalid descriptor 不得产生 L2 read/write。
- Participant completeness：required participant mask 未满足时不得进入 reduce datapath。
- Duplicate participant：同一 participant、同一 sequence 重复到达必须 fault 或按 descriptor policy 处理，不能重复累加。
- Exactly-once completion：每个 accepted collective command exactly one DONE/ERROR/TIMEOUT/RESET completion。
- Output ownership：Collective 写 output range 时不存在第二写 owner；若 SRAM arbiter stall，data/byte enable 保持稳定。
- Stream credit：output stream push 只能在 credit acquired 后发生；reset/drain 后 credit 不泄漏。
- Epoch freshness：input event/token epoch 必须匹配 active command epoch。
- Numeric determinism：固定 op/dtype/input order 下 result deterministic；max/min neutral element 定义由后续规格冻结。
- Deadlock bounded proof：在 L2 ready、stream credit eventually available、participants eventually ready 的假设下 collective 最终完成。

### 8.2 Bring-up

1. Reset/CSR idle/status。
2. Descriptor validation negative tests：bad range、bad op、bad participant mask。
3. Two-tile reduce add small vector，compare golden。
4. Max/min tail element case，compare golden。
5. Broadcast one L2 buffer to multiple tile slots，tile DMA consume。
6. Split-K GEMM partial reduce：BOA partial -> L2 -> Collective -> EVU/Group DMA。
7. MoE combine deterministic routing case。
8. Error token propagation：upstream stage error causes collective ERROR event。
9. Timeout and group reset during WAIT_INPUTS/WRITE_OUTPUTS。
10. PMU readout：active、wait input、bank conflict、output backpressure counters 与 scenario 匹配。

### 8.3 验收标准

- reduce add/max/min 在 supported dtype 下与 golden 对齐。
- Broadcast/multicast completion event 和 output stream token 行为确定。
- Collective 可作为 group task role 与 Stream Queue/Event/Barrier 正确交互。
- reset/drain/error 不产生 stale done event、不泄漏 stream credit、不写坏 output range。
- PMU 能解释 input wait、SRAM bank conflict、NoC/stream backpressure 和 active utilization。
- SVA/formal properties 对配置的 participant count、queue depth 和 FIFO depth 通过。

## 9. 风险、取舍和后续细化方向

风险：

- Numeric policy 不冻结：不同 dtype、rounding、overflow/NaN 规则会导致软件 golden 和 RTL 不一致。
- Participant protocol 复杂：MoE/attention 动态 mask 若未显式 descriptor 化，会出现缺 participant 或重复 participant。
- L2 bank conflict：partial result layout 不合理会让 Collective 与 Group DMA、Tile DMA、BOA storeback 互相阻塞。
- all-reduce scope 扩大：cross-group all-reduce 会牵涉 global NoC 和 runtime scheduling，不适合 First Silicon V1 默认实现。
- Broadcast fanout 过大：大块权重广播若走 event/control path 会阻塞 barrier/event。

取舍：

- First Silicon V1 优先 group 内 reduce/broadcast，global collective 只保留 ABI 扩展空间。
- Collective 只处理跨 tile/group-level partial merge；tile-local reduce 仍由 BOA/EVU 完成。
- 大块数据搬运继续由 Group DMA 管理，Collective 不直接访问 HBM。
- Descriptor 显式表达 participant、shape、dtype、buffer 和 output policy，避免硬件推断高层语义。

后续需要冻结：

- Collective descriptor binary layout、op 编码、dtype/numeric policy。
- Reduction topology、lane 数、pipeline depth、SRAM port 需求。
- Input readiness mode、output stream/broadcast fanout policy。
- All-reduce/reduce-scatter 是否进入 First Silicon V1。
- PMU counter 编号、overflow、event trace 和 runtime ABI。
