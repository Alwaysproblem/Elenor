# Tile Group 设计文档

## 1. 定位、目标和 First Silicon cutline

Tile Group 是 ELENOR 的局部数据复用、局部同步和 group 内 pipeline 执行域。它位于 Device Runtime 与 Compute Tile 之间，承接 group task，将 HBM/DDR/LPDDR 经 Group DMA 预取到 Group Shared SRAM/L2，再通过 Tile DMA、Stream Queue、Barrier/Event 和 Collective Engine 组织多个 Compute Tile 协同执行。

核心目标：

- 降低所有 tile 直接访问全局内存和全局 NoC 的压力。
- 在 group 内提供可验证的 role dispatch：block 粒度流动，而不是 role 全量串行。
- 将 program control、data movement、synchronization、collective 和 PMU attribution 分层，避免 Tile Group Sequencer、Tile UCE、MFE、USE 职责重叠。
- 支撑 Dense Attention、Paged Attention、MoE combine、split-K GEMM、多模型 group partition 等 Architecture V1 目标。

First Silicon V1 cutline：

| 能力            | First Silicon V1                                                        | V1.x / V2 reserved                            |
| --------------- | ----------------------------------------------------------------------- | --------------------------------------------- |
| Group task 执行 | 单 group task、固定 role graph、有限分支                                | priority/preemption、多 context QoS           |
| Group DMA       | 1D/2D/strided HBM 到 L2 prefetch，completion event                      | multicast DMA、gather list、layout transform  |
| Stream Queue    | 单/多 producer-consumer token、credit、EOS/error token、drain           | 复杂 multi-consumer refcount 策略             |
| Barrier/Event   | group barrier、role event、tile done、group done、timeout               | 分层 global barrier、采样 trace               |
| Collective      | group 内 reduce add/max/min、broadcast/multicast                        | cross-group all-reduce、reduce-scatter 强化   |
| PMU             | DMA bandwidth、queue occupancy、event wait、SRAM conflict、NoC VC stall | 完整 stall taxonomy 和 PMU feedback scheduler |

未冻结的容量、端口数、bank 数、队列深度、timeout 默认值和 NoC bandwidth 均写为 `由后续规格冻结` 或 `由 SRAM profile 冻结`，不能在 RTL 中以隐式常量固化。

## 2. 职责、非职责和 ownership

Tile Group owns：

- Tile Group Sequencer：维护 action index，推进 group task，dispatch role，管理 group barrier/event。
- Program Residency Manager：lookup `program_id/template_id`、处理 miss fetch/verify/install、维护 program cache state/epoch/refcount/pinning。
- Stream Queue Engine：role 间 token、credit、backpressure、EOS/error 传播。
- Barrier/Event Engine：group 内 tile 同步、DMA completion、role synchronization、fault propagation。
- Group DMA Engine：HBM/DDR/LPDDR 到 Group Shared SRAM/L2 的预取、storeback 和 completion event。
- Shared SRAM/L2：group 共享 prefetch buffer、stream buffer、partial result buffer、program cache、descriptor/status 区。
- Collective Engine：group 内 reduce、broadcast、multicast、可选 all-reduce。
- Group PMU：group 级 stall attribution、queue occupancy、DMA bandwidth、SRAM/NoC contention。
- Tile Dispatcher：只派发 prepared tile task，附带 local program handle、descriptor window 和 slot frame metadata。

Tile Group 不负责：

- 高层 graph 解释、shape policy 和 package 加载策略；这些由 Device Runtime 和 compiler/runtime package 负责。
- Tile-local kernel PC、L2 到 L1 DMA、BOA/EVU/MFE/USE launch；这些由 Tile UCE 和 tile 内 engine 负责。
- MFE 的数据相关动态地址生成；MFE owns page/segment metadata walk、address generation、prefetch/reorder/stream fill。
- USE 的 state/register file、prefix scan、recurrence、checkpoint/restore ownership。
- 全芯片 memory policy、global arbitration 和跨 context security policy。

关键 ownership 规则：

1. Tile Group Sequencer owns group task dispatch control；hardware 执行 TileGroupTask，不执行高层 graph。
2. Group DMA owns HBM 到 L2 的显式 copy；Tile DMA owns L2 到 L1；二者 completion event 必须区分 producer_id。
3. Stream Queue owns token 生命周期和 credit；Barrier/Event owns event 状态机；不能由 role program 直接篡改队列内部 credit。
4. Shared SRAM/L2 的 allocation 由 descriptor/task frame 显式声明；DMA、Collective、Stream Queue、Tile Dispatcher 的写 owner 必须唯一。
5. PMU primary stall owner 每 cycle 只能有一个，secondary tag 仅用于 debug。

## 3. 微架构和状态机

### 3.1 顶层 block

```text
Tile Group
├── Tile Group Sequencer
├── Task Preparation Engine
├── Program Residency Manager
├── Program Cache / Descriptor Window
├── Tile Dispatcher
├── Stream Queue Engine
├── Barrier / Event Engine
├── Group DMA Engine
├── Shared SRAM / L2 banks
├── Collective Engine
├── Multicast / Broadcast Unit
├── Group PMU / Trace / Fault Record
└── Compute Tile x 由后续规格冻结
```

推荐的内部互连：

- control path：Tile Group Sequencer -> Tile Dispatcher / DMA / Queue / Barrier / Collective。
- event path：所有 engine -> Barrier/Event Engine -> Tile Group Sequencer / Runtime event table。
- data path：Group DMA 和 Collective 访问 Shared SRAM/L2；Tile DMA 通过 L2 port 取数；Stream Queue token payload 指向 L2 或 tile-local slot。
- PMU path：各 engine 输出 active/stall/event，Group PMU 聚合并按 context_id、task_id、role_id、engine_id 标记。

### 3.2 Group task 状态机

```text
RESET
  -> IDLE
  -> ACCEPT_TASK
  -> VALIDATE_DESC
  -> PROGRAM_LOOKUP
  -> PROGRAM_FETCH_ON_MISS
  -> PROGRAM_VERIFY
  -> INIT_RESOURCES
  -> RUN
  -> DRAIN
  -> COMPLETE
  -> IDLE

任一状态遇到 fatal fault:
  -> ERROR_RECORD
  -> DRAIN_OR_KILL
  -> SIGNAL_ERROR
  -> IDLE 或 RESET_REQUIRED
```

状态语义：

- `ACCEPT_TASK`：锁存 context_id、task_id、descriptor pointer、wait/signal event。
- `VALIDATE_DESC`：检查 ABI version、descriptor bytes、CRC/validation mode、tile_mask、SRAM range、queue count。
- `PROGRAM_LOOKUP`：按 `context_id + program_id + program_version` 查询 Program Residency Manager。
- `PROGRAM_FETCH_ON_MISS`：cold launch 时由 Group DMA 将 Tile Program 搬入 Group Program Cache；同一 key 已在抓取时只挂等待，不重复发请求。
- `PROGRAM_VERIFY`：核对 hash/CRC/ABI/epoch，失败则写 fault record，不进入 READY。
- `INIT_RESOURCES`：清零队列 head/tail/credit、初始化 event scoreboard、分配 group SRAM window、配置 PMU epoch。
- `RUN`：Tile Group Sequencer 只在 `program_ready`、descriptor window ready 和依赖满足后按 action index issue 指令，role 以 block token 流动。
- `DRAIN`：停止接收新 block，等待 in-flight DMA、tile task、collective、queue token 回收。
- `COMPLETE`：signal group done event，写入 PMU epoch summary。

### 3.3 Role pipeline semantics

TileGroupTask 内的 role 之间以 block token 连接，不是高层 graph：

```text
Role0(preload/produce) -> S0 -> Role1(compute) -> S1 -> Role2(store/merge)
```

必须满足：

1. Role0 不需要完成全部 block 后 Role1 才开始；只要 S0 有 valid token 且 Role1 有 credit/资源即可并行。
2. 每个 token 携带 `token_id`、`payload_addr`、`payload_bytes`、`producer_id`、`sequence_id` 和 flags。
3. backpressure 是协议的一部分：queue full stall producer，queue empty stall consumer，stall 归因到 `stream_credit_empty_or_full`。
4. EOS 是 token，不是 out-of-band wire；多 producer 时采用 all-EOS policy 或 per-producer EOS policy，policy 由 descriptor 明确。
5. error token 携带 fault_record_slot，并沿 stream 传播到 group task completion。
6. barrier 不替代 stream；barrier 只声明某个同步点所有 participant 达到，stream 声明 producer-consumer 数据可用性。

### 3.4 Reset、drain 和 error 状态机

Reset domain：

| Domain       | 影响范围                                                               | 必须动作                                                                  |
| ------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| tile reset   | 单 tile UCE/engine/L1                                                  | 取消该 tile task，向 group event 写 RESET，释放相关 queue consumer credit |
| group reset  | Tile Group Sequencer、Group DMA、queue、barrier、collective、L2 window | drain 或 kill in-flight，清 queue credit，poison 未完成 event             |
| device reset | 全局 runtime/device queues/所有 group                                  | 由 Device Runtime 统一重建 event table 和 context                         |

Drain 行为：

- graceful drain：停止 dispatch 新 block，等待 DMA/collective/tile done，推送 EOS，回收 credit。
- forced drain：timeout 或 fatal fault 后禁止新 DMA，向未完成 queue 注入 error token，event 置 ERROR 或 TIMEOUT。
- reset drain：不保证 payload 内容有效，但必须保证 token/credit/event 状态确定，不能留下不可见 outstanding transaction。

Error 分类：

- descriptor validation fault。
- SRAM range/permission fault。
- DMA timeout/address fault。
- queue credit leak 或 protocol fault。
- barrier participant mismatch。
- collective numeric/protocol fault。
- NoC poison/backpressure timeout。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Group task launch command 示例

```c
typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t flags;
    uint16_t priority;

    uint32_t context_id;
    uint32_t task_id;
    uint32_t group_id;
    uint32_t role_count;
    uint32_t tile_mask_union;

    uint64_t group_task_iova;
    uint32_t group_task_bytes;
    uint64_t role_binding_iova;
    uint32_t role_binding_bytes;
    uint64_t engine_desc_iova;
    uint32_t engine_desc_bytes;
    uint64_t stream_desc_iova;
    uint32_t stream_desc_bytes;

    uint64_t wait_ref_iova;
    uint32_t wait_ref_count;
    uint32_t wait_ref_crc_or_zero;
    uint32_t signal_event;
    uint32_t signal_sequence;

    uint16_t residency_hint;
    uint16_t cache_policy;
    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_group_task_launch_desc_v0_t;
```

未冻结字段的 alignment、endianness、CRC policy、flag bit 编码由后续规格冻结。

### 4.2 Group task resource descriptor

```c
typedef struct {
    uint32_t queue_count;
    uint32_t event_count;
    uint32_t role_count;
    uint32_t dma_desc_count;

    uint32_t l2_window_base;
    uint32_t l2_window_bytes;
    uint32_t stream_buffer_base;
    uint32_t stream_buffer_bytes;

    uint16_t residency_hint;
    uint16_t cache_policy;
    uint32_t pmu_epoch_id;
    uint32_t sram_bank_hint_mask;
    uint32_t flags;
} elenor_group_task_resource_desc_t;
```

协议要求：

- `l2_window_base/bytes` 必须落在 group SRAM allocation 内，且不覆盖 active resident program、descriptor、event 区。
- `queue_count` 对应 Stream Queue descriptor table；每个 queue 的 producer/consumer mask 必须与 role binding 一致。
- `residency_hint/cache_policy` 只提供 placement/pinning 建议，不改变 correctness；真正正确性由 `program_id + version + hash + epoch` 保证。
- `sram_bank_hint_mask` 是 placement hint，不是安全边界；硬件仍需 range check。
- descriptor patch 只允许由 Runtime/Tile Group Sequencer 在 group task start 前完成，RUN 状态下修改必须通过显式 fence。

### 4.3 MMIO/CSR 建议

| 寄存器                | 描述                                               |
| --------------------- | -------------------------------------------------- |
| `GROUP_CONTROL`       | enable、soft_reset、drain_request、pmu_epoch_start |
| `GROUP_STATUS`        | idle、running、draining、error、reset_required     |
| `TGS_ACTION_INDEX`    | 当前 action index 或 fault action index            |
| `ACTIVE_CONTEXT_TASK` | context_id、task_id、group_id                      |
| `DMA_STATUS`          | in-flight count、last fault、bandwidth bucket      |
| `QUEUE_STATUS_BASE`   | per queue occupancy、credit、EOS/error seen        |
| `EVENT_STATUS_BASE`   | group event scoreboard snapshot                    |
| `BARRIER_STATUS_BASE` | barrier participant mask、arrived mask、epoch      |
| `PMU_SELECT/PMU_READ` | counter select 和读取接口                          |
| `FAULT_RECORD_BASE`   | fault code、producer_id、sequence、timestamp       |

CSR 宽度、地址 map 和 privilege policy 由后续规格冻结。

### 4.4 TileGroupTask dispatch examples

```asm
group_task.accept t0
    init_stream   s0, depth=3, policy=all_eos
    init_stream   s1, depth=2, policy=single_producer

    dma.prefetch  desc=weights_blk0, dst=l2_w0 -> ev_dma0:seq_dma0
    wait.event    ev_dma0 seq=seq_dma0

    dispatch.role role_id=0, event=ev_role0, seq=seq_role0, tile_mask=0x0f, program=tile_kernel_qk, out=s0
    dispatch.role role_id=1, event=ev_role1, seq=seq_role1, tile_mask=0xf0, program=tile_kernel_softmax, in=s0, out=s1

loop_blocks:
    wait.credit   s0
    dma.prefetch  desc=kv_next, dst=l2_kv_next -> ev_dma1:seq_dma1[block_id]
    wait.event    ev_dma1 seq=seq_dma1[block_id]
    wait.stream   s0
    advance.block
    branch.lt     block_id, block_count, loop_blocks

    push.eos      s0
    push.eos      s1
    barrier.group participants=0xff -> ev_bar0:seq_bar0
    wait.event    ev_bar0 seq=seq_bar0
group_task.complete signal=group_done seq=seq_group_done
```

`seq_*[block_id]` 表示每次 block 循环复用 event id 时必须使用新的 expected sequence；固定 sequence 不能跨循环迭代复用。

硬件执行这些 command/descriptor/program，不解释 MLIR 或高层 graph。

## 5. 数据流、控制流和时序路径

### 5.1 数据流

```text
HBM/DDR/LPDDR
  -> Memory Controller / NoC VC1/VC2
  -> Group DMA
  -> Group Shared SRAM / L2
  -> Tile DMA
  -> Tile L1 slot frame
  -> BOA / EVU / MFE tile port / USE
  -> Tile L1
  -> L2 partial/result buffer
  -> Group Collective or Group DMA storeback
```

Group DMA relationship：

- Tile Group Sequencer issues Group DMA descriptors for HBM 到 L2 prefetch/storeback。
- Tile UCE issues Tile DMA for L2 到 L1；Tile UCE 不应直接调度 HBM policy。
- MFE 可以经 global path 参与数据相关动态访问，但 page/segment stream 的 address generation owner 是 MFE；如果写入 L2 stream buffer，需要与 Group DMA 通过 SRAM arbiter 和 descriptor ownership 隔离。
- Group DMA completion event 可作为 role token 生产条件，也可作为 Tile Group Sequencer wait 条件。

### 5.2 控制流

```text
Device Runtime command queue
  -> ELENOR_CMD_LAUNCH_GROUP_TASK
  -> Tile Group Sequencer action index
  -> Tile Dispatcher / Group DMA / Stream Queue / Barrier / Collective
  -> Tile UCE tile task
  -> engine completion event
  -> role event / group done event
  -> Runtime event table
```

控制面原则：

- command/control 使用 NoC VC0，不能被 VC1/VC2 data stream 阻塞。
- collective 使用独立 VC 或内部 tree path，避免 reduce traffic 阻塞 event/barrier。
- Tile Group Sequencer 的 wait 指令必须带 timeout 或受 group task timeout 覆盖。
- event sequence 必须单调递增，避免 reset 后 stale event 被误识别为 done。

### 5.3 关键时序路径

需要重点收敛的路径：

- TGS action decode -> issue arbitration -> engine command valid。
- Queue push/pop -> occupancy/credit update -> backpressure ready。
- Barrier arrived mask reduction -> release event。
- DMA completion -> event scoreboard -> Tile Group Sequencer wait wakeup。
- SRAM arbiter grant -> bank conflict detect -> PMU stall owner latch。
- Collective reduction datapath；大宽度 reduce tree 可能需要 pipeline register。

非关键但必须可观测路径：fault record write、PMU counter increment、debug CSR read。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 SRAM/L2 assumptions

Group SRAM 建议容量来自 Architecture V1：Edge 2 到 8 MB，Balanced 16 到 64 MB，High End 64 到 128 MB。First Silicon V1 应选择能闭合面积、功耗、时序和验证的 SRAM profile；具体容量、bank 数、端口和 ECC policy 由 SRAM profile 冻结。

L2 分区建议：

| 分区                  | 用途                                           | owner                          |
| --------------------- | ---------------------------------------------- | ------------------------------ |
| group task/cache      | TileGroupTask、role binding、descriptor window | Tile Group Sequencer           |
| stream buffer         | token payload、role ring buffer                | Stream Queue / producing role  |
| prefetch buffer       | weights、activation、KV tile                   | Group DMA / MFE                |
| partial result buffer | split-K、attention partial、MoE combine        | BOA/EVU/Collective，单写 owner |
| event/status          | event table、fault record、PMU epoch summary   | Barrier/Event / PMU            |

### 6.2 性能模型

Group 级一阶模型：

```text
T_group_task = max(
  T_group_dma_prefetch - T_overlap_compute,
  T_tile_compute_pipeline,
  T_stream_backpressure,
  T_collective,
  T_storeback
) + T_barrier_event
```

Group SRAM bandwidth 预算：

```text
BW_group_sram_required =
  BW_group_dma_read_write
+ BW_tile_dma_l2_ports
+ BW_stream_payload
+ BW_collective_read_write
+ BW_group_task_desc_event
```

必须满足：

```text
BW_group_sram_required <= BW_group_sram_peak * efficiency
```

`BW_group_sram_peak`、efficiency、bank conflict model 由 SRAM profile 冻结。

### 6.3 PMU counters

必需 counter：

- `group_active_cycles`。
- `tgs_issue_cycles`、`tgs_wait_event_cycles`、`tgs_wait_stream_cycles`。
- `group_dma_bytes_read`、`group_dma_bytes_write`、`group_dma_active_cycles`、`group_dma_wait_memory_cycles`。
- `stream_queue_occupancy_sum`、`stream_credit_empty_cycles`、`stream_credit_full_cycles`、`stream_eos_count`、`stream_error_token_count`。
- `barrier_wait_cycles`、`barrier_release_count`、`event_timeout_count`。
- `collective_active_cycles`、`collective_wait_input_cycles`、`collective_output_backpressure_cycles`。
- `group_sram_bank_conflict_cycles` by consumer。
- `noc_vc0_backpressure_cycles`、`noc_vc1_backpressure_cycles`、`noc_vc2_backpressure_cycles`、`noc_vc3_backpressure_cycles`。
- `tile_dispatch_count`、`tile_done_count`、`tile_error_count`。

Primary stall owner priority：engine active、wait event、wait operand/data、stream credit、SRAM bank、NoC VC、DMA memory、program/descriptor stall、unknown。

## 7. RTL/软件实现建议

RTL 建议：

- Tile Group Sequencer 使用小型 fixed-width decoder + scoreboard，不实现通用 CPU。
- engine issue 采用 ready/valid，所有 command 带 context_id、task_id、role_id、sequence。
- Stream Queue head/tail/credit 使用显式 invariant 检查；credit counter 宽度由 depth 推导，不手写 magic width。
- Barrier/Event Engine 与 Tile Group Sequencer 分离，避免 wait wakeup 和 event write 形成组合环。
- Group DMA descriptor fetch、address generation、NoC request、completion writeback 分 pipeline；completion 必须携带 descriptor sequence。
- Shared SRAM arbiter 输出 grant owner，供 PMU 和 SVA 同时使用。
- Reset controller 为 tile/group/device reset 维护独立 epoch，event/queue token 比较 epoch 防 stale。

软件/firmware 建议：

- compiler 生成 group task action sequence、role binding、stream queue descriptor，以及 `program_id/template_id/program_ref/cache hint`。
- runtime 只提交 command buffer、注册 program section、patch descriptor/launch metadata、等待 event；不在热路径解释高层 graph，也不生成显式 `program.load`。
- firmware validation 检查 descriptor range、tile_mask、queue producer/consumer mask、event count、program version/hash 和 timeout policy。
- bring-up 顺序先 command/event/barrier，再 DMA completion，再 queue token，再 collective，再 PMU attribution。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA/formal points

- Queue invariant：`0 <= occupancy <= depth`，`credit + occupancy == depth`，reset/drain 后 credit 回到 depth。
- Token order：同一 producer 的 `sequence_id` 单调递增；consumer 不得观察重复 valid token。
- EOS rule：all-EOS policy 下所有 producer EOS 到达后 consumer 才收到 stream complete。
- Error propagation：任意 producer error token 最终导致 group task event ERROR 或按 descriptor policy 转换为可恢复状态。
- Barrier safety：release event 只能在 `arrived_mask == participant_mask` 且 epoch 匹配时产生。
- DMA completion：每个 accepted DMA descriptor exactly one completion event，除非 reset epoch kill，kill 必须写 RESET/TIMEOUT。
- SRAM ownership：同一 cycle 同一 L2 address range 不允许两个写 owner 无仲裁写入。
- Event freshness：waiter 只能接受同 epoch、同 sequence 的 event。
- Deadlock freedom bounded proof：在 memory ready、tile ready、collective ready 公平假设下，非 EOS pipeline 最终推进或 timeout。

### 8.2 Bring-up sequence

1. CSR reset/idle/status smoke。
2. launch empty group task，signal group done event。
3. Group DMA 1D copy HBM -> L2 -> HBM，completion event 和 CRC/golden compare。
4. single queue producer/consumer token loop，验证 credit/EOS/error。
5. two-role pipeline overlap，观察 Role1 在 Role0 未完成全部 block 前启动。
6. group barrier participant mask release。
7. group collective reduce add with deterministic data。
8. forced timeout、group drain、soft reset，验证 event/fault/credit 状态。
9. PMU epoch readout 与预期 stall owner 对齐。

### 8.3 验收标准

- Implicit program residency 的 cold/warm path 均可完成。
- command -> Program Residency Manager -> Group DMA -> Stream Queue -> prepared tile dispatch -> event -> PMU 端到端闭环。
- reset/drain/error 行为确定，无 credit leak、stale event 或 orphan DMA。
- SVA/formal properties 在 bounded depth 下通过。
- PMU 能解释至少 DMA memory stall、stream backpressure、barrier wait、SRAM bank conflict、NoC VC backpressure 五类瓶颈。

## 9. 风险、取舍和后续细化方向

主要风险：

- SRAM/NoC contention：Group DMA、Tile DMA、Collective、Stream payload 同时访问 L2，可能压低 BOA/EVU/MFE 有效吞吐。
- Tile Group Sequencer 过度通用化：若演化成小 CPU，会增加验证和时序风险。
- Queue/barrier/event 协议不清：最容易形成不可复现 deadlock 或 reset 后 stale state。
- Collective 与 DMA 共享 SRAM port：reduce tree 和 storeback peak 叠加时可能造成 bank conflict。
- 多 context group partition：SRAM quota、fault isolation、priority policy 若无硬件边界，会影响可靠性。

取舍：

- First Silicon V1 优先固定小而完整的 command/event/DMA/queue/PMU path，而不是追求复杂 scheduling。
- TileGroupTask action encoding 保持控制面指令；数据相关动态访问交给 MFE，tile-local kernel pipeline 交给 Tile UCE。
- Group SRAM 使用 descriptor window 和 placement hint，不承诺 cache coherency。

后续需要冻结：

- Group SRAM profile、bank/port/ECC、arbiter policy。
- TileGroupTask action encoding、CSR map、fault code。
- Stream Queue multi-consumer policy 和 reset/drain 精确时序。
- Collective topology、latency model 和 numeric mode。
- PMU counter 编号、溢出语义和 runtime readout ABI。
