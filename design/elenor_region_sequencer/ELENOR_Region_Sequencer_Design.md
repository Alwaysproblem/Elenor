# Region Sequencer 设计文档

## 1. 定位、目标和 First Silicon cutline

Region Sequencer 是 Tile Group 内的 Region Program 执行控制器。Device Runtime 将 graph schedule lowering 成 Pipeline Region 并提交 `ELENOR_CMD_LAUNCH_REGION`；Region Sequencer 接收 region task，维护 Region PC，初始化 stream/event/barrier 资源，issue Group DMA、dispatch tile stage、等待 event/stream/barrier，并最终 signal group done。

Region Sequencer 的目标不是成为通用微控制器，而是成为可验证的 device pipeline 推进器：

- 执行 command/descriptor/program，不解释高层 graph。
- 在 group 内以 block/token 粒度推进 stage pipeline。
- 控制 HBM/DDR/LPDDR 到 L2 的 Group DMA 预取和 storeback。
- 协调 Stream Queue、Barrier/Event、Collective 与 Tile Dispatcher。
- 为 PMU 提供准确的 primary stall owner。

First Silicon V1 cutline：

| 项目     | First Silicon V1                                 | V1.x / V2 reserved                    |
| -------- | ------------------------------------------------ | ------------------------------------- |
| Program  | 固定宽度 Region Program、有限 opcode、无自修改   | 压缩编码、复杂 predication            |
| Pipeline | 静态 stage graph、block loop、stream wait/credit | 动态 stage 创建、priority scheduling  |
| DMA      | 1D/2D/strided Group DMA prefetch/storeback       | gather list、multicast DMA            |
| Branch   | loop counter、shape variant branch 的有限模式    | 通用分支预测或复杂异常恢复            |
| Sync     | wait/event/barrier/stream/EOS/error              | cross-group barrier、preemption       |
| PMU      | issue、wait、stall、timeout、fault attribution   | trace sampling 和 feedback scheduling |

所有 opcode 编码、寄存器数量、program cache 容量、最大 stage 数、最大 queue 数、timeout 默认值由后续规格冻结。

## 2. 职责、非职责和 ownership

Region Sequencer owns：

- Region PC、region register file、loop counter、stage scoreboard。
- Region Program fetch/decode/issue，以及 Program Residency Manager 协调：lookup、fetch、verify、install、wake waiting regions。
- Region descriptor validation 的硬件检查部分：version、range、count、tile_mask、event/queue id。
- Stream Queue 初始化、`wait.credit`、`wait.stream`、EOS/error policy issue。
- Group DMA descriptor issue、completion wait、DMA fault propagation。
- Tile Dispatcher issue：tile_mask、tile program id、prepared local program handle、descriptor patch window、stream binding。
- Barrier/Event wait/signal sequencing。
- Collective command issue 和 completion wait。
- PMU event generation：active、issue、wait reason、fault PC、timeout。

Region Sequencer 不负责：

- BOA/EVU/MFE/USE datapath 微循环或 tile-local engine launch；这些由 Tile UCE 和 engine sequencer 负责。
- L2 到 L1 的 Tile DMA；Region Sequencer 只准备 L2 buffer、descriptor 和 stream token。
- MFE 的 page/segment 数据相关动态地址生成；Region Sequencer 只 launch 或等待相关 stage/event。
- USE 的 state ownership、recurrence、checkpoint/restore。
- Runtime queue scheduling、context priority、global graph dependency。
- Cache coherency；Shared SRAM/L2 通过显式 descriptor 和 fence 管理。

Ownership 约束：

1. Region Sequencer 是 group 内 program control owner；Tile UCE 是 tile-local program control owner。
2. Region Sequencer 可写 group event/status 区，但不能直接写 tile-local L1 状态，除非通过 Tile Dispatcher/Tile UCE 协议。
3. Region Sequencer issue 的 Group DMA completion event 必须唯一映射到 descriptor sequence。
4. Region Sequencer 不应直接解释 Stream Queue payload 内容，只使用 token metadata。
5. fault_record_slot 由 launch command 指定，Region Sequencer 负责写入 region-level fault 摘要。

## 3. 微架构和状态机

### 3.1 内部模块

```text
Region Sequencer
├── Task Accept / Descriptor Validator
├── Program Residency Manager Interface
├── Region Program Fetch
├── Decode / Issue
├── Region Register File
├── Loop / Branch Unit
├── Stage Scoreboard
├── Stream Wait/Credit Unit
├── Event/Barrier Wait Unit
├── DMA Issue Queue
├── Tile Dispatch Queue
├── Collective Issue Queue
├── Fault / Timeout Unit
└── PMU Event Encoder
```

Region register file 保存 block_id、block_count、descriptor base、L2 buffer index、stage id、event id、queue id 等控制变量。寄存器数量和字段宽度由后续规格冻结。

### 3.2 Region task 状态机

```text
RESET
  -> IDLE
  -> ACCEPT
  -> VALIDATE
  -> PROGRAM_LOOKUP
  -> PROGRAM_FETCH_ON_MISS
  -> PROGRAM_VERIFY
  -> INIT
  -> FETCH
  -> DECODE
  -> ISSUE_OR_WAIT
  -> COMPLETE
  -> IDLE

fault/timeout from VALIDATE..ISSUE_OR_WAIT:
  -> FAULT_RECORD
  -> DRAIN
  -> SIGNAL_ERROR
  -> IDLE 或 RESET_REQUIRED
```

状态说明：

- `ACCEPT`：从 group launch queue 取 region task，锁存 context/region/group。
- `VALIDATE`：检查 descriptor size、range、queue/stage/event count、tile_mask、SRAM window。
- `PROGRAM_LOOKUP`：按 `context_id + program_id + program_version` 查询 Program Residency Manager。
- `PROGRAM_FETCH_ON_MISS`：warm launch 命中 cache 时跳过；cold launch 通过 Group DMA 拉取 region program，并把等待 region 挂到相同 key。
- `PROGRAM_VERIFY`：核对 hash/CRC/ABI/epoch；失败则写 fault record，禁止进入 READY。
- `INIT`：初始化 stream queue、barrier epoch、stage scoreboard、PMU epoch。
- `FETCH/DECODE`：从 resident local program slot / I-cache 取 opcode，decode operand 和 immediate。
- `ISSUE_OR_WAIT`：对 DMA、dispatch、collective、barrier、stream 产生 command；对 wait 指令进入等待。
- `COMPLETE`：所有 in-flight stage/DMA/collective 清空，signal region done。

### 3.3 Stage pipeline semantics

Region Program 负责启动和推进 stage graph。stage 是 tile group 内的一段 kernel pipeline，不是高层 op。数据以 block token 流动：

```text
for block in region.blocks:
  Stage0 产生 S0 token
  Stage1 消费 S0，产生 S1 token
  Stage2 消费 S1，storeback 或 collective merge
```

Region Sequencer 必须支持的 pipeline 行为：

- prefetch ahead：在 Stage0/Stage1 计算当前 block 时，Group DMA 预取 next block。
- block-level overlap：同一 region 内不同 block 可以位于不同 stage。
- stream backpressure：queue full 时停止 signal producer stage 或停止 prefetch 进入该 queue 对应 buffer。
- event order：同一 stage 的 block completion 按 sequence 更新 scoreboard；允许不同 stage overlap。
- EOS drain：Region loop 完成后向 producer queue 推送 EOS，等待 downstream stage drain。
- error short-circuit：任一 stage error token 或 tile error event 触发 region fault policy。

### 3.4 Wait 状态机

```text
WAIT_ENTER
  -> CHECK_READY
  -> PARK
  -> WAKEUP
  -> REVALIDATE
  -> RETIRE
```

wait source 包括 event、stream token、credit、barrier、DMA completion、collective completion。`REVALIDATE` 必须比较 epoch 和 sequence，避免 reset 后 stale wakeup。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Opcode 建议

| Opcode           | 作用                                       | 必要 operand                                          |
| ---------------- | ------------------------------------------ | ----------------------------------------------------- |
| `region.begin`   | 标记 region 开始并绑定 resource desc       | region_id、resource_desc                              |
| `init.stream`    | 初始化 stream queue                        | queue_id、depth、producer_mask、consumer_mask、policy |
| `dma.prefetch`   | issue Group DMA HBM 到 L2                  | dma_desc_id、dst_l2、event_id                         |
| `dma.store`      | issue Group DMA L2 到 HBM                  | dma_desc_id、src_l2、event_id                         |
| `dispatch.stage` | 向 Tile Dispatcher 派发 prepared tile task | stage_id、tile_mask、program_id、stream bindings      |
| `wait.event`     | 等待 event done/error/timeout              | event_id、timeout                                     |
| `wait.stream`    | 等待 queue non-empty 或 stream condition   | queue_id、condition                                   |
| `wait.credit`    | 等待 queue credit                          | queue_id、min_credit                                  |
| `barrier.group`  | group participant barrier                  | participant_mask、event_id                            |
| `collective.run` | issue group collective                     | collective_desc_id、event_id                          |
| `push.eos`       | 向 stream queue 注入 EOS token             | queue_id、producer_id                                 |
| `advance.block`  | 更新 block counter/window                  | block_reg、stride                                     |
| `branch.lt`      | 有界 loop branch                           | lhs、rhs、target_pc                                   |
| `signal.event`   | signal region/tile stage event             | event_id、status                                      |
| `region.end`     | drain 并完成 region                        | signal_event                                          |

编码、立即数宽度、operand 格式由后续规格冻结。

### 4.2 Region Program 示例

```asm
region.begin   region=42, resources=desc_region42
    init.stream   s0, depth=3, producer=stage0, consumer=stage1, policy=all_eos
    init.stream   s1, depth=2, producer=stage1, consumer=stage2, policy=single_producer

    dma.prefetch  desc=input_block0, dst=l2_buf0 -> ev_dma0
    dma.prefetch  desc=input_block1, dst=l2_buf1 -> ev_dma1
    wait.event    ev_dma0

    dispatch.stage stage=0, tile_mask=0x0f, program=tile_kernel_load_qk, out=s0
    dispatch.stage stage=1, tile_mask=0xf0, program=tile_kernel_softmax_av, in=s0, out=s1
    dispatch.stage stage=2, tile_mask=0xff, program=tile_kernel_store, in=s1

loop_blocks:
    wait.credit   s0, min=1
    dma.prefetch  desc=next_block, dst=l2_next -> ev_dma_next
    signal.event  ev_stage0_start, status=ready
    wait.stream   s0, condition=has_token
    signal.event  ev_stage1_start, status=ready
    advance.block block_id, stride=1
    branch.lt     block_id, block_count, loop_blocks

    push.eos      s0, producer=stage0
    push.eos      s1, producer=stage1
    barrier.group participants=0xff -> ev_bar_done
    wait.event    ev_bar_done
region.end signal=ev_region_done
```

### 4.3 Descriptor structures

```c
typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t flags;
    uint16_t priority;

    uint32_t context_id;
    uint32_t region_id;
    uint32_t group_id;
    uint32_t tile_mask;

    uint32_t template_id;
    uint32_t program_id;
    uint32_t program_version;
    uint32_t program_crc_or_hash;

    uint64_t program_iova;
    uint32_t program_bytes;
    uint32_t program_section_id;

    uint64_t region_desc_iova;
    uint32_t region_desc_bytes;
    uint64_t engine_desc_iova;
    uint32_t engine_desc_bytes;

    uint32_t wait_event_base;
    uint16_t wait_event_count;
    uint16_t signal_event;

    uint16_t residency_hint;
    uint16_t cache_policy;
    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_region_launch_desc_v1_t;
```

```c
typedef struct {
    uint32_t stage_id;
    uint32_t tile_mask;
    uint32_t tile_template_id;
    uint32_t tile_program_id;
    uint32_t tile_program_version;
    uint32_t flags;
    uint32_t in_stream_mask;
    uint32_t out_stream_mask;
    uint32_t wait_event_base;
    uint32_t signal_event;
} elenor_region_stage_desc_t;
```

Protocol：

- Descriptor validation failure must not partially initialize queues or dispatch tiles。
- `tile_mask` 必须是当前 group 可用 tile 的子集。
- `program_iova` 只作为 miss fetch backing store；运行态 fetch 只允许来自 resident local slot / I-cache。
- `residency_hint/cache_policy` 是 hint，不是 correctness 语义；正确性由 `program_id + version + hash + epoch` 保证。
- `timeout_cycles == 0` 的语义由后续规格冻结，RTL 不应假设无限等待。

### 4.4 CSR 建议

| CSR                    | 描述                                            |
| ---------------------- | ----------------------------------------------- |
| `RS_CONTROL`           | enable、soft_reset、drain、single_step          |
| `RS_STATUS`            | idle/running/waiting/draining/error             |
| `RS_PC`                | 当前 PC 或 fault PC                             |
| `RS_WAIT_REASON`       | event、stream、credit、barrier、DMA、collective |
| `RS_CONTEXT_REGION`    | context_id、region_id、stage_id                 |
| `RS_TIMEOUT_REMAINING` | 当前 wait timeout 计数                          |
| `RS_FAULT_CODE`        | 最近 fault code                                 |
| `RS_PMU_EVENT`         | debug view of PMU event encoder                 |

## 5. 数据流、控制流和时序路径

### 5.1 Control flow

```text
Runtime command buffer
  -> Group launch queue
  -> Region Sequencer ACCEPT/VALIDATE
  -> PROGRAM_LOOKUP / FETCH_ON_MISS / VERIFY
  -> Region Program FETCH/DECODE
  -> issue Group DMA / dispatch prepared tile task / wait stream / wait event / collective
  -> Tile UCE and group engines produce events
  -> Region Sequencer retires wait
  -> region.end signals runtime event
```

Region Sequencer 只在 group control plane 上 issue，不在 data path 上搬 payload。program backing store 解析和 install 由 Program Residency Manager + Group DMA 完成；payload movement 由 Group DMA、Tile DMA、MFE、Collective 完成。

### 5.2 Data flow relationship

- HBM -> L2：Region Sequencer issue Group DMA prefetch/storeback descriptor。
- L2 -> L1：Region Sequencer dispatch stage，Tile UCE 根据 tile program 和 descriptor issue Tile DMA。
- Stage -> Stage：Region Sequencer 初始化 Stream Queue 并等待 token/credit；payload 地址通常指向 L2 或 tile-local slot，由 descriptor 指明。
- Tile -> Tile：Region Sequencer issue Collective command 或 barrier，不直接读取 partial data。
- State update：USE 管理 state；Region Sequencer 只等待 USE/tile event 或调度包含 USE op 的 tile program。

### 5.3 Timing paths

关键 RTL paths：

- PC + branch compare + next PC mux。
- Decode -> issue valid -> target ready backpressure。
- Event scoreboard CAM/compare -> wait wakeup。
- Stream credit/occupancy compare -> wait wakeup。
- Stage scoreboard all-done reduction。
- Timeout counter compare -> fault transition。

建议将 event/stream/barrier wakeup registered，避免 Region Sequencer issue path 和 queue occupancy path 形成长组合环。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 SRAM/L2 assumptions

Region Sequencer 使用的 SRAM 区域：

| 区域              | 用途                              | 备注                             |
| ----------------- | --------------------------------- | -------------------------------- |
| program cache     | resident Region Program           | 容量由 SRAM profile 冻结         |
| descriptor window | region/stage/queue/dma desc cache | 需 range/CRC validation          |
| event/status      | event scoreboard、stage status    | 避免与 BOA operand hot bank 冲突 |
| stream metadata   | queue head/tail/credit、policy    | payload buffer 可在独立 L2 bank  |
| fault/PMU epoch   | fault record、counter snapshot    | runtime 可读                     |

Region Sequencer fetch 带宽较小，但 wait wakeup 和 CSR/PMU 写入不能阻塞 BOA/EVU/MFE hot path。Program/descriptor/event region 不应与 BOA operand 或 MFE stream buffer 争用同一组 bank。

### 6.2 性能模型

Region Sequencer 本身不应成为 throughput bottleneck。issue rate 模型：

```text
Issue_bw_required = opcode_per_block * blocks_in_flight / T_block
Issue_bw_available = f_rs / cycles_per_issue
```

必须满足：

```text
Issue_bw_available >= Issue_bw_required
```

Pipeline latency：

```text
T_region_control =
  T_init
+ Σ wait_overhead(event/stream/credit/barrier)
+ T_issue_dma_dispatch_collective
+ T_drain
```

Region control overhead 应小于主要 BOA/EVU/MFE/Group DMA 计算和搬运时间；否则 PMU 应显示 `region_issue_stall` 或 `region_wait_*` 异常偏高。

### 6.3 PMU counters

必需 counter：

- `rs_active_cycles`。
- `rs_fetch_stall_cycles`。
- `rs_decode_issue_cycles`。
- `rs_issue_backpressure_cycles` by target：DMA、Tile Dispatcher、Queue、Barrier、Collective。
- `rs_wait_event_cycles`。
- `rs_wait_stream_cycles`。
- `rs_wait_credit_cycles`。
- `rs_wait_barrier_cycles`。
- `rs_wait_dma_cycles`。
- `rs_wait_collective_cycles`。
- `rs_timeout_count`。
- `rs_fault_count` by fault_code。
- `rs_branch_taken_count`、`rs_loop_iteration_count`。
- `rs_region_completed_count`。

Stall attribution：

- Region Sequencer 正在 issue 且 target not ready：owner 为 target backpressure。
- wait event/stream/credit/barrier：owner 为对应 wait reason。
- program fetch miss/cold load：owner 为 program/descriptor stall 或 DMA memory，不能同时计两次 primary stall。

## 7. RTL/软件实现建议

RTL：

- 使用显式 `accepted`/`retired` handshake 追踪每条有副作用指令；fault 时可判断是否需要补偿或 drain。
- opcode decoder 保持 small and boring；复杂行为放 descriptor，不在 decoder 中嵌入 workload-specific special case。
- Stage scoreboard 按 `stage_id` 和 `sequence_id` 组织，避免只用 bitmask 无法处理多 block in flight。
- Wait Unit 分离 source-specific ready 和 common timeout/epoch check。
- 所有 event write 带 producer_id、sequence、timestamp、status。
- Program cache miss 通过 Group DMA 路径完成，复用 DMA completion event 和 fault model。
- CSR single_step 仅用于 debug，不进入 production scheduling contract。

Software/compiler：

- compiler 输出 region program、stage descriptor、queue descriptor、DMA descriptor、collective descriptor，以及 `program_id/template_id/program_ref/cache hint`。
- runtime patch context-level IOVA、shape、descriptor base、event id 和 program backing store metadata，不在 hot path 生成 per-tile program 或显式 `program.load`。
- firmware validation 应在 launch 前检查 ABI version、size、range、producer/consumer mask、tile_mask、program version/hash、timeout policy。
- Region Program 尽量使用固定 kernel template 和 descriptor auto-patch，减少 residency miss 和 program cache 压力。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA/formal verification points

- PC safety：PC 始终落在 resident program bounds；branch target 必须对齐且在 bounds 内。
- Program readiness：`dispatch.stage` 只有在 residency manager 返回 READY、descriptor window ready 且 epoch 匹配时才允许产生 tile 副作用。
- Side-effect retirement：未 accepted 的 issue 不得产生 event/DMA/dispatch 副作用；accepted 指令必须最终 retire、fault 或被 reset epoch kill。
- Wait freshness：wait wakeup 的 event/stream/barrier epoch 必须等于当前 region epoch。
- Timeout liveness：wait 超过 timeout 后必须进入 fault path 并 signal ERROR/TIMEOUT。
- Stage dispatch safety：dispatch 的 tile_mask 必须是 available_tile_mask 的子集。
- DMA exactly-once：每个 Region Sequencer accepted DMA descriptor 必须产生 exactly one completion status，reset kill 除外且必须可观测。
- Deadlock bounded proof：在 queue/DMA/tile/collective fairness 假设下，Region Program loop 要么推进到 `region.end`，要么 timeout。

### 8.2 Bring-up

1. Empty program：`region.begin; region.end`。
2. Branch loop：固定 block_count，验证 loop_iteration_count。
3. DMA prefetch wait：HBM -> L2 completion event。
4. Stream credit wait：producer/consumer queue token loop。
5. Two-stage overlap：Stage1 在 Stage0 未完成全部 block 前启动。
6. Barrier：participant mask release 和 timeout case。
7. Collective issue/wait：reduce add completion。
8. Fault injection：bad descriptor、bad branch target、DMA timeout、queue error token。
9. Reset/drain：RUN 和 WAIT 状态下 soft reset，验证 event RESET、credit 回收、PC 停止。

### 8.3 验收标准

- Cold launch 和 warm launch 均可执行同一 Region Program，且不需要 compiler/runtime 发显式 `program.load`。
- Region Sequencer 能完整推进 `command -> Program Residency Manager -> Group DMA -> stream -> prepared tile dispatch -> event -> group done`。
- PMU counters 能区分 issue backpressure、wait event、wait stream、wait DMA、timeout。
- 所有 SVA/formal safety properties 在配置 depth 下通过。
- Fault path 不产生 stale done event，不泄漏 stream credit，不留下 orphan DMA descriptor。

## 9. 风险、取舍和后续细化方向

风险：

- Opcode 范围失控：Region Sequencer 若承担 workload-specific 算子语义，会变成难验证的小 CPU。
- Wait/event/stream epoch 不严谨：reset 后 stale event 可导致错误完成。
- Program cache 和 descriptor window 与 BOA/MFE hot bank 冲突。
- Stage scoreboard 对多 block in flight 支持不足，导致 pipeline 被迫串行化。
- Timeout policy 不明确，可能掩盖 deadlock 或误杀长 latency DMA。

取舍：

- 使用简单 Region Program + descriptor-driven engines，而不是硬件解释 graph。
- First Silicon V1 先固定有限 opcode 和静态 stage graph，优先验证 command/event/DMA/stream/PMU。
- Branch 支持 loop 和有限 shape variant，复杂 dynamic scheduling 留给 Runtime/Compiler。

后续需要冻结：

- Binary opcode encoding、register file、program alignment。
- Region descriptor ABI、stage descriptor ABI、CSR map。
- Timeout 默认值、zero timeout 语义和 fault code。
- Stage graph 最大规模、queue/stream policy、multi-consumer 行为。
- PMU counter 编号、overflow、read/clear 行为。
