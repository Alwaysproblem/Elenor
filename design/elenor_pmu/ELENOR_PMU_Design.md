# ELENOR PMU 设计文档

## 1. 定位、目标和 First Silicon cutline

ELENOR PMU 是 Architecture V1 中连接硬件执行、runtime 调度、compiler 性能模型和 bring-up 验收的观测面。硬件执行 command、descriptor、Region Program 和 Tile Program；PMU 不解释高层 graph，只记录这些执行对象在 Chip、Tile Group、Compute Tile 和 engine 层面的周期、带宽、队列、事件、错误与 stall 归因。

PMU 的目标不是堆积大量计数器，而是让每一次性能结论都能被唯一归因：是 BOA/EVU/MFE/USE 的有效工作不足，是 DMA/NoC/SRAM 阻塞，是 stream queue backpressure，是 event/barrier 等待，还是 UCE program/descriptor 路径问题。该归因同时服务四类闭环：

1. **bring-up 闭环**：command -> DMA -> engine -> event -> PMU 的路径必须先于复杂 engine 完成。
2. **roofline 闭环**：counter 必须能填入 compute、SRAM bandwidth、HBM bandwidth、stream 和 state 五类瓶颈模型。
3. **workload 指纹闭环**：dense GEMM、dense attention、paged attention、MoE、SSM/recurrence、多模型并发必须有可预期的 PMU 签名。
4. **runtime 调度闭环**：driver/firmware 读取 counter 后能解释 queue occupancy、QoS latency、fault isolation 和 SRAM quota 效果。

First Silicon V1 cutline：

| 范围     | First Silicon V1 必须实现                                                                 | Architecture V1 可预留                     |
| -------- | ----------------------------------------------------------------------------------------- | ------------------------------------------ |
| 计数粒度 | chip/group/tile/engine active、stall、event、DMA、queue、SRAM conflict、NoC VC congestion | 采样 trace、PC histogram、长窗口压缩 trace |
| 归因规则 | 每个 stall cycle 一个 primary owner，允许 secondary debug tag                             | 多级因果图和跨 context 自动诊断            |
| 访问接口 | `elenor_read_counter(counter_id, value)`、snapshot/freeze、context 过滤                   | 采样中断、用户态 mmap counter page         |
| workload | GEMM、softmax/norm、paged attention、MoE routing、USE recurrence 的核心签名               | PMU feedback scheduler 的自动优化策略      |
| 验收     | counter 与 RTL waveform、event timestamp、golden trace 能对齐                             | 由后续规格冻结                             |

## 2. 职责、非职责和 ownership

### 2.1 职责

- Global PMU 聚合 chip 级 command queue、NoC VC、global DMA、memory controller、fault 和 timestamp。
- Group PMU 聚合 Region Sequencer、Group DMA、Stream Queue、Collective、Shared SRAM/L2 和 group barrier。
- Tile PMU 聚合 Tile UCE、Tile DMA、L1 SRAM、BOA、EVU、MFE、USE、local event/barrier。
- Engine PMU 在 engine 本地生成 active/stall/bytes/ops/error 原始事件，并按统一 stall reason 编码上报。
- Firmware 负责 snapshot、overflow 处理、context 过滤、fault record 关联和 driver 可读视图。
- Compiler/runtime 负责把 command id、program id、descriptor id、context id 和 workload phase 标记到可观测边界，便于 PMU 与 executable package 对齐。

### 2.2 非职责

- PMU 不做高层 graph 解释，不根据模型语义推断瓶颈。
- PMU 不替代 SVA、formal、CDC/RDC、lint、STA 或 power signoff。
- PMU 不直接修改调度策略；PMU feedback scheduling 属于 runtime/firmware 策略层。
- PMU 不允许同一 stall cycle 在多个 primary counter 中重复计数。
- PMU 不为调试方便暴露不稳定的内部信号命名作为 ABI；外部 ABI 使用稳定 counter id 和版本号。

### 2.3 ownership 矩阵

| 对象                     | 产生者                    | 聚合者                | 软件 owner              | 验收 owner                   |
| ------------------------ | ------------------------- | --------------------- | ----------------------- | ---------------------------- |
| engine active/stall      | BOA/EVU/MFE/USE/DMA/UCE   | Tile PMU              | firmware/runtime        | RTL + verification           |
| SRAM bank conflict       | L1/L2 SRAM arbiter        | Tile/Group PMU        | compiler memory planner | RTL + physical               |
| stream credit empty/full | Stream Queue Engine       | Group/Tile PMU        | runtime scheduler       | verification                 |
| NoC VC congestion        | router / VC arbiter       | Global PMU            | firmware/runtime        | physical + verification      |
| event wait cycles        | Event/Barrier Unit        | Tile/Group/Global PMU | driver/firmware         | verification                 |
| context/QoS counter      | command queue / scheduler | Global PMU            | driver/runtime          | software + system validation |

## 3. 微架构和状态机

### 3.1 分层微架构

```text
Engine local event source
    |
    v
Local counter bank + stall owner encoder
    |
    v
Tile PMU aggregator
    |
    +--> Tile snapshot RAM
    +--> overflow / saturating status
    |
    v
Group PMU aggregator
    |
    v
Global PMU + firmware snapshot window
    |
    v
Driver/runtime readout
```

每级聚合只接收下级已经编码的计数事件，不重新解释 engine 内部状态。这样能保持 RTL 模块边界清楚，也避免 Global PMU 成为跨层组合逻辑中心。

### 3.2 counter 生命周期状态机

```text
RESET
  |
  v
IDLE
  |  arm(context_id, counter_mask, window)
  v
ARMED
  |  command/event boundary begin
  v
COUNTING
  |  freeze request / window end / fault
  v
FROZEN
  |  firmware snapshot done
  v
DRAINED
  |  clear or re-arm
  v
IDLE
```

- `RESET` 后 counter 值为 0，overflow 标志清零。
- `ARMED` 只配置 mask、context、scope，不计数。
- `COUNTING` 按有效周期计数，counter width 由后续规格冻结；First Silicon 推荐饱和加法并设置 overflow sticky bit。
- `FROZEN` 必须停止更新可读 bank，同时允许硬件在 shadow bank 中继续计数或暂停，具体由后续规格冻结。
- `DRAINED` 将 snapshot 与 fault/event record 对齐。

### 3.3 唯一 stall 归因层级

PMU 采用 primary stall owner 层级。每个 engine 每个周期最多产生一个 primary reason；secondary tag 只进入 debug 视图，不进入 utilization 统计。

```text
0. engine_active
1. engine_wait_event
2. engine_wait_operand
3. stream_credit_empty_or_full
4. sram_bank_conflict
5. noc_backpressure
6. dma_wait_memory
7. uce_program_or_descriptor_stall
8. unknown_or_unclassified
```

优先级解释：

| 层级 | primary owner          | 判定边界                                               | 示例                                   |
| ---- | ---------------------- | ------------------------------------------------------ | -------------------------------------- |
| 0    | active                 | engine 接受并退休有效 work item                        | BOA OPA 正在计算                       |
| 1    | event                  | engine 因 wait_event/barrier 未满足无法发射            | Tile UCE 等待 DMA done event           |
| 2    | operand                | engine 需要的数据未到且不是 stream credit 问题         | BOA A/B operand buffer empty           |
| 3    | stream                 | stream queue full/empty/credit 阻塞 producer/consumer  | MFE producer 因 output queue full 暂停 |
| 4    | SRAM                   | SRAM arbiter 明确拒绝或 replay                         | EVU indexed load bank replay           |
| 5    | NoC                    | router/VC backpressure 阻止 request/response 前进      | DMA read response VC congestion        |
| 6    | DMA/memory             | memory controller 或 DMA outstanding slot 限制         | Group DMA 等 HBM 返回                  |
| 7    | UCE program/descriptor | program fetch、descriptor patch、descriptor cache miss | Tile Program cold load                 |
| 8    | unknown                | 无法分类或非法组合                                     | 应触发 verification 覆盖缺口           |

### 3.4 snapshot 和一致性

- PMU snapshot 必须绑定 `context_id`、`queue_id`、`command_id`、`event_id`、`timestamp`。
- command boundary snapshot 用于单 command profiling；region boundary snapshot 用于 pipeline stage profiling；global periodic snapshot 用于 QoS 和 thermal/power 分析。
- 多级 counter 采用 freeze-then-read 或 double-buffer；不允许 firmware 读取到一半新一半旧的聚合值。
- event timestamp 与 PMU cycle counter 必须同源或有明确转换系数，频率比由后续规格冻结。

## 4. 接口、descriptor、寄存器和协议

### 4.1 counter id 编码

建议 counter id 使用稳定的层级编码：

```text
[31:28] scope      0=global,1=group,2=tile,3=engine
[27:24] block      runtime,dma,noc,sram,stream,boa,evu,mfe,use,uce,event,power
[23:16] instance   group_id/tile_id/engine_id 或 broadcast selector
[15:0]  counter    block-local counter number
```

具体 bit 宽由后续规格冻结；外部软件只依赖枚举和 capability table，不直接假定实例数量。

### 4.2 counter map

| counter id 名称            | scope             | primary owner | 含义                                  | 验证方法                |
| -------------------------- | ----------------- | ------------- | ------------------------------------- | ----------------------- |
| `PMU_GLOBAL_CYCLES`        | Global            | Global PMU    | 全局参考周期                          | timestamp 对齐          |
| `PMU_GLOBAL_CMD_ISSUED`    | Global            | command queue | firmware 接受的 command 数            | command ABI test        |
| `PMU_GLOBAL_CMD_DONE`      | Global            | event table   | done event 数                         | event scoreboard        |
| `PMU_GLOBAL_FAULT_COUNT`   | Global            | fault unit    | fault record 写入数                   | fault injection         |
| `PMU_NOC_VC0_CONGEST`      | Global/Group      | NoC           | command/event VC 阻塞周期             | router random test      |
| `PMU_NOC_VC1_CONGEST`      | Global/Group      | NoC           | DMA read response VC 阻塞周期         | DMA stress              |
| `PMU_NOC_VC2_CONGEST`      | Global/Group      | NoC           | DMA write/stream VC 阻塞周期          | stream stress           |
| `PMU_NOC_VC3_CONGEST`      | Global/Group      | NoC           | collective VC 阻塞周期                | collective test         |
| `PMU_DMA_BYTES_READ`       | Global/Group/Tile | DMA           | DMA 读字节                            | descriptor compare      |
| `PMU_DMA_BYTES_WRITE`      | Global/Group/Tile | DMA           | DMA 写字节                            | descriptor compare      |
| `PMU_DMA_STALL_MEMORY`     | Global/Group/Tile | DMA           | memory 返回受限周期                   | memory model            |
| `PMU_SRAM_BANK_CONFLICT`   | Group/Tile        | SRAM          | bank conflict 或 replay 次数          | SRAM arbiter SVA        |
| `PMU_STREAM_OCCUPANCY_ACC` | Group/Tile        | Stream Queue  | occupancy 累计值                      | queue model             |
| `PMU_STREAM_CREDIT_EMPTY`  | Group/Tile        | Stream Queue  | consumer 等空队列周期                 | constrained random      |
| `PMU_STREAM_CREDIT_FULL`   | Group/Tile        | Stream Queue  | producer 等满队列周期                 | constrained random      |
| `PMU_EVENT_WAIT_CYCLES`    | Global/Group/Tile | Event Unit    | wait_event/barrier 等待周期           | event dependency test   |
| `PMU_UCE_PATCH_STALL`      | Tile              | Tile UCE      | descriptor patch stall                | patch fault/path test   |
| `PMU_UCE_FETCH_STALL`      | Tile              | Tile UCE      | Tile Program fetch/cache stall        | cold/warm launch test   |
| `PMU_BOA_ACTIVE`           | Engine            | BOA           | BOA 有效计算周期                      | GEMM waveform           |
| `PMU_BOA_OPERAND_STALL`    | Engine            | BOA           | A/B operand 不足                      | double-buffer stress    |
| `PMU_BOA_ACC_STALL`        | Engine            | BOA           | accumulator RMW 阻塞                  | SRAM conflict test      |
| `PMU_BOA_WRITEBACK_STALL`  | Engine            | BOA           | output/writeback 阻塞                 | DMA/storeback test      |
| `PMU_EVU_ACTIVE`           | Engine            | EVU           | EVU 有效执行周期                      | softmax/norm test       |
| `PMU_EVU_LSU_REPLAY`       | Engine            | EVU           | gather/scatter 或 strided load replay | bank replay test        |
| `PMU_EVU_MASKED_LANE`      | Engine            | EVU           | mask/tail 无效 lane 累计              | tail corner test        |
| `PMU_MFE_ACTIVE`           | Engine            | MFE           | MFE page/segment 有效工作周期         | MFE trace test          |
| `PMU_MFE_PREFETCH_HIT`     | Engine            | MFE           | prefetch 命中次数                     | paged attention trace   |
| `PMU_MFE_PREFETCH_MISS`    | Engine            | MFE           | prefetch 未命中次数                   | invalid/reorder case    |
| `PMU_MFE_STREAM_STALL`     | Engine            | MFE           | stream fill/backpressure stall        | stream queue test       |
| `PMU_USE_ACTIVE`           | Engine            | USE           | scan/recurrence/state 有效周期        | recurrence golden       |
| `PMU_USE_STATE_HIT`        | Engine            | USE           | state cache 命中                      | state trace             |
| `PMU_USE_STATE_MISS`       | Engine            | USE           | state cache 未命中                    | checkpoint/restore test |
| `PMU_POWER_CG_ELIGIBLE`    | Global/Group/Tile | power monitor | 可 clock-gate 周期估计                | power sim               |
| `PMU_POWER_SRAM_SLEEP`     | Group/Tile        | SRAM power    | SRAM sleep/retention 周期             | power intent test       |

### 4.3 访问协议

```c
typedef struct {
    uint16_t abi_version;
    uint16_t record_size;
    uint32_t context_id;
    uint32_t queue_id;
    uint32_t command_id;
    uint32_t event_id;
    uint64_t begin_timestamp;
    uint64_t end_timestamp;
    uint32_t counter_count;
    uint32_t flags;
} elenor_pmu_snapshot_header_t;
```

字段布局是 PMU 设计建议，最终二进制 ABI 由后续规格冻结。协议要求：

- readout 不能破坏正在运行的 command；需要 freeze/snapshot 语义。
- counter overflow 必须 sticky，并能定位到 snapshot window。
- context isolation 下，非授权 context 不能读取其他 context 的 PMU 记录。
- reset tile/group/device 后，对应 scope counter 进入确定状态并生成可诊断 reset record。

## 5. 数据流、控制流和时序路径

### 5.1 数据路径

PMU 数据路径为低速控制路径，不在 BOA operand、EVU LSU、MFE stream、DMA data path 上插入长组合逻辑。engine 本地只产生短脉冲或小枚举，经寄存后送入 local counter bank。

```text
engine_valid / stall_reason / bytes / ops
    -> local registered event bus
    -> local counter adders
    -> per-scope aggregator FIFO or shadow RAM
    -> firmware snapshot
```

### 5.2 控制路径

- runtime 在 command begin/end 或 region begin/end 发出 PMU marker。
- Tile UCE 在 Tile Program 内支持 `prof.begin`/`prof.end` 类边界，但 PMU 不要求解释 Tile Program 语义。
- fault path 必须在 freeze 前记录 fault slot，使 snapshot 能指向 invalid descriptor、address fault、DMA timeout、event deadlock timeout 或 engine internal fault。

### 5.3 时序路径和物理风险

PMU 容易引入三类时序风险：

| 风险                 | 触发点                                       | 约束                                                    |
| -------------------- | -------------------------------------------- | ------------------------------------------------------- |
| 高扇出 enable/freeze | global snapshot、counter clear、context mask | 必须分层复制寄存器，不从 Global PMU 直接扇出到所有 tile |
| 宽加法器路径         | bytes/ops counter 多路累加                   | local counter bank 分段加法或低频 shadow 聚合           |
| 跨层聚合路径         | tile -> group -> global                      | 使用 registered bus/FIFO，不跨 NoC router 做组合归约    |
| SRAM/NoC 观测侵入    | SRAM arbiter、NoC VC 上增加 debug mux        | counter tap 只能采样已寄存状态，不改变仲裁临界路径      |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 roofline 公式

ELENOR PMU 支持五维 roofline：

```text
Perf = min(BOA_compute, EVU_compute, MFE_stream, USE_state, Memory_bw)
BOA_perf = min(BOA_peak, SRAM_bw * AI_sram, HBM_bw * AI_hbm)
BW_eff = BW_peak * (1 - conflict_rate)
```

PMU 需要给出填入这些公式的观测值：effective ops、active cycles、bytes read/write、SRAM conflict、NoC congestion、stream stall、state cache hit/miss。

### 6.2 per-workload PMU 签名

| workload        | 期望主路径                           | PMU 签名                                                                               | 异常解释                                                                |
| --------------- | ------------------------------------ | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Dense GEMM      | DMA -> BOA -> DMA                    | `BOA_ACTIVE` 高，`BOA_OPERAND_STALL` 低，SRAM conflict 可控                            | operand stall 高表示 double buffer 或 SRAM bank placement 问题          |
| Dense Attention | BOA QK/AV + EVU softmax              | BOA 与 EVU active 分阶段上升，collective stall 与 split-K/score merge 对齐             | EVU stall 高表示 softmax/norm buffer 或 mask/tail 实现问题              |
| Paged Attention | MFE Page Stream + BOA + EVU          | `MFE_PREFETCH_HIT` 增加，`T_prefetch <= T_qk` case 下 BOA operand stall 下降           | prefetch miss 与 BOA stall 同升表示 page depth/stream queue 不足        |
| MoE Dispatch    | MFE Segment + EVU + BOA              | routing imbalance counter、stream full/empty、BOA utilization 随 expert imbalance 变化 | BOA active 低但 MFE stall 高表示 segment gather 或 expert batching 问题 |
| SSM/Mamba/RWKV  | USE scan/recurrence + BOA projection | `USE_ACTIVE`、state hit/miss、event wait 可解释 recurrence phase                       | state miss 高表示 state cache profile 或 checkpoint policy 问题         |
| 多模型并发      | context partition + priority queue   | per-context queue occupancy、QoS latency、SRAM quota 命中、fault isolation             | 一个 context fault 后其他 context counter 不应污染                      |

### 6.3 PPA 成本控制

- Counter bank 面积随实例数增长；First Silicon V1 优先实现可解释瓶颈的核心 counter，采样 trace 预留接口。
- 宽 counter、snapshot RAM、trace FIFO 的 SRAM macro 由 SRAM profile 冻结。
- Counter toggle 会增加动态功耗；非使能 scope 必须 clock-gate，power monitor 记录 clock-gate eligible 周期。
- PMU event bus 不应跨大物理距离高频切换；跨 group 聚合采用低速 snapshot 网络。

### 6.4 PMU 与 verification/bring-up 的关系

PMU 自身也是验证对象。每个 phase exit 不只要求功能正确，还要求 counter 能解释功能路径：Phase 1 GEMM 要看到 BOA active/stall；Phase 3 paged attention 要看到 MFE prefetch 与 BOA stall 对齐；Phase 6 roofline validation 要能用 PMU 解释主要瓶颈。

## 7. RTL/软件实现建议

### 7.1 RTL 建议

- 每个 engine 定义统一 local PMU event record：`active`、`stall_valid`、`stall_reason`、`bytes`、`ops`、`event_id`。
- Stall reason encoder 位于 engine 边界，禁止上级 PMU 读取 engine 内部状态机细节。
- Counter 使用 saturating add + overflow sticky，避免 silent wrap。
- Snapshot bank 和 live bank 分离，或采用 freeze handshake。
- CDC/RDC：PMU 跨 clock/reset domain 的 event pulse 必须经过同步、toggle 或 async FIFO；reset 后 sticky bit、snapshot valid 和 counter enable 必须状态确定。
- SVA：唯一归因、counter freeze 不变、overflow sticky、snapshot atomic、event_id 对齐必须有断言。
- formal：小规模验证 FIFO/arbiter/event dependency/stream credit/EOS/error propagation 与 PMU counter 一致性。

### 7.2 firmware/driver/runtime 建议

- Firmware 提供 capability table，列出支持 counter、scope、width、overflow、filter 能力。
- Driver 暴露只读 PMU ioctl 或 runtime API，不允许用户绕过 context isolation。
- Runtime 在 command buffer 中插入 profiling marker，并在 fault path 自动抓取 snapshot。
- Compiler 生成 canonical trace 时声明预期 PMU signature，使 performance validation 不依赖人工猜测。

### 7.3 EDA、开源和工业工具流

| 阶段      | 开源/早期工具         | 工业工具                         | PMU 关注点                               |
| --------- | --------------------- | -------------------------------- | ---------------------------------------- |
| RTL 仿真  | Verilator、cocotb     | VCS、Xcelium、Questa             | counter 与 waveform/golden trace 对齐    |
| lint      | Verible               | SpyGlass、AscentLint             | 未使用 counter、宽扇出 enable、组合环    |
| formal    | SymbiYosys 可用于小块 | JasperGold、VC Formal            | freeze、overflow、FIFO、arbiter、credit  |
| CDC/RDC   | 由后续规格冻结        | Questa CDC、SpyGlass CDC/RDC     | PMU event crossing、reset sticky state   |
| synthesis | Yosys                 | Design Compiler、Fusion Compiler | counter bank 面积、fanout、clock gating  |
| STA       | OpenSTA               | PrimeTime                        | snapshot enable、counter adder、NoC tap  |
| physical  | OpenROAD              | Innovus、Fusion Compiler         | aggregator placement、routing congestion |
| power     | 由后续规格冻结        | PrimePower、Voltus、PowerArtist  | PMU toggle、clock gating、SRAM sleep     |

## 8. 验证、bring-up 和验收标准

### 8.1 验证层级

| 层级                   | PMU 验证内容                                                                 |
| ---------------------- | ---------------------------------------------------------------------------- |
| Python model           | canonical workload 生成期望 ops/bytes/stall signature                        |
| MLIR/compiler          | descriptor lowering 携带 profiling marker，counter id 与 workload phase 对齐 |
| Runtime ABI            | `elenor_read_counter`、snapshot、context isolation、fault snapshot           |
| RTL unit               | engine local counter、stall encoder、overflow、freeze、SVA                   |
| Tile integration       | UCE + SRAM + DMA + BOA/EVU/MFE/USE counter 对齐                              |
| Group integration      | stream queue、collective、Group DMA、NoC VC counter                          |
| System integration     | driver/firmware/runtime 端到端 readout                                       |
| Performance validation | roofline、per-workload signature、QoS latency/throughput                     |

### 8.2 phase exit criteria

| Phase   | PMU 必须通过                                                                            |
| ------- | --------------------------------------------------------------------------------------- |
| Phase 0 | counter map、stall attribution、snapshot ABI 草案、canonical trace 格式冻结到可评审状态 |
| Phase 1 | command queue 触发 BOA GEMM，event completion 与 BOA active/stall counter 对齐          |
| Phase 2 | EVU softmax/norm/tail 的 active、LSU replay、masked lane counter 可解释误差和 tail 行为 |
| Phase 3 | MFE prefetch hit/miss、stream stall、BOA operand stall 能证明 paged attention overlap   |
| Phase 4 | MoE routing imbalance 与 segment gather/reduce counter、BOA utilization 相关            |
| Phase 5 | USE active、state hit/miss、checkpoint/restore counter 与 recurrence golden 对齐        |
| Phase 6 | multi-context counter isolation、QoS latency/throughput、roofline validation 完成       |

### 8.3 静态和动态检查

- lint：counter width、case default、unused event、unreachable stall reason、implicit latch。
- CDC：engine clock 到 PMU clock、NoC/router clock 到 Global PMU、firmware bus clock crossing。
- RDC：tile/group/device reset 对 counter、sticky、snapshot valid、overflow 的影响。
- SVA：每周期最多一个 primary stall owner；active 与 stall 互斥；freeze 后 snapshot bank 不变。
- formal：小深度证明 stream credit empty/full counter 不丢不重；event wait counter 与 event dependency 一致。
- STA：high fanout freeze/clear/context mask、counter adder、snapshot mux、SRAM/NoC tap。
- power：PMU disabled 时无无意义 toggle；counter bank clock gating；snapshot RAM sleep/retention。

## 9. 风险、取舍和后续细化方向

| 风险                         | 影响                                       | 缓解                                                     |
| ---------------------------- | ------------------------------------------ | -------------------------------------------------------- |
| 归因重复                     | utilization 相加超过实际周期，性能结论失真 | primary owner 层级 + SVA + waveform 对齐                 |
| counter 过多                 | 面积、功耗、验证面膨胀                     | First Silicon V1 只实现核心 counter，trace 预留          |
| 高扇出控制                   | freeze/clear/context mask 破坏时序         | 分层寄存复制、局部 counter bank、低速 snapshot           |
| SRAM/NoC tap 侵入 datapath   | 改变被观测对象的时序和行为                 | 只采样已寄存仲裁状态，不加入临界仲裁组合逻辑             |
| context 泄漏                 | 多租户 profiling 泄露其他模型活动          | context filter、privilege check、fault/reset domain 隔离 |
| PMU 与 software ABI 过早固化 | 后续 counter 扩展困难                      | capability table、版本字段、保留 scope/block 编码        |
| roofline 指标无法闭合        | 性能问题只能猜测                           | 每个 canonical workload 明确 ops/bytes/stall signature   |

需要由后续规格冻结的项目：counter width、snapshot record 二进制布局、采样 trace 格式、PMU clock/domain 划分、power counter 精度、每档配置的 counter bank SRAM 容量。需要由 SRAM profile 冻结的项目：snapshot RAM 容量、counter bank macro、L1/L2 bank conflict event 定义。需要由 PPA exploration 冻结的项目：全局 PMU 聚合网络、trace FIFO 深度、PMU clock gating 粒度和物理 placement 策略。
