# ELENOR Verification Bring-up 设计文档

## 1. 定位、目标和 First Silicon cutline

ELENOR verification/bring-up 的定位是把 Architecture V1 的职责边界、First Silicon V1 cutline、binary ABI、descriptor contract、Tile Slot Frame、Stream Queue、PMU 归因和 workload trace 转化为可执行的验证计划。硬件执行 command、descriptor、TileGroupTask 和 Tile Program；验证环境也必须围绕这些对象构造，不把高层 graph 直接送入硬件 testbench。

核心目标：

1. 先验证系统基础面：command queue、event/barrier、DMA、reset/fault、PMU。
2. 再逐步扩展 BOA、EVU、MFE、USE 和多模型调度，避免每个模块都只完成局部 smoke test。
3. 每个 phase 有明确 exit criteria：功能、错误行为、PMU 证据和 PPA/static check 必须共同达标。
4. golden model、RTL unit、tile integration、group integration、system integration 和 performance validation 使用同一 canonical trace 和 descriptor ABI。

First Silicon V1 cutline：

| 项目     | 必须闭环                                                                                                         | 预留                                                              |
| -------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 控制面   | command queue、event、barrier、timeout、fault record、reset domain                                               | priority/preemption 的完整策略                                    |
| 数据搬运 | DMA 1D/2D/strided、async completion event、L2/L1 基本路径                                                        | multicast/gather list 的复杂模式                                  |
| Engine   | BOA GEMM、EVU elementwise/softmax/norm/tail、MFE Page Stream、MFE Segment Stream、USE scan/recurrence/checkpoint | Sparse Block、Persistent Memory Stream、高级 recurrence transform |
| ABI      | command/event v0、descriptor validation、Tile Slot Frame、Stream Queue token/credit/EOS/error                    | 完整二进制兼容矩阵由后续规格冻结                                  |
| PMU      | active/stall、DMA bandwidth、SRAM conflict、queue/event wait、NoC VC congestion                                  | sampled trace、PMU feedback scheduler                             |
| 静态质量 | lint、CDC/RDC、SVA、formal、STA、power intent 的 phase gate                                                      | signoff 阈值由 PPA exploration 冻结                               |

## 2. 职责、非职责和 ownership

### 2.1 验证职责

- 定义 canonical workload trace：dense GEMM、dense attention、paged attention、MoE routing、SSM/recurrence、多模型并发。
- 建立 Python golden model，用于 tensor result、descriptor side effect、event ordering、PMU signature 的参考。
- 建立 compiler/runtime ABI regression：descriptor binary、command/event/fault/reset、profile marker。
- 建立 RTL unit 与 integration test：engine standalone、tile、group、system。
- 建立 static signoff gate：lint、CDC、RDC、SVA、formal、STA、power check。
- 建立 bring-up sequencing：先 command/event/DMA/PMU，再 engine，再 workload，再 QoS/performance。

### 2.2 非职责

- 验证计划不改变 ELENOR 架构职责：UCE 负责 program control，USE 负责 state，MFE 负责 data-related dynamic memory access。
- testbench 不绕过 command queue 直接拉 datapath 作为 phase exit 证据；可用于早期 unit debug，但不能作为系统验收。
- golden model 不替代 RTL protocol assertion；二者关注不同错误面。
- PMU counter 不替代 waveform/debug，也不替代 SVA/formal 的协议证明。

### 2.3 ownership 矩阵

| 验证对象                 | 设计 owner       | 验证 owner                     | 软件/工具 owner  | 必须产物                                 |
| ------------------------ | ---------------- | ------------------------------ | ---------------- | ---------------------------------------- |
| command/event ABI        | runtime/firmware | system verification            | driver/runtime   | ABI regression、fault injection          |
| Tile UCE program control | tile control RTL | tile integration               | compiler/runtime | Tile Program trace、event ordering       |
| USE state lifecycle      | USE RTL          | engine verification            | model/compiler   | recurrence golden、checkpoint/reset case |
| MFE Page/Segment Stream  | MFE RTL          | engine + workload verification | compiler/runtime | page/segment descriptor trace            |
| Stream Queue             | group RTL        | formal + group integration     | runtime          | token/credit/EOS/error/reset tests       |
| PMU                      | PMU RTL/firmware | performance validation         | driver/runtime   | counter map、signature tests             |
| Physical/static          | RTL/physical     | signoff verification           | EDA              | lint/CDC/RDC/STA/power reports           |

## 3. 微架构和状态机

### 3.1 验证环境分层

```text
Canonical workload trace
    |
    +--> Python golden model
    +--> compiler descriptor/command generator
    +--> RTL testbench stimulus
    +--> PMU expected signature

RTL unit tests
    |
Tile integration
    |
Group integration
    |
System integration with firmware/driver/runtime
    |
Performance validation and bring-up
```

每一层都使用相同的 command、descriptor、event 和 slot frame 概念。这样可以避免 unit test 使用私有接口，而系统 test 又暴露新的 ABI 错误。

### 3.2 bring-up 状态机

```text
SPEC_LOCK
  -> CONTROL_PLANE_ALIVE
  -> DMA_EVENT_ALIVE
  -> BOA_COMMAND_ALIVE
  -> PMU_BASELINE_ALIVE
  -> EVU_ALIVE
  -> MFE_PAGE_ALIVE
  -> PAGED_ATTENTION_ALIVE
  -> MFE_SEGMENT_ALIVE
  -> USE_STATE_ALIVE
  -> MULTI_CONTEXT_ALIVE
  -> PERFORMANCE_CLOSURE
```

- `CONTROL_PLANE_ALIVE`：command queue、event table、timeout、fault record 可观测。
- `DMA_EVENT_ALIVE`：1D/2D DMA 通过 descriptor 发起并产生 completion event。
- `BOA_COMMAND_ALIVE`：BOA GEMM 必须通过 command queue 触发，而不是 testbench 直接驱动 datapath。
- `PMU_BASELINE_ALIVE`：basic counter 与 event timestamp、waveform 对齐。
- 后续状态依次打开 EVU、MFE、USE 和 multi-context，不允许跳过基础闭环。

### 3.3 关键协议状态机

| 协议            | 必测状态                                      | 关键异常                                      |
| --------------- | --------------------------------------------- | --------------------------------------------- |
| command queue   | empty、ready、consume、complete、fault、reset | invalid descriptor、timeout、queue drain      |
| event/barrier   | pending、done、error、timeout、reset          | deadlock timeout、producer id mismatch        |
| DMA             | idle、issue、outstanding、complete、fault     | address fault、stride 越界、reset mid-flight  |
| Stream Queue    | empty、non-empty、full、EOS、error、drain     | credit leak、error token 丢失、循环等待       |
| Tile Slot Frame | bind、access、reuse、release、fault           | slot 权限错误、alignment 错误、bank conflict  |
| PMU snapshot    | arm、counting、freeze、read、clear            | overflow、cross-domain snapshot、context leak |

## 4. 接口、descriptor、寄存器和协议

### 4.1 command/event ABI 验证

command layout 必须验证以下字段：ABI version、command size、context id、queue id、descriptor IOVA、descriptor bytes、descriptor checksum/validation mode、wait/signal event、timeout cycles、fault record slot、privilege/isolation flags。

事件模型必须覆盖：engine completion、DMA completion、role synchronization、tile done、group done、graph done、fault、timeout、reset。每个 event record 必须能关联 producer、sequence、error code 和 timestamp。

### 4.2 descriptor 验证矩阵

| descriptor   | 正常 case                                                     | 异常 case                                     | PMU/事件证据                              |
| ------------ | ------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------- |
| DMA          | 1D、2D、strided、async event                                  | address fault、stride overflow、timeout       | DMA bytes、DMA stall、completion event    |
| BOA          | INT8/BF16 GEMM、double buffer                                 | operand underflow、accumulator conflict       | BOA active/stall、SRAM conflict           |
| EVU          | elementwise、softmax、norm、mask/tail、basic gather           | invalid mask、tail boundary、bank replay      | EVU active、LSU replay、masked lane       |
| MFE Page     | page walk、KV prefetch、reorder、stream fill                  | invalid page、timeout、EOS/error token        | hit/miss、stream stall、BOA operand stall |
| MFE Segment  | offsets decode、segment gather、local reduce、expert batching | duplicate index mode、segment 越界            | routing imbalance、MFE stall              |
| USE          | scan、affine recurrence、checkpoint/restore                   | state slot 权限、reset/fault restore          | USE active、state hit/miss、event wait    |
| Stream Queue | token、credit、backpressure、EOS、error                       | credit leak、full/empty deadlock、reset/drain | occupancy、credit empty/full              |

### 4.3 PMU attribution hierarchy 和 counter map

验证环境必须把 PMU 作为可验证接口，而不是事后调试附件。Stall primary owner 层级：

```text
engine_active
engine_wait_event
engine_wait_operand
stream_credit_empty_or_full
sram_bank_conflict
noc_backpressure
dma_wait_memory
uce_program_or_descriptor_stall
unknown_or_unclassified
```

基础 counter map：

| counter                                   | scope             | 验证用例                                       |
| ----------------------------------------- | ----------------- | ---------------------------------------------- |
| BOA active/stall operand/acc/writeback    | engine/tile       | GEMM、attention、SRAM conflict                 |
| EVU active/LSU replay/masked lane         | engine/tile       | softmax、norm、tail、gather                    |
| MFE active/prefetch hit/miss/stream stall | engine/tile       | paged attention、segment stream                |
| USE active/state hit/miss/checkpoint      | engine/tile       | recurrence、reset/fault restore                |
| DMA bytes/stall/outstanding               | tile/group/global | DMA copy、paged attention prefetch             |
| SRAM bank conflict                        | tile/group        | bank-aware layout、EVU replay                  |
| Stream occupancy/credit empty/full        | tile/group        | role pipeline、EOS/error                       |
| Event wait cycles                         | tile/group/global | barrier、timeout、dependency                   |
| NoC congestion by VC                      | group/global      | command/event isolation、DMA/collective stress |
| command queue occupancy/context count     | global            | multi-context QoS                              |

### 4.4 roofline 和 workload PMU 签名

| workload        | 结果验收                                                | PMU 签名验收                                                           |
| --------------- | ------------------------------------------------------- | ---------------------------------------------------------------------- |
| Dense GEMM      | Python golden bit/容差一致                              | BOA active 高，operand stall 低，DMA bytes 与 descriptor 一致          |
| Dense Attention | QK/softmax/AV 结果一致                                  | BOA/EVU 阶段切换清楚，collective stall 与 split/reduce 对齐            |
| Paged Attention | page reorder、KV prefetch、attention 输出一致           | `T_prefetch <= T_qk` case 下 BOA operand stall 降低，MFE hit/miss 合理 |
| MoE Dispatch    | 8/16 expert routing 与 combine golden 一致              | routing imbalance 与 BOA utilization、MFE stall 可解释                 |
| SSM/Recurrence  | Mamba/RWKV 类 recurrence golden 通过                    | USE active、state hit/miss、checkpoint/restore counter 有效            |
| 多模型并发      | context isolation、fault isolation、priority queue 通过 | per-context queue occupancy、QoS latency/throughput 可量化             |

## 5. 数据流、控制流和时序路径

### 5.1 数据流验证

数据流从 host/runtime command buffer 开始，经过 firmware consume、descriptor validation、DMA/engine launch、event completion、PMU snapshot，最后回到 driver/runtime。每个 test 必须记录：输入 tensor/descriptor、command sequence、event sequence、输出 tensor、fault record、PMU snapshot。

### 5.2 控制流验证

- Tile Group Sequencer 推进 Group Task 和 role dispatch。
- Tile UCE 推进 Tile Program、launch/wait/branch、stream token、descriptor patch、L2/L1 DMA、engine orchestration。
- USE 只验证 state compute/state lifecycle，不把 Tile Program 主控制流归给 USE。
- MFE 只验证 page/segment metadata walk、address generation、prefetch、reorder、stream fill，不把任意图遍历归给 MFE。

### 5.3 时序路径风险验证

| 风险                            | 验证策略                                                                                     |
| ------------------------------- | -------------------------------------------------------------------------------------------- |
| 高扇出 reset/event/enable       | lint fanout report、synthesis constraint、STA、gate-level smoke，由 PPA exploration 冻结阈值 |
| SRAM boundary timing            | SRAM macro wrapper SVA、read/write latency parameter sweep、OpenSTA/PrimeTime path review    |
| NoC router critical path        | VC arbitration constrained random、NoC congestion PMU、STA critical path review              |
| Stream Queue combinational loop | formal credit proof、lint combo-loop check、deadlock timeout test                            |
| PMU freeze fanout               | 分层 freeze tree、CDC/RDC、STA fanout check                                                  |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置维度

Verification 必须覆盖 Edge、Balanced 和 High End 的参数化方向，但 First Silicon V1 推荐聚焦能闭合面积、时序、功耗和验证的 SRAM profile。具体 Tile 数、Group 数、SRAM 容量、NoC 拓扑、clock target、功耗目标由 PPA exploration 冻结；测试环境必须支持参数扫描，但 phase exit 使用被冻结的 first-silicon profile。

### 6.2 PPA/性能模型检查

- SRAM bandwidth 预算：`BW_sram_required <= BW_sram_peak * efficiency`。
- Bank conflict：`BW_eff = BW_peak * (1 - conflict_rate)`。
- Paged Attention latency：`T_total = T_page_walk + T_prefetch + T_qk + T_softmax + T_av + T_writeback - T_overlap`。
- 当 test 声称性能瓶颈时，必须给出 PMU counter 证据，不只给 runtime latency。

### 6.3 PMU 与 phase gate

Phase gate 中 PMU 是硬性验收项：Phase 1 要有 BOA active/stall；Phase 2 要有 EVU active/stall；Phase 3 要有 MFE prefetch hit/miss 与 BOA operand stall 关系；Phase 4 要有 routing imbalance 与 utilization；Phase 5 要有 USE state counter；Phase 6 要有 roofline validation 和 QoS counter。

## 7. RTL/软件实现建议

### 7.1 RTL 验证实现

- 使用统一 transaction record：command id、descriptor id、event id、context id、tile/group id、engine id。
- Scoreboard 同时比对功能结果、event ordering、fault record 和 PMU counter。
- 对 Stream Queue、event dependency、arbiter、FIFO、reset/drain 使用 formal 证明小状态空间性质。
- 对 BOA/EVU/MFE/USE 使用 constrained random 覆盖 shape、stride、mask、tail、page、segment、state cache。
- 对 CDC/RDC 使用工具检查加 hand-written assertions：跨域 pulse、async FIFO、reset release、sticky bit。

### 7.2 软件验证实现

- Compiler regression 使用 lit/FileCheck 检查 ELENOR dialect lowering、descriptor binary、profile marker。
- Runtime ABI tests 覆盖 submit/wait/signal/reset/read_counter、timeout、fault isolation。
- Driver tests 覆盖 memory allocation/pinning、IOMMU fault、interrupt handling、PMU readout、context isolation。
- Firmware tests 覆盖 command queue consume、descriptor validation、event scheduling、shape branch、profiling snapshot、reset/drain。

### 7.3 EDA、开源和工业工具流

| 类别                 | 开源/早期原型            | 工业实现                         | phase gate                           |
| -------------------- | ------------------------ | -------------------------------- | ------------------------------------ |
| RTL simulation       | Verilator、cocotb        | VCS、Xcelium、Questa             | 每个 phase 必跑相关 unit/integration |
| waveform/debug       | GTKWave                  | Verdi                            | fault 和 PMU mismatch debug          |
| compiler regression  | MLIR/LLVM、lit/FileCheck | 同左加 CI farm                   | descriptor ABI 与 trace 对齐         |
| synthesis smoke      | Yosys                    | Design Compiler、Fusion Compiler | area/timing 趋势                     |
| physical exploration | OpenROAD                 | Innovus、Fusion Compiler         | high fanout、SRAM/NoC placement      |
| STA                  | OpenSTA                  | PrimeTime                        | phase timing gate                    |
| lint                 | Verible                  | SpyGlass、AscentLint             | RTL quality gate                     |
| CDC/RDC              | 由后续规格冻结           | Questa CDC、SpyGlass CDC/RDC     | clock/reset crossing gate            |
| formal               | SymbiYosys 小块          | JasperGold、VC Formal、360 DV    | FIFO/arbiter/event/stream proof      |
| equivalence          | 由后续规格冻结           | Formality、Conformal             | synthesis ECO gate                   |
| power                | 由后续规格冻结           | PrimePower、Voltus、PowerArtist  | clock gating、SRAM sleep             |

## 8. 验证、bring-up 和验收标准

### 8.1 验证层级

| 层级                   | 方法                                                  | 验收                                      |
| ---------------------- | ----------------------------------------------------- | ----------------------------------------- |
| Python model           | golden reference、random tensor、workload trace       | canonical trace 可复现                    |
| MLIR/compiler          | FileCheck、descriptor ABI、binary descriptor golden   | command/descriptor 与 source intent 对齐  |
| Runtime ABI            | command ring、event table、fault record、reset domain | submit/wait/fault/reset/read_counter 正确 |
| RTL unit               | Verilator/VCS、constrained random、SVA                | engine/protocol 单元覆盖                  |
| Tile integration       | command queue + SRAM + UCE/USE + engine               | tile 内 DMA/engine/event/PMU 闭环         |
| Group integration      | stream queue、collective、broadcast、DMA overlap      | role pipeline、EOS/error/reset 闭环       |
| System integration     | driver + firmware + runtime end-to-end                | workload 从 command 到 output 闭环        |
| Performance validation | PMU counter + benchmark + roofline                    | 瓶颈可由 counter 解释                     |

### 8.2 phase exit criteria

| Phase   | 必须通过的功能                    | 必须通过的验证                                     | 性能/PMU 门槛                               | 静态 gate                     |
| ------- | --------------------------------- | -------------------------------------------------- | ------------------------------------------- | ----------------------------- |
| Phase 0 | ABI/cutline/ownership/spec freeze | spec checklist、golden trace format                | 无性能门槛，counter map 可评审              | lint/formal plan 可评审       |
| Phase 1 | BOA GEMM through command queue    | RTL sim、Python golden、command/event/fault        | BOA active/stall 合理                       | lint、基础 CDC/RDC、STA smoke |
| Phase 2 | EVU softmax/norm/tail             | random tensor、mask/tail corner                    | EVU active/stall、LSU replay 可归因         | lint、SVA、STA smoke          |
| Phase 3 | MFE Page Stream + paged attention | reorder、timeout、EOS/error token                  | BOA operand stall 与 MFE prefetch 对齐      | formal stream、CDC/RDC        |
| Phase 4 | MoE Segment Stream                | routing imbalance、segment reduce golden           | BOA utilization vs imbalance 可解释         | NoC/SRAM stress STA           |
| Phase 5 | USE scan/recurrence               | Mamba/RWKV golden、checkpoint/restore              | USE state cache counter 有效                | reset/fault RDC、formal event |
| Phase 6 | multi-model                       | context isolation、fault injection、priority queue | QoS latency/throughput、roofline validation | power/timing/area exploration |

### 8.3 静态检查验收

- lint：Verible、SpyGlass、AscentLint；检查 latch、unreachable state、width truncation、unused signal、case 完整性、组合环。
- CDC：clock domain crossing，覆盖 PMU event、NoC/router、DMA/memory、firmware bus、debug path。
- RDC：reset domain crossing，覆盖 tile/group/device reset、fault reset、counter sticky、stream drain。
- SVA：command/event ordering、descriptor validation、stream credit、EOS/error propagation、stall attribution、snapshot atomic。
- formal：FIFO、arbiter、event dependency、stream queue credit、reset/drain、deadlock freedom 的局部性质。
- static timing：high fanout control、SRAM macro boundary、NoC router、counter freeze tree、descriptor patch path。
- power：clock gating、operand gating、SRAM sleep/retention、PMU disabled toggle、reset/power intent consistency。

## 9. 风险、取舍和后续细化方向

| 风险                            | 影响                                        | 缓解                                                   |
| ------------------------------- | ------------------------------------------- | ------------------------------------------------------ |
| 只测 datapath 不测 command path | silicon bring-up 时 runtime/firmware 不可用 | phase exit 要求 through command queue                  |
| 验证面过大                      | 每个模块都只有局部完成                      | bring-up 顺序严格阶段化，canonical trace 复用          |
| PMU 失真                        | 性能结论不可行动                            | counter 与 waveform/golden trace 对齐，唯一归因 SVA    |
| Stream Queue 死锁               | pipeline stage 卡死且难定位                 | formal credit proof、timeout、EOS/error/reset tests    |
| reset/fault 行为不确定          | 多 context 隔离失败                         | reset domain tests、fault injection、RDC gate          |
| 高扇出和 SRAM/NoC 时序晚发现    | 后端收敛风险                                | early synthesis/OpenSTA/OpenROAD + industrial STA gate |
| 开源工具与工业 signoff 差异     | 原型通过但实现不可签核                      | 每个 phase 保留工业工具映射和等价检查                  |

需要由后续规格冻结的项目：最终 command/event 二进制兼容矩阵、descriptor checksum 策略、coverage metric、simulation seed policy、fault injection taxonomy、PMU snapshot ABI。需要由 SRAM profile 冻结的项目：L1/L2 latency、bank count、macro port、ECC/parity、SRAM sleep/retention 行为。需要由 PPA exploration 冻结的项目：clock target、NoC topology、high fanout 阈值、STA/power signoff corner、floorplan 和 physical hierarchy。
