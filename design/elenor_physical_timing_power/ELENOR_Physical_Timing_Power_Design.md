# ELENOR Physical Timing Power 设计文档

## 1. 定位、目标和 First Silicon cutline

ELENOR Physical/Timing/Power 规划负责把 Architecture V1 中的 Tile、Tile Group、NoC、SRAM、DMA、BOA、EVU、MFE、USE、Tile UCE、PMU 和 runtime 控制面约束成可实现、可验证、可收敛的 First Silicon V1 物理方案。该文档不替代 RTL specification，也不冻结工艺库、SRAM macro 或封装；未冻结的定量值使用 `由 PPA exploration 冻结` 或 `由 SRAM profile 冻结`。

核心目标：

1. 让 First Silicon V1 的 SRAM/NoC/clock/reset/power 选择能闭合面积、时序、功耗和验证。
2. 在架构阶段提前暴露高扇出控制、SRAM macro 边界、NoC router、counter/trace、stream queue 和 descriptor patch 的关键路径风险。
3. 建立开源原型与工业 EDA flow 的映射，使 early exploration 的结论能迁移到 signoff 级工具。
4. 让 PMU 与 power/timing validation 形成闭环：性能瓶颈、带宽冲突、clock gating 效果和 workload signature 必须可观测。

First Silicon V1 cutline：

| 范围         | First Silicon V1 必须闭合                                                                       | Architecture V1 / 后续预留          |
| ------------ | ----------------------------------------------------------------------------------------------- | ----------------------------------- |
| SRAM         | 一个可实现的 Tile L1 + Group L2 profile、bank/port/latency、macro wrapper、sleep/retention 策略 | High End 3D SRAM/chiplet/eDRAM 配置 |
| NoC          | command/event、DMA read response、DMA write/stream、collective 的 VC 划分和基本 QoS             | 完整多模型 preemption/QoS fabric    |
| Timing       | high fanout、SRAM boundary、NoC router、descriptor patch、PMU freeze、stream credit 关键路径    | 频率目标由 PPA exploration 冻结     |
| Power        | clock gating、operand gating、SRAM sleep、PMU disabled toggle、workload power estimation        | DVFS、thermal feedback scheduler    |
| Verification | lint、CDC/RDC、SVA、formal、STA、power check 纳入 phase gate                                    | signoff checklist 由后续规格冻结    |

## 2. 职责、非职责和 ownership

### 2.1 职责

- 定义 Tile/Group/Chip 物理层级和 floorplan 假设，避免跨层长路径无界增长。
- 定义 SRAM profile 模板：容量、bank、端口、读写 latency、ECC/parity、sleep/retention、BIST/repair、macro placement。
- 定义 NoC topology 和 VC 规划：command/event、DMA read response、DMA write/stream、collective 分离。
- 定义 clock/reset/power domain 原则：Tile、Group、Global、firmware/debug/PMU domain 的 crossing 和 reset sequencing。
- 定义 static signoff gate：lint、CDC、RDC、SVA、formal、STA、power、equivalence。
- 定义 PMU/PPA 观测：active/stall、SRAM conflict、NoC congestion、DMA bandwidth、queue occupancy、power gating 状态。

### 2.2 非职责

- 不在本文件冻结具体工艺节点、标准单元库、SRAM compiler、封装和电压角。
- 不把 High End 的 1.5 GB 级片上 SRAM 当作普通 6T SRAM 默认实现。
- 不把 PMU counter 当作 signoff 替代品；PMU 是 silicon/runtime 观测，不替代 STA/power/CDC/RDC。
- 不改变架构 ownership：UCE 负责 program control，USE 负责 state，MFE 负责 data-related dynamic memory access。

### 2.3 ownership 矩阵

| 对象                | 架构/RTL owner   | Physical owner          | Verification owner      | PMU/PPA 证据                    |
| ------------------- | ---------------- | ----------------------- | ----------------------- | ------------------------------- |
| Tile L1 SRAM        | Tile RTL         | physical + memory       | RTL/static verification | bank conflict、latency、sleep   |
| Group L2 SRAM       | Group RTL        | physical + memory       | group integration       | DMA bandwidth、stream occupancy |
| NoC/router          | NoC RTL          | physical                | NoC verification        | VC congestion、latency          |
| high fanout control | RTL owner        | physical                | lint/STA/CDC/RDC        | fanout report、timing slack     |
| clock/reset domain  | SoC RTL          | physical + verification | CDC/RDC                 | crossing report、reset tests    |
| PMU aggregation     | PMU RTL/firmware | physical                | PMU verification        | counter toggle、freeze timing   |
| power intent        | RTL + firmware   | physical/power          | power verification      | gating ratio、SRAM sleep cycles |

## 3. 微架构和状态机

### 3.1 物理层级建议

```text
Chip
├── Host / Runtime / Global PMU / Global DMA
├── Global NoC spine
└── Tile Group x N
    ├── Region Sequencer / Group PMU / Barrier
    ├── Group DMA / Collective / Stream Queue
    ├── Group Shared SRAM L2 macros
    └── Compute Tile x M
        ├── Tile UCE / local event / Tile PMU
        ├── L1 SRAM macros + slot frame banks
        ├── BOA cluster
        ├── EVU
        ├── MFE
        └── USE
```

物理设计必须保持层级边界：Tile 内关键数据路径尽量本地闭合；Group 内共享 SRAM 和 stream queue 不向全芯片直接拉长；Global NoC 只承载跨 group 通信和 runtime/control aggregation。

### 3.2 clock/reset/power 状态机

```text
POWER_OFF
  -> POWER_RAMP
  -> RESET_ASSERTED
  -> RESET_RELEASE_SYNC
  -> SRAM_INIT_BIST
  -> IDLE_CLOCK_GATED
  -> ACTIVE
  -> QUIESCE
  -> RETENTION_OR_SLEEP
  -> WAKE_RESTORE
  -> ACTIVE
```

- `RESET_RELEASE_SYNC` 必须按 clock domain 同步释放，禁止 async release 直接进入 state machine。
- `SRAM_INIT_BIST` 覆盖 macro init、repair、ECC/parity 初始化，具体能力由 SRAM profile 冻结。
- `QUIESCE` 需要 command queue、stream queue、DMA outstanding、event table 和 PMU snapshot 达到确定状态后才能进入 sleep/retention。
- `WAKE_RESTORE` 后 descriptor cache、program SRAM、state cache 和 event/status region 的有效性由后续规格冻结。

### 3.3 timing closure 状态机

```text
RTL_CLEAN
  -> SYNTHESIS_SMOKE
  -> FLOORPLAN_ESTIMATE
  -> SRAM_MACRO_INTEGRATION
  -> CTS_AND_ROUTE_ESTIMATE
  -> STA_MULTI_CORNER
  -> POWER_ANALYSIS
  -> ECO_LOOP
  -> PHASE_EXIT
```

每个 phase 不能只看功能仿真。即使早期使用开源 flow，也必须产生 synthesis/timing/power 趋势，避免后端才发现架构级不可收敛路径。

## 4. 接口、descriptor、寄存器和协议

### 4.1 SRAM profile contract

| 字段            | 含义                                                 | 冻结方式                |
| --------------- | ---------------------------------------------------- | ----------------------- |
| Tile L1 容量    | 每 tile local SRAM 总容量                            | 由 SRAM profile 冻结    |
| Group L2 容量   | 每 group shared SRAM 总容量                          | 由 SRAM profile 冻结    |
| bank count      | L1/L2 bank 数量，First Silicon 推荐至少 16 bank 方向 | 由 SRAM profile 冻结    |
| port model      | read/write/RMW 端口和冲突语义                        | 由 SRAM profile 冻结    |
| latency         | read/write、sleep wake、BIST 时间                    | 由 SRAM profile 冻结    |
| ECC/parity      | data/tag/metadata 保护策略                           | 由后续规格冻结          |
| sleep/retention | SRAM low-power mode 和状态保持                       | 由 PPA exploration 冻结 |
| placement rule  | macro aspect ratio、bank adjacency、channel keepout  | 由 PPA exploration 冻结 |

SRAM arbitration 必须暴露 bank conflict event 给 PMU，并能按 consumer 归因：BOA operand、BOA accumulator、EVU LSU、MFE stream、DMA、USE state、UCE program/descriptor/event。

### 4.2 NoC/VC contract

| VC  | 用途               | timing/power 关注                           |
| --- | ------------------ | ------------------------------------------- |
| VC0 | command and event  | 低延迟、高可靠，不能被大数据流阻塞          |
| VC1 | DMA read response  | 带宽与 response ordering                    |
| VC2 | DMA write / stream | MFE/stream backpressure、burst power        |
| VC3 | collective         | reduce/broadcast latency、router congestion |

NoC router 必须提供 VC congestion counter、credit/backpressure 状态和 poison/error packet 预留。Command/event VC 的优先级策略由后续规格冻结，但不得出现数据流占满后 event timeout 无法传播的结构性风险。

### 4.3 PMU attribution hierarchy 和 counter map

Physical/PPA flow 使用与 PMU 设计一致的 primary stall owner：

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

PPA 相关 counter map：

| counter                           | scope             | PPA 用途                                          |
| --------------------------------- | ----------------- | ------------------------------------------------- |
| `PMU_SRAM_BANK_CONFLICT`          | tile/group        | SRAM bank/port/layout 评估                        |
| `PMU_NOC_VC*_CONGEST`             | group/global      | NoC topology、VC、router placement 评估           |
| `PMU_DMA_BYTES_READ/WRITE`        | tile/group/global | memory bandwidth 和 burst power 估算              |
| `PMU_STREAM_OCCUPANCY_ACC`        | tile/group        | stream buffer depth 和 SRAM footprint 评估        |
| `PMU_BOA/EVU/MFE/USE_ACTIVE`      | engine            | power activity factor 和 roofline                 |
| `PMU_EVENT_WAIT_CYCLES`           | tile/group/global | control/event path latency                        |
| `PMU_UCE_FETCH_STALL/PATCH_STALL` | tile              | program SRAM、descriptor cache、patch path timing |
| `PMU_POWER_CG_ELIGIBLE`           | tile/group/global | clock gating 机会估算                             |
| `PMU_POWER_SRAM_SLEEP`            | tile/group        | SRAM sleep/retention policy 评估                  |

### 4.4 register/protocol physical guardrail

- 跨 Tile/Group 的 control register write 必须经分层 decode，不允许单一 global register 高扇出驱动所有 tile。
- Descriptor patch path 只能更新 descriptor data，不允许 patch running program text。
- PMU freeze/clear、reset、clock enable、event broadcast 必须采用本地复制寄存器或树形分发。
- SRAM macro wrapper 必须固定 latency contract，RTL 不能依赖 memory compiler 的隐式时序。

## 5. 数据流、控制流和时序路径

### 5.1 关键数据流

| 数据流                    | 路径                                                   | 物理关注                                 |
| ------------------------- | ------------------------------------------------------ | ---------------------------------------- |
| HBM/DDR/LPDDR -> Group L2 | memory controller -> Global DMA/NoC -> Group DMA -> L2 | NoC VC1/VC2 带宽、L2 macro placement     |
| Group L2 -> Tile L1       | Group DMA/Tile DMA -> tile router port -> L1           | group-to-tile distance、burst timing     |
| L1 -> BOA                 | L1 operand banks -> BOA operand buffer/OPA             | bank-aware layout、短路径、高吞吐        |
| L1 -> EVU                 | vector buffer -> EVU LSU                               | indexed/gather replay、bank conflict     |
| MFE stream                | MFE page/segment -> stream buffer -> BOA/EVU           | stream queue backpressure、metadata SRAM |
| USE state                 | state slot/cache -> USE -> checkpoint/restore          | small random access、protected region    |
| PMU/control               | local events -> Tile/Group/Global PMU                  | low-speed registered aggregation         |

### 5.2 高风险时序路径

| 路径                            | 风险                                             | 规避原则                                                  |
| ------------------------------- | ------------------------------------------------ | --------------------------------------------------------- |
| high fanout reset/enable/freeze | 扇出过大、skew、hold 修复困难                    | 分层 reset tree、local enable flop、clock gating cell     |
| event/barrier broadcast         | 控制面跨 group 传播延迟影响 timeout              | event hierarchy、VC0 保护、timestamp 校准                 |
| SRAM macro boundary             | macro setup/hold、address decode、bank mux       | wrapper register、latency parameter、bank-local arbiter   |
| NoC router arbitration          | VC select/credit/update 组合路径长               | pipeline router、registered credit、localized arbitration |
| stream queue credit loop        | producer/consumer 组合回环                       | credit state 寄存，formal deadlock check                  |
| descriptor patch                | UCE patch + address template + slot frame decode | 预计算、分拍、patch cache                                 |
| PMU aggregation                 | wide counter add、global freeze                  | local counter bank、shadow snapshot、低速聚合             |
| clock gating enable             | enable 组合生成复杂                              | integrated clock gate 前寄存 enable                       |

### 5.3 控制流和 reset/drain

Reset/drain 是 physical 和 verification 的交界点。进入 tile/group/device reset 前，必须定义 DMA outstanding、stream queue token、event table、descriptor cache、program SRAM、USE state cache 和 PMU snapshot 的确定状态。Reset 不应依赖高层 software 猜测硬件内部残留状态。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置模型

ELENOR 可覆盖 Edge、Balanced、High End，但 First Silicon V1 应选择一个能闭合的 SRAM/NoC profile。推荐以 Balanced-small 方向作为现实起点：Tile L1、Group SRAM、Tile/Group 数量、NoC topology、clock target 和 memory bandwidth 的精确值由 SRAM profile 和 PPA exploration 冻结。

### 6.2 roofline 和 PMU 签名

物理规划必须支撑性能模型：

```text
Perf = min(BOA_compute, EVU_compute, MFE_stream, USE_state, Memory_bw)
BOA_perf = min(BOA_peak, SRAM_bw * AI_sram, HBM_bw * AI_hbm)
BW_eff = BW_peak * (1 - conflict_rate)
```

Per-workload PMU/PPA 签名：

| workload        | 物理瓶颈假设                                         | PMU 签名                                               | PPA 行动                                        |
| --------------- | ---------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------- |
| Dense GEMM      | L1 bank/BOA operand path、DMA overlap                | BOA active 高，SRAM conflict 低，DMA bytes 对齐        | 调整 bank placement、operand buffer adjacency   |
| Dense Attention | BOA/EVU 切换、collective、softmax buffer             | BOA/EVU active 分阶段，collective/NoC stall 可见       | 优化 EVU buffer 和 collective router            |
| Paged Attention | MFE prefetch、KV stream、NoC/DMA bandwidth           | MFE hit/miss、BOA operand stall、stream occupancy 相关 | 调整 stream buffer、prefetch depth、VC2 带宽    |
| MoE             | segment gather、expert imbalance、collective combine | routing imbalance、MFE stall、BOA utilization 下降     | 优化 segment buffer、NoC collective placement   |
| SSM/Recurrence  | USE state cache small random access                  | USE state hit/miss、event wait、state active           | state cache macro 和 protected region placement |
| 多模型并发      | SRAM quota、NoC QoS、fault isolation                 | per-context occupancy、VC congestion、QoS latency      | partition floorplan、VC/QoS 策略                |

### 6.3 power 模型

Power exploration 需要至少分解为：

- BOA/EVU/MFE/USE active dynamic power。
- SRAM read/write/leakage/sleep/retention power。
- NoC/router dynamic power by VC traffic。
- DMA burst power 和 memory interface activity。
- Clock tree power，尤其 high fanout control 和 always-on firmware/debug/PMU domain。
- PMU counter/trace toggle power。

具体功耗数值由 PPA exploration 冻结；早期必须使用 activity factor + PMU signature 形成相对趋势，而不是只看理论 peak TOPS。

### 6.4 SRAM/NoC timing 与容量约束

SRAM 容量越大，macro placement、wire length、bank mux、sleep/retention control 和 BIST/repair 都会增加复杂度。Group SRAM 超过常规片上 SRAM 合理范围时，必须明确 eDRAM、3D SRAM、HBM cache、chiplet SRAM die 或外部近存层，不允许在 First Silicon V1 中隐式假设。

NoC 拓扑必须按 workload traffic 做压力测试。VC0 command/event 的隔离是可靠性要求，不是性能优化选项；VC2 stream/DMA write 与 VC3 collective 不能形成不可打破的 backpressure 环。

## 7. RTL/软件实现建议

### 7.1 RTL physical-friendly coding

- pipeline 所有跨层长路径；Tile 内数据路径和 Group/Global 控制路径分离。
- 所有 high fanout control 使用本地寄存复制或时钟门控 enable，不直接组合扇出。
- SRAM macro wrapper 明确 request/response latency、byte enable、ECC/parity、sleep/wake、BIST 接口。
- NoC router credit 和 VC arbitration 寄存化，避免 combinational credit loop。
- PMU counter tap 采样已寄存状态，不影响 SRAM arbiter 或 NoC router 临界路径。
- reset release 同步化，RDC 可检查；sticky fault/counter state reset 行为明确。

### 7.2 软件/firmware 配合

- Firmware 在进入低功耗或 reset 前执行 quiesce：停止新 command、drain safe command、freeze PMU、记录 fault/event、发起 reset/sleep。
- Driver 暴露 SRAM/NoC/PMU capability table，使 runtime 不假设固定配置。
- Runtime 根据 PMU counter 调整 command batching、stream depth、SRAM quota 和 context priority，但 First Silicon V1 只要求可观测和手动调参。
- Compiler memory planner 输出 bank placement hint、slot frame layout 和 descriptor template patch 信息，降低 SRAM conflict。

### 7.3 EDA、开源和工业工具流

| Flow 阶段             | 开源/早期         | 工业 signoff                     | 关注点                                               |
| --------------------- | ----------------- | -------------------------------- | ---------------------------------------------------- |
| RTL lint              | Verible           | SpyGlass、AscentLint             | latch、width、fanout、combo loop、CDC pragma         |
| Simulation            | Verilator、cocotb | VCS、Xcelium、Questa             | reset/drain、power mode、PMU/PPA signature           |
| Synthesis             | Yosys             | Design Compiler、Fusion Compiler | area、critical path、clock gating inference          |
| STA                   | OpenSTA           | PrimeTime                        | high fanout、SRAM boundary、NoC router、PMU freeze   |
| Floorplan/route       | OpenROAD          | Innovus、Fusion Compiler         | SRAM macro placement、NoC channel、congestion        |
| CDC/RDC               | 由后续规格冻结    | Questa CDC、SpyGlass CDC/RDC     | clock/reset crossing、power domain crossing          |
| Formal                | SymbiYosys 小块   | JasperGold、VC Formal            | FIFO、arbiter、stream credit、reset sequencing       |
| Equivalence           | 由后续规格冻结    | Formality、Conformal             | synthesis/ECO equivalence                            |
| Power                 | 由后续规格冻结    | PrimePower、Voltus、PowerArtist  | clock gating、operand gating、SRAM sleep、IR/EM 输入 |
| Physical verification | 由后续规格冻结    | Calibre/Pegasus                  | DRC/LVS/antenna，具体由工艺 flow 冻结                |

## 8. 验证、bring-up 和验收标准

### 8.1 验证层级

| 层级                   | Physical/Timing/Power 验收                                       |
| ---------------------- | ---------------------------------------------------------------- |
| Python/workload model  | 产生 ops/bytes/traffic/activity factor 和预期 PMU signature      |
| Compiler/runtime       | 产生 bank placement、slot frame、profile marker、command traffic |
| RTL unit               | SRAM wrapper、NoC router、clock gating、reset、PMU tap SVA       |
| Tile integration       | L1/BOA/EVU/MFE/USE/UCE timing proxy、bank conflict PMU           |
| Group integration      | L2、Stream Queue、Collective、Group DMA、NoC VC congestion       |
| System integration     | quiesce/reset/sleep/wake、driver PMU readout、fault isolation    |
| Performance validation | roofline、bandwidth、NoC congestion、power trend 与 PMU 对齐     |
| Static signoff         | lint、CDC/RDC、formal、STA、power、equivalence、physical checks  |

### 8.2 phase exit criteria

| Phase   | Physical/Timing/Power exit                                                                              |
| ------- | ------------------------------------------------------------------------------------------------------- |
| Phase 0 | First-silicon SRAM profile 和 NoC VC 假设可评审；high-risk path list 建立                               |
| Phase 1 | command/event/DMA/BOA skeleton 通过 synthesis smoke、lint、基础 CDC/RDC、STA smoke                      |
| Phase 2 | EVU mask/tail/gather 相关 SRAM bank replay path 有 SVA/STA 证据                                         |
| Phase 3 | MFE Page Stream + paged attention 的 stream buffer、NoC VC2、SRAM footprint 有 PMU/PPA 证据             |
| Phase 4 | Segment/MoE 的 routing imbalance、collective/NoC congestion 有 stress 和 timing review                  |
| Phase 5 | USE state cache、checkpoint/restore、reset/fault RDC 行为确定                                           |
| Phase 6 | multi-context isolation、roofline validation、timing/power/area exploration 完成，主要瓶颈可由 PMU 解释 |

### 8.3 静态检查清单

- **lint**：未寄存 high fanout、组合环、未覆盖 case、隐式 latch、width truncation、unused reset/power signal。
- **CDC**：Tile/Group/Global/firmware/PMU clock crossing，NoC router crossing，DMA/memory crossing。
- **RDC**：tile reset、group reset、device reset、fault reset、power wake reset 的 crossing 和 release sequence。
- **SVA**：SRAM request/response latency、NoC credit、stream credit、event ordering、PMU stall attribution、clock gating enable stable。
- **formal**：FIFO、arbiter、event dependency、stream credit、deadlock freedom、reset drain 的局部证明。
- **STA**：high fanout reset/enable/freeze、SRAM boundary、NoC router、descriptor patch、PMU snapshot、clock gating enable。
- **power**：clock gating、operand gating、SRAM sleep/retention、PMU disabled toggle、activity annotation、power intent consistency。

## 9. 风险、取舍和后续细化方向

| 风险                          | 影响                                           | 缓解                                                             |
| ----------------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| SRAM 容量目标过高             | 面积、功耗、时序不可闭合                       | First Silicon 选择可实现 profile，High End 作为后续配置          |
| Group SRAM/NoC 距离过长       | DMA/stream/collective latency 增加             | 层级 floorplan、bank-local access、router pipeline               |
| 高扇出控制晚发现              | CTS/hold/ECO 成本高                            | RTL 阶段强制 local enable、early synthesis fanout gate           |
| NoC VC 不隔离                 | event/barrier 被数据流阻塞，系统不可用         | VC0 command/event 保护、PMU VC congestion、stress test           |
| PMU/trace 侵入 datapath       | 观测逻辑改变时序                               | local tap 已寄存状态、低速聚合、counter bank 本地化              |
| power gating 破坏状态         | wake 后 descriptor/state/event 不一致          | quiesce protocol、RDC、SVA、firmware reset/drain                 |
| 开源 exploration 误导 signoff | 原型收敛不代表工业实现收敛                     | 保留工业工具映射，phase gate 使用趋势而非绝对签核替代            |
| PPA 与 compiler/runtime 脱节  | bank conflict 和 NoC congestion 无法从软件规避 | slot frame bank hint、PMU feedback、canonical workload signature |

需要由后续规格冻结的项目：clock/reset/power domain 划分、low-power state ABI、BIST/repair policy、physical signoff checklist、NoC QoS 细节、event timeout 与 clock 关系。需要由 SRAM profile 冻结的项目：macro 容量、bank、端口、latency、ECC/parity、sleep/retention、placement rule。需要由 PPA exploration 冻结的项目：frequency target、voltage/corner、floorplan、NoC topology、clock tree、power budget、IR/EM 约束、trace/PMU 面积和功耗预算。
