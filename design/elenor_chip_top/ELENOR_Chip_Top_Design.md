# ELENOR Chip Top 设计文档

## 1. 定位、目标和 First Silicon cutline

ELENOR Chip Top 是整颗 ELENOR Device 的顶层集成边界，负责把 Host Interface、Runtime Processor、Global Scheduler、Global DMA、Memory Controller、Collective、Global PMU、NoC/Router 和 Tile Group 阵列组合成一个可复位、可枚举、可提交 command、可产生 event、可读 PMU、可隔离 fault 的 device。它不是高层 graph interpreter；硬件只消费 command buffer、descriptor、Region Program 和 Tile Program。

顶层目标有三类：

1. **控制面闭环**：host doorbell 进入 device 后，command queue、event table、barrier、fault record 和 interrupt 必须形成确定协议。
2. **数据面闭环**：Global DMA/Memory Controller/NoC/Tile Group 的 HBM/DDR/LPDDR -> L2 -> L1 路径必须能执行 descriptor-driven copy，并产生 completion event。
3. **观测面闭环**：Global PMU 能唯一归因 command queue、DMA、NoC、event wait、SRAM/engine 汇聚来的 stall，不重复计数同一 cycle。

Architecture V1 允许描述完整芯片形态，包括多 Tile Group、多 NoC virtual channel、全局 collective、多 context 和 PMU feedback scheduling。First Silicon V1 只要求以下 cutline：

| 范围              | First Silicon V1 必须实现                                                             | Architecture V1 / 后续规格                          |
| ----------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------- |
| Host path         | doorbell、command ring、event/fault MSI/MSI-X 或 SoC interrupt                        | CXL.cache/coherent attach、复杂虚拟化               |
| Runtime Processor | command consume、descriptor validation、shape branch 基础路径、reset/drain            | 高级 priority/preemption、PMU feedback scheduling   |
| Global Scheduler  | queue dispatch、region launch、event/barrier、basic resource map                      | 多模型 QoS、跨 group 动态重分配                     |
| Global DMA        | 1D/2D/strided copy、async completion event、timeout fault                             | multicast、gather list、复杂 layout transform       |
| Memory            | HBM/DDR/LPDDR 控制器接口、IOMMU/IOVA 透传检查                                         | coherent host memory policy                         |
| NoC               | VC0 command/event、VC1 read response、VC2 write/stream、VC3 collective 预留或最小实现 | hierarchical mesh 参数、QoS aging、adaptive routing |
| PMU               | queue occupancy、DMA bandwidth、NoC VC congestion、event wait、fault counter          | trace sampling、feedback scheduler                  |
| Reset             | device/group/tile reset domain、drain protocol、fault record 保留                     | partial context preemption                          |

所有位宽、队列深度、Tile Group 数、NoC 拓扑参数、SRAM profile、时钟频率和功耗域划分在本文中不冻结，写为 `由后续规格冻结` 或 `由 PPA exploration 冻结`。

顶层概念图：

```text
Host / System SoC
        |
        v
+----------------------------------------------------------------------------+
| ELENOR Chip Top                                                            |
|                                                                            |
|  +------------------+      +------------------+      +------------------+  |
|  | Host Interface   |      | Runtime Processor|      | Global PMU       |  |
|  | PCIe/CXL/AXI     |<---->| RISC-V / uCtrl   |<---->| Trace/Error      |  |
|  +--------+---------+      +---------+--------+      +---------+--------+  |
|           |                          |                         ^           |
|           v                          v                         |           |
|  +------------------------------------------------------------------------+|
|  | Global Scheduler / Command Queue / Event Fabric / Fault Records        ||
|  +------------------------+--------------------------+--------------------+|
|                           |                          |                     |
|                           v                          v                     |
|  +------------------+     +------------------+       +------------------+  |
|  | Global DMA       |<--->| Memory Controller|<----->| Collective       |  |
|  | copy/stride/event|     | HBM/DDR/LPDDR    |       | reduce/bcast     |  |
|  +--------+---------+     +---------+--------+       +---------+--------+  |
|           |                         |                          |           |
|           +-------------------------+--------------------------+           |
|                                     v                                      |
|                              NoC / Router                                  |
|       VC0 command/event, VC1 read, VC2 write/stream, VC3 collective        |
|                                     |                                      |
|                       +-------------+-------------+                        |
|                       v                           v                        |
|                Tile Group 0                 Tile Group N-1                 |
+----------------------------------------------------------------------------+
```

## 2. 职责、非职责和 ownership

### 2.1 顶层职责

| 职责          | 顶层实现要求                                                                                                   |
| ------------- | -------------------------------------------------------------------------------------------------------------- |
| 集成边界      | 定义所有全局模块端口、clock/reset/power domain、DFT hook、scan/test mode 和 CSR aperture。                     |
| 地址空间      | 统一 Host IOVA、device physical address、HBM address、CSR address 和 NoC target id 的解码规则。                |
| 命令入口      | 把 Host Interface 写入的 doorbell/queue tail 转成 Runtime Processor 和 Global Scheduler 可消费的 work item。   |
| 事件出口      | 将 engine completion、DMA completion、region done、tile/group fault 汇聚为 event table 状态和 host interrupt。 |
| 数据路径      | 连接 Global DMA、Memory Controller、NoC 和 Tile Group，保证 command/control 与 bulk data 不互相饿死。          |
| 错误隔离      | 按 context、queue、group、tile、engine 记录 fault，并触发 reset/drain，不把单模型 fault 扩散到无关 context。   |
| PMU 汇聚      | 对 global PMU、本地 group/tile PMU 和 NoC counter 做时间戳对齐、snapshot、clear-on-read 或 latch-on-event。    |
| Bring-up 支撑 | 提供最小 CSR、loopback、DMA copy、event interrupt、PMU readout 和 reset smoke path。                           |

### 2.2 非职责

- 不解释 PyTorch/ONNX/MLIR 高层 graph。
- 不在顶层实现 BOA/EVU/MFE/USE 计算语义。
- 不把某个绝对 SRAM 地址绑定到算子语义；Tile L1 由 Slot Frame 和 descriptor 管理。
- 不替代 driver 的 memory allocation/pinning/IOMMU 配置。
- 不在 Global Scheduler 中实现任意通用 CPU 调度策略；First Silicon V1 只做 command/region 级确定调度。
- 不在 NoC 中修复无效 descriptor 或非法访问；NoC 只传播错误响应和 poison/error metadata。

### 2.3 Ownership matrix

| 对象                   | Owner                                             | Consumer                               | 顶层约束                                                          |
| ---------------------- | ------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------- |
| command ring tail/head | Host Interface / Runtime Processor                | Global Scheduler                       | head 更新必须在 descriptor validation 后发生。                    |
| event table            | Global Scheduler / Event Fabric                   | Host Interface、Runtime Processor、PMU | event_id 必须全局唯一或带 context_id namespace。                  |
| fault record           | 产生 fault 的模块写入，Global Scheduler 分配 slot | driver/runtime                         | fault 必须包含 command_id、context_id、queue_id 和 source id。    |
| global address map     | Chip Top                                          | DMA、MFE global path、Host Interface   | CSR/HBM/NoC apertures 不可重叠。                                  |
| reset domain           | Chip Top reset controller                         | 所有模块                               | reset 必须定义 pending event、queue credit、DMA inflight 的结果。 |
| PMU timestamp          | Global PMU                                        | group/tile PMU、driver                 | timestamp source 必须单调，跨 clock domain 使用同步快照。         |

## 3. 微架构和状态机

### 3.1 顶层子模块

建议 Chip Top 用薄集成层加显式 fabric adapter，不把协议转换散落到各模块：

```text
chip_top
├── clk_rst_pwr_ctrl
├── csr_decode_and_firewall
├── host_interface_wrapper
├── runtime_processor_subsystem
├── global_scheduler_subsystem
│   ├── command_queue_array
│   ├── event_fabric
│   ├── fault_record_file
│   └── resource_map
├── global_dma_subsystem
├── memory_controller_wrapper
├── noc_top
├── collective_top
├── global_pmu
└── tile_group_array
```

### 3.2 Device lifecycle 状态机

```text
POR
  -> RESET_ASSERTED
  -> RESET_RELEASE_SYNC
  -> CSR_ENUM
  -> FW_BOOT
  -> IDLE
  -> RUNNING
  -> DRAINING
  -> IDLE
  -> FAULTED
  -> RESET_ASSERTED 或 DRAINING
```

| 状态           | 进入条件                     | 可接受操作                               | 退出条件                                        |
| -------------- | ---------------------------- | ---------------------------------------- | ----------------------------------------------- |
| POR            | power good 未稳定            | 无                                       | pll/clock/power stable，由 PPA exploration 冻结 |
| RESET_ASSERTED | global reset 或 fatal reset  | CSR 只读 boot status                     | reset synchronizer complete                     |
| CSR_ENUM       | host 枚举 BAR/AXI aperture   | 读 capability、写 scratch                | driver enable device                            |
| FW_BOOT        | runtime firmware load/boot   | firmware mailbox、PMU basic counter      | firmware ready event                            |
| IDLE           | 无 inflight command          | queue bind、context bind、PMU snapshot   | doorbell 或 command available                   |
| RUNNING        | 至少一个 queue active        | command consume、DMA、NoC、tile dispatch | queues empty 或 fault                           |
| DRAINING       | reset/drain requested        | 禁止新 command，完成可安全 command       | drain done 或 timeout                           |
| FAULTED        | fatal fault 或 drain timeout | fault readout、reset domain select       | reset request                                   |

### 3.3 Reset/drain 微架构

顶层 reset controller 维护三级 reset：device、group、tile。First Silicon V1 必须至少支持 device reset 和 group/tile soft reset request 透传。reset/drain 语义：

1. Global Scheduler 停止受影响 queue 取新 command。
2. Event Fabric 将相关 pending event 标为 blocked 或 reset，具体编码由后续规格冻结。
3. Global DMA 停止发起新 burst，对已发出事务等待响应或 timeout。
4. Stream Queue、Tile Group、NoC 对受影响 domain 执行 drain，回收 credit。
5. Fault Record File 记录 reset reason、last command、timeout source。
6. Host Interface 产生 interrupt 或 mailbox bit。

### 3.4 CDC/RDC 策略

- Host Interface、Runtime Processor、Global DMA、NoC、Memory Controller、Tile Group 可以处于不同 clock domain；跨域只允许使用 async FIFO、toggle synchronizer、req/ack bridge 或 gray-coded counter。
- reset deassert 必须在每个 clock domain 内同步；reset assert 可以异步，但对 SRAM macro、PLL、NoC link training 的顺序由 PPA exploration 冻结。
- event table、fault record、PMU snapshot 跨域读写必须使用 shadow/latch 机制，不允许 host 直接采样多 bit 未同步状态。

## 4. 接口、descriptor、寄存器和协议

### 4.1 顶层端口类别

| 类别       | 示例信号                                                    | 说明                                                 |
| ---------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| Host       | PCIe/CXL/AXI transaction、interrupt、reset sideband         | 具体协议由后续规格冻结。                             |
| Memory     | HBM/DDR/LPDDR controller AXI-like port                      | burst、QoS、ECC/error response 需要向 DMA/PMU 暴露。 |
| Tile Group | command/event NoC port、data NoC port、PMU port、reset port | group_id 编码由后续规格冻结。                        |
| DFT/debug  | JTAG、scan enable、mbist、trace drain                       | 不进入性能数据路径。                                 |
| Power      | power good、isolation enable、retention request             | power domain 划分由 PPA exploration 冻结。           |

### 4.2 Capability CSR 示例

```c
typedef struct {
    uint32_t device_id;
    uint16_t abi_major;
    uint16_t abi_minor;
    uint32_t feature_bits;
    uint32_t num_groups;
    uint32_t tiles_per_group;
    uint32_t num_queues;
    uint32_t event_table_entries;
    uint64_t hbm_bytes_or_zero;
    uint32_t noc_vc_mask;
    uint32_t reset_domain_mask;
} elenor_chip_capability_v0_t;
```

必须保留 version、size、feature_bits。driver 不能根据寄存器偏移猜测 feature；必须读 capability 后再启用 queue、DMA、PMU 或 reset domain。

### 4.3 顶层 CSR aperture 建议

| Offset | 名称             | 属性 | 说明                                               |
| ------ | ---------------- | ---- | -------------------------------------------------- |
| 0x0000 | CHIP_ID          | RO   | device id、revision、implementation id。           |
| 0x0008 | ABI_VERSION      | RO   | command/event/descriptor ABI 兼容范围。            |
| 0x0010 | DEVICE_CONTROL   | RW   | enable、quiesce、interrupt enable。                |
| 0x0018 | DEVICE_STATUS    | RO   | lifecycle state、fw ready、fatal fault。           |
| 0x0020 | RESET_REQUEST    | WO   | bitmask: device/group/tile，由后续规格冻结。       |
| 0x0028 | RESET_STATUS     | RO   | reset busy、last reset reason。                    |
| 0x0100 | QUEUE_BASE_LO/HI | RW   | command queue IOVA base。                          |
| 0x0110 | QUEUE_SIZE       | RW   | ring entries，必须 power-of-two 或由后续规格冻结。 |
| 0x0118 | DOORBELL         | WO   | queue_id + tail snapshot。                         |
| 0x0200 | EVENT_BASE_LO/HI | RW   | event table IOVA 或 device-resident table base。   |
| 0x0300 | FAULT_BASE_LO/HI | RW   | fault record buffer。                              |
| 0x0400 | PMU_CONTROL      | RW   | snapshot、clear、filter context/group。            |
| 0x0800 | SCRATCH          | RW   | bring-up scratch/loopback。                        |

### 4.4 Command/Event ABI 对顶层的要求

顶层必须接受 `elenor_command_v0_t` 中的以下字段并传递给 owner：abi_version、cmd_size、type、flags、context_id、queue_id、desc_iova、desc_bytes、desc_crc_or_zero、wait_event_base、wait_event_count、signal_event、timeout_cycles、fault_record_slot。Chip Top 不解析 engine 私有 descriptor，但必须校验：

- command header 可读、大小合法、version 支持。
- desc_iova 属于当前 context/IOMMU domain。
- wait/signal event 落在 context event namespace 内。
- timeout_cycles 不为非法保留值；具体范围由后续规格冻结。
- fault_record_slot 可写或由硬件分配。

Event Fabric 输出 `PENDING/DONE/ERROR/TIMEOUT/RESET`，用于 DMA completion、stage synchronization、tile done、group done 和 graph done。

## 5. 数据流、控制流和时序路径

### 5.1 Cold launch 控制流

```text
Host Runtime
  -> load package / allocate HBM / upload program + descriptor
  -> write queue base/event base/fault base CSR
  -> ring doorbell
Host Interface
  -> validates doorbell ordering
  -> notifies Runtime Processor / Scheduler
Runtime Processor
  -> consumes command header
  -> validates version/context/descriptor bounds
Global Scheduler
  -> allocates event/fault slots
  -> launches Pipeline Region to target Tile Groups
Global DMA / NoC
  -> moves program/data to Group SRAM or Tile path
Tile Group
  -> executes Region Program and Tile Program
Event Fabric
  -> writes event done/error and interrupts host
```

关键 ordering：descriptor writes 必须在 doorbell 前对 device 可见；event write 必须在 interrupt 前对 host 可见；fault record write 必须在 event error 前对 host 可见。

### 5.2 数据面路径

```text
Host pinned memory / HBM / DDR / LPDDR
    -> Host Interface or Memory Controller
    -> Global DMA read channel
    -> NoC VC1 response / VC2 write-stream
    -> Group Shared SRAM / L2
    -> Tile DMA / MFE tile port
    -> Tile L1 Slot Frame
```

Chip Top 必须给 DMA、MFE global path 和 Memory Controller 统一 backpressure 语义。若 data stream 堵塞，不允许阻塞 VC0 command/event 到不可恢复；NoC 至少将 command/event 与 bulk data 分 VC。

### 5.3 关键时序路径

| 路径                         | 风险                         | 缓解                                                      |
| ---------------------------- | ---------------------------- | --------------------------------------------------------- |
| doorbell -> scheduler wakeup | 高 fanout queue wakeup       | doorbell decode 后用 per-queue pending bit，分层 wakeup。 |
| event done -> interrupt      | 跨域、写后中断顺序           | event shadow write ack 后再触发 interrupt。               |
| reset request -> all domains | reset fanout 大              | reset tree 分层，domain-local synchronizer。              |
| NoC route select             | VC arbitration critical path | route compute pipeline，credit ready 提前寄存。           |
| PMU aggregation              | counter bus fan-in 大        | hierarchical snapshot，group summary register。           |
| CSR decode                   | aperture 大导致组合路径长    | 分块 decode + registered response。                       |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置参数

| 参数            | 建议表达                                                      | 当前状态                                 |
| --------------- | ------------------------------------------------------------- | ---------------------------------------- |
| num_tile_groups | capability CSR + synthesis parameter                          | 由后续规格冻结                           |
| tiles_per_group | capability CSR + static netlist                               | 由后续规格冻结                           |
| command queues  | per context/per priority                                      | 由后续规格冻结                           |
| event entries   | global table or per context table                             | 由后续规格冻结                           |
| NoC topology    | crossbar / 2D mesh / hierarchical mesh                        | 由 PPA exploration 冻结                  |
| NoC VC          | VC0 command/event、VC1 read、VC2 write/stream、VC3 collective | Architecture V1 建议，深度由后续规格冻结 |
| SRAM profile    | Tile L1、Group SRAM、program cache                            | 由 SRAM profile 冻结                     |
| clock domains   | host/core/noc/mem/group                                       | 由 PPA exploration 冻结                  |

### 6.2 顶层 PMU counter

| Counter                        | 归因 owner       | 用途                                             |
| ------------------------------ | ---------------- | ------------------------------------------------ |
| chip_cycles                    | none             | 时间基准。                                       |
| queue_occupancy_cycles         | Global Scheduler | 判断 host/runtime 是否供给不足或阻塞。           |
| queue_blocked_by_event_cycles  | Event Fabric     | dependency 等待。                                |
| event_wait_cycles              | Event Fabric     | 与 engine local wait 对齐。                      |
| dma_bytes_read/write           | Global DMA       | 带宽模型。                                       |
| dma_wait_memory_cycles         | Global DMA       | primary stall owner: `ELENOR_STALL_DMA_MEMORY`。 |
| noc_congestion_cycles_vc0..vc3 | NoC              | 区分 command/event 与 data stream 拥塞。         |
| interrupt_count                | Host Interface   | 过度 interrupt 或 event coalescing 调优。        |
| fault_count_by_source          | Fault Fabric     | bring-up 和可靠性。                              |
| reset_count_by_domain          | Reset Controller | fault recovery 验证。                            |

PMU 唯一归因规则：同一 cycle 只能有一个 primary stall owner；global PMU 可以保留 secondary tag 供 debug，但不进入 utilization 主统计。group/tile 本地 counter 与 global timestamp 对齐后才能汇总。

### 6.3 PPA 策略

- Control plane 以低延迟和确定性优先，不为极低面积牺牲 fault/event 可观测性。
- Bulk data path 以带宽和 backpressure 稳定性优先；descriptor/control 不走大数据仲裁队列。
- Clock gating 以 queue empty、DMA idle、NoC VC idle、Tile Group idle 为粒度。
- Power gating 只允许在没有 inflight command、event pending、DMA transaction、NoC packet、stream token 的 domain 上启用。
- Counter、trace、fault record 默认低开销，trace sampling buffer 容量由后续规格冻结。

## 7. RTL/软件实现建议

### 7.1 RTL 切分

- `elenor_chip_top.sv` 只实例化模块、连接接口、绑定 top-level assertions，不包含复杂 arbitration。
- `elenor_csr_firewall.sv` 负责 CSR decode、privilege/context check、illegal access fault。
- `elenor_reset_ctrl.sv` 负责 lifecycle state、reset request、drain timeout、domain reset handshake。
- `elenor_event_fabric.sv` 负责 event table update、interrupt coalescing、event wait wakeup。
- `elenor_fault_fabric.sv` 负责 fault slot allocation、record write、fatal policy。
- `elenor_pmu_aggregator.sv` 负责 counter snapshot、clear、filter 和 timestamp alignment。

接口建议使用 SystemVerilog `interface` 封装 valid/ready、credit、event、fault、PMU sample，避免 top-level 端口失控。

### 7.2 软件 bring-up 顺序

1. 读 CHIP_ID/ABI_VERSION/capability。
2. 写 SCRATCH 并读回，验证 CSR path。
3. 配置 event/fault buffer 和 interrupt。
4. 提交 no-op command，观察 event done。
5. 提交 DMA 1D copy，比较 host-visible buffer。
6. 读取 PMU snapshot，验证 queue/DMA/event counter 非零且 fault counter 为零。
7. 注入 invalid descriptor，验证 fault record 和 event error。
8. 触发 reset/drain，验证 queue 停止、event reset、device 回到 IDLE。

### 7.3 Assertions

- doorbell 后若 queue 非空，scheduler 最终看到 pending bit，除非 reset/fault。
- event 从 DONE/ERROR/TIMEOUT/RESET 不允许回到 PENDING。
- fault record valid 前必须写完 source/context/command/event 字段。
- NoC VC0 credit 不允许被 VC1/VC2 永久阻塞。
- reset request 后受影响 domain 不允许再接受新 command。
- PMU clear 与 snapshot 同周期时行为必须确定，由后续规格冻结。

## 8. 验证、bring-up 和验收标准

### 8.1 单元验证

| 单元             | 场景                                                                            |
| ---------------- | ------------------------------------------------------------------------------- |
| CSR firewall     | 合法读写、非法 aperture、权限错误、跨 context 访问。                            |
| Reset controller | device reset、group reset、tile reset、drain timeout、reset during DMA。        |
| Event fabric     | wait/signal、event dependency、timeout、interrupt ordering。                    |
| Fault fabric     | invalid descriptor、address fault、DMA timeout、NoC poison、fault buffer full。 |
| PMU aggregator   | snapshot consistency、clear-on-read、counter overflow、timestamp alignment。    |
| NoC top          | VC isolation、credit exhaustion、backpressure recovery、poison propagation。    |

### 8.2 系统 bring-up 验收

First Silicon V1 顶层验收必须覆盖：

1. command queue + event + barrier 最小闭环。
2. DMA 1D/2D/strided copy + completion event。
3. BOA GEMM 必须通过 command queue 触发，而不是 testbench 直接拉 datapath。
4. basic PMU counter 可读，且能解释 queue occupancy、DMA bandwidth、NoC congestion、event wait。
5. invalid descriptor、address fault、DMA timeout 能产生 fault record 和 event error。
6. reset/drain 后 queue、event、DMA、NoC、Tile Group 状态确定。
7. CDC/RDC、lint、SVA、formal FIFO/arbiter/event dependency 通过对应检查。

### 8.3 覆盖点

- 所有 lifecycle 状态和状态转换。
- 所有 reset domain 和 reset reason。
- command type 至少覆盖 DMA、BARRIER、EVENT_WAIT、EVENT_SIGNAL、LAUNCH_REGION、RESET_DOMAIN。
- event status 覆盖 PENDING、DONE、ERROR、TIMEOUT、RESET。
- NoC VC 覆盖 command/event 与 data 同时拥塞。
- PMU primary stall owner 覆盖 WAIT_EVENT、NOC_VC、DMA_MEMORY、UNKNOWN。

## 9. 风险、取舍和后续细化方向

| 风险                              | 影响                                    | 缓解                                                                         |
| --------------------------------- | --------------------------------------- | ---------------------------------------------------------------------------- |
| 顶层集成过早绑定高端配置          | First Silicon 面积/时序/功耗无法闭合    | capability 化，多档参数，First Silicon 选择现实 SRAM/NoC profile。           |
| command/event 被 data stream 阻塞 | 系统不可调试、event timeout 假阳性      | VC0 独立、event path 高优先级、PMU 记录 VC congestion。                      |
| reset/drain 语义不完整            | fault recovery 不确定，context 隔离失败 | reset 前停止取新 command，drain inflight，fault/event 顺序固定。             |
| PMU 多重计数                      | 性能结论不可用                          | primary stall owner 规则，global/local timestamp 对齐。                      |
| CSR/ABI version 演进混乱          | driver/firmware 不兼容                  | capability + abi_major/minor + cmd_size + feature_bits。                     |
| CDC/RDC 漏洞                      | silicon 间歇性故障                      | 所有跨域接口结构化，CDC/RDC 工具和 SVA 双重验证。                            |
| NoC 拓扑过度复杂                  | 验证面扩大                              | First Silicon V1 保留清晰 VC 语义，routing/QoS 参数由 PPA exploration 冻结。 |

后续规格需要冻结：CSR 完整表、interrupt 格式、fault record 二进制 layout、NoC topology/VC depth、reset domain 编码、capability feature bits、PMU counter id、clock/power domain、DFT/MBIST 接口和 SRAM/NoC PPA profile。