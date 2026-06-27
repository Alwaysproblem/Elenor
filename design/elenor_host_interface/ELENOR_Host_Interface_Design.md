# ELENOR Host Interface 设计文档

## 1. 定位、目标和 First Silicon cutline

Host Interface 是 Host/System SoC 与 ELENOR Device 的协议边界，负责设备枚举、CSR 访问、command queue doorbell、event/fault interrupt、host memory IOVA 访问入口、context isolation 和 bring-up debug path。它把 PCIe/CXL/AXI 等外部协议转换成 ELENOR 内部 CSR、queue、DMA 和 event 协议；它不解释高层 graph，也不执行 command，只保证 host 提交的 command buffer 能以确定顺序到达 Runtime Processor/Global Scheduler。

First Silicon V1 的目标是稳定控制面，而不是实现所有 host coherent 特性：

| 类别      | First Silicon V1                                                              | Architecture V1 / 后续规格                            |
| --------- | ----------------------------------------------------------------------------- | ----------------------------------------------------- |
| 外部协议  | PCIe 或 SoC AXI attachment 二选一，具体由后续规格冻结                         | PCIe/CXL/AXI 多协议 SKU                               |
| CSR       | capability、queue config、doorbell、event/fault base、PMU read、reset request | SR-IOV、advanced virtualization CSR                   |
| Queue     | host memory command ring 或 device-resident queue pointer                     | 多 priority queue、doorbell batching、queue migration |
| Interrupt | MSI/MSI-X 或 SoC interrupt，event/fault coalescing 基础策略                   | per-context vector、adaptive interrupt moderation     |
| DMA 地址  | IOVA/context domain 检查，descriptor bounds validation 配合 runtime           | full ATS/PASID/coherent memory policy                 |
| Error     | illegal CSR、bad doorbell、IOMMU/address fault、event/fault reporting         | poison recovery、link-level RAS 扩展                  |
| PMU       | host interface latency、doorbell count、interrupt count、fault count          | full trace packet export                              |

Host Interface 的验收路径是：driver 枚举 device -> 读 capability -> 配 queue/event/fault buffer -> 写 command ring -> ring doorbell -> 设备执行 no-op/DMA -> event done -> interrupt -> host 读 PMU/fault 状态。

模块框图：

```text
Host / System Interconnect
        |
        v
+------------------------------------------------------------------+
| ELENOR Host Interface                                            |
|                                                                  |
| +----------------+   +----------------+   +-------------------+  |
| | Protocol Front |-->| CSR Aperture   |-->| CSR Firewall      |  |
| | PCIe/CXL/AXI   |   | BAR/AXI decode |   | context/privilege |  |
| +-------+--------+   +--------+-------+   +---------+---------+  |
|         |                     |                     |            |
|         |                     v                     v            |
|         |             +---------------+     +---------------+    |
|         +------------>| Queue Manager |---->| Doorbell FIFO |----+--> Runtime/Scheduler
|                       +-------+-------+     +---------------+    |
|                               |                                  |
|                               v                                  |
|                       +---------------+                          |
|                       | Event/Fault   |<-------------------------+-- Event/Fault Fabric
|                       | Interrupt     |                          |
|                       +-------+-------+                          |
|                               |                                  |
|                     +---------v----------+                       |
|                     | Host DMA/IOMMU     |<----------------------+-- Global DMA requests
|                     | Translation Assist |                       |
|                     +---------+----------+                       |
|                               v                                  |
|                         PMU/Debug                                |
+------------------------------------------------------------------+
```

## 2. 职责、非职责和 ownership

### 2.1 职责

| 职责              | 具体要求                                                                                  |
| ----------------- | ----------------------------------------------------------------------------------------- |
| 枚举和 capability | 暴露 device id、ABI version、feature bits、queue/event/fault/PMU 能力。                   |
| CSR 访问          | 对 host 读写执行地址 decode、字节使能、权限检查、非法访问 fault。                         |
| Queue 配置        | 接收 queue base/size/head/tail/context/domain 参数，保证 ring 地址合法。                  |
| Doorbell          | 将 host 写 doorbell 转换为内部 queue pending event，保留 write ordering。                 |
| Event/fault 通知  | 将 Event Fabric 的 done/error/timeout/reset 转成 host interrupt 或 polling-visible 状态。 |
| Host memory path  | 为 command fetch、descriptor fetch、DMA host memory 访问提供 IOVA/domain metadata。       |
| Context isolation | queue、event、fault、PMU readout 必须按 context_id 或 privilege 过滤。                    |
| Debug bring-up    | scratch、loopback、mailbox、link status、interrupt test、PMU snapshot。                   |

### 2.2 非职责

- 不消费 command payload，不解析 BOA/EVU/MFE/USE/DMA 私有 descriptor。
- 不调度 Group Task；调度由 Runtime Processor 和 Global Scheduler 执行。
- 不决定 Tile Group partition、SRAM quota 或 QoS policy。
- 不实现 Host OS driver 的 memory pinning；只消费 driver 配置后的 IOVA/domain。
- 不把 host interrupt 当作 event table 的唯一真相；event table 状态由 Event Fabric owns。

### 2.3 Ownership

| 对象                 | Owner                            | Host Interface 行为                                                                |
| -------------------- | -------------------------------- | ---------------------------------------------------------------------------------- |
| BAR/AXI CSR aperture | Host Interface                   | decode、firewall、route 到内部 CSR bus。                                           |
| queue base/size      | Driver 写入，Host Interface 保存 | 校验 alignment/size，输出给 Queue Manager。                                        |
| queue head           | Runtime Processor owns           | Host Interface 可提供 host-readable shadow。                                       |
| queue tail           | Host writes                      | Host Interface latch tail snapshot 并生成 doorbell。                               |
| command body         | Host memory / Runtime fetch      | Host Interface 不修改 command body。                                               |
| event table          | Event Fabric owns                | Host Interface 只做 interrupt、polling read 或 DMA-visible cache management hook。 |
| fault record         | Fault Fabric owns                | Host Interface 提供 host read path 和 interrupt cause。                            |
| IOMMU domain         | Driver/IOMMU owns                | Host Interface 保存 domain id/PASID-like metadata，具体字段由后续规格冻结。        |

## 3. 微架构和状态机

### 3.1 子模块切分

```text
host_interface
├── protocol_frontend
│   ├── pcie_or_axi_rx
│   ├── pcie_or_axi_tx
│   └── ordering_adapter
├── csr_aperture_decode
├── csr_firewall
├── queue_manager
│   ├── queue_context_table
│   ├── doorbell_decode
│   ├── doorbell_fifo
│   └── queue_head_tail_shadow
├── interrupt_controller
│   ├── event_vector_map
│   ├── fault_vector_map
│   └── coalescing_timer
├── host_address_domain_table
├── pmu_debug_block
└── internal_bus_adapter
```

### 3.2 Link/device 状态机

```text
LINK_DOWN
  -> LINK_TRAINING
  -> LINK_READY
  -> DEVICE_DISABLED
  -> DEVICE_ENABLED
  -> QUEUE_BOUND
  -> ACTIVE
  -> QUIESCE
  -> DEVICE_ENABLED 或 ERROR
```

| 状态            | 含义                               | 允许动作                                |
| --------------- | ---------------------------------- | --------------------------------------- |
| LINK_DOWN       | 外部链路不可用                     | 只更新 sticky link fault。              |
| LINK_TRAINING   | PCIe/CXL link 或 AXI attach 初始化 | 禁止 command doorbell。                 |
| LINK_READY      | 可访问基础 CSR                     | 读 capability、写 scratch。             |
| DEVICE_DISABLED | driver 尚未 enable                 | 配置 event/fault/queue base。           |
| DEVICE_ENABLED  | device 可接收 queue bind           | enable interrupt、reset request。       |
| QUEUE_BOUND     | 至少一个 queue 配置完成            | ring doorbell。                         |
| ACTIVE          | 有未完成 command 或 event          | 正常运行。                              |
| QUIESCE         | driver 请求停止新 command          | doorbell 被拒绝或延迟，由后续规格冻结。 |
| ERROR           | fatal host-side error              | 只允许 fault readout/reset。            |

### 3.3 Doorbell 状态机

```text
IDLE
  -> WRITE_ACCEPTED
  -> ORDERING_FENCE
  -> TAIL_LATCHED
  -> FIFO_PUSH
  -> SCHEDULER_ACK
  -> IDLE
```

关键点：

1. Host 对 command buffer/descriptor/event memory 的写入必须在 doorbell 前对 device 可见。外部协议的 posted write ordering 不足时，driver 需要 read flush 或 Host Interface 需要 ordering_adapter；选择由后续规格冻结。
2. Doorbell FIFO entry 至少包含 queue_id、context_id、tail_snapshot、sequence、timestamp、flags。
3. 若 FIFO full，CSR 写入返回 backpressure/error 的具体行为由外部协议决定；内部必须计数 `doorbell_fifo_full`。
4. Scheduler ack 只表示内部已接收 doorbell，不表示 command 完成。

### 3.4 Interrupt 状态机

```text
EVENT_OR_FAULT_PENDING
  -> VECTOR_SELECT
  -> COALESCE_WINDOW
  -> INTERRUPT_ASSERT
  -> HOST_ACK_OR_MASKED
  -> IDLE_OR_PENDING
```

Interrupt Controller 必须支持 polling-only 模式，便于 bring-up 和仿真；也必须支持 fault interrupt 优先于 normal completion。event coalescing 的计时窗口、计数阈值由后续规格冻结。

## 4. 接口、descriptor、寄存器和协议

### 4.1 CSR 表建议

| Offset | 名称              | 属性   | 说明                                                     |
| ------ | ----------------- | ------ | -------------------------------------------------------- |
| 0x0000 | HI_CAP0           | RO     | protocol type、max payload、interrupt mode。             |
| 0x0008 | HI_STATUS         | RO     | link ready、device enable、queue active、fatal error。   |
| 0x0010 | HI_CONTROL        | RW     | device enable、quiesce、polling mode、interrupt enable。 |
| 0x0018 | HI_SCRATCH        | RW     | bring-up loopback。                                      |
| 0x0100 | QUEUE_BASE_LO/HI  | RW     | 当前选择 queue 的 command ring base。                    |
| 0x0110 | QUEUE_SIZE        | RW     | entries 或 bytes，单位由后续规格冻结。                   |
| 0x0118 | QUEUE_CONTEXT     | RW     | context_id、domain_id、priority。                        |
| 0x0120 | QUEUE_HEAD_SHADOW | RO     | runtime consumed head。                                  |
| 0x0128 | QUEUE_TAIL_SHADOW | RO     | last doorbell tail。                                     |
| 0x0130 | DOORBELL          | WO     | queue_id、tail 或 increment。                            |
| 0x0200 | EVENT_BASE_LO/HI  | RW     | event table host-visible base 或 mirror base。           |
| 0x0210 | EVENT_CONFIG      | RW     | event count、interrupt vector base。                     |
| 0x0300 | FAULT_BASE_LO/HI  | RW     | fault record ring base。                                 |
| 0x0310 | FAULT_STATUS      | RO/W1C | sticky fault bits。                                      |
| 0x0400 | INTR_STATUS       | RO/W1C | event/fault/mailbox cause。                              |
| 0x0408 | INTR_MASK         | RW     | interrupt mask。                                         |
| 0x0500 | PMU_SELECT        | RW     | counter select/filter。                                  |
| 0x0508 | PMU_VALUE_LO/HI   | RO     | selected counter value。                                 |

### 4.2 Doorbell entry 内部格式示例

```c
typedef struct {
    uint32_t queue_id;
    uint32_t context_id;
    uint32_t domain_id;
    uint32_t tail_snapshot;
    uint32_t doorbell_seq;
    uint32_t flags;
    uint64_t host_timestamp_or_zero;
} elenor_doorbell_internal_t;
```

字段含义：

- `queue_id` 选择 command ring。
- `context_id` 用于 event/fault/PMU namespace。
- `domain_id` 绑定 IOMMU/PASID-like 地址空间，编码由后续规格冻结。
- `tail_snapshot` 是 host 看到的 ring tail，Queue Manager 不能自行递增越过该值。
- `doorbell_seq` 用于检测重复 doorbell、lost doorbell 和 debug trace。
- `flags` 可表示 polling、interrupt suppress、high priority 等，具体 bit 由后续规格冻结。

### 4.3 Queue configuration descriptor

```c
typedef struct {
    uint64_t ring_base_iova;
    uint32_t ring_entries;
    uint32_t entry_stride;
    uint32_t context_id;
    uint32_t queue_id;
    uint32_t domain_id;
    uint32_t priority;
    uint32_t flags;
} elenor_host_queue_desc_v0_t;
```

Host Interface 校验：base alignment、entries 范围、entry_stride 至少覆盖 command header、context/queue 不冲突、domain 已启用。command entry 的 ABI version/cmd_size 由 Runtime Processor 校验；Host Interface 不重复解析。

### 4.4 Event/fault interrupt 协议

Event/Fault Fabric 输入：

```text
event_valid
event_id
context_id
queue_id
status
producer_id
fault_record_slot
interrupt_hint
```

Host Interface 输出：

```text
interrupt_assert(vector)
intr_status[event|fault|mailbox]
optional event_mirror_write
```

Ordering 要求：fault record 写入完成早于 event error 可见；event table 状态写入完成早于 interrupt assert。若使用 host memory mirror，write completion 语义由外部协议和 driver flush 策略共同冻结。

### 4.5 Host memory ordering

- Doorbell write 前，host 必须保证 command buffer 和 descriptor 已写入并对 device 可见。
- Device 写 event/fault 后，Host Interface 必须保证 interrupt 不早于 event/fault visibility。
- Queue head shadow 更新不能早于 Runtime Processor 真正完成 command header consume。
- PMU snapshot read 需要 latch；host 连续读 low/high 寄存器不能撕裂。

## 5. 数据流、控制流和时序路径

### 5.1 Command submission 流程

```text
Driver/runtime
  -> fill command ring entries
  -> fill descriptors / executable package metadata
  -> memory fence or protocol flush
  -> write DOORBELL(queue_id, tail)
Host Interface
  -> accept CSR write
  -> check device/queue/context state
  -> latch tail_snapshot
  -> push doorbell FIFO
Runtime Processor / Scheduler
  -> pop doorbell
  -> fetch command header
  -> validate ABI/context/descriptor bounds
  -> execute or reject with fault event
```

Host Interface 的 critical path 不应包含 host memory command fetch；doorbell path 只传递 tail snapshot。command fetch 可以由 Runtime Processor DMA/read master 或专用 queue fetcher 完成。

### 5.2 Completion 流程

```text
Engine / DMA / Scheduler
  -> event update request
Event Fabric
  -> writes event table
  -> optional fault record write
  -> sends interrupt request
Host Interface
  -> vector map + coalescing
  -> assert interrupt
Driver/runtime
  -> reads event/fault/PMU
  -> advances user-visible completion
```

### 5.3 Fault 流程

```text
bad CSR / bad doorbell / address fault / timeout
  -> source-specific fault code
  -> fault fabric slot allocation
  -> affected queue stop or device fatal policy
  -> event error if command-related
  -> interrupt fault vector
  -> runtime reset tile/group/device as policy
```

Host Interface 自身产生的 fault 包括非法 CSR、未 enable doorbell、queue size/alignment 错误、domain 未绑定、doorbell FIFO overflow、host protocol error。它不生成 engine internal fault，只转发。

### 5.4 时序路径

| 路径                            | 优化建议                                                               |
| ------------------------------- | ---------------------------------------------------------------------- |
| CSR write decode -> response    | 分块 decode，非法访问快速返回，跨域写入用 posted command FIFO。        |
| Doorbell write -> FIFO push     | queue_id/context table 读出寄存，tail latch 与权限检查分级。           |
| Event input -> interrupt assert | event write ack 后异步进入 interrupt domain，vector select 寄存。      |
| PMU read                        | select 阶段 latch counter，下一拍读 value。                            |
| Host address domain lookup      | context/domain table 小容量 CAM 或 RAM，输出寄存后给 queue fetch/DMA。 |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置

| 参数                               | 状态                    |
| ---------------------------------- | ----------------------- |
| 外部协议                           | 由后续规格冻结          |
| BAR/CSR aperture 大小              | 由后续规格冻结          |
| queue 数量                         | 由后续规格冻结          |
| queue entry stride                 | 由后续规格冻结          |
| doorbell FIFO depth                | 由 PPA exploration 冻结 |
| interrupt vector 数                | 由后续规格冻结          |
| coalescing threshold/window        | 由后续规格冻结          |
| context/domain table entries       | 由后续规格冻结          |
| protocol clock 与 core clock ratio | 由 PPA exploration 冻结 |

### 6.2 PMU counters

| Counter                       | 说明                           | Primary owner                            |
| ----------------------------- | ------------------------------ | ---------------------------------------- |
| hi_csr_reads / hi_csr_writes  | CSR 压力和 driver 行为         | Host Interface                           |
| hi_bad_csr_access             | 非法访问                       | Host Interface fault                     |
| hi_doorbell_count             | 提交次数                       | Host Interface                           |
| hi_doorbell_to_sched_cycles   | doorbell 到 scheduler ack 延迟 | queue/control path                       |
| hi_doorbell_fifo_full_cycles  | FIFO backpressure              | Host Interface                           |
| hi_interrupt_count_event      | completion interrupt 数        | Interrupt Controller                     |
| hi_interrupt_count_fault      | fault interrupt 数             | Interrupt Controller                     |
| hi_interrupt_coalesced_events | coalescing 效果                | Interrupt Controller                     |
| hi_queue_head_lag             | tail-head 差值采样             | Global Scheduler / Host Interface shadow |
| hi_host_read/write_bytes      | Host memory path 流量          | Host Interface / DMA                     |
| hi_ordering_flush_cycles      | ordering_adapter 等待          | Host Interface                           |

这些 counter 需要和 global PMU 的 command queue occupancy、event wait、DMA bandwidth 对齐。Host Interface 的 primary stall 只覆盖 host 边界；若 command 因 engine/DMA 阻塞，归因不应落到 Host Interface。

### 6.3 PPA 取舍

- Doorbell latency 对小 batch 和 dynamic shape path 敏感，Queue Manager 应避免大组合 CAM。
- Interrupt coalescing 降低 host overhead，但会增加 tail latency；First Silicon V1 应支持关闭 coalescing。
- Protocol frontend 可复用 vendor IP，但 CSR firewall、queue/event/fault 语义必须在 ELENOR RTL 中可验证。
- Host Interface 不应成为 DMA bulk data 的唯一瓶颈；host memory copy path 与 CSR/doorbell path 分开仲裁。
- Power gating 前必须确认没有 posted CSR、doorbell FIFO、pending interrupt 或 host memory transaction。

### 6.4 Clock/reset/power/timing 考虑

- Host protocol clock、core control clock、interrupt clock 可以不同；doorbell FIFO、event interrupt FIFO、PMU snapshot 必须使用结构化 CDC，不允许裸多 bit 跨域。
- Reset release 顺序建议为 protocol frontend -> CSR aperture -> queue manager -> interrupt controller；assert 可异步，deassert 必须在本域同步。
- Quiesce 或 reset 时，Host Interface 必须先阻止新 doorbell，再等待 doorbell FIFO drain，最后清理 pending interrupt；host-visible sticky fault 不应被普通 reset 静默清除。
- Power gating 前必须确认没有 posted CSR write、doorbell FIFO entry、host memory read/write、pending interrupt 和 PMU snapshot transaction。
- Timing closure 优先关注 CSR decode fanout、queue context table lookup、doorbell accept path、interrupt vector select 和 domain table lookup；这些路径应寄存切分。

## 7. RTL/软件实现建议

### 7.1 RTL 建议

- CSR 字段使用 `hw_rw/hw_set/w1c/ro` 明确访问语义，避免 software/hardware 同时写同一 bit 的未定义行为。
- Queue context table 写入时必须校验 queue disabled；active queue 不允许修改 base/size/domain，除非进入 quiesce。
- Doorbell FIFO entry 加 sequence 和 sticky overflow bit，便于定位 lost doorbell。
- Interrupt status 使用 W1C，mask 只屏蔽 interrupt assert，不屏蔽 event/fault 状态更新。
- PMU read 使用 snapshot register，避免 high/low split read 撕裂。
- Host address domain table 输出给 DMA/queue fetch 时带 generation，reset/context destroy 后旧请求必须失效。

### 7.2 Driver/runtime 建议

1. 初始化时先读 capability 和 ABI version，不匹配则禁止提交 command。
2. 分配 command ring、event table、fault record buffer，满足 alignment 和 cacheability 要求。
3. 写 queue config 后读回 shadow，确认 Host Interface 已接受。
4. 提交 command 前执行 host memory fence/flush。
5. Doorbell 后可以 polling event 或等待 interrupt；bring-up 首选 polling-only 降低变量。
6. fault interrupt 后先读 fault record，再决定 reset tile/group/device。
7. reset 后重新读取 queue head/tail shadow 和 PMU reset_count，确认状态一致。

### 7.3 SVA/形式化建议

- active queue 的 base/size/domain 不可被修改。
- doorbell FIFO 不可丢 entry；overflow 必须产生可见 fault 或 backpressure。
- interrupt assert 前 event/fault visible 条件必须成立。
- W1C bit 清除不能清掉同周期新来的 fault，除非规格定义优先级。
- queue_id/context_id 越界必须拒绝 doorbell。
- PMU high/low 读在 snapshot 模式下保持一致。

## 8. 验证、bring-up 和验收标准

### 8.1 单元验证

| 单元                      | 覆盖                                                                       |
| ------------------------- | -------------------------------------------------------------------------- |
| Protocol frontend adapter | read/write ordering、byte enable、unaligned illegal access、backpressure。 |
| CSR firewall              | RO/WO/RW/W1C、privilege/context filter、非法 aperture。                    |
| Queue Manager             | queue bind/unbind、doorbell sequence、FIFO full、head/tail wrap。          |
| Interrupt Controller      | event/fault priority、mask、W1C、coalescing、polling-only。                |
| Address Domain Table      | domain enable/disable、generation mismatch、context isolation。            |
| PMU Debug                 | snapshot、overflow、clear、filter。                                        |

### 8.2 Bring-up 序列

1. link ready 后读 HI_CAP0/HI_STATUS。
2. 写 HI_SCRATCH，读回。
3. 配置 event/fault base，打开 polling-only。
4. 绑定 queue，提交 no-op command，poll event done。
5. 打开 interrupt，提交 no-op command，验证 interrupt cause。
6. 提交 DMA 1D copy command，验证 data 和 event。
7. 注入 bad doorbell queue_id，验证 bad doorbell fault。
8. 注入 invalid descriptor command，验证 Host Interface 转发 fault/event，不误归因为 CSR fault。
9. 执行 quiesce/reset，验证 active queue 停止且 doorbell 被拒绝或延迟，行为与规格一致。

### 8.3 验收标准

- command/event/DMA/PMU-first 路径可在仿真和 FPGA/silicon bring-up 中闭环。
- Host Interface 不需要理解 engine descriptor 即可正确传递 command。
- 所有 host-visible fault 都有可读 fault record 或 sticky status。
- polling 和 interrupt 两种 completion 模式都可用。
- context A 不能读写 context B 的 queue/event/fault/PMU 资源。
- doorbell、event、fault 的 ordering 有 SVA 和集成测试覆盖。
- CDC/RDC 对 protocol clock、core clock、interrupt clock 的跨域路径全部收敛。

## 9. 风险、取舍和后续细化方向

| 风险                              | 影响                            | 缓解                                                                    |
| --------------------------------- | ------------------------------- | ----------------------------------------------------------------------- |
| 外部协议选择过早扩散到内部 RTL    | 后续 SKU 难维护                 | 用 protocol_frontend 隔离 PCIe/CXL/AXI，内部统一 CSR/queue/event 协议。 |
| Doorbell ordering 不清晰          | command fetch 读到旧 descriptor | driver fence + ordering_adapter 规则写入 ABI，bring-up 测试覆盖。       |
| Interrupt coalescing 掩盖 fault   | debug 复杂                      | fault vector 优先，fault coalescing 可关闭。                            |
| Queue active 时重配置             | command ring 损坏               | active queue 禁止重配置，quiesce 后再改。                               |
| Context isolation 漏洞            | 多模型或多进程互相污染          | context/domain table、generation、CSR filter、PMU filter 全链路验证。   |
| Host Interface 被误归因为性能瓶颈 | PMU 结论错误                    | 只统计 host 边界 stall，engine/DMA/NoC stall 由对应 owner 统计。        |
| Vendor IP 黑盒行为不可验证        | ordering/interrupt 边界风险     | 在 wrapper 处定义可验证协议和 assertion，黑盒只承担物理链路。           |

后续规格需要冻结：外部协议、BAR/AXI aperture、queue entry stride、doorbell write 格式、interrupt vector 分配、event/fault host mirror 语义、IOMMU/domain 字段、coalescing 策略、host memory cacheability、W1C 同周期优先级和 PMU counter id。
