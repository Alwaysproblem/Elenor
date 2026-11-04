# ELENOR Global DMA 设计文档

## 1. 定位、目标和 First Silicon cutline

Global DMA 是 ELENOR 芯片级大粒度数据搬运引擎，负责 host/HBM/DDR/LPDDR 与 device 内部 memory hierarchy 之间的 descriptor-driven copy。它主要服务于 HBM/DDR/LPDDR -> Group Shared SRAM/L2、program/descriptor/weight upload、activation prefetch、workspace spill/fill 和 host-visible buffer copy。Global DMA 不执行高层 graph，不做 Tile L2 -> L1 的细粒度搬运；L2 -> L1 由 Tile DMA/Tile UCE 管理。MFE 拥有数据相关的动态内存访问，如 page/segment walk、KV prefetch/reorder、stream fill；Global DMA 只执行 descriptor 中已确定的地址/stride/size 搬运。

First Silicon V1 的 DMA cutline 与架构评审一致：

```text
1D/2D/strided copy > async event > multicast > gather list
```

| 能力       | First Silicon V1                                             | Architecture V1 / 后续规格                       |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------ |
| Descriptor | src、dst、bytes、src_stride、dst_stride、rows、flags         | versioned binary layout、extended address mode   |
| Copy       | 1D contiguous、2D rows、strided rows                         | scatter/gather list、multicast、layout transform |
| Completion | async completion event、error event、timeout                 | chained DMA program、batch descriptor prefetch   |
| Address    | IOVA/device physical/HBM aperture 基础检查                   | ATS/PASID/coherent host memory policy            |
| QoS        | simple arbitration、NoC VC mapping                           | priority、deadline、bandwidth reservation        |
| PMU        | bytes、requests、latency、stall by memory/NoC/descriptor     | per-context bandwidth shaping                    |
| Error      | invalid descriptor、address fault、timeout、NoC/memory error | poison recovery、retry policy 扩展               |

Global DMA block diagram：

```text
Global Scheduler / Runtime Processor
        |
        v
+----------------------------------------------------------------------------+
| Global DMA                                                                 |
|                                                                            |
| +------------------+   +------------------+   +-------------------------+  |
| | Launch Queue     |-->| Descriptor Fetch |-->| Descriptor Validator    |  |
| +---------+--------+   +---------+--------+   +-----------+-------------+  |
|           |                      |                        |                |
|           v                      v                        v                |
| +------------------+   +------------------+   +-------------------------+  |
| | Address Generator|-->| Read Issue       |-->| Write Issue             |  |
| | 1D/2D/stride     |   | HBM/host/L2      |   | HBM/host/L2             |  |
| +---------+--------+   +---------+--------+   +-----------+-------------+  |
|           |                      |                        |                |
|           v                      v                        v                |
| +------------------+   +------------------+   +-------------------------+  |
| | Data Buffer      |<->| NoC/Mem Adapter  |<->| Completion/Fault/PMU    |  |
| | reorder/ecc tag  |   | VC1/VC2          |   | event/timeout/counter   |  |
| +------------------+   +------------------+   +-------------------------+  |
+----------------------------------------------------------------------------+
        |                      |
        v                      v
Memory Controller         NoC / Group SRAM / Host Interface
```

## 2. 职责、非职责和 ownership

### 2.1 职责

| 职责               | 实现要求                                                                                    |
| ------------------ | ------------------------------------------------------------------------------------------- |
| Launch 接收        | 从 Global Scheduler 接收 DMA command，绑定 context、queue、event、fault slot。              |
| Descriptor fetch   | 从 desc_iova 读取 DMA descriptor，校验 version/size/alignment/bounds。                      |
| Address generation | 根据 src/dst/bytes/stride/rows 生成 read/write burst。                                      |
| Data buffering     | 解耦 read response 与 write request，吸收 NoC/Memory backpressure。                         |
| Completion         | 所有 row/burst 完成后 signal event；错误时写 fault record 并 signal error。                 |
| Timeout            | launch、descriptor fetch、read、write、drain 均可被 timeout 观察。                          |
| Ordering           | 对同一 descriptor 内 row/burst 的可见顺序满足 flags 定义；默认 copy completion 前数据可见。 |
| PMU                | 统计 bandwidth、outstanding、stall、error、latency、row/burst 指纹。                        |

### 2.2 非职责

- 不执行 Tile L2 -> L1 DMA；Tile DMA 由 Tile UCE 控制。
- 不负责 page table walk、KV page reorder、segment gather/reduce；这些属于 MFE。
- 不解释 tensor semantic，不做自动 tiling/fusion。
- 不隐式修改 USE state slot；state slot 只能通过明确 command 或 checkpoint/restore path 修改。
- 不在 descriptor 错误时静默裁剪 copy；必须 fault。
- 不保证 cache coherence，除非外部 host/coherent policy 在后续规格中冻结。

### 2.3 Ownership

| 对象                      | Owner                                       | Global DMA 行为                                    |
| ------------------------- | ------------------------------------------- | -------------------------------------------------- |
| DMA command header        | Scheduler/Runtime                           | 接收已校验 context/queue/event 基本字段。          |
| DMA descriptor body       | DMA owns validation and execution           | 解析 copy 相关字段。                               |
| Source/destination memory | Memory Controller/Host Interface/Group SRAM | 按地址 aperture 发起读写，不拥有 allocation。      |
| Completion event          | Event Fabric owns                           | DMA 发送 done/error update。                       |
| Fault record              | Fault Fabric owns                           | DMA 提供 source、address、descriptor、burst 信息。 |
| NoC VC mapping            | NoC owns arbitration                        | DMA 将 read response/write stream 映射到 VC1/VC2。 |
| PMU counter               | DMA local + Global PMU                      | DMA 生成 local counter 和 snapshot。               |

## 3. 微架构和状态机

### 3.1 子模块

```text
global_dma
├── launch_queue
├── descriptor_prefetch
├── descriptor_validator
├── address_aperture_checker
├── row_generator
├── burst_splitter
├── read_request_engine
├── read_response_tracker
├── data_fifo_or_line_buffer
├── write_request_engine
├── completion_tracker
├── timeout_controller
├── fault_encoder
├── pmu_block
└── noc_mem_host_adapters
```

### 3.2 Descriptor lifecycle

```text
IDLE
  -> LAUNCH_ACCEPT
  -> DESC_FETCH
  -> DESC_VALIDATE
  -> ISSUE_ROWS
  -> DRAIN_READS
  -> DRAIN_WRITES
  -> COMPLETE
  -> IDLE
           \-> FAULT
           \-> TIMEOUT
           \-> RESET_DRAIN
```

| 状态          | 行为                                        | 错误条件                            |
| ------------- | ------------------------------------------- | ----------------------------------- |
| LAUNCH_ACCEPT | latch context/event/fault/desc pointer      | launch queue full。                 |
| DESC_FETCH    | 读取 descriptor header/body                 | desc address fault、fetch timeout。 |
| DESC_VALIDATE | 检查 bytes/rows/stride/flags/address        | invalid descriptor。                |
| ISSUE_ROWS    | 生成 row 和 burst                           | aperture violation、overflow。      |
| DRAIN_READS   | 等待读响应                                  | read timeout、memory error。        |
| DRAIN_WRITES  | 等待写响应或 write ack                      | write timeout、NoC poison。         |
| COMPLETE      | signal event done，更新 PMU                 | event update failure。              |
| FAULT/TIMEOUT | 写 fault record，signal event error/timeout | fault buffer full 需 sticky fatal。 |

### 3.3 Address generation

First Silicon V1 支持：

- 1D：`rows = 1`，`bytes` 为总长度，stride 忽略或必须等于 bytes，具体由后续规格冻结。
- 2D：每行 `bytes` 字节，共 `rows` 行，行起点递增 `src_stride` / `dst_stride`。
- strided：允许 `src_stride != bytes` 或 `dst_stride != bytes`。

地址计算：

```text
src_row_addr = src + row * src_stride
dst_row_addr = dst + row * dst_stride
row_bytes    = bytes
```

实现必须检测：`src + row * stride + bytes` 溢出、跨越非法 aperture、alignment 不满足 flags、rows 为零、bytes 为零时的行为。零长度 copy 是否产生 event done 由后续规格冻结。

### 3.4 Burst splitter

Burst splitter 将 row 切成 memory/NoC 支持的最大 burst：

```text
burst_addr  = row_addr + burst_offset
burst_bytes = min(max_burst_bytes_aligned, row_bytes - burst_offset)
```

max burst、alignment、boundary crossing 由 Memory Controller/NoC profile 冻结。Splitter 必须避免跨越禁止边界，例如 4KB/IOMMU page 或 NoC target aperture，边界大小由后续规格冻结。

### 3.5 Completion tracker

Completion tracker 维护：issued_reads、completed_reads、issued_writes、completed_writes、row_done、descriptor_done。Completion event 只能在所有写可见条件满足后 signal。若 read response 可乱序，data buffer 必须带 tag 或使用 in-order 限制；First Silicon V1 推荐限制单 descriptor 内写出顺序，降低验证面。

## 4. 接口、descriptor、寄存器和协议

### 4.1 DMA descriptor v0

架构评审给出的最小 descriptor：

```c
typedef struct {
    uint64_t src;
    uint64_t dst;
    uint32_t bytes;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t rows;
    uint32_t flags;
} elenor_dma_desc_t;
```

本文建议实现时加 version/size/context 保留字段，但二进制 layout 由后续规格冻结：

```c
typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t op;
    uint16_t flags_hi;
    uint64_t src;
    uint64_t dst;
    uint32_t bytes;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t rows;
    uint32_t flags;
    uint32_t reserved0;
} elenor_dma_desc_v0_ext_t;
```

`op` First Silicon V1 只需要 COPY。multicast/gather list 使用 flags/op 预留，但未实现的必需位必须产生 unsupported descriptor fault。

### 4.2 Launch interface

```text
valid/ready
context_id
queue_id
command_id
desc_iova
desc_bytes
desc_crc_or_zero
signal_event
signal_sequence
fault_record_slot
timeout_cycles
priority_or_qos
address_domain
```

Global DMA 接受 launch 后返回 `launch_accepted`，最终通过 Event Fabric 返回 done/error/timeout。Scheduler 不应在 launch_accepted 后立即 retire command，除非 command 语义明确为 fire-and-forget；First Silicon V1 推荐等待 DMA completion event。

### 4.3 Memory/NoC interface

| 通道         | 方向                | VC/目标         | 内容                                                  |
| ------------ | ------------------- | --------------- | ----------------------------------------------------- |
| read_req     | DMA -> Memory/NoC   | request path    | address、bytes、tag、domain、qos。                    |
| read_rsp     | Memory/NoC -> DMA   | VC1             | data、tag、error、last。                              |
| write_req    | DMA -> Memory/NoC   | VC2             | address、data、byte_enable、tag、domain、qos。        |
| write_rsp    | Memory/NoC -> DMA   | response path   | tag、error。                                          |
| event_update | DMA -> Event Fabric | VC0 或 sideband | event_id、sequence、status、producer_id、fault_slot。 |
| pmu_sample   | DMA -> Global PMU   | sideband        | bytes、stall、latency、error。                        |

### 4.4 DMA CSR

| Offset | 名称                 | 属性   | 说明                                            |
| ------ | -------------------- | ------ | ----------------------------------------------- |
| 0x0000 | DMA_CAP              | RO     | max outstanding、max burst、2D/stride support。 |
| 0x0008 | DMA_CONTROL          | RW     | enable、quiesce、pmu enable。                   |
| 0x0010 | DMA_STATUS           | RO     | idle、busy、faulted、draining。                 |
| 0x0018 | DMA_FAULT_STATUS     | RO/W1C | sticky fault bits。                             |
| 0x0020 | DMA_TIMEOUT_DEFAULT  | RW     | 默认 timeout。                                  |
| 0x0100 | DMA_DEBUG_DESC_LO/HI | RO     | last fault descriptor pointer。                 |
| 0x0110 | DMA_DEBUG_ADDR_LO/HI | RO     | last fault address。                            |
| 0x0200 | DMA_PMU_SELECT/VALUE | RW/RO  | local PMU counter。                             |

CSR debug launch 可用于 bring-up，但生产路径应通过 Scheduler launch，避免绕过 command/event ABI。

## 5. 数据流、控制流和时序路径

### 5.1 1D copy 流程

```text
Scheduler
  -> DMA launch(desc_iova, event_id)
Descriptor Fetch
  -> read descriptor
Validator
  -> rows=1, bytes valid, aperture valid
Burst Splitter
  -> generate read bursts
Read Engine
  -> issue reads
Data Buffer
  -> collect read responses
Write Engine
  -> issue writes
Completion Tracker
  -> wait write responses
Event Fabric
  -> signal DONE
```

### 5.2 2D/strided copy 流程

```text
for row in 0 .. rows-1:
    src_row = src + row * src_stride
    dst_row = dst + row * dst_stride
    split row_bytes into bursts
    issue read/write sequence according to outstanding window
```

Rows 可以 pipeline，但必须保留 fault 精确性：fault record 至少包含 row index、burst offset、source/destination indicator、address、command_id、descriptor pointer。

### 5.3 Program/descriptor load path

Cold launch 中 Tile Program 从 HBM 到 Tile Program SRAM。Global DMA 负责 HBM -> L2/Group SRAM 的大粒度 prefetch；Tile Program 到 Tile SRAM 可能由 Group/Tile DMA 完成，具体路径由后续规格冻结。Global DMA 必须支持 program/descriptor region 的 read-only 或 executable permission metadata，以便 invalid write 产生 fault。

### 5.4 Data path 与 NoC VC

- DMA read response 使用 VC1，避免和 command/event 混杂。
- DMA write / stream 使用 VC2。
- completion event 使用 VC0 或专用 sideband，不能被 VC2 大流量永久阻塞。
- collective 使用 VC3，不应与 DMA write 同 VC，除非 profile 明确证明不会影响 event/barrier。

### 5.5 关键时序路径

| 路径                   | 风险                            | 缓解                                             |
| ---------------------- | ------------------------------- | ------------------------------------------------ |
| descriptor validate    | 多字段 overflow/address compare | 分拍 validate，先 basic header 后 address rows。 |
| burst split            | boundary/alignment 组合复杂     | 预计算 row_end，burst boundary 用寄存器切片。    |
| outstanding tracker    | tag CAM 大                      | 限制 outstanding 或用 circular tag FIFO。        |
| data buffer write/read | 宽数据 mux                      | banked FIFO，read/write channel decouple。       |
| PMU bytes/stall update | counter fan-in                  | per-channel local counter，snapshot 汇总。       |
| fault encoder          | 多 source priority              | one-hot source latch，首 fault wins。            |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 参数

| 参数                         | 状态                    |
| ---------------------------- | ----------------------- |
| max_burst_bytes              | 由后续规格冻结          |
| max_outstanding_reads/writes | 由 PPA exploration 冻结 |
| data buffer depth            | 由 SRAM profile 冻结    |
| descriptor prefetch depth    | 由 PPA exploration 冻结 |
| rows/bytes bit width         | 由后续规格冻结          |
| address aperture count       | 由后续规格冻结          |
| alignment requirement        | 由后续规格冻结          |
| timeout granularity          | 由后续规格冻结          |
| NoC VC depth                 | 由 PPA exploration 冻结 |

### 6.2 性能模型

DMA copy 理想时间：

```text
T_dma = max(T_read, T_write) + T_desc + T_issue + T_completion
T_read  = bytes_total / read_bandwidth_effective
T_write = bytes_total / write_bandwidth_effective
bytes_total = bytes * rows
```

有效带宽受 outstanding、NoC congestion、memory latency、alignment、row stride 影响：

```text
BW_effective = BW_peak * efficiency
              * (1 - noc_backpressure_rate)
              * (1 - memory_stall_rate)
```

2D/strided copy 的短 row 会降低 burst efficiency；PMU 必须记录 average burst bytes 和 row_count，避免只看总 bytes。

### 6.3 PMU counters

| Counter                    | 说明                          | Primary stall owner       |
| -------------------------- | ----------------------------- | ------------------------- |
| dma_desc_count             | descriptor 完成数             | none                      |
| dma_bytes_read/write       | 实际读写字节                  | none                      |
| dma_read_req_count         | read burst 数                 | none                      |
| dma_write_req_count        | write burst 数                | none                      |
| dma_avg_burst_bytes        | burst efficiency              | none                      |
| dma_active_cycles          | launch 到 completion 活跃周期 | engine_active             |
| dma_wait_memory_cycles     | memory response 不足          | `ELENOR_STALL_DMA_MEMORY` |
| dma_wait_noc_vc1_cycles    | read response VC backpressure | `ELENOR_STALL_NOC_VC`     |
| dma_wait_noc_vc2_cycles    | write stream VC backpressure  | `ELENOR_STALL_NOC_VC`     |
| dma_data_fifo_full_cycles  | write side 慢导致 read 停     | DMA internal backpressure |
| dma_data_fifo_empty_cycles | read side 慢导致 write 停     | DMA internal backpressure |
| dma_timeout_count          | timeout                       | fault                     |
| dma_fault_invalid_desc     | descriptor 错误               | fault                     |
| dma_fault_address          | address/aperture 错误         | fault                     |

Global PMU 汇总时，DMA stall 只有在 DMA command 已 active 且等待 memory/NoC/data buffer 时归因给 DMA。若 Scheduler 未发 launch，则不算 DMA stall。

### 6.4 PPA 策略

- First Silicon V1 优先保证 descriptor validation、fault 精确性和 event ordering，不追求复杂 gather/multicast。
- data buffer 深度由 bandwidth-delay product 决定，具体由 SRAM profile 冻结。
- outstanding window 过大会增加 tag CAM 和验证面；可按 Edge/Balanced/High End profile 分档。
- strided copy 短 row 场景可用 row coalescing hint，但必须保持地址语义，hint 格式由后续规格冻结。
- DMA clock gating 以 launch_queue empty、no outstanding、data buffer empty 为条件。

### 6.4 Clock/reset/power/timing 考虑

- Global DMA 可与 Memory Controller、NoC、Scheduler 位于不同 clock domain；launch、read response、write response、event update 和 PMU sample 必须明确 CDC 边界。
- Reset/drain 时先关闭 launch_queue accept，再停止 descriptor prefetch，最后等待 read/write outstanding 清零或 timeout；data buffer 中未写出的 payload 必须被丢弃并记录 reset/error 状态。
- SRAM/FIFO macro 的 reset、sleep、retention 语义由 SRAM profile 冻结；DMA 不得在 macro sleep 期间接受新 descriptor。
- Clock gating 条件为 launch_queue empty、descriptor engine idle、outstanding counter 为零、data buffer empty、event update FIFO empty；任何条件跨域输入都必须同步。
- Timing closure 优先关注 address overflow/boundary check、burst split、outstanding tag match、data buffer mux、NoC ready fanout 和 PMU counter update。

## 7. RTL/软件实现建议

### 7.1 RTL 建议

- Descriptor validator 与 address generator 分离；validator 只产生合法 internal plan，generator 不再处理非法情况，减少重复判断。
- 所有地址加法使用扩展位检测 overflow。
- Fault 采用 first-fault-wins；后续错误计入 secondary counter，不覆盖首 fault record。
- Completion event 写入必须在所有 write response 完成后发生。
- Reset/drain 时停止接收新 launch，已发事务等待完成或 timeout；data buffer 中数据丢弃前必须记录 reset fault 或 reset event。
- Descriptor fetch 可以复用 read engine，但必须用高优先级或独立小通道，避免大 copy 饿死 descriptor fetch。

### 7.2 Firmware/runtime 建议

- Runtime 生成 DMA descriptor 时显式设置 rows/stride，不依赖硬件猜测 contiguous/2D。
- bytes、rows、stride 必须避免 32-bit 溢出；大 copy 由 runtime 拆分成多个 descriptor，拆分规则由后续规格冻结。
- Program/descriptor load 与 activation prefetch 使用不同 event_id，便于 fault 定位。
- 对 warm launch，runtime patch descriptor 后必须执行 descriptor cache invalidate/flush 规则。
- 多 context 下，DMA descriptor 必须带 context/domain，address fault 不应影响其他 context queue。

### 7.3 Assertions

- launch accepted 后最终 DONE/ERROR/TIMEOUT/RESET，除非 global reset。
- descriptor invalid 不允许产生 memory read/write burst。
- address overflow 必须 fault。
- completion event 不早于最后一个 write response。
- read/write outstanding counter 不下溢、不超过配置上限。
- reset/drain 后 no outstanding 且 data buffer empty 或状态被清除并记录。
- unsupported flags 必须 fault，不能忽略。

## 8. 验证、bring-up 和验收标准

### 8.1 单元验证

| 单元                 | 场景                                                                     |
| -------------------- | ------------------------------------------------------------------------ |
| Descriptor validator | legal 1D/2D/stride、zero bytes、zero rows、unsupported flags、bad size。 |
| Address generator    | overflow、unaligned、aperture crossing、boundary split、max row。        |
| Burst splitter       | aligned burst、short tail、page boundary、row boundary。                 |
| Read/write engines   | backpressure、out-of-order response、error response、timeout。           |
| Data buffer          | full/empty、tag match、reset with data pending。                         |
| Completion tracker   | all writes done、read error、write error、same-cycle completion/fault。  |
| PMU                  | bytes count、stall owner、overflow、snapshot。                           |

### 8.2 Bring-up sequence

1. CSR 读 DMA_CAP/DMA_STATUS，确认 idle。
2. 通过 Scheduler 提交 1D HBM->HBM 或 host->HBM copy，验证 completion event。
3. 提交 2D copy，rows/stride 覆盖不连续地址。
4. 提交 strided copy，src_stride 与 dst_stride 不同。
5. 注入 invalid descriptor size/flags，验证 invalid descriptor fault。
6. 注入 address fault，验证 fault record 包含 address 和 descriptor pointer。
7. 注入 memory/NoC backpressure，验证 DMA 不丢数据且 PMU stall 增加。
8. 注入 timeout，验证 event TIMEOUT 和 queue recovery。
9. reset/drain during active DMA，验证状态确定。

### 8.3 验收标准

- DMA 1D/2D/strided copy 与 golden memory compare 一致。
- DMA completion event、error event、timeout event 与 command_id/context_id/fault_slot 对齐。
- DMA 不能绕过 command queue 成为不可审计 side path；debug CSR launch 仅用于 bring-up。
- invalid descriptor/address/timeout 不产生 silent data corruption。
- PMU 能解释 DMA bandwidth、memory stall、NoC VC stall、descriptor fault。
- event / queue / DMA / reset / timeout 系统稳定性先于复杂 engine 扩展完成。
- CDC/RDC、FIFO/arbiter formal、SVA protocol assertion 覆盖 DMA 关键路径。

## 9. 风险、取舍和后续细化方向

| 风险                           | 影响                                   | 缓解                                                               |
| ------------------------------ | -------------------------------------- | ------------------------------------------------------------------ |
| DMA 承担 MFE 动态内存职责      | ownership 混乱，paged attention 难验证 | DMA 只做确定地址 copy；page/segment walk 和 stream fill 属于 MFE。 |
| Completion 早于数据可见        | host/runtime 读到旧数据                | write response 完成后才 signal event，ordering SVA 覆盖。          |
| Strided copy 溢出或跨 aperture | 数据破坏或安全问题                     | 扩展位 overflow 检测、aperture checker、fault record。             |
| Outstanding 过大               | tag CAM/时序/验证复杂                  | First Silicon 限制 window，profile 分档。                          |
| NoC data 阻塞 event            | 系统 hang 或误 timeout                 | VC1/VC2 与 VC0 分离，event sideband 或高优先级。                   |
| PMU 只统计 bytes 不统计效率    | 无法解释短 row/stride 性能             | average burst、row_count、stall by memory/NoC/data_fifo。          |
| reset/drain 丢失 inflight 状态 | recovery 不确定                        | stop new launch、drain or timeout、fault/event 顺序固定。          |

后续规格需要冻结：DMA descriptor binary layout、zero-length copy 语义、alignment/boundary 规则、address aperture 表、host coherent policy、max burst/outstanding/data buffer depth、unsupported flags policy、completion/fault 同周期优先级、debug launch 权限、PMU counter id、reset/drain 精确行为和 multicast/gather list 的阶段边界。
