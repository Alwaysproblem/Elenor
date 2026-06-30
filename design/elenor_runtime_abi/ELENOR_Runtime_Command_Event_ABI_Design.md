# ELENOR Runtime Command/Event ABI 设计文档

## 1. 定位、目标和 First Silicon cutline

Runtime Command/Event ABI 是 host runtime、kernel driver、firmware runtime 和硬件队列之间的二进制契约。它把 compiler 生成的 executable package 转换成 device 可消费的 command sequence、descriptor pointer、event dependency、fault record 和 PMU attribution 信息。

设计原则：

- 硬件执行 command、descriptor 和 program，不执行高层 graph。
- Command/Event ABI 只表达 launch、wait、signal、barrier、DMA、engine descriptor、reset 和 fault；不承载高层 tensor algebra。
- ABI v0 结构体是样例，不是最终冻结定义；field、alignment、endianness、size、versioning、兼容策略由后续规格冻结。
- Architecture V1 定义长期边界；First Silicon V1 优先打通 command queue、event、barrier、DMA、BOA GEMM、fault record 和 PMU basic counter。

First Silicon V1 切线：

| 领域            | 必须实现                                                                         | 可预留                            |
| --------------- | -------------------------------------------------------------------------------- | --------------------------------- |
| Command queue   | fixed-size ring、doorbell、sequence、context_id、queue_id                        | priority/preemption               |
| Command type    | launch_group_task、DMA、BOA descriptor、event wait/signal、barrier、reset_domain | full MFE Segment、multi-model QoS |
| Event table     | pending/done/error/timeout/reset、producer、sequence、timestamp                  | sampled trace                     |
| Fault record    | invalid descriptor、address fault、timeout、engine fault                         | NoC poison、ECC policy            |
| Memory ordering | command fetch fence、descriptor visibility、completion visibility                | relaxed ordering hint             |
| PMU             | command queue occupancy、event wait、DMA bandwidth、BOA active/stall             | full stall taxonomy               |

## 2. 职责、非职责和 ownership

### 2.1 职责

Runtime ABI 负责：

1. 定义 host 到 device 的 command binary layout、queue protocol、doorbell protocol 和 completion event。
2. 定义 event id、event sequence、event status、timeout 和 wait/signal 语义。
3. 定义 descriptor pointer、descriptor bytes、validation checksum、version 和 fault reporting。
4. 定义 context isolation：context_id、queue_id、IOMMU domain、privilege flags。
5. 定义 reset domain：tile、group、device 和 queue drain 的可观察行为。
6. 定义 runtime API 与 kernel driver ioctl/firmware command 的映射。

### 2.2 非职责

Runtime ABI 不负责：

- 不定义 MLIR dialect 的完整语法。
- 不规定 BOA/EVU/MFE/USE 内部微架构。
- 不在 ABI 中嵌入任意 graph interpreter。
- 不替代 executable package 的 section format。
- 不让 USE 承担 program control；program control 归 UCE/Tile Group Sequencer/Runtime，USE 管 state/scan/recurrence。

### 2.3 Ownership

| 对象           | Owner                    | Producer                  | Consumer         | 备注                                     |
| -------------- | ------------------------ | ------------------------- | ---------------- | ---------------------------------------- |
| Command ring   | Kernel driver / firmware | User runtime              | Device Runtime   | driver 建立，runtime 填充，firmware 消费 |
| Doorbell       | Kernel driver            | User runtime 或 driver    | Host interface   | 必须有 ordering fence                    |
| Event table    | Firmware runtime         | Device Runtime / engines  | Runtime / driver | event_id + sequence 防复用错误           |
| Fault record   | Firmware runtime         | Firmware / engine / MMU   | Driver / runtime | 记录 command_id、descriptor、domain      |
| Descriptor ABI | Compiler + firmware      | Runtime patch 后提交      | Engines          | v0 样例，最终由后续规格冻结              |
| Reset domain   | Firmware / driver        | Runtime 请求或 fault 触发 | Hardware queues  | drain 语义必须确定                       |

## 3. 微架构和状态机

### 3.1 Command queue 状态机

```text
Empty
  -> HostWritesCommand
  -> HostPublishesTail
  -> DoorbellRung
  -> FirmwareFetch
  -> Validate
  -> Dispatch
  -> WaitingDependencies
  -> Running
  -> Completing
  -> EventSignaled
  -> Retired
```

错误路径：

```text
Validate -> Faulted -> EventError -> QueueStoppedOrDrained
Running  -> Timeout -> EventTimeout -> ResetDomainDecision
Running  -> EngineFault -> EventError -> FaultRecordVisible
```

关键不变量：

- host 在更新 tail/doorbell 前必须保证 command 和 descriptor bytes 对 device 可见。
- firmware fetch command 后必须校验 `abi_version`、`cmd_size`、`type`、`context_id`、`queue_id`、descriptor range 和 timeout policy。
- event signal 必须在所有被该 event 表示的 memory write 对 host/runtime 可见之后发生。
- queue reset 后 pending command、credit、event 和 fault record 状态必须 deterministic。

### 3.2 Event 状态机

```text
FREE -> PENDING -> DONE
              |-> ERROR
              |-> TIMEOUT
              |-> RESET
```

Event 复用规则：

- `event_id` 定位 table entry，`sequence` 区分复用轮次。
- wait 方必须匹配 expected sequence，不能只检查 status。
- reset 将未完成 event 写为 RESET 或 ERROR，具体策略由后续规格冻结，但不能保持静默 pending。

### 3.3 Fault 状态机

```text
NoFault
  -> DetectFault
  -> FreezeProducer
  -> WriteFaultRecord
  -> SignalErrorEvent
  -> NotifyDriver
  -> DrainOrReset
  -> RecoverOrDestroyContext
```

Fault handling 不允许吞错后继续使用同一 descriptor。runtime 可以选择 reset tile、reset group 或 reset device，但选择必须受 fault domain 和 isolation policy 约束。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Object model

```text
RuntimeContext
  context_id
  iommu_domain
  queues[]
  event_table
  fault_record_ring
  loaded_packages[]
  pmu_session

CommandQueue
  queue_id
  ring_base_iova
  ring_entries
  head
  tail
  doorbell_id
  priority
  state

Command
  header
  descriptor_ref
  wait_refs[]  // event_id + expected_sequence
  signal_event + signal_sequence
  timeout_policy
  fault_record_slot

Event
  event_id
  sequence
  status
  producer
  timestamp
  error_code
```

### 4.2 ABI v0 command layout 示例

```c
#define ELENOR_ABI_VERSION_EXAMPLE 1u

typedef struct {
    uint32_t event_id;
    uint32_t expected_sequence;
} elenor_event_ref_v0_t;

typedef struct {
    uint16_t abi_version;
    uint16_t cmd_size;
    uint16_t type;
    uint16_t flags;

    uint32_t context_id;
    uint32_t queue_id;

    uint64_t desc_iova;
    uint32_t desc_bytes;
    uint32_t desc_crc_or_zero;

    uint64_t wait_ref_iova;        /* elenor_event_ref_v0_t[]; 0 means no waits */
    uint32_t wait_ref_count;
    uint32_t wait_ref_crc_or_zero;

    uint32_t signal_event;
    uint32_t signal_sequence;

    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_command_v0_t;
```

Command type 示例：

```c
typedef enum {
    ELENOR_CMD_LAUNCH_GROUP_TASK = 1,
    ELENOR_CMD_DMA = 2,
    ELENOR_CMD_BOA_GEMM = 3,
    ELENOR_CMD_EVU_KERNEL = 4,
    ELENOR_CMD_MFE_PAGE_STREAM = 5,
    ELENOR_CMD_MFE_SEGMENT_STREAM = 6,
    ELENOR_CMD_USE_SCAN = 7,
    ELENOR_CMD_USE_UPDATE = 8,
    ELENOR_CMD_BARRIER = 9,
    ELENOR_CMD_BRANCH_SHAPE = 10,
    ELENOR_CMD_EVENT_WAIT = 11,
    ELENOR_CMD_EVENT_SIGNAL = 12,
    ELENOR_CMD_RESET_DOMAIN = 13,
} elenor_cmd_type_v0_t;
```

最小 ABI 必须预留：ABI version、command size、context id、queue id、address space/IOMMU domain、descriptor length、descriptor checksum/validation mode、timeout、memory ordering、completion event、fault record pointer、privilege/isolation flags。

### 4.3 Event layout 示例

```c
typedef enum {
    ELENOR_EVENT_PENDING = 0,
    ELENOR_EVENT_DONE = 1,
    ELENOR_EVENT_ERROR = 2,
    ELENOR_EVENT_TIMEOUT = 3,
    ELENOR_EVENT_RESET = 4,
} elenor_event_status_v0_t;

typedef struct {
    uint32_t event_id;
    uint32_t status;
    uint32_t producer_id;
    uint32_t sequence;
    uint32_t error_code;
    uint32_t timestamp_lo;
    uint32_t timestamp_hi;
} elenor_event_v0_t;
```

### 4.4 Fault record 示例

```c
typedef struct {
    uint32_t fault_id;
    uint32_t context_id;
    uint32_t queue_id;
    uint32_t command_index;
    uint32_t command_type;
    uint32_t event_id;
    uint32_t event_sequence;
    uint32_t fault_code;
    uint64_t descriptor_iova;
    uint32_t descriptor_offset;
    uint32_t producer_id;
    uint32_t domain; /* tile/group/device/queue */
    uint32_t detail0;
    uint32_t detail1;
} elenor_fault_record_v0_t;
```

Fault code 示例：invalid descriptor、unsupported ABI、address fault、DMA timeout、event dependency timeout、stream protocol error、engine internal fault、reset during execution。

### 4.5 Runtime APIs

```c
int elenor_context_create(elenor_device_t *dev, elenor_context_t **ctx);
int elenor_queue_create(elenor_context_t *ctx, const elenor_queue_attr_t *attr, elenor_queue_t **queue);
int elenor_submit(elenor_queue_t *queue, const elenor_command_v0_t *cmds, uint32_t num_cmds);
int elenor_wait(elenor_context_t *ctx, uint32_t event_id, uint32_t sequence, uint64_t timeout_ns);
int elenor_event_query(elenor_context_t *ctx, uint32_t event_id, elenor_event_v0_t *out);
int elenor_fault_read(elenor_context_t *ctx, uint32_t slot, elenor_fault_record_v0_t *out);
int elenor_reset_domain(elenor_context_t *ctx, const elenor_reset_request_t *req);
int elenor_read_counter(elenor_context_t *ctx, uint32_t counter_id, uint64_t *value);
```

Kernel driver 可以把这些 API 映射成 ioctl 或 shared queue doorbell。API 语义必须保证 command bytes 与 descriptor bytes 在 submit 前可见，event completion 后 output buffer 对 host 可见。

## 5. 数据流、控制流和时序路径

### 5.1 Runtime load/patch/submit flow

```text
User runtime:
  load executable package
  validate ABI tuple
  allocate buffers and event table entries
  patch descriptor context/base IOVA
  write command sequence
  publish queue tail
  ring doorbell

Kernel driver:
  pin memory and map IOVA
  enforce context isolation
  deliver interrupt or poll completion

Firmware runtime:
  fetch command
  validate command and descriptor
  dispatch group task or engine command
  update event table
  write fault record on error

Hardware:
  consume descriptor/program
  execute DMA/BOA/EVU/MFE/USE work
  signal engine completion
```

### 5.2 Descriptor visibility protocol

提交前必须满足：

```text
host writes descriptor
host writes command.desc_iova/desc_bytes
host memory fence
host updates queue tail
host rings doorbell
firmware reads command
firmware validates descriptor
```

Warm launch 中如果 descriptor cache 可能保留旧数据，runtime 必须执行明确的 descriptor invalidate/flush command 或使用 sequence-tagged descriptor cache。具体机制由后续规格冻结。

### 5.3 Wait/signal protocol

- `wait_ref_iova + wait_ref_count` 表达等待集合；每个元素必须携带 `event_id + expected_sequence`。
- `signal_event + signal_sequence` 在 command 成功完成时写 DONE；失败写 ERROR/TIMEOUT/RESET，同一 event id 的旧 sequence 不得覆盖新轮次。
- 连续 event id 只是 runtime 分配优化；硬件 wait 方不得只依赖 `event_id` 或 base/count 推断正确性。
- timeout 计数域必须声明：device cycle、queue cycle 或 wall-clock tick 由后续规格冻结。

### 5.4 Dynamic shape branch

Runtime dynamic shape 采用三层：compiler multi-versioning、runtime branch_shape、EVU mask/MFE descriptor 处理 tail/ragged。

```c
if (seq_len <= profile->small_seq_limit) {
    elenor_submit(queue, small_cmds, small_count);
} else if (seq_len <= profile->medium_seq_limit) {
    elenor_submit(queue, medium_cmds, medium_count);
} else {
    elenor_submit(queue, paged_cmds, paged_count);
}
```

`small_seq_limit`、`medium_seq_limit` 等阈值由后续规格冻结或由 compiler profile 生成。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 ABI 配置字段

| 字段                    | 目的                       | 冻结方式                |
| ----------------------- | -------------------------- | ----------------------- |
| queue_depth             | command ring 容量          | 由后续规格冻结          |
| max_wait_refs           | 单 command 直接等待 ref 数 | 由后续规格冻结          |
| event_table_entries     | event table 大小           | 由后续规格冻结          |
| fault_record_slots      | fault ring 大小            | 由后续规格冻结          |
| descriptor_alignment    | descriptor fetch 对齐      | 由后续规格冻结          |
| command_cacheline_bytes | command ring stride        | 由后续规格冻结          |
| doorbell_latency        | submit latency model       | 由 PPA exploration 冻结 |

### 6.2 Performance model

```text
T_submit = T_write_cmd + T_patch_desc + T_fence + T_doorbell
T_dispatch = T_fw_fetch + T_validate + T_dependency_wait + T_issue
T_completion = T_engine_done + T_event_write + T_interrupt_or_poll
```

PMU 必须能解释：

- command queue occupancy。
- firmware validate stall。
- event wait cycles。
- descriptor fetch stall。
- DMA bandwidth。
- engine active/stall。
- stream queue credit empty/full。
- reset/drain cycles。

### 6.3 PMU 唯一归因

每个 stall cycle 只能有一个 primary owner。推荐归因顺序：engine_active、engine_wait_event、engine_wait_operand、stream_credit、SRAM bank conflict、NoC backpressure、DMA wait memory、UCE program/descriptor stall、unknown。secondary tag 可用于 debug，但不进入 primary utilization 统计。

## 7. RTL/软件实现建议

### 7.1 Driver

- command ring 使用 cacheline-aligned entry，禁止 host 与 firmware 同时写同一字段。
- context 创建时建立 IOMMU domain、event table、fault ring、PMU access mask。
- submit path 只做必要校验，深度 ABI 校验尽量在 package load 或 firmware validate 中完成。
- interrupt handler 只采集 completion/fault summary，复杂恢复交给 threaded handler 或 runtime。

### 7.2 Firmware

- validate command header 后再读取 descriptor。
- unsupported ABI 或 descriptor size mismatch 必须生成 fault record。
- timeout path 必须先冻结受影响 queue，再决定 drain/reset。
- reset domain 必须清理 stream queue credit、pending event 和 local descriptor cache。

### 7.3 RTL

- doorbell write 与 command fetch 之间需要 ordering guarantee。
- event table write 需要原子地更新 status 和 sequence，避免 host 读到混合状态。
- fault record 写入必须先于 ERROR event 对 host 可见，或 event 中携带可重试读取机制。
- command/event NoC traffic 应使用独立 VC，避免大 DMA 阻塞 control plane。

### 7.4 Pass pipeline/dialect 策略

Runtime ABI 由 compiler 后端生成 command template，再由 runtime 实例化：

```text
stablehlo-to-linalg
shape-specialize
elenor-engine-partition
elenor-kernel-library-select
elenor-descriptor-template
elenor-runtime-dialect-lowering
elenor-command-buffer-pack
elenor-abi-validate
```

Runtime dialect 只表达可打包为 ABI 的对象：

```mlir
elenor.runtime.command @decode_step {
  type = #elenor.cmd<launch_group_task>
  task = @paged_attention_group_task
  wait = [#elenor.event<input_ready>]
  signal = #elenor.event<decode_done>
  timeout = #elenor.timeout<profile_default>
}
```

Descriptor dialect 不应读取 runtime queue 细节；Runtime dialect 不应重写 BOA/EVU/MFE/USE engine 语义。

### 7.5 Kernel library strategy

ABI 中的 command 指向 program_id/kernel_id，而不是内联 tile microcode。kernel library 条目必须包含：kernel_id、kernel_abi_version、required_descriptor_abi、slot_frame_abi、supported dtype/layout、PMU fingerprint。runtime submit 时校验 package 选择的 kernel 与 firmware resident kernel 是否兼容。

## 8. 验证、bring-up 和验收标准

### 8.1 Golden tests

| 测试                  | 覆盖                             | 期望                           |
| --------------------- | -------------------------------- | ------------------------------ |
| command layout golden | struct packing、endianness、size | bytes 与 golden 一致           |
| event sequence reuse  | event_id 复用                    | wait 不误判旧 completion       |
| descriptor mismatch   | desc_bytes/version 错误          | ERROR event + fault record     |
| queue ring wrap       | head/tail wrap                   | command 不丢失不重复           |
| timeout               | command 永不完成                 | TIMEOUT event + reset decision |
| reset drain           | pending event/stream credit      | deterministic RESET/ERROR      |
| memory ordering       | descriptor patch 后 submit       | firmware 读到新 descriptor     |
| PMU attribution       | wait/operand/stream stall        | primary owner 唯一             |

### 8.2 Verification plan

1. C ABI pack/unpack 单测：header、command、event、fault record。
2. Firmware queue model：random command sequence、wait/signal/barrier、fault injection。
3. Driver/runtime integration：submit、poll、interrupt、reset、fault read。
4. RTL smoke：command queue + event + DMA 1D/2D + PMU。
5. BOA through command queue：禁止 testbench 绕过 runtime 直接拉 datapath。
6. Paged attention trace：MFE Page Stream、EOS/error token、BOA/EVU event chain。
7. Multi-context isolation：context_id、queue_id、IOMMU fault、fault domain。

### 8.3 验收标准

- command/event/DMA/PMU basic loop 独立闭环。
- 所有 ABI 错误都有可读 fault record，不允许 silent hang。
- event wait 支持 timeout，timeout 后 reset/drain 行为确定。
- descriptor validation 能区分 unsupported version、bad size、bad address、bad checksum。
- runtime submit flow 与 executable package load/patch flow 一致。

## 9. 风险、取舍和后续细化方向

| 风险                            | 影响                         | 缓解                                                           |
| ------------------------------- | ---------------------------- | -------------------------------------------------------------- |
| ABI v0 冻结过早                 | 后续 descriptor 扩展破坏兼容 | version、size、flags、reserved 字段分层                        |
| Event 复用错误                  | false completion，数据损坏   | event_id + sequence，wait 必须匹配 sequence                    |
| Descriptor cache stale          | warm launch 使用旧参数       | flush/invalidate 或 sequence-tagged cache                      |
| Reset 语义不清                  | fault 后无法恢复             | tile/group/device reset domain 明确定义                        |
| Command queue 被 DMA 阻塞       | control plane deadlock       | NoC VC 分离，command/event 优先级                              |
| Fault record 不完整             | driver 无法恢复              | 强制记录 context、queue、command、descriptor、producer、domain |
| Runtime/firmware 重复解释 graph | 复杂度失控                   | ABI 只接收 command/descriptor/program                          |

后续必须冻结：command ring exact entry stride、doorbell register semantics、event table memory ordering、fault code 编码、timeout clock domain、descriptor cache coherence、reset/drain policy 和 multi-context privilege flags。未冻结参数统一写为由后续规格冻结。
