# ELENOR Driver/Firmware/Runtime 设计文档

## 1. 定位、目标和 First Silicon cutline

Driver/Firmware/Runtime 是 ELENOR 软件控制面。它把 compiler 生成的 executable package 装载到 device，完成内存分配、IOMMU 映射、descriptor patch、command submit、event wait、fault recovery、PMU 读取和 multi-context 隔离。

分层原则：

```text
Framework runtime
  -> Compiler generated executable package
  -> User-mode runtime library
  -> Kernel driver
  -> Firmware / RISC-V runtime
  -> ELENOR hardware queues
```

First Silicon V1 目标不是完整 QoS scheduler，而是可靠闭环：command -> descriptor validation -> DMA -> BOA/EVU/MFE/USE launch subset -> event -> fault record -> PMU。

First Silicon V1 切线：

| 层                 | 必须实现                                                                            | 预留                                    |
| ------------------ | ----------------------------------------------------------------------------------- | --------------------------------------- |
| User runtime       | package load、buffer bind、descriptor patch、submit/wait、PMU read                  | graph-level dynamic scheduler           |
| Kernel driver      | device init、memory pin/map、command queue、interrupt、fault read、reset            | preemption、多租户 QoS                  |
| Firmware runtime   | command consume、descriptor validation、event scheduling、shape branch、reset/drain | priority scheduling、advanced profiling |
| Hardware interface | doorbell、event table、fault record、PMU counter                                    | sampled trace、complex reset policy     |

ABI v0 结构体只作为样例，不是最终冻结定义；field、alignment、queue depth、timeout unit、fault code 和 ioctl 编码由后续规格冻结。

## 2. 职责、非职责和 ownership

### 2.1 User-mode runtime 职责

- 打开并验证 executable package。
- 创建 context 和 command queue。
- 分配/绑定 device buffer、weight、workspace、KV cache 和 state buffer。
- 执行 context-level descriptor patch：base IOVA、context_id、queue_id、event base、fault slot。
- 注册 `program_id -> section_id / iova / version / hash` 元数据，供 launch descriptor 引用。
- 选择 dynamic shape multi-version command path 和 residency hint。
- 提交 command buffer，等待 event，读取 fault record 和 PMU counter。

### 2.2 Kernel driver 职责

- device discovery、BAR/AXI resource init、firmware boot。
- memory allocation、pinning、IOMMU mapping、cache coherence hook。
- command queue creation、doorbell mapping、interrupt registration。
- context isolation、privilege check、reset domain mediation。
- fault notification、PMU access control、debug snapshot。

### 2.3 Firmware runtime 职责

- command queue consume。
- command/descriptor/package ABI tuple validation。
- event table update、barrier、wait/signal。
- shape branch dispatch。
- GroupTaskLaunchDesc patch/fanout、program residency contract orchestration、reset/drain epoch 管理。
- fault record 写入、queue stop、reset/drain。
- profiling snapshot 与 PMU attribution aggregation。

### 2.4 非职责

- Driver 不做 graph lowering。
- Firmware 不解释高层 graph，不生成 tile program。
- User runtime 不绕过 ABI 直接控制 BOA datapath。
- USE 不负责 program control；USE 管 state/scan/recurrence，Tile UCE 管 tile program PC、launch、wait、branch 和 descriptor patch。
- MFE 管数据相关动态内存访问，例如 page table walk、segment offset、stream fill。

### 2.5 Ownership matrix

| 对象               | User runtime       | Kernel driver | Firmware                    | Hardware               |
| ------------------ | ------------------ | ------------- | --------------------------- | ---------------------- |
| Package validation | ABI/manifest 初验  | 安全策略      | command-time 校验           | 不关心 package         |
| IOVA binding       | 请求/记录          | 建立/销毁     | 校验访问                    | 发起读写               |
| Descriptor patch   | context/base/shape | 不改语义      | tile/group/local patch 协调 | engine 消费            |
| Command ring       | 填 entry           | 分配保护      | fetch/update head           | doorbell/queue logic   |
| Event table        | wait/query         | map/protect   | 写状态                      | engine completion 输入 |
| Fault record       | 读取解释           | 通知/恢复     | 写 record                   | fault source           |
| PMU                | session/query      | 权限/汇聚     | snapshot                    | counter source         |

## 3. 微架构和状态机

### 3.1 Context 生命周期

```text
Uncreated
  -> Created
  -> MemoryBound
  -> PackageLoaded
  -> QueueReady
  -> Running
  -> Faulted
  -> Recovering
  -> Destroyed
```

`Faulted` 不等于 device 全局不可用。fault domain 可以是 command、queue、tile、group、device。runtime 必须根据 fault record 决定 reset domain。

### 3.2 Submit 状态机

```text
PrepareBindings
  -> PatchDescriptors
  -> BuildCommands
  -> PublishRing
  -> RingDoorbell
  -> FirmwareValidate
  -> DispatchGroupTask
  -> WaitEvents
  -> CompleteOrFault
```

### 3.3 Firmware command loop

```text
while queue is enabled:
  read queue head/tail
  fetch command
  validate header and descriptor
  resolve dependencies
  dispatch command
  monitor timeout
  update event or fault record
  advance head
```

Firmware validate 必须在 dispatch 前完成，不能让 engine 在明显非法 descriptor 上启动。

### 3.4 Reset/drain 状态机

```text
FaultDetected
  -> StopAffectedQueue
  -> FreezeNewDispatch
  -> DrainSafeCommands
  -> MarkPendingEvents
  -> ResetTileOrGroupOrDevice
  -> ClearStreamCreditAndDescriptorCache
  -> ResumeOrDestroyContext
```

reset 后必须处理：stream token、credit、pending event、local descriptor cache、program residency metadata、PMU snapshot。未定义状态会造成下一次 warm launch 不可信。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Object model

```text
ElenorDevice
  caps
  firmware_version
  memory_regions
  global_pmu

ElenorContext
  context_id
  iommu_domain
  queues[]
  event_table
  fault_ring
  loaded_packages[]
  state_handles[]

ElenorQueue
  queue_id
  priority
  ring
  doorbell
  state

LoadedPackage
  package_id
  program_registry
  descriptor_table
  relocation_bindings
  hot_program_policy

RuntimeLaunch
  entry_id
  command_span
  event_span
  timeout_policy
```

### 4.2 Binary layout/versioning 示例

Driver/Firmware 共享的 control-plane layout 必须带 version 与 size。以下结构体是 ABI v0 样例，不是最终冻结定义。

```c
typedef struct {
    uint16_t abi_version;
    uint16_t struct_bytes;
    uint32_t context_flags;
    uint32_t requested_queues;
    uint32_t event_table_entries;
    uint32_t fault_record_slots;
    uint64_t user_va_or_handle;
} elenor_create_context_v0_example_t;

typedef struct {
    uint16_t abi_version;
    uint16_t struct_bytes;
    uint32_t queue_id;
    uint32_t command_count;
    uint64_t command_ring_iova;
    uint32_t tail;
    uint32_t submit_sequence;
} elenor_submit_v0_example_t;
```

Versioning 规则：driver ioctl、firmware mailbox、command ABI 和 package ABI 分开编号；kernel driver 必须拒绝 unsupported major version，minor-compatible 变更只能追加字段并依赖 `struct_bytes` 做 bounds check。exact ioctl number、mailbox opcode、cacheability 和 alignment 由后续规格冻结。

### 4.3 Driver ioctl/API 示例

```c
int elenor_ioctl_create_context(int fd, struct elenor_create_context *arg);
int elenor_ioctl_alloc_memory(int fd, struct elenor_alloc_memory *arg);
int elenor_ioctl_map_iova(int fd, struct elenor_map_iova *arg);
int elenor_ioctl_create_queue(int fd, struct elenor_create_queue *arg);
int elenor_ioctl_submit(int fd, struct elenor_submit *arg);
int elenor_ioctl_wait_event(int fd, struct elenor_wait_event *arg);
int elenor_ioctl_read_fault(int fd, struct elenor_read_fault *arg);
int elenor_ioctl_reset_domain(int fd, struct elenor_reset_domain *arg);
int elenor_ioctl_read_pmu(int fd, struct elenor_read_pmu *arg);
```

### 4.4 User runtime API 示例

```c
int elenor_runtime_create(elenor_device_t *dev, elenor_runtime_t **rt);
int elenor_load_package(elenor_runtime_t *rt, const void *pkg, uint64_t bytes, elenor_loaded_package_t **out);
int elenor_bind_buffer(elenor_loaded_package_t *pkg, uint32_t binding_id, elenor_buffer_t *buf);
int elenor_patch_descriptors(elenor_loaded_package_t *pkg, const elenor_launch_params_t *params);
int elenor_launch(elenor_loaded_package_t *pkg, uint32_t entry_id, const elenor_launch_params_t *params, elenor_event_handle_t *done);
int elenor_wait_event(elenor_runtime_t *rt, elenor_event_handle_t event, uint64_t timeout_ns);
int elenor_read_pmu_snapshot(elenor_runtime_t *rt, elenor_pmu_snapshot_t *snapshot);
```

### 4.5 Doorbell/register 协议示例

```c
typedef struct {
    uint32_t queue_id;
    uint32_t tail;
    uint32_t sequence;
    uint32_t flags;
} elenor_doorbell_write_v0_t;
```

Doorbell 前要求 host memory fence；doorbell 后 firmware 从 ring fetch command。doorbell register exact offset、write combining policy、cacheability 由后续规格冻结。

### 4.6 Descriptor patch protocol

Patch 分层：

| Patch 类型                    | Owner                   | 时机                  |
| ----------------------------- | ----------------------- | --------------------- |
| static shape/dtype/layout     | Compiler                | package build         |
| context/base IOVA/residency   | User runtime / firmware | load/bind             |
| tile_id/group_id/slot offset  | Tile UCE auto-patch     | tile launch           |
| page list/segment offset      | MFE                     | data stream execution |
| state slot/checkpoint pointer | USE / Tile UCE          | state lifecycle       |

runtime 只 patch 已声明 relocation entry，禁止扫描 arbitrary descriptor bytes。

## 5. 数据流、控制流和时序路径

### 5.1 Runtime load/patch/submit flow

```text
1. runtime opens package and validates ABI tuple
2. driver creates context, queue, event table and IOMMU domain
3. runtime allocates device buffers for program, descriptor, weight and workspace
4. runtime uploads immutable sections
5. runtime patches context-level relocations
6. runtime builds command buffer from command templates
7. driver publishes ring tail and rings doorbell
8. firmware fetches and validates commands
9. firmware dispatches group task
10. Tile Group Sequencer dispatches prepared tile tasks
11. Tile UCE launches DMA/BOA/EVU/MFE/USE and waits events
12. firmware signals completion event or fault event
13. runtime waits, reads PMU/fault if needed
```

### 5.2 Cold vs warm path

Cold path：launch descriptor 触发 Tile Group Program Residency Manager 的 miss fetch/verify/install；runtime/firmware 只提供 `program_id/program_iova/version/hash`，不发显式 load 指令。

Warm path：program resident，只更新 descriptor/event/context/shape metadata。warm path 不允许 patch running program text。若 descriptor cache 有 stale 风险，必须做 invalidate/flush 或 sequence-tagged descriptor。

### 5.3 Interrupt vs polling

- latency-sensitive small batch 可 polling event table，但必须有 timeout。
- long-running group task 使用 interrupt 或 hybrid polling。
- fault interrupt 优先级高于 normal completion。
- PMU sampling 不应阻塞 command/event VC。

### 5.4 Fault handling flow

```text
engine or firmware detects fault
  -> write fault record
  -> mark event ERROR/TIMEOUT/RESET
  -> interrupt driver
  -> runtime reads fault record
  -> runtime decides reset tile/group/device/context destroy
  -> driver issues reset_domain
  -> firmware drains queue and clears local state
```

错误类型：invalid descriptor、unsupported ABI、address fault、DMA timeout、event deadlock timeout、stream protocol error、engine internal fault、optional SRAM ECC、optional NoC poison。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 Driver/runtime 配置

| 配置                | 含义                                                               | 冻结方式             |
| ------------------- | ------------------------------------------------------------------ | -------------------- |
| max_contexts        | 最大并发 context                                                   | 由后续规格冻结       |
| queues_per_context  | 每 context queue 数                                                | 由后续规格冻结       |
| command_ring_depth  | ring entry 数                                                      | 由后续规格冻结       |
| event_table_entries | event 数                                                           | 由后续规格冻结       |
| fault_record_slots  | fault ring 槽位                                                    | 由后续规格冻结       |
| max_pinned_bytes    | pin memory 上限                                                    | 由后续规格冻结       |
| tile_l1_bytes       | slot frame 校验                                                    | 由 SRAM profile 冻结 |
| group_sram_bytes    | program residency cache、descriptor window 和 stream buffer sizing | 由 SRAM profile 冻结 |

### 6.2 Submit latency model

```text
T_submit_host = T_patch_desc + T_write_cmd + T_fence + T_doorbell
T_fw = T_fetch + T_validate + T_dependency_wait + T_dispatch
T_total_launch = T_submit_host + T_fw + T_residency_miss
```

Warm launch 应主要受 `T_patch_desc + T_write_cmd + T_doorbell` 影响；cold launch 额外受 residency miss 的 fetch/verify/install 延迟和 descriptor upload 影响。

### 6.3 PMU

Driver/Firmware 需要暴露：

- command queue occupancy。
- event wait cycles。
- firmware validate cycles。
- descriptor fetch/patch stall。
- DMA bandwidth。
- engine active/stall。
- stream queue occupancy、credit empty/full。
- SRAM bank conflict。
- NoC congestion by VC。
- reset/drain cycles。

PMU snapshot 必须带 context_id、queue_id、time window、counter version。counter exact id 由后续规格冻结。

## 7. RTL/软件实现建议

### 7.1 Runtime library

- 以 package entry 为 launch API，不暴露硬件内部 engine launch 给 framework。
- 使用 binding table 管理 tensor/state/KV cache，避免每次 submit 构造完整 descriptor。
- dynamic shape 选择 compiler 生成的多版本 command path。
- fault 后默认停止复用 warm descriptor，直到 reset/drain 成功。

### 7.2 Kernel driver

- 使用 per-context IOMMU domain，queue/fault/event memory 不跨 context 共享。
- mmap doorbell 时限制权限，防止用户写其他 queue。
- interrupt handler 只做最小确认和唤醒，复杂恢复在安全上下文执行。
- reset_domain 需要 firmware ack，不能 host 单方面认为恢复完成。

### 7.3 Firmware

- command loop 小而确定，避免引入复杂 graph scheduler。
- Group task scheduling 以 group task 为粒度，不逐 tile 做全局调度。
- descriptor validation 失败必须在启动 engine 前报告。
- profiling aggregation 不得改变 command/event 可观察顺序。

### 7.4 Pass pipeline/dialect 策略

Driver/Firmware/Runtime 消费 compiler 产物，不执行 lowering，但需要定义与 Runtime dialect 对接的 ABI：

```text
elenor-runtime-dialect
  -> command template
  -> event dependency graph
  -> descriptor relocation table
  -> package manifest
  -> user runtime load/patch/submit
```

Clean MLIR-like 示例：

```mlir
elenor.runtime.entry @decode_token(%input, %kv, %state) {
  %shape_class = elenor.runtime.shape_class %input.seq_len
  elenor.runtime.branch %shape_class {
    case #elenor.shape<short>: elenor.runtime.launch @attention_short
    case #elenor.shape<long>:  elenor.runtime.launch @attention_paged
  }
}
```

### 7.5 Kernel library strategy

Runtime 不生成 tile program，而选择 package 中绑定的 tile kernel library 条目。Firmware 管理 resident kernel cache：matmul_boa_v1、evu_unary_v1、softmax_v1、page_attention_qk_v1、page_attention_pv_v1、fallback_use_v1 等。kernel 版本、descriptor ABI、slot frame ABI 必须同时匹配。

## 8. 验证、bring-up 和验收标准

### 8.1 Golden tests

| 测试                    | 覆盖                                  | 验收                                                                         |
| ----------------------- | ------------------------------------- | ---------------------------------------------------------------------------- |
| runtime package load    | header、section、ABI tuple            | 拒绝不兼容 package                                                           |
| descriptor patch golden | binding table -> descriptor bytes     | 与 golden binary 一致                                                        |
| command ring            | wrap、multi-submit、doorbell sequence | 不丢失不重复                                                                 |
| event wait              | done/error/timeout/reset              | 状态与 sequence 正确                                                         |
| fault injection         | invalid descriptor/address/timeout    | fault record 完整                                                            |
| reset domain            | tile/group/device                     | pending event 和 credit deterministic                                        |
| PMU read                | counter snapshot                      | context/queue attribution 正确                                               |
| cold/warm launch        | program miss/hit                      | trace 可区分，warm 不 reload program，cold miss 通过隐式 residency path 完成 |

### 8.2 Bring-up 顺序

1. firmware boot + driver probe。
2. context/queue/event/fault memory 建立。
3. command queue + event + barrier 最小闭环。
4. DMA 1D/2D copy + completion event。
5. BOA GEMM through command queue。
6. PMU basic counter。
7. EVU mask/tail/softmax。
8. MFE Page Stream + EOS/error token。
9. Paged Attention end-to-end。
10. USE scan/recurrence checkpoint。
11. MoE Segment Stream。
12. multi-context isolation。

### 8.3 Verification plan

- User runtime unit：load、bind、patch、submit、wait、fault read。
- Driver unit：ioctl validation、IOMMU mapping、interrupt、reset。
- Firmware model：random command/event/fault sequences。
- RTL integration：command queue + DMA + event + PMU。
- End-to-end：GEMM、dense attention、paged attention、MoE、SSM canonical traces。
- Fault campaign：bad ABI、bad descriptor, stale event sequence、DMA timeout、stream credit leak。

### 8.4 验收标准

- 所有 fault 都能产生 event 和 fault record，不允许 host 永久等待。
- Reset/drain 后下一次 command 行为确定。
- Runtime load/patch/submit flow 与 package/ABI 文档一致。
- PMU counter 可将 stall 归因到 command/event、DMA、engine、stream、SRAM 或 NoC。
- First Silicon V1 禁止绕过 command queue 证明 datapath；必须通过 runtime 控制面触发。

## 9. 风险、取舍和后续细化方向

| 风险                     | 影响                   | 缓解                                                      |
| ------------------------ | ---------------------- | --------------------------------------------------------- |
| Firmware 过度复杂        | 变成小 OS，验证困难    | 只消费 command/descriptor/program，group task 粒度调度    |
| Driver/runtime 责任混乱  | 安全漏洞或重复 patch   | binding、IOMMU、descriptor patch 分层                     |
| Fault 后 warm state 污染 | 下一次运行错误         | fault 后 invalidate descriptor/program residency metadata |
| Interrupt 风暴           | latency/CPU overhead   | polling/interrupt hybrid，PMU 采样限频                    |
| Queue reset 不确定       | deadlock 或 event 泄漏 | reset/drain 状态机和 golden test                          |
| PMU 归因不唯一           | 性能优化误导           | primary owner 规则，secondary tag 仅 debug                |
| Multi-context 过早复杂   | First Silicon 收敛困难 | V1 只做隔离和基本 priority，QoS 后续扩展                  |

后续应冻结：ioctl ABI、doorbell register、queue entry stride、cache coherence、timeout 单位、fault code、reset domain、PMU counter id、firmware image loading 和 secure boot policy。未冻结参数统一写为由后续规格冻结。
