# ELENOR Global Scheduler 设计文档

## 1. 定位、目标和 First Silicon cutline

Global Scheduler 是 ELENOR chip-level 控制面的核心，位于 Host Interface/Runtime Processor 与 Tile Group/Global DMA/Collective/Event Fabric 之间。它消费已经通过 Runtime Processor 基础校验的 command queue entry，管理全局 event/barrier/fault/resource map，启动 Pipeline Region，并将带有 `program_id/template_id/program_ref` 的 RegionLaunchDesc 转发到 Tile Group Region Sequencer。它不解释高层 graph；graph schedule 已由 compiler/runtime 降到 command buffer、descriptor table、Region Program 和 Tile Program。

Global Scheduler 的设计目标：

1. **确定性调度**：同一 command sequence 在相同 event/descriptor 条件下产生相同 region dispatch 顺序和 event 结果。
2. **事件驱动**：command wait_event、signal_event、DMA completion、tile/group done、barrier 统一进入 Event Fabric。
3. **资源绑定**：把 context、queue、Tile Group mask、program reference、descriptor_iova、fault_record_slot 绑定到可审计的 region task。
4. **First Silicon 稳定优先**：先闭合 command/event/barrier/DMA/PMU，再扩展 multi-model priority、preemption 和 PMU feedback scheduling。

First Silicon V1 cutline：

| 能力            | First Silicon V1                                                      | Architecture V1 / 后续规格               |
| --------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| Command consume | 接收 Runtime Processor 输出的 validated command header                | 多级 hardware command parser             |
| Queue policy    | round-robin 或 fixed priority，策略由后续规格冻结                     | 多模型 QoS、aging、deadline              |
| Event/barrier   | wait/signal、completion event、timeout、barrier                       | event dependency graph 优化              |
| Region launch   | LAUNCH_REGION、DMA、BARRIER、EVENT_WAIT/SIGNAL、RESET_DOMAIN 基础命令 | full graph schedule hardware assist      |
| Resource map    | static group mask、queue/context binding                              | dynamic group partition、SRAM quota 调整 |
| Fault           | invalid state、timeout、resource conflict、downstream fault 汇聚      | per-context recovery policy 扩展         |
| PMU             | queue occupancy、event wait、scheduler stall、dispatch latency        | PMU feedback scheduling                  |

模块框图：

```text
Runtime Processor / Queue Fetcher
        |
        v
+----------------------------------------------------------------------------+
| Global Scheduler                                                           |
|                                                                            |
| +------------------+   +------------------+   +-------------------------+  |
| | Command Arbiter  |-->| Command Decoder  |-->| Dependency/Event Check  |  |
| +---------+--------+   +---------+--------+   +-----------+-------------+  |
|           |                      |                        |                |
|           v                      v                        v                |
| +------------------+   +------------------+   +-------------------------+  |
| | Resource Map     |-->| Region Launcher  |-->| Event/Barrier Fabric    |  |
| | group/context    |   | group tasks      |   | wait/signal/timeout     |  |
| +---------+--------+   +---------+--------+   +-----------+-------------+  |
|           |                      |                        |                |
|           v                      v                        v                |
| +------------------+   +------------------+   +-------------------------+  |
| | DMA Launcher     |   | Collective Launch|   | Fault/PMU               |  |
| +---------+--------+   +---------+--------+   +-----------+-------------+  |
|           |                      |                        |                |
|           v                      v                        v                |
|      Global DMA             NoC/Collective             Host Interface      |
|           |                      |                                         |
|           +---------------------> Tile Group Region Sequencers             |
+----------------------------------------------------------------------------+
```

## 2. 职责、非职责和 ownership

### 2.1 职责

| 职责                    | 说明                                                                                                                                                                  |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| command dispatch        | 从多个 queue 中选择 ready command，维护 queue head 更新条件。                                                                                                         |
| dependency check        | 检查 wait_event 是否 DONE，ERROR/TIMEOUT/RESET 是否阻断 command。                                                                                                     |
| event allocation/update | 对 command signal_event、region done、DMA done、barrier done 统一更新。                                                                                               |
| region task launch      | 向 Tile Group Region Sequencer 发送 RegionLaunchDesc：program_id/template_id/program backing store、descriptor pointer、group mask、stream config 和 residency hint。 |
| DMA task launch         | 将 ELENOR_CMD_DMA 转换成 Global DMA descriptor launch，并绑定 completion event。                                                                                      |
| barrier                 | 管理 group/tile/global barrier 参与者、timeout 和 fault propagation。                                                                                                 |
| resource map            | 管理 context->queue->group partition、active command、inflight region。                                                                                               |
| timeout                 | 以 command timeout_cycles 或默认 policy 生成 timeout event/fault。                                                                                                    |
| fault propagation       | 将 downstream fault 映射到 command/event/fault record。                                                                                                               |
| PMU                     | 统计 queue occupancy、event wait、dispatch latency、scheduler backpressure。                                                                                          |

### 2.2 非职责

- 不解释高层 graph，不执行 MLIR/ONNX/PyTorch 语义。
- 不执行 Tile Program；Tile Program PC、launch/wait/branch 归 Tile UCE。
- 不拥有 USE state；USE owns state，UCE owns tile program control。
- 不直接管理 MFE 的数据相关动态内存访问；MFE owns page/segment walk、address generation、stream fill。
- 不执行 Global DMA data movement；只发 launch 和接收 completion/fault。
- 不修改 program text；warm launch 只能更新 descriptor/context/shape metadata，并遵守 descriptor cache coherence。

### 2.3 Ownership matrix

| 对象               | Owner                                               | Scheduler 权限                                                                      |
| ------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| command header     | Runtime/Queue Fetcher validates，Scheduler consumes | 读 type/wait/signal/timeout/context/desc pointer。                                  |
| descriptor body    | Engine/DMA/Region Sequencer consumes                | Scheduler 只做 bounds/version/owner 级检查，不解析 engine 私有字段。                |
| event table        | Event Fabric/Scheduler                              | 创建、wait、signal、timeout、reset。                                                |
| barrier state      | Scheduler                                           | 维护 participant mask、arrival count、timeout。                                     |
| region task        | Scheduler owns until accepted by Group              | 生成 task_id、group_id、program_id/template_id、descriptor_iova 和 residency hint。 |
| group resource map | Scheduler                                           | 分配/释放 group，记录 context ownership。                                           |
| queue head         | Runtime/Scheduler handshake                         | command accepted 或 rejected with event 后更新。                                    |
| fault record slot  | Fault Fabric owns，Scheduler 填 source metadata     | 写 command_id/context/queue/source/timeout。                                        |

## 3. 微架构和状态机

### 3.1 子模块

```text
global_scheduler
├── queue_ready_table
├── command_arbiter
├── command_decode_stage
├── dependency_checker
├── event_scoreboard
├── barrier_manager
├── resource_map
├── region_task_builder
├── dma_task_builder
├── collective_task_builder
├── timeout_wheel
├── completion_router
├── fault_router
└── scheduler_pmu
```

### 3.2 Command pipeline

```text
Q_READY
  -> ARBITRATE
  -> HEADER_ACCEPT
  -> DEP_CHECK
  -> RESOURCE_CHECK
  -> ISSUE
  -> WAIT_COMPLETION 或 COMPLETE_IMMEDIATE
  -> SIGNAL_EVENT
  -> RETIRE
```

每一级建议寄存化，避免 wait_event fan-in、resource conflict check 和 NoC ready 组合成一条长路径。

| 阶段            | 输入                                     | 输出                             | fault 条件                                                   |
| --------------- | ---------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| Q_READY         | queue pending/head/tail                  | selected queue                   | queue disabled、context reset。                              |
| HEADER_ACCEPT   | command header                           | internal command record          | unsupported type、bad ABI 已由 Runtime 捕获时可直接 reject。 |
| DEP_CHECK       | wait_event_base/count                    | ready 或 blocked                 | wait event ERROR/TIMEOUT/RESET。                             |
| RESOURCE_CHECK  | group mask、queue policy、inflight slots | grant 或 stall                   | resource conflict、quota exceeded。                          |
| ISSUE           | command type                             | region/dma/barrier/event task    | downstream not ready timeout。                               |
| WAIT_COMPLETION | completion event                         | done/error/timeout               | command timeout。                                            |
| RETIRE          | final status                             | queue head advance、signal event | event table write failure。                                  |

### 3.3 Event scoreboard

Event scoreboard 保存 event status、producer、sequence、context、waiter list 或 waiter bitset。First Silicon V1 可以采用固定大小 event table，event_id namespace 可为 global 或 per context，选择由后续规格冻结。

状态机：

```text
FREE
  -> PENDING
  -> DONE
  -> ERROR
  -> TIMEOUT
  -> RESET
```

规则：

- terminal status 不允许回到 PENDING。
- wait on DONE 可立即放行。
- wait on ERROR/TIMEOUT/RESET 必须阻断 command，并生成 dependent command fault 或 skipped status，具体编码由后续规格冻结。
- signal_event 如果重复写 terminal event，必须按 duplicate signal policy 处理；policy 由后续规格冻结，但必须可观测。
- event update 必须带 producer_id 和 sequence，方便定位 stale completion。

### 3.4 Barrier manager

Barrier 对象字段：barrier_id、context_id、participant_mask、arrived_mask、generation、timeout_cycles、signal_event、fault_policy。状态：

```text
BARRIER_FREE
  -> BARRIER_ARMED
  -> BARRIER_PARTIAL
  -> BARRIER_RELEASED
  -> BARRIER_TIMEOUT
  -> BARRIER_RESET
```

Barrier 必须支持 group-level 和 global-level；tile-local barrier 由 Tile Group/Tile UCE 处理，Scheduler 只接收 group completion 或 error propagation。

### 3.5 Timeout wheel

command timeout_cycles 可直接映射到 timeout wheel entry，粒度由后续规格冻结。实现建议：

- short timeout 用小 wheel，long timeout 用 coarse counter。
- timeout entry 绑定 command_id、context_id、queue_id、event_id、source type。
- completion 到达时取消 timeout；若 timeout 与 completion 同周期，优先级由后续规格冻结，必须有 SVA 覆盖。

## 4. 接口、descriptor、寄存器和协议

### 4.1 输入 command record

Runtime Processor 输出给 Scheduler 的内部 record 示例：

```c
typedef struct {
    uint16_t abi_version;
    uint16_t cmd_size;
    uint16_t type;
    uint16_t flags;
    uint32_t context_id;
    uint32_t queue_id;
    uint32_t command_id;
    uint64_t desc_iova;
    uint32_t desc_bytes;
    uint32_t wait_event_base;
    uint16_t wait_event_count;
    uint16_t signal_event;
    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_sched_cmd_record_v0_t;
```

Scheduler 消费 command header，不假设 descriptor body 格式。对于 `ELENOR_CMD_LAUNCH_REGION`，descriptor 指向 region launch descriptor；对于 `ELENOR_CMD_DMA`，descriptor 指向 DMA descriptor；对于 event/barrier command，descriptor 可为空或指向扩展参数。

### 4.2 Region launch descriptor 示例

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

校验要求：Scheduler 只检查 ABI version、descriptor bounds、tile_group_mask/tile_mask 合法、group ownership、wait/signal event 范围、program_id/version/hash 的基本一致性，以及 hint 字段不含未支持必需位。program ready 由下游 Tile Group Sequencer 的 Program Residency Manager 保证，不由 Scheduler 发出显式 `program.load`。

### 4.3 Task 发往 Tile Group 的协议

```text
valid/ready
task_id
context_id
queue_id
region_id
group_id
tile_mask
template_id
program_id
program_version
program_crc_or_hash
program_iova
program_bytes
program_section_id
region_desc_iova / region_desc_bytes
engine_desc_iova / engine_desc_bytes
wait_event_base / wait_event_count
completion_event
fault_record_slot
timeout_cycles
residency_hint / cache_policy
flags
```

Tile Group accept 后，Scheduler 可认为 region task inflight；region done/error 通过 completion_router 返回。Scheduler 不进入 Tile Program 细节，也不管理 group/tile local program slot。

### 4.4 Scheduler CSR

| Offset | 名称                  | 属性  | 说明                                                          |
| ------ | --------------------- | ----- | ------------------------------------------------------------- |
| 0x0000 | SCHED_CAP             | RO    | queue count、event entries、barrier entries、inflight depth。 |
| 0x0008 | SCHED_CONTROL         | RW    | enable、quiesce、policy select。                              |
| 0x0010 | SCHED_STATUS          | RO    | active queues、blocked queues、fatal fault。                  |
| 0x0100 | QUEUE_ENABLE_MASK     | RW    | 可运行 queue bitmask。                                        |
| 0x0108 | QUEUE_PRIORITY        | RW    | 每队列 priority，编码由后续规格冻结。                         |
| 0x0200 | EVENT_STATUS_WINDOW   | RO    | debug 读取 event table window。                               |
| 0x0300 | BARRIER_STATUS_WINDOW | RO    | debug barrier 状态。                                          |
| 0x0400 | RESOURCE_GROUP_OWNER  | RO/RW | context->group 静态绑定或 debug override。                    |
| 0x0500 | TIMEOUT_DEFAULT       | RW    | command 默认 timeout。                                        |
| 0x0600 | PMU_SELECT/PMU_VALUE  | RW/RO | scheduler PMU。                                               |

Active command 运行时修改 policy、resource map、queue enable 的行为必须受 quiesce 保护。

## 5. 数据流、控制流和时序路径

### 5.1 LAUNCH_REGION 流程

```text
Command Arbiter
  -> select ready queue
Command Decoder
  -> sees ELENOR_CMD_LAUNCH_REGION
Dependency Checker
  -> wait events done?
Resource Map
  -> group mask available and owned by context?
Region Task Builder
  -> build per-group task
NoC VC0
  -> send task to group Region Sequencer
Completion Router
  -> collect group done/error
Event Fabric
  -> signal command event
Queue Retire
  -> advance queue head
```

Region task 可跨多个 group；completion policy 可为 all-groups done 或 first-error abort，First Silicon V1 推荐 first-error records fault、stop affected region、signal event error。

### 5.2 DMA command 流程

```text
Scheduler
  -> dependency/resource check
  -> sends DMA descriptor pointer to Global DMA
Global DMA
  -> validates DMA descriptor details
  -> performs copy
  -> returns done/error/timeout
Scheduler/Event Fabric
  -> signal event and retire command
```

Scheduler 不参与 DMA burst 级调度，但负责 DMA command 的 event dependency、timeout、fault_record_slot 和 queue retire。

### 5.3 Event wait/signal 流程

- `ELENOR_CMD_EVENT_WAIT`：若 event terminal DONE，command immediate complete；若 PENDING，queue blocked 并挂入 waiter；若 ERROR/TIMEOUT/RESET，产生 dependent fault 或 error event。
- `ELENOR_CMD_EVENT_SIGNAL`：写 event DONE 或指定状态，用于 host/runtime software event。权限由 context filter 决定。
- engine completion event：由 completion_router 写入，唤醒 waiters。

### 5.4 关键时序路径

| 路径                  | 风险                      | 缓解                                                             |
| --------------------- | ------------------------- | ---------------------------------------------------------------- |
| multi-queue arbitrate | queue 数多时 fan-in 大    | 分层仲裁，per-priority ready bitmap。                            |
| wait_event_count scan | 多 event 依赖组合路径长   | 限制 First Silicon wait_event_count 或分拍检查，由后续规格冻结。 |
| event waiter wakeup   | 高 fanout wakeup          | waiter bitmap 分块，queue pending bit 寄存。                     |
| resource map check    | context/group mask CAM    | static partition 用 RAM lookup + mask compare。                  |
| timeout wheel cancel  | completion 同周期 cancel  | tag match 寄存，定义优先级。                                     |
| completion_router     | DMA/group/collective 多源 | source arbiter + event update FIFO。                             |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 参数

| 参数                      | 状态                    |
| ------------------------- | ----------------------- |
| queue count               | 由后续规格冻结          |
| event table entries       | 由后续规格冻结          |
| barrier entries           | 由后续规格冻结          |
| inflight region tasks     | 由 PPA exploration 冻结 |
| wait_event_count max      | 由后续规格冻结          |
| group mask width          | 由后续规格冻结          |
| timeout wheel granularity | 由后续规格冻结          |
| arbitration policy        | 由后续规格冻结          |

### 6.2 PMU counters

| Counter                           | 说明                        | Stall owner                  |
| --------------------------------- | --------------------------- | ---------------------------- |
| sched_cycles_active               | 至少一个 command inflight   | engine_active/control_active |
| sched_queue_occupancy             | queue 非空周期              | command queue occupancy      |
| sched_queue_blocked_event         | queue 因 wait_event 阻塞    | `ELENOR_STALL_WAIT_EVENT`    |
| sched_queue_blocked_resource      | group/resource 不可用       | scheduler_resource           |
| sched_dispatch_count_region       | region launch 数            | none                         |
| sched_dispatch_count_dma          | DMA command 数              | none                         |
| sched_dispatch_latency_cycles     | command ready 到 issue 延迟 | scheduler                    |
| sched_event_update_count          | event 更新数                | none                         |
| sched_barrier_wait_cycles         | barrier 未齐周期            | `ELENOR_STALL_WAIT_EVENT`    |
| sched_timeout_count               | timeout 次数                | fault                        |
| sched_downstream_backpressure_vc0 | NoC VC0 not ready           | `ELENOR_STALL_NOC_VC`        |
| sched_fault_count                 | scheduler source fault      | fault                        |

PMU 归因：如果 command 已 issue 并等待 engine/DMA completion，stall owner 是 engine/DMA/event，不是 Scheduler；如果 command 因 event dependency 未满足而不能 issue，owner 是 wait_event；如果 NoC VC0 无 credit，owner 是 NoC VC。

### 6.3 性能模型

控制面吞吐需满足：

```text
Scheduler_issue_rate >= min(queue_fetch_rate, region_accept_rate, dma_accept_rate, event_update_rate)
```

对小 kernel 或 dynamic shape path，launch overhead 会进入端到端 latency：

```text
T_launch = T_doorbell + T_queue_fetch + T_dep_check + T_resource + T_region_dispatch
```

First Silicon V1 不要求硬件消除所有 launch overhead，但必须用 PMU 拆分上述项，避免把调度瓶颈误判为 BOA/EVU/MFE 计算瓶颈。

### 6.4 Clock/reset/power/timing 考虑

- Scheduler 建议工作在 core/control clock；NoC、Runtime Processor、Global DMA、Tile Group completion 可能来自不同 domain，所有 launch/completion/event update 使用 valid/ready bridge 或 async FIFO。
- Reset/drain 时先停止 queue arbitration，再取消未 issue command 的 pending 状态；已 issue 的 DMA/region task 等待 completion、timeout 或 downstream reset ack。
- Event table、barrier table、resource map 和 timeout wheel reset 后必须进入确定状态；terminal event 是否保留给 host 读取由后续规格冻结。
- Clock gating 粒度可按 queue_ready_table idle、event update FIFO empty、no inflight task、timeout wheel idle 划分；gating 条件必须排除同周期新 doorbell/wakeup。
- Timing closure 优先关注 multi-queue arbitration、wait_event scan、waiter wakeup fanout、resource mask compare、timeout cancel 和 completion_router arbitration。

## 7. RTL/软件实现建议

### 7.1 RTL 建议

- Scheduler 内部 command record 使用固定宽度结构，所有字段在 HEADER_ACCEPT 后保持不变。
- Event table 写口集中到 Event Fabric，Scheduler 通过统一 update FIFO 写入，避免多个模块同时改同一 event。
- queue blocked reason 单独编码，PMU 和 debug CSR 共用同一来源。
- Resource map 初版使用静态 group partition，禁止 active context 运行中修改 group ownership。
- Timeout wheel entry 使用 generation/tag，防止 event_id reuse 后旧 timeout 命中新 command。
- Completion router 对每个 source 保留 source_id、task_id、sequence，stale completion 进入 fault path。

### 7.2 Firmware/runtime 建议

- Runtime Processor 先做 ABI/cmd_size/context/domain/descriptor bounds 校验，再交给 Scheduler。
- Compiler/runtime 生成的 command sequence 应显式表达 wait_event/signal_event，避免 Scheduler 推断高层依赖。
- 多模型 First Silicon 可用 static group partition + queue priority，不引入 preemption。
- timeout_cycles 应由 runtime 按 workload profile 设置；未设置时使用 Scheduler default。
- reset/drain command 应先 quiesce affected queue，再请求 reset domain。

### 7.3 Assertions

- terminal event 不可回到 PENDING。
- command retire 前必须写 signal_event 或 fault event，除非 command 类型定义为 no-signal。
- queue head 不能越过未完成 command。
- resource map grant 的 group 必须属于 command context。
- timeout entry 被 cancel 后不能再产生 timeout fault。
- duplicate completion 必须被检测，不能重复 signal event。
- NoC task valid 在 ready 前保持 stable。

## 8. 验证、bring-up 和验收标准

### 8.1 单元验证

| 单元               | 场景                                                                         |
| ------------------ | ---------------------------------------------------------------------------- |
| Command Arbiter    | 多 queue ready、priority、round-robin fairness、queue disable。              |
| Dependency Checker | zero wait、single wait、multi wait、ERROR/TIMEOUT dependency。               |
| Event Scoreboard   | signal、waiter wakeup、duplicate signal、event reuse generation。            |
| Barrier Manager    | all participants arrive、missing participant timeout、reset during barrier。 |
| Resource Map       | static group partition、conflict、context reset、invalid group mask。        |
| Timeout Wheel      | completion before timeout、timeout before completion、same-cycle priority。  |
| Completion Router  | DMA done、group done、fault、stale task_id、source arbitration。             |

### 8.2 Bring-up 顺序

1. no-op command through queue，event done。
2. EVENT_SIGNAL/EVENT_WAIT command pair，验证 waiter wakeup。
3. BARRIER command 单 group 和多 group，验证 barrier done。
4. DMA command 1D copy，completion event。
5. LAUNCH_REGION 到一个 Tile Group，Region Sequencer 返回 done。
6. LAUNCH_REGION 多 group all-done。
7. 注入 invalid group mask，验证 fault record。
8. 注入 event timeout，验证 event TIMEOUT、queue stop/drain。
9. 读取 PMU，确认 queue occupancy、event wait、dispatch latency、NoC backpressure 可解释。

### 8.3 验收标准

- command queue + event + barrier 最小闭环通过。
- DMA 1D/2D/strided copy 能通过 Scheduler 发起并产生 completion event。
- BOA GEMM 通过 command queue/LAUNCH_REGION 触发，而不是绕过 Scheduler。
- event completion、timeout、fault record 闭环。
- resource map 能隔离两个 context 的 group partition。
- Scheduler PMU 能区分 queue empty、wait_event、resource stall、NoC VC0 backpressure。
- reset/drain 后 event、barrier、timeout、resource map 状态确定。

## 9. 风险、取舍和后续细化方向

| 风险                             | 影响                                           | 缓解                                                                       |
| -------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------- |
| Scheduler 变成 graph interpreter | 硬件复杂度失控，compiler/runtime contract 模糊 | 只消费 command/descriptor/program，Region Program 运行在 Group Sequencer。 |
| Event dependency fan-in 过大     | 时序不收敛                                     | 限制 wait_event_count，分拍检查，waiter bitmap 分块。                      |
| Timeout 与 completion 竞态       | 偶发错误 event                                 | 定义同周期优先级，generation/tag，SVA 覆盖。                               |
| 多模型 QoS 过早复杂              | First Silicon 验证面扩大                       | First Silicon 用 static partition + simple priority，PMU feedback 放后续。 |
| Resource map 动态修改            | context 污染或 use-after-reset                 | active context 禁止修改，quiesce 后更新。                                  |
| PMU 归因错误                     | 性能优化方向错误                               | primary stall owner 规则，Scheduler 只统计控制面阻塞。                     |
| Barrier deadlock                 | region 永久挂起                                | timeout、fault propagation、reset/drain 明确定义。                         |

后续规格需要冻结：event_id namespace、event table size、wait_event_count 上限、duplicate signal policy、barrier participant 编码、timeout 同周期优先级、resource map CSR、queue arbitration policy、region launch descriptor binary layout、Scheduler PMU counter id 和 reset/drain 对 blocked queue 的精确语义。