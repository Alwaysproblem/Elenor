# ELENOR Global Scheduler 设计文档

## 1. 定位、目标和 First Silicon cutline

Global Scheduler 规格描述的是未来芯片的**系统级命令提交与 event/dependency 边界**，不是当前软件 `pipeline_validator` 中的 Group 内 ready-action scheduler。当前可执行主路径是 `source xDSL → compile_program → immutable CompiledProgram → independent verify/load(actual bindings) → LoadedProgram → Simulator.run`；CPU Device interpreter 执行已编译的 `nexus.*` 控制序列，不在 Runtime 进行 graph lowering 或同步修复。

当前单 Group 参考模型中，`CpuDeviceController`、`DevicePort` 与 `GroupPortAdapter` 表达 CPU 提交和消息边界；GroupPort 接受请求时只登记有界 pending root metadata，root 之后完整准入 L2 Arena、事件/控制预算和 Group slot。Group 内另有**每 root** `TileGroupSequencer` 维护 registration cursor/fence/completion，以及**Group 共享** `GroupScheduler` 调度已注册 action。它们不是本文件所描述的多 Group 芯片全局 Scheduler，也没有证明该硬件模块已经实现。

本模块硬件目标负责系统级 command/event/dependency、资源映射及向 Group 的消息提交；它消费预先形成的 command/descriptor，不生成 Tile Program、不 lower graph，也不代替 Group root admission。命令结构、CSR、跨 Group 仲裁和物理实现仍是草案，未冻结的字段由后续规格冻结。

设计目标：

1. **确定性提交**：相同的 command、依赖与下游接受条件产生可审计的提交顺序和 terminal event。
2. **事件驱动**：command wait_ref、`signal_event + signal_sequence`、DMA completion、Group/root completion 与故障按显式事件身份交接。
3. **边界清晰**：系统级 queue / dependency / Group 消息准入不与 Group 内 action ISSUE、Tile Task admission 或 UCE issue 混为一个 scheduler。
4. **First Silicon 草案优先**：先冻结 command/event/barrier/DMA/PMU 的硬件协议；priority、preemption、PMU feedback 与多 Group 动态分配仍是扩展项。

下表列出待实现硬件目标，不代表本轮已验证的 RTL 能力：

| 能力             | 硬件规格目标                                               | 后续扩展                                 |
| ---------------- | ---------------------------------------------------------- | ---------------------------------------- |
| Command consume  | 接收已准备的 command header                                | 多级 hardware command parser             |
| Queue policy     | 有界队列仲裁，具体策略由后续规格冻结                       | 多模型 QoS、aging、deadline              |
| Event/barrier    | wait/signal、completion、timeout 与明确的 participant 合同 | event dependency graph 优化              |
| Group submission | 发送 Group 消息并接收 accepted/completion                  | 多 Group 动态分区、跨 Group load-balance |
| Resource map     | queue/context 与可用 Group 的静态绑定                      | 动态 SRAM quota 调整                     |
| Fault / PMU      | 下游 fault 汇聚与本地 queue/event/backpressure 观测        | per-context recovery、反馈调度           |

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
| | Resource Map     |-->| Group Task Launcher |-->| Event/Barrier Fabric    |  |
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
|           +---------------------> Tile Group Sequencers                    |
+----------------------------------------------------------------------------+
```

## 2. 职责、非职责和 ownership

本节区分目标硬件 Global Scheduler 与当前软件参考实现：本表中的全局队列、跨 Group 资源映射和硬件 event fabric 均未由单 Group 模型实现。`GroupScheduler` 的有限 action 表与单 cycle ISSUE/REGISTER 属 Group ownership，不能当作本模块的当前执行对象。

### 2.1 职责

本表描述目标硬件 Scheduler 的职责草案；当前软件参考模型中对应职责分别由 CPU/Device interpreter、GroupPort 与 Group 内组件承担（见 §1 与 §2.3），不存在一个已实现的芯片级调度器实体。

| 职责                    | 说明                                                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- |
| command dispatch        | 从多个 queue 中选择 ready command，维护 queue head 更新条件。                                                 |
| dependency check        | 检查 wait_ref 中每个 `event_id + expected_sequence` 是否 DONE，ERROR/TIMEOUT/RESET 是否阻断 command。         |
| event allocation/update | 对 command `signal_event + signal_sequence`、group task done、DMA done、barrier done 统一更新。               |
| group task message      | 向 Group 发送硬件 v0 descriptor 草案：root identity、资源/事件引用与 hint。消息接受不等于完整 root 资源准入。 |
| DMA task launch         | 将 ELENOR_CMD_DMA 转换成 Global DMA descriptor launch，并绑定 completion event。                              |
| barrier                 | 管理 group/tile/global barrier 参与者、timeout 和 fault propagation。                                         |
| resource map            | 管理 context->queue->group partition、active command、inflight group task。                                   |
| timeout                 | 以 command timeout_cycles 或默认 policy 生成 timeout event/fault。                                            |
| fault propagation       | 将 downstream fault 映射到 command/event/fault record。                                                       |
| PMU                     | 统计 queue occupancy、event wait、dispatch latency、scheduler backpressure。                                  |

### 2.2 非职责（目标硬件 + 当前模型共同边界）

- 不解释高层 graph，不执行 MLIR/ONNX/PyTorch 语义。
- 不执行 Tile Program；Tile Program PC、launch/wait/branch 归 Tile UCE。
- 不拥有 USE state；USE owns state，UCE owns tile program control。
- 不直接管理 MFE 的数据相关动态内存访问；MFE owns page/segment walk、address generation、stream fill。
- 不执行 Global DMA data movement；只发 launch 和接收 completion/fault。
- 不修改 program text；warm launch 只能更新 descriptor/context/shape metadata，并遵守 descriptor cache coherence。

### 2.3 Ownership matrix

| 对象                  | Owner / 边界                          | Scheduler 权限                                                                                      |
| --------------------- | ------------------------------------- | --------------------------------------------------------------------------------------------------- |
| CPU 指令/提交序列     | 已加载程序中的 `CpuDeviceController`  | 执行依赖、pending/outstanding 与 completion；不 lowering graph。                                    |
| Device 消息           | `DevicePort` 协议边界                 | 传递稳定 request identity；不等同于 Group root 已 ACTIVE。                                          |
| pending root metadata | 当前模型 `GroupPortAdapter`           | 有界排队；只保留 metadata，不预占 Group slot/L2 Arena。                                             |
| Group root resources  | Group root admission                  | 完整提交 L2 Arena、event/control budget 与 execution slot；不是全局 Scheduler 的本地 action table。 |
| system command queue  | 目标硬件 Scheduler（草案）            | 消费已准备 header、检查系统级依赖并决定消息发送。                                                   |
| Group action table    | Group 共享 `GroupScheduler`（模型内） | 注册 action 后 ISSUE；与系统级 command queue 分离。                                                 |
| event table           | Event Fabric / Group runtime model    | 维护 event producer/consumer、generation、sequence 与 terminal status。                             |
| barrier state         | 对应同步域的 owner                    | participant 与 epoch 必须显式；不得将 Context `nest.barrier` 推广成 Device barrier。                |
| group task message    | Scheduler until message accepted      | 转发硬件 v0 descriptor 草案；消息接受不表示模型中的完整 root 准入已完成。                           |
| group resource map    | 目标硬件 Scheduler（草案）            | system-level Group ownership；不可混称 Group slot 或 Tile UCE context。                             |
| fault record slot     | Fault Fabric（硬件草案）              | 写来源与请求身份；布局仍由后续规格冻结。                                                            |

## 3. 微架构和状态机

本节的子模块、pipeline、event/barrier/timeout 结构均为**目标硬件草案**，用于说明未来 Scheduler 的内部组织；当前参考实现中没有这些 RTL 模块。软件模型中相应机制分别是：CPU 的 pending/outstanding/completion 管理、GroupPort 的 SAME/COMPATIBLE pending 队列、Group EventTable 的 owner/generation/sequence，以及 Group `GroupScheduler` 的单条 ISSUE/REGISTER。

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
├── group_task_builder
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

此 pipeline 是系统级硬件命令处理草案。软件参考执行时，CPU/Device 只递交已编译的操作；Group 内的 ready-action ISSUE 由另一层共享 `GroupScheduler` 执行，不能将本 pipeline 描述为 graph lowering 或 Group action PC。

| 阶段            | 输入                                           | 输出                              | fault 条件                                                   |
| --------------- | ---------------------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| Q_READY         | queue pending/head/tail                        | selected queue                    | queue disabled、context reset。                              |
| HEADER_ACCEPT   | command header                                 | internal command record           | unsupported type、bad ABI 已由 Runtime 捕获时可直接 reject。 |
| DEP_CHECK       | wait_ref list (`event_id + expected_sequence`) | ready 或 blocked                  | wait event ERROR/TIMEOUT/RESET 或 sequence mismatch。        |
| RESOURCE_CHECK  | group mask、queue policy、inflight slots       | grant 或 stall                    | resource conflict、quota exceeded。                          |
| ISSUE           | command type                                   | group_task/dma/barrier/event task | downstream not ready timeout。                               |
| WAIT_COMPLETION | completion event                               | done/error/timeout                | command timeout。                                            |
| RETIRE          | final status                                   | queue head advance、signal event  | event table write failure。                                  |

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
- wait on DONE 只有在 `event_id + expected_sequence` 匹配时可立即放行；sequence mismatch 必须继续等待、转 stale fault 或按 reset/drain policy 处理。
- wait on ERROR/TIMEOUT/RESET 且 sequence 匹配时必须阻断 command，并生成 dependent command fault 或 skipped status，具体编码由后续规格冻结。
- `signal_event + signal_sequence` 如果重复写 terminal event，必须按 duplicate signal policy 处理；policy 由后续规格冻结，但必须可观测。
- event update 必须带 producer_id 和 sequence，方便定位 stale completion。

### 3.4 Barrier manager

Barrier 对象字段：barrier_id、context_id、participant_mask、arrived_mask、generation、timeout_cycles、signal_event、signal_sequence、fault_policy。状态：

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

本节的所有 C 结构、CSR offset 与 command enum 都是**硬件 ABI v0 草案**，只定义目标芯片的消息边界；它们不是当前 `CompiledProgram`（schema 2 / compiler ABI v2 软件 artifact）的编码，也不是已实现的寄存器。字段宽度、CRC、endianness、物理搬运与命令集由后续规格冻结。

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
    uint64_t wait_ref_iova;
    uint32_t wait_ref_count;
    uint32_t wait_ref_crc_or_zero;
    uint32_t signal_event;
    uint32_t signal_sequence;
    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_sched_cmd_record_v0_t;
```

Scheduler 消费 command header，不假设 descriptor body 格式。对于 `ELENOR_CMD_LAUNCH_GROUP_TASK`，descriptor 指向 group task launch descriptor；对于 `ELENOR_CMD_DMA`，descriptor 指向 DMA descriptor；对于 event/barrier command，descriptor 可为空或指向扩展参数。

### 4.2 Group task launch descriptor 示例

```c
typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t flags;
    uint16_t priority;

    uint32_t context_id;
    uint32_t task_id;
    uint32_t group_id;
    uint32_t role_count;
    uint32_t tile_mask_union;

    uint64_t group_task_iova;
    uint32_t group_task_bytes;
    uint64_t role_binding_iova;
    uint32_t role_binding_bytes;
    uint64_t engine_desc_iova;
    uint32_t engine_desc_bytes;
    uint64_t stream_desc_iova;
    uint32_t stream_desc_bytes;

    uint64_t wait_ref_iova;
    uint32_t wait_ref_count;
    uint32_t wait_ref_crc_or_zero;
    uint32_t signal_event;
    uint32_t signal_sequence;

    uint16_t residency_hint;
    uint16_t cache_policy;
    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_group_task_launch_desc_v0_t;
```

校验要求：Scheduler 只检查硬件消息/descriptor 草案中的 ABI version、bounds、tile mask / role 数量及 wait/signal identity 等系统级字段；它不作 source graph lowering、不插入隐式依赖，也不取代 Group 的完整 root resource admission。descriptor v0 与当前 `CompiledProgram` / `ExecTileGroupTask` 软件 DTO 是不同命名空间，不是当前软件产物的二进制布局。

### 4.3 Task 发往 Tile Group 的协议

```text
valid/ready
task_id
context_id
queue_id
group_id
role_count
tile_mask_union
group_task_iova / group_task_bytes
role_binding_iova / role_binding_bytes
engine_desc_iova / engine_desc_bytes
stream_desc_iova / stream_desc_bytes
wait_ref_iova / wait_ref_count / wait_ref_crc_or_zero
signal_event / signal_sequence
completion_event
fault_record_slot
timeout_cycles
residency_hint / cache_policy
flags
```

在硬件目标中，Scheduler 只能在明确的下游消息接受后将请求记为已提交/inflight；Group 返回的 root done/error 才终结它。当前参考模型中的 CPU request 在 GroupPort 接受 pending metadata 后仍占 Device outstanding；只有 Group root 完整 admission 才占 Group execution slot 并拥有 L2。Scheduler 不进入 Group action ISSUE、Tile Program PC 或 UCE eligible-head 仲裁。

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

### 5.1 已准备命令的系统级提交（硬件目标草案）

```text
已加载的可执行程序 / 已准备 command record
  -> CPU/Device interpreter 执行 nexus submit / depends_on / await
  -> DevicePort 接受消息（CPU outstanding 仍有效）
  -> GroupPort 接收有界 pending root metadata（软件参考模型）
  -> Group root 完整 admission：L2 / event-control budget / Group slot
  -> per-root Sequencer registration cursor / fence
  -> Group 共享 ready-action ISSUE / REGISTER
  -> Grid Route -> 每 Tile Task admission / UCE 执行
  -> root completion 经 DevicePort 返回 CPU
```

`Global Scheduler` 的硬件职责映射在 DevicePort 消息边界之外仍待明确的 command/CSR ABI 与 RTL 规格；它不承担编译期 lowering，也不拥有 Group 内 ready-action 表。系统级 message queue acceptance 与 Group root 资源准入是不同事件：前者不代表后者已经成功，也不因 pending 而部分占有 L2 或 execution slot。

下列 v0 command/descriptor 结构只说明预期的硬件消息载荷和事件关联，不是当前 Python/JSON executable artifact 的 codec，也不是已经实现的寄存器 ABI。若单条硬件命令面向多 Group，其 completion aggregation / first-error policy 需由后续规格单独冻结；当前模型只覆盖 1 Group。

### 5.2 DMA command 流程

```text
system Scheduler (target hardware)
  -> dependency/resource check on explicit command contract
  -> send DMA descriptor pointer
Global DMA (target hardware)
  -> validate descriptor and perform supported transfer
  -> return done/error/timeout
Event Fabric
  -> signal explicit event and retire command
```

Scheduler 不参与 DMA burst 级调度。当前参考模型的传输合同有限：`full_memory` 按实现支持的腿推进，连续 row-major、等字节数才合法；该模型不证明任意 strided/2D 物理 DMA。

### 5.3 Event wait/signal 流程

- 系统级 `EVENT_WAIT` / `EVENT_SIGNAL` 是待冻结硬件命令合同，不等同于源 IR 的任意 Context 内 `nest.await` / `nest.barrier`。
- 当前 CPU `nexus.await` 等待其显式 producer request completion；Group `nest.await` lower 为该 root 后续注册的 fence；Group `nest.barrier` 等待同 Context 更早 action。两者均不是隐式全 Group/Device barrier。
- engine / DMA completion 按 event identity、generation/sequence 与 owner 交接；任何硬件事件表命名、容量、错误码与 CSR 编码仍待规格冻结。

### 5.4 关键时序路径

| 路径                  | 风险                            | 缓解                                                                                    |
| --------------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| multi-queue arbitrate | queue 数多时 fan-in 大          | 分层仲裁，ready bitmap 参数由硬件规格冻结。                                             |
| wait_ref_count scan   | 多 event 依赖组合路径长         | 限制或分拍检查；每项比较 event_id + sequence，由后续规格冻结。                          |
| event waiter wakeup   | 高 fanout wakeup                | waiter bitmap 分块，queue pending bit 寄存。                                            |
| resource map check    | context/group mask CAM          | static partition 用 RAM lookup + mask compare。                                         |
| message acceptance    | 下游只收 metadata，资源尚未准入 | 分别记录 message accepted 与 Group root active/completion，不把它们算成同一 handshake。 |
| completion_router     | DMA/group/collective 多源       | source arbiter + event update FIFO；跨时钟与容量仍待硬件规格冻结。                      |

硬件接口未来可允许单 Group 或多 Group 目标，但当前参考模型只覆盖 1 Group；多 Group completion aggregation、跨 Group 调度公平性及物理消息时序均由后续规格冻结。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 参数

| 参数                      | 状态                    |
| ------------------------- | ----------------------- |
| queue count               | 由后续规格冻结          |
| event table entries       | 由后续规格冻结          |
| barrier entries           | 由后续规格冻结          |
| inflight group tasks      | 由 PPA exploration 冻结 |
| wait_ref_count max        | 由后续规格冻结          |
| group mask width          | 由后续规格冻结          |
| timeout wheel granularity | 由后续规格冻结          |
| arbitration policy        | 由后续规格冻结          |

### 6.2 PMU counters

此处 PMU 表为目标硬件 counter 草案，不能与模拟器同名/相似字段直接等同。模型中 CPU 可记录 `device_outstanding_full` 与 Port backpressure 的 `device_admission_wait`；GroupPort 的 root admission wait 通过 `context_admission_wait` / 具体 wait reason trace 观察；Group action credit/ordering 与 Tile `WAIT_CONTEXT_LIMIT` / UCE eligible-head 阻塞分别归不同层。并发 stall 属不同组件视角，不应求和为唯一全局 cycle。

| Counter                           | 说明                        | Stall owner                  |
| --------------------------------- | --------------------------- | ---------------------------- |
| sched_cycles_active               | 至少一个 command inflight   | engine_active/control_active |
| sched_queue_occupancy             | queue 非空周期              | command queue occupancy      |
| sched_queue_blocked_event         | queue 因 wait_ref 依赖阻塞  | `ELENOR_STALL_WAIT_EVENT`    |
| sched_queue_blocked_resource      | group/resource 不可用       | scheduler_resource           |
| sched_dispatch_count_group_task   | group task launch 数        | none                         |
| sched_dispatch_count_dma          | DMA command 数              | none                         |
| sched_dispatch_latency_cycles     | command ready 到 issue 延迟 | scheduler                    |
| sched_event_update_count          | event 更新数                | none                         |
| sched_barrier_wait_cycles         | barrier 未齐周期            | `ELENOR_STALL_WAIT_EVENT`    |
| sched_timeout_count               | timeout 次数                | fault                        |
| sched_downstream_backpressure_vc0 | NoC VC0 not ready           | `ELENOR_STALL_NOC_VC`        |
| sched_fault_count                 | scheduler source fault      | fault                        |

PMU 归因：如果 command 已 issue 并等待 engine/DMA completion，stall owner 是 engine/DMA/event，不是 Scheduler；如果 command 因 wait_ref 依赖未满足而不能 issue，owner 是 wait_event；如果 NoC VC0 无 credit，owner 是 NoC VC。

### 6.3 性能模型

控制面吞吐需满足：

```text
Scheduler_issue_rate >= min(queue_fetch_rate, group_task_accept_rate, dma_accept_rate, event_update_rate)
```

对小 kernel 或 dynamic shape path，launch overhead 会进入端到端 latency：

```text
T_launch = T_doorbell + T_queue_fetch + T_dep_check + T_resource + T_group_task_dispatch
```

First Silicon V1 不要求硬件消除所有 launch overhead，但必须用 PMU 拆分上述项，避免把调度瓶颈误判为 BOA/EVU/MFE 计算瓶颈。

### 6.4 Clock/reset/power/timing 考虑

- Scheduler 建议工作在 core/control clock；NoC、Runtime Processor、Global DMA、Tile Group completion 可能来自不同 domain，所有 launch/completion/event update 使用 valid/ready bridge 或 async FIFO。
- Reset/drain 时先停止 queue arbitration，再取消未 issue command 的 pending 状态；已 issue 的 DMA/group task 等待 completion、timeout 或 downstream reset ack。
- Event table、barrier table、resource map 和 timeout wheel reset 后必须进入确定状态；terminal event 是否保留给 host 读取由后续规格冻结。
- Clock gating 粒度可按 queue_ready_table idle、event update FIFO empty、no inflight task、timeout wheel idle 划分；gating 条件必须排除同周期新 doorbell/wakeup。
- Timing closure 优先关注 multi-queue arbitration、wait_ref scan、waiter wakeup fanout、resource mask compare、timeout cancel 和 completion_router arbitration。

## 7. RTL/软件实现建议

本节建议针对未来硬件实现。当前软件参考实现中，系统级命令/依赖/事件职责由 `CpuDeviceController`、`GroupPortAdapter` 与 Group 组件的既有结构承担；若硬件落地，需要把 §4 的消息边界映射为 RTL 接口，而不是把软件类名直接当作模块名。

### 7.1 RTL 建议

- Scheduler 内部 command record 使用固定宽度结构，所有字段在 HEADER_ACCEPT 后保持不变。
- Event table 写口集中到 Event Fabric，Scheduler 通过统一 update FIFO 写入，避免多个模块同时改同一 event。
- queue blocked reason 单独编码，PMU 和 debug CSR 共用同一来源。
- Resource map 初版使用静态 group partition，禁止 active context 运行中修改 group ownership。
- Timeout wheel entry 使用 generation/tag，防止 event_id reuse 后旧 timeout 命中新 command。
- Completion router 对每个 source 保留 source_id、task_id、sequence，stale completion 进入 fault path。

### 7.2 Firmware/runtime 建议

- Runtime Processor 先做 ABI/cmd_size/context/domain/descriptor bounds 校验，再交给 Scheduler。
- Compiler/runtime 生成的 command sequence 应显式表达 wait_ref 和 `signal_event + signal_sequence`，避免 Scheduler 推断高层依赖。
- 多模型 First Silicon 可用 static group partition + queue priority，不引入 preemption。
- timeout_cycles 应由 runtime 按 workload profile 设置；未设置时使用 Scheduler default。
- reset/drain command 应先 quiesce affected queue，再请求 reset domain。

### 7.3 Assertions

- terminal event 不可回到 PENDING。
- command retire 前必须写 `signal_event + signal_sequence` 或 fault event，除非 command 类型定义为 no-signal。
- queue head 不能越过未完成 command。
- resource map grant 的 group 必须属于 command context。
- timeout entry 被 cancel 后不能再产生 timeout fault。
- duplicate completion 必须用 `event_id + sequence` 检测，不能重复 signal event。
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
5. LAUNCH_GROUP_TASK 到一个 Tile Group，Tile Group Sequencer 返回 done。
6. LAUNCH_GROUP_TASK 多 group all-done。
7. 注入 invalid group mask，验证 fault record。
8. 注入 event timeout，验证 event TIMEOUT、queue stop/drain。
9. 读取 PMU，确认 queue occupancy、event wait、dispatch latency、NoC backpressure 可解释。

### 8.3 验收标准

以上单元测试、bring-up 与验收条目均是待实施的硬件 Scheduler 计划；当前参考验证覆盖的是编译产物、只读加载、单 Group 运行合同及可观察生命周期，不构成多 Group Global Scheduler RTL、CSR、CDC 或 command ABI 的验收证据。

- command queue + event + barrier 最小闭环通过。
- DMA 1D/2D/strided copy 能通过 Scheduler 发起并产生 completion event。
- BOA GEMM 通过 command queue/LAUNCH_GROUP_TASK 触发，而不是绕过 Scheduler。
- event completion、timeout、fault record 闭环。
- resource map 能隔离两个 context 的 group partition。
- Scheduler PMU 能区分 queue empty、wait_ref dependency、resource stall、NoC VC0 backpressure。
- reset/drain 后 event、barrier、timeout、resource map 状态确定。

## 9. 风险、取舍和后续细化方向

| 风险                             | 影响                                           | 缓解                                                                              |
| -------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------- |
| Scheduler 变成 graph interpreter | 硬件复杂度失控，compiler/runtime contract 模糊 | 只消费 command/descriptor/program，TileGroupTask 由 Tile Group Sequencer 执行。   |
| Event dependency fan-in 过大     | 时序不收敛                                     | 限制 wait_ref_count，分拍检查，waiter bitmap 分块，event_id + sequence 同时比较。 |
| Timeout 与 completion 竞态       | 偶发错误 event                                 | 定义同周期优先级，generation/tag，SVA 覆盖。                                      |
| 多模型 QoS 过早复杂              | First Silicon 验证面扩大                       | First Silicon 用 static partition + simple priority，PMU feedback 放后续。        |
| Resource map 动态修改            | context 污染或 use-after-reset                 | active context 禁止修改，quiesce 后更新。                                         |
| PMU 归因错误                     | 性能优化方向错误                               | primary stall owner 规则，Scheduler 只统计控制面阻塞。                            |
| Barrier deadlock                 | group task 永久挂起                            | timeout、fault propagation、reset/drain 明确定义。                                |

后续规格需要冻结：event_id namespace、event table size、wait_ref_count 上限、duplicate signal policy、barrier participant 编码、timeout 同周期优先级、resource map CSR、queue arbitration policy、group task launch descriptor binary layout、Scheduler PMU counter id 和 reset/drain 对 blocked queue 的精确语义。
