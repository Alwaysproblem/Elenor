# Tile Group 设计文档

## 1. 定位、目标和 First Silicon cutline

Tile Group 是一个 Group execution 域，管理 root Context 的 L2 Arena、事件/控制预算与 Group execution slot，并提供多个 root 间共享的有限 action issue。当前 `pipeline_validator` 是调度/资源语义参考实现，验证范围为 1 Group × 4 Tile；本文的 RTL、物理 SRAM/NoC、ABI/CSR 与真实引擎实现仍是硬件设计草案。

控制对象按以下层级组织：

```text
Graph → Context(root invocation) → Grid(finite dispatch) → Task(per Tile)
      → Tile Program → Engine
```

Root pending 与 root ACTIVE 不同：GroupPort 先只接收有界 pending metadata；只有完整资源准入才取得 Group execution slot、L2 Arena、event/control budget 与 ownership。当前模型 `group.active_context_capacity` 默认 8，但它是 Group root execution slot 数，不是 CPU outstanding 上限，也不是每 Tile UCE context 数。

目标调度结构：

```text
每 root TileGroupSequencer: registration cursor + fence + completion
                ↓
Group 共享 GroupScheduler: bounded action table / scan / credit / issue
                ↓
有限 Grid Route → 每 Tile 独立 Task admission → Tile UCE
```

Tile Group 的设计目标是维持本层 ownership，而非复现第二套 graph scheduler：

- 多个 root Context 在满足完整 L2/Profile/event/control 资源合同后可并存；准入失败只等待，不预占半套资源。
- 每 root 依序登记 action；Group 共享调度器在有界扫描窗口中 issue ready action，注册与完成解耦。
- `dispatch_role` 登记有限 Grid Route；各 Tile 独立申请 Task 资源，单 Tile 阻塞不撤销其他 Tile 已提交工作。
- Tile Task 的 L1 Arena、Frame、UCE pin、父 L2 pin 与 `requested_contexts_per_tile=R` lease 在 Task commit 时一致取得，Task 安全退休后归还。

| 能力                     | 当前软件参考模型合同                                                   | 硬件/物理边界                                 |
| ------------------------ | ---------------------------------------------------------------------- | --------------------------------------------- |
| Root admission           | 有界 pending metadata；完整 Group 资源准入；SAME/COMPATIBLE FIFO heads | GroupPort RTL/握手/深度待规格冻结             |
| Group ready-action issue | S0 oldest-head；S1 有限窗口跳过 PENDING；S2 当前与 S1 相同             | action RAM、scan 宽度与 issue pipeline 待 PPA |
| Tile Task admission      | 有界 Grid Route；每 Tile 独立 commit；R lease                          | 多 Tile 规模和跨 Group 资源组织未验证         |
| UCE                      | 每 Tile 参数化 1..8 contexts 模型；eligible-head RR 单指令 issue       | 硅片 context 数及执行电路未冻结               |
| Stream Queue             | 可选 credit/backpressure/EOS/error overlay                             | FIFO/CDC/物理 payload 组织未验证              |

容量、bank、队列深度、timeout、NoC bandwidth、物理 Tile 数以及 ABI 编码均由后续规格或 PPA exploration 冻结。

## 2. 职责、非职责和 ownership

Tile Group ownership：

- **GroupPort/root admission**：pending root metadata 与完整资源准入分离；准入成功时一次性取得 Group slot、L2 Arena、event/control 预算及 root identity。Group slot 不等于 Tile UCE context。
- **每 root `TileGroupSequencer`**：持有该 root 的 action registration cursor、await/barrier fence、queued/inflight action 与 completion 状态；它不拥有共享 issue 带宽。
- **Group 共享 `GroupScheduler`**：持有有限 ready-action 表、有限 scan/credit 并对所有 root 的已登记 action 仲裁；每 cycle 最多一条 ISSUE 与一条 REGISTER，completion/control poll 独立推进。
- **Grid Route / Task admission**：dispatch 登记有限 Route；每 Tile 按 L1 Profile、slot、逐 bank 容量/碎片、Frame、UCE pin、父 L2 pin 和 R lease 完整提交一项 Task。
- **Tile UCE**：每 Tile 拥有物理执行 context 状态及 eligible-head RR 单指令 issue；context 数是 Tile 内资源，不是 Group slot 或 `context_id` 隔离域。
- **L2 / ProfileController / EventTable**：Group 拥有 root L2 Arena 与共享 backing；Profile writer 和 event/generation 生命周期按当前实现单写者/所有权合同执行。
- **DMA、Collective、Stream Queue、PMU**：各资源有独立 owner/credit/event；Stream Queue 仅是可选 producer-consumer overlay，不是所有 role 的唯一通路。

Tile Group 不负责：

- CPU `nexus.*` 提交、Device outstanding、系统级 queue policy 或高层 graph lowering。
- 编译期依赖修复、现场 Profile 选择或运行时改变已编译执行合同。
- Tile-local UCE PC、L1 Arena/Frame 分配、引擎微循环或数据相关 MFE 地址生成。
- 尚未建模的多 Group 全芯片 arbitration、priority、preemption、迁移和硬件 QoS。

关键 ownership 规则：

1. pending root metadata 不占 Group slot、L2 Arena 或完整 event/control 资源；root admission 才完整 commit。
2. 各 root Sequencer 只管理本 root registration/fence/completion；共享 `GroupScheduler` 才拥有 Group action issue 表与带宽。
3. 每 cycle `GroupScheduler.step` 先处理 completion/control，再至多 ISSUE 一项旧 action，最后 REGISTER 一项新 action；completion credit 回收不受 action 表满阻塞。
4. 每 Tile Task commit 包括 L1 Arena、Frame、UCE pin、父 L2 pin 与 R lease；准备路径可 abort/rollback 未对外可见的部分，已接受 work 不作任意事务回滚。
5. 同一父 Context 的 R lease 按 `(parent binding_id, launch_generation, tile_id)` 跨 Grid 计数；route pending 不占 lease，安全 Task retirement / cancel-confirm 才释放。
6. PMU wait/credit/resource attribution 必须区分 CPU outstanding、Group slot、Group action credit、Tile slot/UCE context 与 R 限额，不能混为一个“context stall”。

## 3. 微架构和状态机

### 3.1 控制与执行结构

```text
GroupPort
├── bounded pending root metadata
└── full root admission: Group slot + L2 Arena + event/control budget
    └── root Context 0..N
        └── TileGroupSequencer per root
            ├── registration cursor / fence / completion
            └── registered actions ───────┐
                                           v
Group-shared GroupScheduler: bounded action table / scan / credit / ISSUE
                                           |
                                           v
bounded Grid Route → independent Task admission on each selected Tile
                                           |
                                           v
                         Tile UCE contexts / engines / transfers
```

`GroupScheduler` 属于一个 Group，持有共享 action table 与 issue 带宽；它不是芯片级多 Group Scheduler。每个 `TileGroupSequencer` 只保留自己的 registration cursor、fence 和完成状态，不拥有独立 issue pipeline。

### 3.2 Root admission 与 ready-action 规则

```text
CPU submit → PORT_PENDING → ROOT_ADMISSION → ACTIVE_ROOT
           → register actions / issue ready actions / admit Grid Tasks
           → retire Tasks / Grids / root Arena → ROOT_COMPLETE
```

- `GroupPortAdapter.try_submit` 接收成功只占有界 pending metadata。只有完整 root admission 才提交 Group slot、L2 Arena、event/control budget 与 ownership；等待期间不占半套资源。
- Group slot 与 CPU outstanding 分离：请求可已在 CPU 侧记为 ACTIVE/outstanding，但 root 的 `active_cycle` 尚未到达。`device_context_count` 限制 Device outstanding；`group.active_context_capacity` 默认 8，限制 Group execution slots。
- Root admission 按活跃 L2 Profile 比较 SAME / COMPATIBLE 两类 FIFO 队首。SAME 优先；SAME 队首受阻时可由 COMPATIBLE 队首补位，但不越过同类别队首装箱。
- 每个 root 的 registration cursor 在 action 成功登记后前进，不等 action 完成；`nest.await` lower 为 `WAIT_EVENT` 并 fence 该 root 的后续 registration，`nest.barrier` lower 为 `BARRIER_GROUP` 并等待该 Context 更早 action 完成。二者都不是隐式全 Group/Device barrier。
- Group `step` 先 poll completion/control，再最多 ISSUE 一条旧 action，最后 REGISTER 一条新 action。完成事件独立回收 event/adapter/inflight credit，即使 action table 满也不能阻止完成推进。
- **S0** 只允许各 root 在扫描窗口内的最老 action 参与 issue；**S1** 可在有限 `scan_width` 窗口内跳过依赖仍为 PENDING 的 action；**S2** 当前与 S1 同实现，不是第三种已实现策略。`scan_width` 独立于 UCE context 数和 engine ingress queue depth。
- REGISTER 与 ISSUE 是不同仲裁层；即使各层有 RR，也不由此承诺全局 fairness 或无饥饿。

### 3.3 Grid Route 与逐 Tile Task admission

`dispatch_role` 登记有限 Grid Route。`placement` 是 Tile mask，非空 `task.range` 数量必须等于 `popcount(placement)`；Task 按选中 Tile 的顺序映射。`logical_tasks` 汇总该 root 所有 dispatch 的逻辑 Task 数。

`_step_task_admission` 每 Tile 每 tick 至多完整 commit 一个 Task。Task admission 受当前 L1 Profile、UCE pin/slot、L1 逐 bank 容量与碎片、Frame、父 L2 pin、控制资源及 `requested_contexts_per_tile=R` 限制。R 按 `(parent binding_id, launch_generation, tile_id)` 计数，跨该 parent 的全部 Grid；Route pending 不占 lease，Task safe retirement 或已确认取消隔离后才归还。

Task commit 一致拥有 L1 Arena、Frame、UCE pin、父 L2 pin 与 R lease。准备失败可以 abort/rollback 尚未对外可见的部分；已接受的计算不做任意事务回滚，而通过 drain/cancel-confirm 收敛。一个 Tile 暂时阻塞不会撤销其他 Tile 已提交的 Task；没有跨 Tile gang commit、迁移、抢占或 task stealing。

每 Task 在所选 Tile 运行相同 Tile Program 模板，依 `tile_id`、TaskIdentity 和 descriptor binding 区分实际工作。`context=N`（dispatch pin）只 pin 每个所选 Tile 的同号 UCE execution context；它不表示 CPU request、Group slot 或 ABI isolation `context_id`。

### 3.4 Completion、retirement 与故障

Dispatch `input_released` 表示全部相关输入读取已结束；`output_ready` 表示相关 L1→L2 输出可见；Grid done 表示所选 Task 全部安全退休。它们不可互换，`output_ready` 不是 HBM 持久化，`input_released` 也不是 Arena 退休。

Task terminal 后仍需等待在途 transfer/view 关闭、失效 L1 views、退休整块 Task Arena、释放 Frame、父 L2 pin 与 R lease；所有 Grid Task 安全退休后才发 Grid completion。Root 再收敛其 actions/events、L2 owner 与 claims 后结束。pending root 取消与已准入 work 的 drain/cancel-confirm 属不同故障路径；硬件 reset 波形、ECC/CDC 与故障 CSR 仍未建模。

## 4. 接口、descriptor、寄存器和协议

### 4.1 硬件 launch descriptor（v0 草案）

下列 C record 是未冻结的**硬件 ABI v0 草案**，仅示意 System Scheduler 到 Group 的消息字段；不是当前 `CompiledProgram` codec。当前软件对象是 immutable artifact 中的 `ExecTileGroupTask` / `ExecTileRoleBinding` DTO；不从 DTO 推定硬件 binary layout。

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

具体字段宽度、对齐、CRC、endianness、编码、物理 descriptor window 与 command/event handshake 均由后续规格冻结。`context_id` 是隔离/ABI namespace，不自动等同 Group slot、Tile UCE context 或 Task R lease。

### 4.2 Resource descriptor 草案与真实准入合同

旧版 resource descriptor 的 `l2_window_base/bytes`、`sram_bank_hint_mask` 只是硬件 v0 草案字段；它们不代表当前软件按 buffer 动态分配 L2，也不覆盖 compile-time Arena/Profile 合同。L2 root Arena 在 root 准入时完整保留并 no-rebind；Tile Task L1 Arena 在 Task commit 时保留。容量证明按每个 bank 与碎片布局检查，slot/frame/UCE context/Group execution slot/R lease 是不同资源。

### 4.3 Event 与 action 协议

Group action 的候选对象包含 dispatch、transfer、profile/maintenance control、event wait/signal、barrier、release 与完成相关 action；其软件表示不等同于这里的硬件 op encoding。

- Event 身份必须携带 owner、generation、sequence 与 producer/consumer 关系；旧 generation 的完成不能唤醒新 owner。
- `nest.await` 对应 `WAIT_EVENT`；只等待显式依赖并 fence 本 Context 后续 action registration。
- `nest.barrier` 对应 `BARRIER_GROUP`；只等待该 Context 更早 action，不自动构成全 Group/Device barrier。
- 普通 action dependencies 只等待真实生产者；event/error/timeout/reset 的状态转移必须可观察。
- Profile reconfiguration 与 range maintenance 是显式 action/control 事务；Group `SAME` / `COMPATIBLE` admission 不等于切档，也不产生 Profile generation 变更。
- Action table、event capacity、control credit 与物理信号编码均是独立待冻结参数。

### 4.4 Tile route 与 Stream overlay

`dispatch.role` 绑定静态 `ExecTileRoleBinding`，先建立 bounded Grid Route，再向各个 `tile_id` 独立申请完整 Task admission。一个 Tile 受阻不撤销已 commit 的其他 Tile。Tile Program 的引擎细节和 descriptor encoding 属未建模微架构。

Stream Queue 是可选 producer-consumer overlay，仅在 workload 明确声明时使用其 credit、backpressure、EOS/error 语义。显式 L2 view、event dependency、`input_released` / `output_ready` 可直接表达阶段数据交接；所有 role 不要求经由 token FIFO 串联。FIFO/CDC、token payload 的物理放置与多消费者电路仍待规格冻结。

### 4.5 CSR 草案

以下寄存器名是硬件 bring-up 草案，不是软件快照字段或已实现 CSR：

| CSR 草案               | 目标观察内容                                                   |
| ---------------------- | -------------------------------------------------------------- |
| `GROUP_CONTROL/STATUS` | enable、reset/drain 请求与状态                                 |
| `GROUP_PENDING_STATUS` | pending root metadata 与等待类别                               |
| `GROUP_ROOT_STATUS`    | root identity、Group slot、Profile 与生命周期                  |
| `TGS_ACTION_STATUS`    | 注册 cursor、Group action table occupancy、ISSUE/REGISTER 状态 |
| `GRID_ROUTE_STATUS`    | Route、逐 Tile Task admission / retirement 与 wait reason      |
| `EVENT_STATUS_BASE`    | event owner、generation、sequence、terminal state              |
| `FAULT_RECORD_BASE`    | 来源、request/root/Task identity 与 fault status               |
| `PMU_SELECT/PMU_READ`  | snapshot 选择与读取；counter ID 待冻结                         |

CSR address、位宽、权限、计数溢出与清除语义均由后续规格冻结。

## 5. 数据流、控制流和时序路径

### 5.1 数据路径

```text
HBM / Global binding
  → supported transfer legs → Group L2 Arena / shared backing
  → per-Tile Task view / Tile DMA → Tile L1 Arena and Frame
  → Tile Program / Engine
  → L1→L2 completion and phase signal
  → optional final HBM storeback
```

L2 Arena 属于 root invocation；Task 持有独立 L1 Arena。L2 `release` 只使 view 失效，不回收 owner Arena；L2 layout 当前 no-rebind。L1 可按编译期已证明的 Task 内 lifetime 复用静态 slot/offset，Task 退休才回收整个 L1 Arena 并归还 UCE pin、父 L2 pin 与 R lease。`input_released`、`output_ready` 和 Grid done 是不同事件；`output_ready` 不表示 HBM 持久化。

模型 transfer 按声明合同及受限 fidelity 推进，并遵守 owner/generation/Profile gate。模型的连续 row-major、等字节数 transfer 规则不代表任意转置、strided DMA 或物理 NoC 已实现。Stream Queue 仅在 workload 使用时承载 token/credit，不是 L2 view/event 交接的强制唯一通路。

### 5.2 控制路径

```text
source xDSL → compile_program → immutable CompiledProgram
  → independent verify / load_program(actual bindings) → LoadedProgram
  → CPU Device interpreter: submit / depends_on / await
  → DevicePort → GroupPort pending-root metadata
  → full Group root admission
  → per-root registration cursor / fence / completion
  → Group-shared ready-action scan and single ISSUE
  → bounded Grid Route
  → independent per-Tile Task admission and UCE issue
  → engine/transfer completion → Task/Grid/root completion
  → CPU completion harvest
```

模型中的 CPU step → Group step → CPU harvest 顺序意味着 Group 当前 cycle 的完成最早在下一 CPU step 唤醒依赖；这是模拟推进顺序，不是 Host/FPGA 同步时钟承诺。Root completion、Grid completion 与 Task phase signal 是不同层次。

### 5.3 关键时序路径

需分层观测 root pending 与完整准入 wait reason；Group action dependency/ordering/credit 与有限 scan；每 Tile R lease、slot、L1 逐 bank capacity/fragmentation、Frame/UCE pin；传输和 engine queue credit；Task/Grid/root retirement。并发 stall 是各组件 owner 视角，不直接相加成唯一 Group cycle。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 当前模型参数与硬件资源边界

| 参数                                             | 当前软件模型默认 / 范围                          | 硬件状态                         |
| ------------------------------------------------ | ------------------------------------------------ | -------------------------------- |
| `device_context_count`                           | 默认 1；Device outstanding 上限 1..8             | 硬件 CPU/queue credit 待冻结     |
| `group.active_context_capacity`                  | 默认 8 个 Group root execution slots             | Group physical slots 待 PPA      |
| `group.context_pending_capacity`                 | 默认 32 个 pending metadata entries              | pending queue 深度待冻结         |
| `group.action_capacity` / `context_action_quota` | 默认 16 / 8                                      | action table / root quota 待 PPA |
| `group.scan_width` / policy                      | 默认 4；S0/S1/S2，S2 当前等同 S1                 | scan / issue 电路待 PPA          |
| `context_count`                                  | 默认 1；Tile UCE model 支持 1..8 contexts / Tile | silicon UCE context 数待 PPA     |
| engine ingress queues                            | 各引擎按独立深度约束                             | 各物理 queue depth 待冻结        |

CPU outstanding、Group execution slot、Tile UCE context、Group scan window、engine queue depth、Frame slot 和 R lease 是独立容量。Group/L1/L2 Profile 的实际字节、bank 组织、端口、时钟、ECC、NoC 拓扑与 Queue/credit 实现由后续 Registry/规格/PPA 冻结。

### 6.2 性能与容量模型边界

Group execution progress 受共享 action ISSUE、registration quota/table、依赖/显式 fence、event/inflight/adapter credit、root/Task admission 与 transfer/engine progress 共同约束。当前 Group 每 cycle 至多 ISSUE 一个 action 且至多 REGISTER 一个 action；这两个限额属于不同阶段。S0/S1 影响 ready-action 排序，不等于吞吐、全局公平或 starvation-free 保证。

静态容量检查按已编译 Arena layout 和每 bank Profile 合同证明可行性，不预测 runtime 的 occupancy、碎片、排队次序或实际 latency。等待必须由真实 resource/credit/event 归因，不通过丢完成、隐式同步或 Runtime 修图恢复进展。

### 6.3 PMU / snapshot 归因

模型按组件记录 root admission wait（容量、fragmentation、slot、control resource）、Group action dependency/ordering/credit、Grid/Tile `WAIT_CONTEXT_LIMIT` 与 Task admission、Arena/claim/pin/R lease 生命周期、Profile transaction/ACK 和传输腿。具体 counter 名称以模型 snapshot/PMU 为准，不等同硬件 CSR。

硬件 PMU counter ID、primary/secondary stall arbitration、timestamp、跨时钟域 snapshot 和 overflow/clear 语义尚未冻结。不同 owner 的并发 stall 不能相加成唯一 global cycle；`uce_issue` 也不代表传输完成。

## 7. RTL/软件实现建议

当前软件职责映射为 `GroupPortAdapter`（pending metadata / root admission message edge）、每 root `TileGroupSequencer`、Group 共享 `GroupScheduler`、Tile Group 的 Grid Route / Task commit，以及每 Tile `TileUCE`。这些类是调度/资源语义参考实现，不是对应硬件模块已经 RTL 化的证据。

硬件实现可沿以下边界展开，但不得把软件字段名直接冻结为 CSR/ABI：

- Group ingress / resource admission：pending metadata、Profile 分类队首、完整资源提交与 wait reason 编码。
- Per-root registration：cursor、fence、event owner/generation 与 root completion。
- Shared ready-action scheduler：有限 action table、S0/S1 scan、adapter/inflight credit 与独立 completion reclaim。
- Grid/Tile admission：bounded Route、每 Tile 原子可见 commit、R ledger 与安全退休。
- Event/Profile/PMU/Reset：独立 owner、ACK/generation、fault/cancel-confirm 与跨域合同。
- 数据面与引擎内部 RTL 由 Global DMA、Memory/NoC、Tile UCE 与对应 Engine 规格确定，不从本调度模型外推。

所有 action opcode/字段宽度、queue depth、SRAM macro、时钟域与 CSR 均为 `由后续规格冻结` 或 `由 PPA exploration 冻结`。

## 8. 验证、bring-up 和验收标准

### 8.1 调度与资源合同验证点

- Root request 可经历 pending metadata 但不预占 Group slot、L2 Arena 或半套 event/control 预算；完整 admission 后才成为 ACTIVE。
- CPU outstanding 上限、Group root execution capacity 与每 Tile UCE context 数能分别施加 backpressure，wait reason 不混淆。
- SAME/COMPATIBLE FIFO-head 准入保持同类别队首；S0、S1 及当前与 S1 等价的 S2 遵循有限 scan 窗口规则。
- Per-root registration cursor 按 source 顺序前进；`nest.await` 与 `nest.barrier` 仅 fence/等待当前 Context 规定的 action frontier。
- `GroupScheduler` 每 cycle 至多 ISSUE/REGISTER 各一项；action table 满时 completion 仍归还 credit/event。
- `dispatch_role` 建立 bounded Grid Route；每 Tile 独立、每 tick 最多 Task commit；一个 Tile 阻塞不撤销另一个 Tile 已提交 Task。
- R lease 跨父 Context 的 Grid 计数，只在 Task 安全退休/cancel-confirm 后释放；`free`、`input_released`、`output_ready` 不能误退休 Arena/lease。
- Grid done、root done、Profile ACK、transfer visibility 与 fault/cancel-confirm 的 owner/generation/sequence 可区分。

### 8.2 软件参考运行与硬件 bring-up 边界

`pipeline_validator` 的实际路径使用 `compile_program → independent verification/load → Simulator.run`，并在 1 Group × 4 Tile 范围检查报告、snapshot、资源不变量及所配置的 transfer fidelity。代表性 ready-action、L2 sharing、R/Task admission、Profile 与依赖场景由仓库 `examples/run.sh` 驱动。此规格同步本身不声称已经重新运行这些场景。

硬件 bring-up 仍需独立验证 command/event/descriptor ABI、Group slot/root admission、RTL SRAM/NoC、真实 DMA、Tile/engine execution、CDC/RDC、SVA/formal、fault/reset 与 PMU。Host 可见 buffer、物理带宽、张量算术 golden 和多 Group 行为不是当前 Group 模型的证明范围。

### 8.3 硬件目标验收条目

- 硬件 message acceptance 与完整 root resource admission 有明确、可分辨的 handshake/state。
- 多 root 能按有限注册/issue/完成合同运行，S0/S1 策略无隐式全局 barrier；S2 不要求实现独立策略。
- Grid Route / per-Tile commit 与 R lease 在资源冲突、取消、fault、reset 下无半提交泄漏或错误回收。
- Task/Grid/root completion 及 event/Profile generations 无 stale completion、orphan backing、Arena/Frame/pin/lease 泄漏。
- Stream Queue overlay（若实现）通过独立 credit/EOS/error/CDC 验证，不要求它成为所有 role 的唯一数据通路。
- 只有真实 RTL、物理/时序分析与目标板 bring-up 完成后，才可宣称对应硅片规格已验收。

## 9. 风险、取舍和后续细化方向

主要风险：

- SRAM/NoC contention：Group DMA、Tile DMA、Collective、Stream payload 同时访问 L2，可能压低 BOA/EVU/MFE 有效吞吐。
- Tile Group Sequencer 过度通用化：若演化成小 CPU，会增加验证和时序风险。
- Queue/barrier/event 协议不清：最容易形成不可复现 deadlock 或 reset 后 stale state。
- Collective 与 DMA 共享 SRAM port：reduce tree 和 storeback peak 叠加时可能造成 bank conflict。
- 共享 issue/资源准入边界不清：若 root pending 与 active 资源混淆，或 R/Frame/pin/lease 误释放，会出现不可归因的等待或泄漏。

取舍：

- First Silicon V1 优先固定小而完整的 command/event/DMA/queue/PMU path，而不是追求复杂 scheduling。
- TileGroupTask action encoding 保持控制面指令；数据相关动态访问交给 MFE，tile-local kernel pipeline 交给 Tile UCE。
- L2 Arena 按编译期合同整体预留（no-rebind）；L1 静态 lifetime 复用；不引入运行时动态 pool 或修图。

后续需要冻结：

- Group SRAM profile、bank/port/ECC、arbiter policy。
- 硬件 action encoding、CSR map、fault code（软件 ExecGroupActionOp/DTO 不是其编码）。
- Stream Queue multi-consumer policy 和 reset/drain 精确时序。
- Collective topology、latency model 和 numeric mode。
- PMU counter 编号、溢出语义和 runtime readout ABI。
