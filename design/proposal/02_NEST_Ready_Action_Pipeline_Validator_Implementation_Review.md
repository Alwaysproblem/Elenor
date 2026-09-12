# NEST Ready-Action：Pipeline Validator 现状审查与落地设计

| 项目     | 内容                                                                                                                              |
| -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 日期     | 2026-09-12                                                                                                                        |
| 状态     | **供 Review 的实现设计；本次只新增报告，不修改 validator 实现**                                                                   |
| 输入     | [01_NEST_Ready_Action_Architecture_Proposal.md](01_NEST_Ready_Action_Architecture_Proposal.md)                                    |
| 代码范围 | `pipeline_validator/`，以及 `examples/run.sh` 和相关可执行 MLIR 场景                                                              |
| 证据等级 | **已实现**＝当前代码存在对应行为；**实测**＝本次实际运行并检查结果；**[INFERENCE]**＝影响推导，尚非性能测量；**建议**＝待实现契约 |
| 验收边界 | Python/xDSL 功能与周期模型，不包括 tensor 数值正确性、RTL 对拍、Fmax、面积或功耗结论                                              |

## 1. Review 结论

**值得落地的是：在现有三层 IR 和内存生命周期机制上，增加有界 Group action 可见窗口，并把登记、发射和完成分开。不是重新搭建一套 NEST，也不是简单增加 context 数。**

当前仓库已不是只有设计文档或空 validator：存在可执行 `nexus.program → nest.context → tile.program`、1–8 个可配置 context、真实 L1/L2 分配、逐腿传输、阶段信号聚合和可运行的 NEST 子图案例。不能按旧 README 摘要或 proposal 中“本轮没有指定真实源码仓库”的前提从零规划。

### 1.1 已有基础，不应重复实现

- **三层执行入口**：Device 顺序解释 submit/await/return；Group 执行 context；Tile 多 context 执行程序。[E01][E02]
- **真实 L2 admission**：一次 plan/commit 全部声明的 L2 buffers，失败零部分分配；区分临时容量不足和永久不可满足。[E05]
- **生命周期保护**：真实 read/write actual 集、logical-task phase 聚合、owner/generation、consumer pins、未完成 transfer 检查、显式 L2 release 和 Tile L1 free。[E06][E07][E09]
- **完成不等于 PC 到末尾**：已发出的 grid、phase 和 Group DMA/collective 必须排空；Device 等待 slot 完成。[E01][E04]
- **可观测执行路径**：MFE load/store、Group transfer、HBM/NoC/bank legs、admission 和 context 生命周期 trace。[E12]

### 1.2 真正缺失的核心

1. **没有共享 ready-action 前后端**：每个 active context 一套 `TileGroupSequencer`，每周期各自 step；同一 context 只有当前 action 可见。[E03][E04]
2. **普通依赖仍阻塞 Group PC**：`depends_on` 被 lowering 成前置 `WAIT_EVENT`，无法让后方独立 action 越过等待。[E03]
3. **没有有界 action/event/inflight 联合 admission**：HBM/NoC 等执行资源有限，不代表软件 transaction/event 容器也有限。[E10][E11]
4. **Device 没有 submit 自带的异步依赖描述符**：当前依赖用 `nexus.await` 阻塞 CPU IR；L2 admission waiter 已占 device slot。[E01][E02][E05]
5. **Tile 选择与资源检查尚未完全合一**：当前 READY 选择、held launch 重试和 engine queue 机制是有用基础，但不是 proposal 的统一 instruction eligibility。[E08]

### 1.3 不能照搬的三个默认值

| Proposal 默认                           | 当前事实                                                                                           | 本报告建议                                                                                                                 |
| --------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| D-03：整 context reservation 保持到终止 | 当前每个 L2 对象 eager 分配；经完整 release preflight 后可在 context 结束前 final-free，并唤醒别人 | **保留已证明的最后使用点归还**。不要为了“符合 proposal”强制退化为全程占有；未来物理块复用才区分租约结束与 reservation 结束 |
| §9：一个 Group 仅一个兼容 program epoch | 当前允许不同程序占据不同 Tile context；已有 mixed-program 示例                                     | 作为显式策略实验，而不是无条件替换默认并发行为。必须把 epoch 串行化与 ready-action 效果分开                                |
| §11：独立 JSON/解释器先行、以后接 MLIR  | 已有 xDSL 自定义 assembly、verifier、lowering 和 Simulator                                         | 直接扩展现有链路；JSON 用于配置/结果，不另建竞争性的可执行 IR                                                              |

**推荐审查顺序：依赖与资源安全 → 共享 S0 基线 → S1 ready-action → Tile eligibility → 可选 epoch/队列扩展。** 未实测前，不承诺 S1 比现状或 S0 更快。

## 2. 当前真实执行链路

```text
xDSL Module
  nexus.program                       Simulator._run_model
    submit / await / return           CPU PC；空闲 device slot；done_events
               |
               v
  nest.context                        每次 launch 新建/准备 sequencer
    静态 L2 bundle admission          ADMISSION_WAIT 已占 device slot
               |
               v
  每 context 一个 TileGroupSequencer  Group 每周期遍历全部 active sequencers
    action_index + _pending           depends_on → WAIT_EVENT → 真 action
               |
      +--------+----------------+
      |                         |
  Group transfer             dispatch_role
  transaction 容器           对全部选中 Tile 做原子 L1/context admission
      |                         |
  有限 HBM/NoC/bank 服务      Tile UCE contexts + 有限 engine queues
      |                         |
      +----完成/phase 聚合------+----→ release / drain / context_done
```

### 2.1 三个容易混淆的 context

| 当前对象                                | 实际含义                                                      | 与 proposal 的差异                                                    |
| --------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------- |
| `device_context_count` / Device slot    | 限制同时占用 slot 的 context launch，包括 L2 admission waiter | 不是独立的 pending descriptor 容量，也不是仅计算 ACTIVE 的 Group slot |
| `nest.context` 实例                     | 有独立参数映射、launch generation、sequencer 和 L2 ownership  | 已有状态基础，但没有共享登记前端与 action 表                          |
| Tile UCE `context_count` / `context_id` | 物理 Tile 执行 context；当前 dispatch 可以显式 pin            | 不是 proposal 的逻辑 `tile_context_queue_id`                          |

`Simulator.__init__` 使用 `max(context_count, device_context_count)` 建立 TileGroup。[E01] 因此设置更多 Device slots 会隐式增加实际 Tile context 数，**不能用当前两个 CLI 参数直接证明独立的 Group/Tile 资源扫描**。报告和未来实验必须记录 effective count，而不只记录用户输入。

### 2.2 已有异步能力不等于 ready-action

当前 prefetch/store 可提交给 transfer manager 后继续推进，Tile 引擎也可后台运行；因此“异步执行”和“完成跟踪”不是缺失功能。缺的是：**把尚不能发射的 action 留在有限窗口内，同时继续登记同 context 的后继独立动作**。

旧 `action_index` 的含义也不统一：DMA/dispatch 接收成功后前进；WAIT action 设置 `_pending` 后前进，但之后整个 sequencer 等待；末尾 drain 再决定 done。不能只改这个变量的名字，就宣称完成 C-04。

## 3. Proposal 决策逐项对照

下表的“部分”表示已经有可复用基础，**不是对应完整契约已经成立**。

| ID   | 当前状态                  | 代码事实与缺口                                                                                                                  | 落地判断                                                                    |
| ---- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| C-01 | 已实现三层主体            | xDSL 有 nexus/nest/tile；Device 控制在 `Simulator._run_model`，而非 `runtime/device_runtime.py` 的延迟包装 [E01][E02]           | 复用；CPU 指令带宽仍是模型假设                                              |
| C-02 | 部分实现                  | Tile 多 context、顺序 PC、异步 engine queues 已有；READY 与完整 eligibility 未合一 [E08]                                        | 补充当前指令资源谓词，不重建 UCE                                            |
| C-03 | 未实现目标形态            | 多 sequencer 可并行推进，但没有共享登记表，更没有非 FIFO action 选择 [E03][E04]                                                 | 核心改造                                                                    |
| C-04 | 未实现目标形态            | action_index 主要在执行/接受后前进；无独立 REGISTER [E04]                                                                       | 引入 submission_pc、queued 和 inflight 生命周期                             |
| C-05 | 部分实现                  | 一个 sequencer 等待不阻塞其他 sequencer；不存在共享窗口满的语义 [E04]                                                           | 在共享前端建立局部配额和全局表满两类背压                                    |
| C-06 | 部分实现                  | verifier 检查 SSA/event/访问/release；普通依赖转 WAIT；没有完整可重排 action DAG 和运行时 producer-bound 检查 [E03][E07]        | 必须在 S1 前补齐，不能直接删除 WAIT                                         |
| C-07 | 大部分内存基础已实现      | 所有 L2 对象的 eager atomic bundle 包括输出；未覆盖有界 action/event/inflight 预算，亦无复用布局证明 [E05][E10][E11]            | 保留 allocator，补充资源计划与控制预算                                      |
| C-08 | 大部分实现                | 独立 prefetch、dispatch 三结果、Group store completion、task-bound phases 已有；重复信号和零任务策略与 proposal 不同 [E03][E06] | 保留事件含义，明确冲突再迁移                                                |
| C-09 | 部分实现且 ownership 不同 | L1 allocator 在 Tile，但 Group `dispatch_role` 负责跨 Tile plan/commit/frame/context bind [E09]                                 | 先抽出 Tile-owned admission API，保留原子 rollback                          |
| C-10 | 部分边界一致              | 未新增通用 MPMD/stream；但无兼容 program epoch 门控，dispatch context_id 是物理 pin [E08][E09]                                  | 不能把 residency epoch 当执行 epoch；逻辑 queue 单独设计                    |
| D-01 | 部分实现                  | Device 顺序控制、Group 多份状态已有；submit deps、共享前端和有限后端没有 [E01–E04]                                              | 复用状态，引入共享 scheduler                                                |
| D-02 | 未实现                    | submit 无依赖 operands；等待 L2 的 launch 占 slot；没有 WAIT_DEPS pending 集合 [E01][E02][E05]                                  | pending、依赖门控、active admission 三者分离                                |
| D-03 | 策略不同，不宜原样默认    | eager bundle 已保证未来声明对象；release 会提前归还实际 extents [E05][E06]                                                      | 当前安全早退优于强制持有；需要新增 reserved/live 指标，而不是重写 allocator |
| D-04 | 部分完成跟踪基础          | `_outstanding_jobs`、transaction/role 跟踪存在；没有 action 槽位到有限 inflight 的原子转移 [E04][E11]                           | 有界 adapter 接收事务是核心                                                 |
| D-05 | 未实现                    | 每个 active sequencer 各自最多一 action/cycle；Group 总宽度随 context 数增长 [E04]                                              | 公平 S0/S1 共用 register=1、issue=1 的模型                                  |
| D-06 | 未实现 Group 策略         | Group 按 active list 顺序 step；Tile 是当前 READY 优先与重试机制，不是 Group eligible action RR [E04][E08]                      | 公平 RR 为起点；不先加评分                                                  |
| O-01 | 部分实现                  | Tile/Device count 可配 1–8，真实 memory 参数可配；没有 action/event/inflight/scan_width 配置 [E10][E13]                         | 增加模型配置，数字视为实验参数，不冻结硬件                                  |
| O-02 | 已有相关早退，其他未实现  | L2 release、Tile free 有最后使用点保护；不等于通用动态分配、跨 context 借用或多发射 [E06][E09]                                  | 保留现有早退；动态复用、压力评分、多发射后置                                |

## 4. 详细落地设计

本节的类型名、字段和策略名均为**建议接口**，不是当前已有 API，也不是已冻结 binary ABI。优先修改现有文件，沿用 xDSL 和 execution DTO 分层。

### 4.1 Device：submit dependencies 与 pending/active 分离

**现状问题**：当前 submit 遇到无空闲 slot 就停住 CPU PC；依赖只能写成 `nexus.await`。若 A→C，且 C 后面还有与 A 无关的 B，把 await 放在 C 前面会连 B 的提交一起阻塞。[E01][E02]

**建议**：

1. 给 `NexusSubmitContextOp` 增加 `depends_on` 的 Nexus event operands；同时修改 parse/print、`_verify_nexus_program`、`lower_model_ir` 和 `ExecDeviceOp`。
2. CPU submit 成功的定义改为“有限 pending descriptor 已接收”，不是“已经得到 active slot/L2”。pending 满才对 CPU 背压。
3. pending descriptor 保存 context 实例 ID、不可变 actual 参数映射、依赖 handles、提交周期；依赖失败产生 FAILED，不进入 memory admission。
4. `WAIT_DEPS` 不占 active slot/L2/L1/Tile context；依赖成功后进入 `WAIT_ADMISSION`。检查所有依赖已就绪的 pending，不让未就绪队首挡住后面的独立提交。
5. 依赖就绪集合内部，**第一版仍可保持现有内存 FIFO 策略**；不要把 Device dependency bypass 顺便变成 utilization-first memory admission。
6. 外部 completion handle 使用 launch instance，而非可复用 slot；消费者引用消失后才能回收。
7. `nexus.await` 仍保留 CPU 真正需要同步的位置，不能全部改成异步。

建议状态：`SUBMITTED → WAIT_DEPS → WAIT_ADMISSION → ACTIVE → DRAINING → COMPLETE`；错误走 `FAULT_DRAINING → FAILED`。pending metadata 与 active state 表必须分别配置容量、分别计数。

**[INFERENCE] 正面影响**：减少 CPU 提交队首阻塞；依赖未满足者不抢占稀缺 active slots；有利于较大子图提前提交。

**[INFERENCE] 代价**：外部事件引用寿命增加；pending 数量必须有界；错误需要沿依赖传播。它不能消除 L2 容量 FIFO 队首阻塞，更不能替代 Group 同 context 重排。

### 4.2 IR/Verifier：从 wait 列表转向可重排依赖

**绝不能只删 `ir_lowering.py` 的 WAIT 插入。** 当前顺序执行可能隐含保护内存访问；一旦 action 可乱序，源码先后不再是保护。[E03][E07]

建议在 `ExecGroupAction` 中增加以下语义，不把整个图复制进每个 action：

```text
ActionTemplate（编译产物，可共享）
  ordinal, kind, immutable_payload_ref
  dependency_event_slots[]
  output_event_slots[]
  read/write/release buffer effects

RegisteredAction（运行实例，有界）
  action_instance = (context_instance, region_generation, ordinal)
  owner/context_state_ref
  template_ref + immutable_parameter_instance_ref
  bound input/output event handles
  allocation/reservation generation references
```

落地要点：

- dispatch/store/release 的 `depends_on` 成为 descriptor 边，而不是前置阻塞 WAIT。
- `nest.await` 保留为**显式提交 fence**：前端到此等待真实控制条件；此前已登记动作继续执行。不能为提高 overlap 偷删用户显式 fence。
- compiler/verifier 计算 alias 后的 RAW/WAR/WAW、prefetch→reader、writer→store、所有最后访问→release、物理复用→下一 producer 等边。现有 `bindings/ins/outs` 的真实访问集合必须复用，不能按 in/out 分配标签猜依赖。
- 多个 prefetch/写者访问同一个对象时，需要区分版本或加入有语义根据的访问边；不能把两个 event ID 不同解释成数据独立。
- 跨 context 若使用 HBM 同一实际范围，必须有显式 Device 依赖；不自动推断整个全局 DAG，也不把地址相同但本来无依赖的输入误串行。
- 登记时要求依赖 producer 已 bound，或为已成功兑现的 external event；SSA 定义在前只是编译期条件，不等于生产者已经进窗口。
- 大 context 以有限窗口流式登记，不要求全 region 同时驻留。第一版保持当前静态展开，不顺便引入运行时循环和跨迭代推测。
- `nest.return` lowering 应表达 submission closed / terminal requirements，不能让一个提前 signal 的“完成”绕过尚未登记或尚未完成的 action。

**[INFERENCE] 正面影响**：同一 context 的独立分支真正暴露给 scheduler；依赖成为可检查对象，S0/S1 能共享合法性定义。

**[INFERENCE] 代价**：新增依赖会改变旧时序；漏边导致真实 use-before-ready/overwrite，过度加边则抹平 ready-action 收益。应拒绝不充分的依赖描述，而不是统一加全序边掩盖问题。

### 4.3 Group：共享前端、有限窗口和独立 completion

建议将现有 `TileGroupSequencer` 的 context 状态与每周期调度控制分开；不要继续“一 context 一份可独立发射的 controller”再声称共享宽度为一。

```text
每周期（建议冻结的模型顺序）
  1. 应用本周期可见的 execution completions / fault / credit 归还
  2. 从周期开始时已登记的 action 中检查有限候选，至多 ISSUE 一条
  3. 共享前端轮转选择一个 context，至多 REGISTER 一条
  4. 更新 submission_closed / drain / cleanup / external completion

本周期新 REGISTER 的 action 最早下周期 ISSUE。
表满只阻止 REGISTER；完成与资源回收从不要求表内空位。
```

这是一种保守周期约定，不是声称真实 RTL 必须这样分级。必须固定 Tile 与 DMA 完成何时对 Group 可见，不能依 Python 遍历顺序产生隐式多级穿透。

建议 context 状态最小集合：

```text
context_instance, submission_pc, submission_closed, lifecycle_state
parameter_instance_ref, resource_plan/reservation_handle
queued_count, inflight_count, live_grids, terminal_events
window_quota, registration_age / arbitration cursor
```

发射谓词至少包含：producer SUCCESS、owner 可执行、buffer generation/ownership 正确、reservation 覆盖、adapter credit、inflight metadata credit、所选 epoch policy 允许。

策略只改变选择，不改变合法性：

- **S0**：每 context 最早未发射 action 可参与选择。
- **S1**：同一有限表内所有被有限扫描覆盖的 eligible action 可参与。
- **S2**：增大但仍有限的窗口；其他配置保持一致。

Group 前端与后端各自 RR，维护局部 quota 和全局容量。表满时切换 context 不会凭空创造槽位；一个 context 配额满则可以登记其他 context。有限扫描的游标必须持续推进，避免固定扫描低编号 slot 造成饥饿。

**[INFERENCE] 正面影响**：同 context 中被等待挡住的独立 prefetch/store/dispatch 可以先发；小控制窗口不必等比例增加 tensor 驻留。

**[INFERENCE] 代价**：多 context 现状可同周期执行多条 Group action，新的共享 issue=1 可能更慢；有限扫描增加 wakeup→issue 延迟；规则饱和流水可能没有收益。不能把相对 Legacy 的所有周期变化归因于乱序选择。

### 4.4 Adapter/inflight：有限资源和原子 ISSUE

当前 `TransferManager.submit` 直接将 transaction 写入字典；下游 stage credit 有限，但**入口元数据没有容量拒绝**。[E11] 只设置 action_window_size 会把无限缓冲从一个位置移到另一个位置。

建议接收契约：

```text
try_accept(action) -> ACCEPTED(inflight_handle) | BACKPRESSURE | FAULT
```

- BACKPRESSURE：action 留表；不推进完成计数，不持有半份 credit，不产生成功 event。
- ACCEPTED：参数快照和 owner/tag 已保存；inflight 记录与 adapter credit 同一 commit 生效；action 槽位随即归还。
- FAULT：停止该 owner 新登记/发射，走 drain/reset；不能把永久错误伪装成永远重试。
- prefetch/store/dispatch 有各自有限接收 credit；一种 adapter 被堵不阻塞其他类型的 eligible action。
- `RELEASE/JOIN` 可在本地 commit 完成，但同样消耗本周期发射预算；不必长期占 engine inflight。
- completion 独立于 action 表；表满时仍能 retire transaction、完成 event、解除 pins 并释放资源。
- 已 ISSUE 的 dispatch 必须能依靠已具备的数据和本地执行完成；禁止等待一个必须由未来尚未登记 Group action 才能生成的资源。

**[INFERENCE] 正面影响**：结果不再依赖无限 transaction/event 元数据；能区分 frontend、adapter 和 engine 真正瓶颈。

**[INFERENCE] 代价**：周期数可能增加，这是去掉理想化缓冲而不是 ready-action 性能退化的唯一证据。故障注入和 credit 守恒需要覆盖跨表转移。

### 4.5 Memory admission：保留安全早退，不盲目套整 context 持有

当前 `try_admit_l2_buffers` 在 launch 时分配 `task.l2_buffers` 的全部对象，因此输入、输出等声明空间不是等执行到 store 才临时找。[E05] 对当前“不允许 release 后再访问同一对象”的静态程序，这已经是强资源保证。

当前 `release_l2` 的作用不是仅 unpin：完整 preflight 后可 final-free，触发 FIFO admission retry。[E06] §9 实测表明 B 可在 A context 完成前获得 A 已不用的地址并运行。这不等于不安全的“借用还会再用的额度”。

这份静态保护确实存在于 `_verify_release_graph`：`workload_ir.py:543–547` 拒绝 release 之后再次使用该 allocation，605–618 要求 release 精确依赖全部 reader phases、prefetch 和 store completions；runtime 再检查 phases、pins 与在途访问。**不能因为 DTO 没有 `proven_final` 布尔字段，就推断当前完全没有 last-use 证明。** 但这个证明针对当前不复用的对象，不自动扩展成未来 reservation block 的复用证明，也不能靠一个由输入随意填写的布尔值替代 verifier。

建议分两层落实：

1. **当前 IR 的直接方案**：继续整 bundle upfront 分配；每对象在所有最终依赖完成后永久退役，允许归还。`reserved_remaining` 可随经过证明的对象终结减少。
2. **未来增加复用 layout 时**：引入 reservation block 与 buffer lease 的区别。A 的 lease 结束后若同 context 的 B 还要复用该 block，则不对外归还；编译器证明该物理 block 未来也不再使用后，才结束 reservation。

资源 accounting 至少区分：

| 指标            | 定义                                                            |
| --------------- | --------------------------------------------------------------- |
| reserved bytes  | 尚未撤销的资源承诺，包含已使用部分                              |
| allocated bytes | 当前实际对象/块占有的 extents；当前已有相应 peak                |
| live-data bytes | 处于已定义的数据有效生命周期中的数据范围；需按 range/alias 去重 |
| free bytes      | 未被其他承诺占用、真正可供新 admission 使用的容量               |

`reserved + used` 不能重复扣容量；`allocated` 也不能直接当作有效 tensor 数据。没有真实数值模型时，live-data 只是协议有效性，不是测得的有效非零数据量。

新增 admission 检查：event/参数预算、active slot、最小控制窗口保证；不可满足请求立即失败，临时不足进入等待。不要在本次同时做分阶段 admission、跨 context 借用或运行时任意分配。

**[INFERENCE] 正面影响**：保留既有跨 context 重叠；显式承诺可支撑重排生命周期证明。

**[INFERENCE] 不足**：eager 分配所有独立对象仍偏保守，长 context 的静态 footprint 大；真正的物理复用需要新的编译器证明，不能靠 allocator 更积极地分配来补。

### 4.6 Tile：统一 eligibility，逐步迁移 local admission ownership

当前 Tile context 数支持 1–8，不需要把“支持三个以上 context”再列为未实现。[E13] 当前 UCE 已有 engine queue、held launch、等待 context 切换；应在这些机制上改造。[E08]

建议先建立无副作用的当前指令检查：

```text
probe_head(context) -> ELIGIBLE | WAIT_EVENT | WAIT_ENGINE_QUEUE |
                       WAIT_STREAM | WAIT_LOCAL_RESOURCE | FAULT
```

然后 RR 从 eligible heads 中选择一条 commit。**engine 正忙不必然不可发射**：若可进入有限命令队列，仍可接受；谓词应检查接收 credit，而不是强制等 engine idle。Context 内 PC 顺序、event hazards 和当前有限 queue 深度必须保留。

L1 ownership 分两步：

- 当前 dispatch 对所有目标 Tile 原子 plan/commit/prepare/pin/bind，已有 late failure rollback；这项安全能力必须保留。[E09]
- 把每 Tile 的 L1/context 选择和 reservation 移到 Tile-owned prepare/commit/abort API；Group 只组织请求和跨 Tile commit barrier。第一步可保留当前 gang dispatch，不强行同时引入 task stealing。

L1 当前“空间不足时 fault”与 proposal 的 local BACKPRESSURE 需要区分：非法或空池也放不下是永久错误；合法但暂时不足才应等待，而且等待不能残留部分分配。

**[INFERENCE] 正面影响**：减少一次选中后才发现不可发射的空泡；避免 Group 直接管理每个 task 的局部状态；让更多 context 数成为有意义的独立实验参数。

**[INFERENCE] 代价**：候选探测次数和 Python 仿真时间增加；要防止 probe 本身改变 queue/event 状态；不能承诺执行单元已饱和时增加吞吐。

### 4.7 Grid、逻辑 queue 和 SPMD epoch

当前 verifier 要求 task 数等于 placement popcount，runtime 把逻辑 task 映射到选中 Tile；并不是可 oversubscribe 的逻辑 task 队列。[E07][E09]

还存在显式的 Device→Tile 绑定耦合：`_prepare_context_launch` 会把未 pin 的 dispatch 自动设为 `binding.context_id = slot_index`，并要求 slot_index 在 Tile context 范围内（`tile_group.py:1980–1996`）。所以即使源码不写 physical pin，当前 model 路径也不是完全自由的 Tile local placement；逻辑 queue 迁移必须改这条生产路径，不能只增加一个未使用的 queue_id 字段。

因此分清三个任务：

1. **Ready-action 最小落地**不必先改 task mapping：保留当前 dispatch 作为原子宏动作。
2. **逻辑 queue 落地**则需新增有限 grid/task 投递记录、唯一领取、Tile local admission 和退出计数；`context_id` 不能简单改名 queue_id 后继续拿它索引物理 contexts。
3. **SPMD epoch policy**使用兼容 program/ABI key，并统计已接收但未完成的 tasks；已有 program residency reset epoch 不是这个语义。

建议先以现有 mixed-program 行为保持 Legacy 观察基线，再单独实验 `same_program_epoch`。若严格采用 proposal epoch，必须预期 BOA/EVU 混合程序重叠减少，修改相关例子的预期并报告原因；prefetch/store 不应被计算 epoch 一律冻结。

通用 oversubscribed barrier 不在本次范围；集体操作需要显式 gang 保证或阶段完成同步。当前 collective 是一周期控制窗口，不具备可用于吞吐分析的 reduce datapath。[E09]

### 4.8 事件、零任务、重复和错误

应保留现有 `GridInstanceId + TaskIdentity` 和 `_live_launches` stale 判定。[E06] 需要补充：

- 有界事件实例槽、producer binding、消费者引用计数、owner/generation 参数化信号；不能让无限字符串集合充当已实现的有界事件硬件。
- `UNBOUND/PENDING/SUCCESS/ERROR` 的区分应覆盖 S0/S1 共用的依赖谓词。现有 EventTable 的 signal 不携带 producer 提交时的 expected generation，不能仅凭类里有 sequence 就认定完整 handle 协议完成。[E10]
- 当前 duplicate phase signal 计数后忽略；proposal 要求可靠传输中重复是协议错误。建议目标契约采用“fault，但绝不二次计数”，同时迁移当前幂等测试；stale retired launch 信号仍可丢弃并计数，不与 live duplicate 混同。
- 当前空 context 可以正常完成；不等于零 task dispatch 完成事件已实现。`workload_ir.py:236–238` 明确拒绝 `num_tasks <= 0`，并另有 task-count/placement 相等约束。第一版可继续明确拒绝零 task dispatch，编译器将静态零工作降为空 context；若要求完整 proposal 零 grid 语义，则必须实现三个里程碑的即时、且仅一次完成并新增用例。
- 不能为了前端关闭，提前返回 context_done。至少要求 `submission_closed && queued==0 && inflight==0 && grids_drained && terminal_events_success && cleanup_done`。
- V1 可沿用现有 Group fault/reset drain 范围，不在本次引入精确回滚或每 context 独立故障隔离；失败链必须不产生成功 completion。

### 4.9 已有 barrier 必须先澄清，不能当现成 JOIN

`NestBarrierOp` 在方言中名为 `nest.barrier`，lowering 产生 `BARRIER_GROUP`，但当前执行分支只增加 `tgs_barrier` 计数并推进 action_index，没有等待任何 event/grid/job（`tile_group_sequencer.py:258–260`）。这是可从源码确认的语义缺口；**本次未运行专门 barrier 反例，不把它写成已实测的故障**。

引入可重排后，这个问题不能被顺序 PC 掩盖：不能把现有 barrier opcode 直接当作 proposal 的 JOIN 或 region fence。建议 P0 冻结为明确的 context 内前驱完成集合及后继控制边；若当前 IR 无法表达所需集合，则 verifier 清晰拒绝，而不是静默当 no-op。跨 context 同步仍通过显式 Device 依赖，不扩展成通用跨 task barrier。

同时保留 `_retire_sequencer` 的最后防线（`tile_group.py:1674–1728`）：正常完成仍有未释放 L2 handle 或残留 grid pin 时转 invariant fault，不能为了让新 scheduler 完成而静默清掉残留状态。

## 5. 改进的收益与不足：不能混在一张加速比里

| 改动                              | [INFERENCE] 可能优化                           | [INFERENCE] 主要不足/代价                       | 应观察的证据                                            |
| --------------------------------- | ---------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------- |
| Device pending deps               | 解除 CPU await 引起的无关提交阻塞              | 更长 event 寿命、pending 压力                   | WAIT_DEPS 不占 active/L2；独立 B 在 A 完成前被接纳      |
| Group S1 小窗口                   | 越过同 context 的依赖等待，提前启动独立分支    | 更大窗口/唤醒/扫描成本；不一定有足够独立性      | ordering-stall 下降及真实 ISSUE 提前，不是只看 REGISTER |
| 共享 issue=1                      | 模型带宽明确，便于硬件取舍                     | 可能比多 sequencer Legacy 慢                    | Legacy 与 S0 的差异单列                                 |
| 有限 event/inflight/adapter       | 消除无限元数据带来的虚假 overlap               | 暴露新瓶颈、周期可能上升                        | 满表时 completion 推进；峰值从不超容量                  |
| Tile eligible-head RR             | 把不可发射 context 让给其他 load/store/compute | 探测/仲裁成本，频繁切换可能无收益               | 同 cycle 有其他可接收指令时不空转；lane credit 守恒     |
| 保留 L2 静态最后使用点归还        | 减少跨 context 容量等待                        | 需要完整未来访问证明                            | release 之后无引用；旧 generation 无法影响重用对象      |
| 强制整 context 持有（不推荐默认） | 简化 reservation 寿命                          | 否定现有早退重叠，大 context footprint 长期占用 | `l2-admission-wait` 中 B 延迟；与调度收益分离           |
| same-program epoch（可选）        | 简化硬件 program 兼容管理                      | 混合 BOA/EVU 程序可能串行                       | epoch_wait 与 mixed-program overlap 单列                |
| 逻辑 task queue                   | 允许 task domain 大于物理槽，解耦 pin          | 唯一领取、部分投递聚合、队列容量验证复杂        | 未投递 tasks 仍在聚合 domain；无重复领取                |
| reserved/live 分离                | 避免把容量承诺误当有效数据利用                 | range alias/lifetime accounting 成本            | 两类占用独立曲线；不用 occupancy 最大化当目标           |

不能把“更多 active context”“更多 L2”“更多 Group issue 宽度”和“更大窗口”同时打开，再把收益全部标成 ready-action。

## 6. 不适合直接引入的想法

| 想法                                                             | 判定                       | 原因与替代                                                                      |
| ---------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------- |
| 再做独立 JSON interpreter，之后接 MLIR                           | **不适合当前项目**         | 重复现有 xDSL/verifier/lowering；改现有可执行链路                               |
| 本次以 RTL 对拍、P&R、Fmax、SAIF/板级功耗作为 validator 签出条件 | **超出当前交付边界**       | 仓库没有对应 RTL 实现/工具证据；软件签出与未来硬件签出分开                      |
| 大窗口等同很小面积、Python 全表扫描等同免费硬件                  | **不能采用此推断**         | 没有端口、读延迟、扇出、布线证据；有限 scan_width 建模，物理结论后置            |
| 通用 ROB、重命名、推测、精确回滚、全局 DAG 最优调度              | **不适合 V1**              | proposal 自身也是非目标；会改变项目问题而非验证有限调度                         |
| 把 stream/MPMD pipeline、atomics、Scatter/Gather 重设计一起加入  | **不适合本次范围**         | 已有 Gather/stream 路径是保留对象，不是本次重排必需改动                         |
| 无条件以一个 program epoch 串行全部 mixed-program 工作           | **不适合直接作为默认切换** | 改变既有并发能力；作为可选硬件策略独立评估                                      |
| 无限 pending/action/event/transaction 用于展示理想吞吐           | **不能作为实现验收**       | 可作明确标注的离线上界，但不进入 S0/S1 公平结果                                 |
| 优先实现动态 reservation 借用/阶段准入/压力评分/多发射           | **暂缓，而非永久否定**     | 需要新资源证明或测量基础；先完成静态安全与有界后端                              |
| 把 `input_released` 当整 context capacity 可归还                 | **语义不成立**             | phase 只说明某输入访问阶段结束；当前显式 release 另有完整保护                   |
| 直接删除 context_id pin/现有 epoch 行为而不迁移案例              | **不适合机械修改**         | pin 是当前真实物理约束，改变它会改变实验图和资源争用；逻辑 queue 需完整接口迁移 |

## 7. 分阶段实施与文件级工作包

以下是后续实现计划，不表示本次已经修改代码。P0–P3 完成最小可评估 ready-action；P4/P5 是互相独立的进一步切片，不能用它们拖延或掩盖 Group 核心语义问题。

### P0：冻结安全契约和保存 Legacy 证据

**涉及**：`IR_SPEC.md`、`execution_ir.py` 设计、当前 examples 与 trace 分析方法。

- 冻结 action 类型、依赖语义、显式 await fence、REGISTER/ISSUE 时点、buffer 最后访问、terminal completion 和错误策略。
- 区分当前提前 final-free 与未来复用 lease；明确零任务选择、duplicate fault 选择和 epoch 是否启用。
- 澄清 `nest.barrier` 的实际同步契约；当前 no-op 不可直接复用为 JOIN/fence，也不可拿它作 S0/S1 正确性证据。
- 保存当前源码 hash、完整 hw/sim/bindings、场景命令和 Legacy 输出；不把旧多 sequencer 直接命名 S0。
- 明确现有 `runtime`/`full_memory` 的 allocation/pin/rollback/generation 断言必须保留；`timing_only` 不能提供内存安全验收。

**退出标准**：每个 future action 有明确输入/输出事件、访问集合和资源需求；没有靠源码顺序才能解释的隐式 hazard。

### P1：依赖 DTO 与有界资源契约

**涉及**：`dialects/elenor.py`、`workload_ir.py`、`ir_lowering.py`、`execution_ir.py`、`runtime/event_table.py`、`memory/transfer.py`、`config.py`。

- 增加 action deps/outputs/effects 和实例化 event handle；保留模板共享和 launch 参数独立。
- verifier 建立可检查依赖、拒绝 producer 未绑定/内存边不完整的输入。
- 设计有限 adapter/event/inflight 接收，BACKPRESSURE 与 FAULT 分离。
- 在 `config.py` 定义 pending/active/action/quota/event/inflight/scan/register/issue 容量；硬件数量由后续规格冻结，实验选择不等于冻结值。

**退出标准**：每种资源可独立压满；拒绝接收无副作用；完成通路在 action 表满时仍推进；错误不走 success。

### P2：共享 S0，不先开启同 context 重排

**涉及**：`tile_group_sequencer.py`、`tile_group.py`、`simulator.py`、`pmu.py`、`trace.py`、`report.py`。

- 将 context 状态与共享 per-cycle controller 分离；所有 active context 共用 register=1/issue=1。
- 同一后端落实 REGISTER→QUEUED→ISSUE→INFLIGHT→COMPLETE；S0 仅允许每 context 最早未发射 action 参与。
- 统一周期可见性、submission_closed、context_done 和 fault drain。
- 输出四时点、control/data/inflight 三窗口和分层 stall。

**退出标准**：现有有效案例在新资源假设下完成；安全断言不退化；相对 Legacy 的周期变化能解释为带宽/容量/延迟变化。不能要求所有旧周期数保持不变。

### P3：开启 S1/S2 与公平评估

**涉及**：同一个 Group scheduler 的 candidate selection、`config.py`、CLI、报告与场景。

- 在共享 S0 机制上增加 ready-action eligible 集合与 RR；S2 只改变有限窗口配置。
- 引入同 context 独立分支反例、窗口满而完成可推进、某 adapter 满但另一类可 issue 的场景。
- 同输入 DAG、同 memory admission、同 Tile count/engine queues、同 event/inflight 容量比较 S0/S1；大 context 不要求全图入表。

**退出标准**：正确性和有界性全满足；至少能解释一个 head-of-line 场景是否受益，以及一个饱和/无独立性场景为何不受益。没有收益也允许结论为保留 S0，不通过修改数据边制造收益。

### P4：Device pending deps 与 Tile eligibility

这两项可以在 P1 接口稳定后分别实施，并各自做消融，不必互相等待。

| 切片                          | 文件                                                                                                         | 退出标准                                                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| Device deps/pending           | `dialects/elenor.py`、`workload_ir.py`、`ir_lowering.py`、`execution_ir.py`、`simulator.py`、`tile_group.py` | A→C、B 独立：C 可先 pending，但不占 active/L2；B 不受 C 的依赖队首阻塞；依赖 ERROR 不准入                           |
| Tile eligible/local admission | `tile.py`、`tile_group.py`、`memory/l1_slot_frame.py`、`config.py`                                           | 当前 MMA 无接收 credit 时可选其他 context load/store；临时 L1 不足背压零部分占有；late bind failure 仍原子 rollback |

需要先解除 `max(context_count, device_context_count)` 的隐式耦合并确定资源策略，才能对两个维度做独立容量扫描；不是简单去掉 max 后让旧 gang dispatch 随机失败。

### P5：逻辑 task queue / epoch / reservation 复用实验

**进入条件**：P3 已说明瓶颈在哪里；确实需要 oversubscription、兼容 epoch 或 footprint 优化。

- 逻辑 queue 完整替换 physical pin 语义时，迁移 parser、DTO、dispatch、local admission、全部受影响案例和测试，不保留同名字段两种解释。
- 保留完整 expected task domain，包括尚未投递者；zero grid 按最终决策执行。
- 复用 layout 必须有 release→reuse 边与 reservation block 生命周期证明。
- epoch、内存策略和 ready-action 分开配置和报告。

**退出标准**：新契约的正负场景通过；对旧能力的限制有明确记录和收益/代价证据。RTL/物理实现仍是另一项目阶段。

## 8. 验收场景与性能报告规范

### 8.1 必须覆盖的行为

| 场景                                                | 必须观察到的行为                                                      | 防止的真实错误                                 |
| --------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------- |
| 同 context：慢 A→dispatch A，后有独立 prefetch B    | S1 可先 ISSUE B；S0 受 head 限制；A 的依赖仍满足后才 dispatch         | 只改登记顺序却没有执行重叠；或错误越过真实依赖 |
| A/B 双输入到达时间不同                              | dispatch 等待两者；一方 ready 不冒充双方 ready                        | 合并 prefetch/丢依赖                           |
| W=1 小窗口、先 producer 再 consumer                 | 生产者 issue/complete 后持续推进；窗口满不阻塞 completion             | 登记/完成循环依赖                              |
| C0 quota 满、C1 有额度；随后共享表全满              | 先可切 C1；全满只停前端，后端继续                                     | 把局部和全局阻塞混为一谈                       |
| store adapter 无 credit，prefetch adapter 有 credit | 后者合法 action 仍可 issue                                            | 单共享输出寄存器造成跨类型 HOL                 |
| ISSUE 中任一资源不足/注入 late failure              | 无半个 inflight/credit/descriptor 消失；可重试或 fault drain          | 资源事务非原子                                 |
| admitted context 覆盖全部输出/workspace             | 执行不在末尾再申请未预算输出；不可满足请求立即拒绝                    | 输入挤占输出容量                               |
| 3/4 logical task signals；未来扩展时还有未投递 task | 不提前 phase complete/release                                         | 按当前 resident 数而非完整 domain 聚合         |
| live duplicate、stale launch、wrong owner           | duplicate 按新策略 fault且不二次计数；stale 不影响新代；foreign fault | event/slot 重用混淆                            |
| output_ready 后慢 HBM store                         | context_done 晚于最终 store completion                                | 将 Tile 输出可见误作 Host 输出可见             |
| early release 后另一 context 复用相同地址           | 新 allocation ID/generation 不受旧完成影响                            | use-after-free 与迟到完成                      |
| delayed reader 与 early writer/store                | store 完成不释放仍有 reader 的对象；release 等完整访问集合            | 按 allocation role 猜使用方向                  |
| 暂时不足与空池也不足                                | 前者 WAIT，后者 FAULT，均无部分分配                                   | 无界等待永久不可能请求                         |
| pending C 等 A，B 独立                              | C 不占 active/L2，B 可以推进                                          | Device dependency HOL                          |
| MMA queue 满且其他 context load 可接收              | 同周期候选选择不只看 READY                                            | Tile 选中后才发现不能发射                      |
| epoch 不兼容但 DMA 可执行                           | 只阻止不兼容 dispatch，不冻结 prefetch/store                          | 把计算 epoch 扩成全 Group fence                |
| 空 context 与零 task dispatch                       | 分别验证；若后者不支持则清晰拒绝                                      | 用空案例代替零 grid 聚合验证                   |
| 显式 barrier/fence 遇到未完成前驱                   | 后继遵循已冻结的控制边；不支持的 barrier 清晰拒绝                     | 只计数并跳过同步的现有语义缺口                 |
| 任何 phase/transfer fault 遇到满表                  | 停止新 work、排空/取消、无 success，资源零泄漏                        | fault 清理依赖新 action 空位                   |

现有相关测试可作为基础，例如 `test_partial_signals_do_not_complete_phase`、`test_duplicate_signal_does_not_advance`、`test_stale_launch_signal_ignored`、`test_late_context_bind_failure_rolls_back_all_tiles`、`test_early_store_preserves_delayed_reader`、`test_full_run_ab_overlap_release_wakes_b`。这里只是源码覆盖清单，**本次未运行 pytest，不把测试名称当作本次通过记录**。

### 8.2 公平配置

| 维度                 | Legacy                              | S0/S1 比较要求                                                                |
| -------------------- | ----------------------------------- | ----------------------------------------------------------------------------- |
| Group 控制带宽       | 每 active sequencer 一 action/cycle | 同一共享 register/issue 宽度                                                  |
| admission            | 当前 eager bundle + final-free FIFO | 完全一致；是否保留早退必须相同                                                |
| actual Tile contexts | 当前 max 耦合后的数量               | 使用实际数量，不只比较 CLI 输入                                               |
| 数据容量与执行延迟   | 当前硬件 profile                    | HBM/L1/L2/NoC/DMA、engine queues、residency 状态相同                          |
| 元数据               | 存在无界容器                        | action/event/inflight/adapter/pending 全部有限且相同                          |
| 输入图               | 当前案例                            | 相同数据边、内存边、buffer 范围和 synthetic work；不能通过删 await/边变更问题 |
| epoch                | 当前无兼容 key gate                 | S0/S1 必须同策略；开关另做消融                                                |

S2 增大窗口时必须明确新增的控制状态成本；若同时增加 event/inflight 容量，应作为另一资源配置，不能称“只扩大可见性”。

### 8.3 指标与定义

- makespan：使用明确起止周期约定。当前 report 的 `cycles` 是 Simulator 循环停止时的 cycle 值，不直接假定等于从 0 起的执行 tick 数。[E01]
- 每 launch 的 submit→complete、admit→complete、WAIT_DEPS、WAIT_ADMISSION、active/drain 时长；足够多请求时才汇报分位尾延迟和稳态完成间隔。
- engine accepted/start/complete、busy、queue-full；Tile issue attempt 与实际接受分开。
- reserved/live/allocated occupancy；action/inflight/event/pending 峰值和容量；scan/wakeup 延迟。
- frontend stall、dependency wait、adapter credit wait、inflight full、epoch wait、memory admission wait 分层记录；多个 context 的 cycle 累加不能伪装为单 Group 时间。
- ordering stall 仅在存在其他**合法且资源允许**的候选、但被 head 策略挡住时成立。只看 `WAIT_EVENT` 或低利用率不能归因 ordering。
- 超出有限窗口、需要未来图信息的机会损失可以离线计算，但标为 offline oracle，不冒充 PMU。
- `clock_mhz` 是模型输入，不是 achieved Fmax；不得由模拟 cycles 推导已测得能耗或实际硅片吞吐。

## 9. 本次实际运行证据

### 9.1 环境与方法

运行环境：conda `elenor-validator`，Python **3.11.15**，xDSL **0.69.0**，PyYAML **6.0.3**。全部使用真实 CLI/示例入口，开启 `--memory-trace` 和 Perfetto JSON；不修改输入 IR。

每项复现命令形式如下；目录需已存在，例如使用 `/tmp`。硬件、binding 和 count 的场景默认见 `examples/run.sh:231–337`，不要脱离脚本自行改小工作量。

```bash
bash examples/run.sh pow-dual-context --json --memory-trace --trace-json /tmp/review-pow.json
bash examples/run.sh l2-admission-wait --json --memory-trace --trace-json /tmp/review-admission.json
bash examples/run.sh nest-n04-fifo-hol --json --memory-trace --trace-json /tmp/review-fifo.json
bash examples/run.sh nest-n09-impossible --json --memory-trace --trace-json /tmp/review-impossible.json
bash examples/run.sh nest-t03-single-zero --json --memory-trace --trace-json /tmp/review-empty.json
bash examples/run.sh nest-s04-single --json --memory-trace --trace-json /tmp/review-branch.json
```

### 9.2 实测汇总

| 场景                 | CLI exit          | report cycles | completed | L2 peak allocated bytes | memory issued/completed | 证据边界                                                            |
| -------------------- | ----------------- | ------------- | --------- | ----------------------- | ----------------------- | ------------------------------------------------------------------- |
| pow-dual-context     | 0                 | 55,160        | true      | 262,144                 | 20/20                   | 两个 Device/Tile context 的现有异步运行，不是 ready-action          |
| l2-admission-wait    | 0                 | 29,052        | true      | 262,144                 | 16/16                   | 真实 release 唤醒和提前重叠                                         |
| nest-n04-fifo-hol    | 0                 | 43,412        | true      | 196,608                 | 30/30                   | 当前严格容量 FIFO 的 HOL                                            |
| nest-n09-impossible  | **1（预期负例）** | 8             | false     | 0                       | 0/0                     | `faulted: L2 capacity fault during context admission`，不是超时等待 |
| nest-t03-single-zero | 0                 | 2             | true      | 0                       | 0/0                     | **空 context，无 dispatch**；不能证明 zero-task grid 语义           |
| nest-s04-single      | 0                 | 38,337        | true      | 262,144                 | 52/52                   | 当前同 context 子图基线；未运行尚不存在的 S1                        |

所有六项输出 `credit_invariant_ok=true`。外部 IR 的 CLI checks 在本次这些输入上只有 task_completed 和 credit_invariant；负例 task_completed=false 是期望结果。**CLI exit 0 不证明图依赖、phase/release 时序或 tensor 数值全部正确。**

### 9.3 从本次 trace 检查的具体不变量

**A. 已有提前释放，不能误判为缺失**：`l2-admission-wait`。

| 事件                                       | cycle  |
| ------------------------------------------ | ------ |
| A admission、B 进入 ADMISSION_WAIT         | 0      |
| A 的 a_input final-free；B admission       | 6,984  |
| B first Group action                       | 6,985  |
| B context_done                             | 12,683 |
| A 最终 HBM write 与 Group store completion | 29,044 |
| A 的 a_output release                      | 29,045 |
| A context_done                             | 29,051 |

实际检查：B admission 后下一周期才 first action；B 在 A context_done 前完成；A 的最终 store 完成不晚于 context_done。这证明当前 early final-free 支持跨 context 重叠，**不证明新的 reservation-layout 复用或 ready-action 已实现**。

**B. 当前内存 FIFO 队首阻塞**：`nest-n04-fifo-hol`。

- 容量 196,608 B；Holder 输入 65,536 B、输出 131,072 B；Head 要 131,072 B，Tail 要 65,536 B。
- Holder 输入在 cycle **3,571** 释放；此时空间足够 Tail，但不够队首 Head。
- Head/Tail 均到 cycle **35,025** 才 admitted，并均在 **35,026** 执行 first action。
- 3,571→35,025 的 **31,454 cycle** 窗口反映当前 admission 策略，不是 S1 可以自动消除的 Group ordering stall。

**C. 实际执行路径**：

| 场景              | `MFE:load` 完整 X slices | `MFE:store` 完整 X slices | Group `global_store` summaries |
| ----------------- | ------------------------ | ------------------------- | ------------------------------ |
| pow-dual-context  | 8                        | 8                         | 2                              |
| l2-admission-wait | 8                        | 4                         | 1                              |
| nest-n04-fifo-hol | 12                       | 12                        | 3                              |
| nest-s04-single   | 28                       | 16                        | 4                              |

对这四项逐个 Group store 检查了 `completion_cycle <= owner context_done`。对 admission 场景，还按同一 transaction_id 核对了最终 `hbm_write` X slice 的完成周期为 29,044。

当前 Group DMA summary 的显示名称是 `context / buffer`，真正语义在 `args.summary_kind=group_transfer`、`args.op=prefetch/global_store` 与 transaction_id；**不能继续只 grep 历史 `dma.store:*` 显示名来判定写回是否存在**。[E12] 分析时只取 `ph=X`，因为同名 memory leg 也有 s/t/f flow 事件。

`uce_issue` 是尝试记录，不代表 engine 已接受；唯一性也要按 `(pid, tid, ctx_id, pc)` 的 lane 范围解释。本次 admission、FIFO、s04 分别出现 `uce_unknown_event=4/4/12`；代码会把 Group DMA completion 广播到 active Tile，而 Tile 对非本地 owner、非 `ev_dma_` 名称计此数。[E08][E12] 这不是本次已证明的数据损坏，但说明名称驱动路由不应作为新 typed-event 协议的最终实现。

### 9.4 本次没有声称的证据

- 没有实现或测量 S0/S1/S2，不提供加速比。
- 没有运行项目 pytest/pre-commit；本次为只读实现调查和文档交付，运行的是上列真实 CLI 场景和 trace 不变量检查。
- 没有穷举所有 NEST 子图，也没有用一次 completed=true 代替其全量图正确性审计。
- 没有 tensor 数值、随机压力、RTL、FPGA、频率、面积或功耗验证。

## 10. 建议 Review 时优先确认的决策

| 决策                            | 推荐值                                                     | 选择其他方案的主要影响                                   |
| ------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| 是否落地有限 Group ready-action | 是，P0→P3；先共享 S0 再 S1                                 | 只增加 context 数无法暴露同 context 独立分支             |
| L2 是否强制持有到 context_done  | **否**；保留经证明的静态最后使用点归还                     | 强制持有更简单，但现有提前 admission overlap 会退化      |
| 是否同时扩展动态 task queue     | 否，最小落地保留现有 gang mapping                          | 同时做会扩大 local admission/aggregation 的风险面        |
| 是否默认强制 same-program epoch | 否，先做可选独立实验                                       | 会改变已有 mixed-program 并发，不能归因给 ready-action   |
| zero-task dispatch              | 最小路径静态消除为空 context；若签完整 proposal 再明确实现 | 空 context 不能冒充零 grid 的里程碑验收                  |
| live duplicate signal           | 目标改为协议 fault、不二次计数                             | 保持幂等会继续偏离 proposal，但可作为明确 transport 策略 |
| 第一版资源与扫描数值            | 有限、可配置、小窗口扫描；不冻结硬件值                     | 过早固定大窗口容易隐藏端口/唤醒成本                      |
| 本次验收范围                    | 可执行 xDSL + 有限资源周期模型 + 真实 trace                | RTL/PPA 是后续独立交付，不能由 Python 仿真代替           |

上述是可 review 的建议，不要求现在为了产出报告逐项确认；后续执行时应按 review 结论冻结，避免混用互不相容的默认值。

## 11. 代码证据索引

行号对应本次读取的源码；链接指向当前文件，后续修改可能使行号移动。代码事实优先于 README、旧 proposal 和历史审查结论。

| 编号 | 文件与范围                                                                                                                                                                                                                              | 支持的结论                                                                                                                 |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| E01  | [simulator.py](../../pipeline_validator/simulator.py)，145–156、250–414                                                                                                                                                                 | 三层入口；max context 耦合；Device PC、slot、await、done_events、fault drain                                               |
| E02  | [dialects/elenor.py](../../pipeline_validator/dialects/elenor.py)，1199–1308；[execution_ir.py](../../pipeline_validator/execution_ir.py)，346–364                                                                                      | Nexus IR 已有，但 submit/DTO 无 dependencies 字段                                                                          |
| E03  | [ir_lowering.py](../../pipeline_validator/ir_lowering.py)，139–368                                                                                                                                                                      | prefetch、dispatch 三事件、depends_on 转 WAIT、return signal                                                               |
| E04  | [tile_group_sequencer.py](../../pipeline_validator/tile_group_sequencer.py)，47–185、189–318；[tile_group.py](../../pipeline_validator/tile_group.py)，1526–1539、1638–1728                                                             | 每 context 顺序执行；多 sequencer step；非 PC-only drain、retirement 不变量；barrier 无等待行为；没有共享 action window    |
| E05  | [tile_group.py](../../pipeline_validator/tile_group.py)，501–543、729–784、1904–1962；[memory/allocator.py](../../pipeline_validator/memory/allocator.py)                                                                               | 全 L2 bundle、临时/永久容量、strict FIFO、waiter 生命周期                                                                  |
| E06  | [tile_group.py](../../pipeline_validator/tile_group.py)，545–665、1256–1394；[execution_ir.py](../../pipeline_validator/execution_ir.py)，203–260                                                                                       | release preflight、grid identity、all-tasks、duplicate ignore、stale handling                                              |
| E07  | [workload_ir.py](../../pipeline_validator/workload_ir.py)，236–238、394–400、500–622                                                                                                                                                    | 零任务拒绝、task 数与 placement；真实访问集合、最后 use 与 writer/store/release 验证，不等于完整 reorder DAG               |
| E08  | [tile.py](../../pipeline_validator/tile.py)，196–199、252–328、541–592                                                                                                                                                                  | physical context 接受、READY/held-launch 策略、issue attempt、unknown event 计数                                           |
| E09  | [tile_group.py](../../pipeline_validator/tile_group.py)，967–1015、1036–1072、1100–1254、1980–1996；[tile.py](../../pipeline_validator/tile.py)；[memory/l1_slot_frame.py](../../pipeline_validator/memory/l1_slot_frame.py)            | Group 协调 L1/context admission 与 rollback；默认 Device slot→Tile context pin；collective 控制窗口；Tile-local free/frame |
| E10  | [runtime/event_table.py](../../pipeline_validator/runtime/event_table.py)，63–131                                                                                                                                                       | dict-backed event、auto-register、sequence API，不是有限实例 handle 完整实现                                               |
| E11  | [memory/transfer.py](../../pipeline_validator/memory/transfer.py)，394–397、501–508、828–832、1003–1005、1093–1105                                                                                                                      | transaction 容器接收无上限，实际 stage 竞争与完成另行推进                                                                  |
| E12  | [tile_group.py](../../pipeline_validator/tile_group.py)，1418–1485；[report.py](../../pipeline_validator/report.py)，17–115、118–136；[cli.py](../../pipeline_validator/cli.py)，182–198；[trace.py](../../pipeline_validator/trace.py) | DMA summary 标签/事件路由；报告已有指标与外部 IR checks 边界                                                               |
| E13  | [config.py](../../pipeline_validator/config.py)，439–465；[cli.py](../../pipeline_validator/cli.py)，103–115、151–154                                                                                                                   | context 1–8 配置与当前 simulation 参数                                                                                     |
| E14  | [examples/run.sh](../../examples/run.sh)，231–337；[n04_fifo_hol.mlir](../../examples/scenarios/nest_subgraphs/n04_fifo_hol.mlir)，1–4；[t03_single_zero.mlir](../../examples/scenarios/nest_subgraphs/t03_single_zero.mlir)，3–11      | 本次运行入口、容量负例/FIFO 场景、空 context 的真实内容                                                                    |

## 12. 源码身份与可复现边界

本次关键文件 SHA-256（不是 commit ID，也不代表整个仓库已冻结）：

```text
pipeline_validator/simulator.py
2295b751222f78059fe20d5e17f1acf19bf0b7acd5e392f5ab12d88ada462c99
pipeline_validator/tile_group_sequencer.py
64f32c8931afc584ee638c67bc522d259b298f27219302151437b3ab4676afed
pipeline_validator/tile_group.py
910dedb3038e3c5c193737ab43b4a0760ac0fcfbb03c7ea17ebe9f9959da0512
pipeline_validator/tile.py
b49e77f2ec5cba94926f44f6613328c81bd4a812a1a3fc30e6c4c05d38a025da
pipeline_validator/ir_lowering.py
3b2335853908db02a2dd94a103ec79a1d283c2104972236db774a60a618180f4
pipeline_validator/workload_ir.py
474425f6c576d225e3a35efc068315a1f0ba42f777e8f68713c07b96e97b7af6
pipeline_validator/config.py
fd500b68ed56fb44be2f877bd9629709c6e3b547677da1a3af41ea79a12eb308
pipeline_validator/hardware_config.yaml
c7dab7dc398a9b187403dbc2ea900e6d8ac3617a488ad0b0ef9b4f93cf373d31
examples/run.sh
db85275a7956ca41bfb9949898b6ddd314ebc4c1011aa3be97d53eb375487c0b
```

Proposal 链接的 `02_NEST_Ready_Action_Agent_Implementation_Guide.md` 在本次 `design/proposal/` 目录中不存在；本报告没有把它当作已读取或已冻结的依据。本报告文件也不冒充该指南，只提供针对**当前代码**的落地审查。
