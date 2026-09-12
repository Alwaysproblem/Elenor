# NEST 三层混合调度架构 Proposal

**Device CPU IR → Group NEST Context + Ready-Action → Tile Multi-Context**

| 项目     | 内容                                                                             |
| -------- | -------------------------------------------------------------------------------- |
| 文档版本 | 1.0                                                                              |
| 日期     | 2026-09-12                                                                       |
| 状态     | 架构提案；包含已确认边界与建议的 V1 实现契约，尚非 RTL 验证报告                  |
| 配套文档 | [Agent 实现指南](02_NEST_Ready_Action_Agent_Implementation_Guide.md)             |
| 目标读者 | 架构、编译器、runtime、FPGA/RTL、验证与性能分析 agent                            |
| 范围     | 本次对话的三层调度、事件和资源契约；不重新设计矩阵单元、向量单元或存储器数据通路 |

> **核心结论：编译器规定合法依赖与资源计划，memory admission 保证资源，Group ready-action 决定当前先发射谁，Tile multi-context 隐藏局部执行等待。**
>
> 本文的硬件规模、队列深度、仲裁策略与阶段划分是建议，不是用户已经确认的全部参数。所有 IR、字段与 pass 名均为提议的语义接口，不宣称已存在于代码库或上游 MLIR。

## 1. 决策状态与阅读规则

本文使用三类标签：**C：本轮已确认/明确保留的边界；D：为了让 V1 可实现而给出的默认设计；O：待测量或后续选择。** “必须”描述遵循本提案时应满足的契约，不表示用户已经批准每个微架构细节。

| ID   | 状态 | 决策                                                                                                          |
| ---- | ---- | ------------------------------------------------------------------------------------------------------------- |
| C-01 | C    | Device 是 CPU 执行 IR；`nest.context` 是 Group 层；tile program 是 Tile 层，只有三个执行层级。                |
| C-02 | C    | Tile 采用任务级 multi-context；context 内保持顺序执行语义，异步操作可在后台进行。                             |
| C-03 | C    | Group 前端可以交错登记不同 context；action 表不是严格执行 FIFO。                                              |
| C-04 | C    | 提交 PC 在成功登记时前进，不能把登记、发射、完成合并。                                                        |
| C-05 | C    | 单个 context 提交阻塞可换其他 context；共享表全满时换 context 无效，但后端与执行单元仍可推进。                |
| C-06 | C    | 完整依赖的拓扑登记排除“消费者挡住尚未登记的必要生产者”这一类登记依赖死锁。                                    |
| C-07 | C    | memory admission 是实际资源承诺；正确覆盖后续需求后，可排除讨论中的输入挤占输出空间型内存容量死锁。           |
| C-08 | C    | A/B prefetch 分开；保留 `input_released`、`output_ready`、`store_done`、`grid_done` 的不同含义。              |
| C-09 | C    | Group 负责 L2 生命周期与需求描述；实际 L1 分配、物理 Tile context 绑定由 Tile local admission 完成。          |
| C-10 | C    | 保留 SPMD epoch 与逻辑 Tile context queue 边界；不引入通用跨 Tile MPMD pipeline，不自动引入 stream。          |
| D-01 | D    | Device 顺序控制流 + 异步提交依赖；Group 使用共享提交前端、多份 context 状态和有限 ready-action 后端。         |
| D-02 | D    | V1 跨 Group context 的依赖在 admission 前门控；未满足依赖者只占 pending 元数据，不占 active context/L2 资源。 |
| D-03 | D    | V1 首先采用整 context 的静态可兑现资源预算；分阶段准入和跨 context 借用额度暂不实现。                         |
| D-04 | D    | action 槽位在目标适配器接收命令、且在途状态建立后释放；完成由独立表跟踪。                                     |
| D-05 | D    | V1 每 Group 每周期最多登记一条、最多提交一个 action 到执行适配器；两者可并行。                                |
| D-06 | D    | 公平轮转作为初始策略；不先加入复杂关键路径评分、推测或全局最优求解。                                          |
| O-01 | O    | context 数、action 窗口、inflight/event 容量、扫描宽度、bank/pool 划分与目标频率。                            |
| O-02 | O    | 静态最后使用点提前归还 reservation、局部动态分配、压力感知优先级和多发射。                                    |

优先级：本轮明确结论 > 本文 C 项 > 本文 D 项 > 历史实现习惯。旧版“PC 等到 action 发射成功才前进”的 Group sequencer 不能未经区分地套到新前端。

## 2. 问题、目标与非目标

### 2.1 要解决什么

纯 context-head 选择只能看到每个 context 的当前位置。同一 context 中，较早动作的等待可能挡住后面独立的 prefetch、dispatch 或 store。增加 context 数可以绕开一部分等待，但不必然暴露同一任务的所有独立分支。

本设计在 Group 层增加有限动作可见范围，同时保留编译器显式管理内存、任务生命周期和 Tile 顺序程序的优点。

### 2.2 优化目标

优先保证正确性和可推进性，再优化稳态吞吐、单请求延迟、尾延迟、资源利用率和每次推理能量。**SRAM 占用率是诊断指标，不是需要最大化的独立目标。**

分开三个窗口：

| 窗口         | 限制什么                               | 不代表什么                  |
| ------------ | -------------------------------------- | --------------------------- |
| 控制可见窗口 | 已登记且尚未发射的 action 数量         | 不意味着已经搬入对应 tensor |
| 数据驻留窗口 | 已承诺/正在使用的 L2、L1 空间          | 不意味着这些动作已发射      |
| 在途执行窗口 | DMA、dispatch 等已接收但尚未完成的操作 | 不等于 action 表仍被占用    |

已登记动作越多，不要求 tensor buffer 同比例增加；但 event、descriptor 和在途状态始终是有限资源。

### 2.3 V1 非目标

不实现 CPU 式寄存器重命名、推测执行、通用 ROB 精确回滚；不实现全芯片任意 DAG 搜索；不加入通用硬件死锁检测/解除器；不重新定义 gather/cache 路径或加入 atomics/stream；不声称已得到面积、频率、功耗或加速比。

## 3. 三层职责

| 层级   | 程序/状态                                             | 调度对象                                      | 不负责的事情                                        |
| ------ | ----------------------------------------------------- | --------------------------------------------- | --------------------------------------------------- |
| Device | `nexus.program` 或等价 CPU IR                         | context 提交、依赖、最终完成                  | 逐个 DMA/Tile 指令的实时选择                        |
| Group  | `nest.context`；每实例有提交 PC、参数、资源及事件状态 | 已登记的 prefetch/dispatch/store/release/join | 物理 Tile context ID、实际 L1 地址与每 task 执行 PC |
| Tile   | `nest.tile.program` 或已有 tile program               | 多个任务 context 的当前可发射指令             | 全局模型图调度、Group L2 生命周期                   |

文本结构图：

```text
CPU / Device
    顺序执行 → submit(context, deps) → pending descriptors
                                            |
                                  deps gate + admission
                                            |
Group                   +-------------------+-------------------+
    context state RAM   | C0: submit_pc ...  | C1: submit_pc ... |
                        +-------------------+-------------------+
                                       共享提交前端
                                            |
                       有限 action 表（带 context/event/buffer 标签）
                                            |
                         ready-action 选择 + 原子 issue commit
                           /                |                 \
                    prefetch adapter   dispatch adapter   store adapter
                                            |
Tile                 local admission → physical context binding
                                多 context 当前指令选择
                                            |
                               load / compute / store engines

完成事件：执行端 → 独立 completion 通路 → event 表 / 资源回收 / context done
```

“前端/后端”只是 Group 内部模块，不是第四层 IR。

## 4. Device 层运行逻辑

Device 顺序执行 CPU 程序，提交较粗粒度 context 和依赖，而不是在每个 Group action 上同步。以下仅为语义示意：

```mlir
nexus.program @run(%A, %B, %C) {
  %a_done = nexus.submit_context.async @ctx_A(%A)
  %b_done = nexus.submit_context.async @ctx_B(%B)
  %c_done = nexus.submit_context.async @ctx_C(%C)
                depends_on(%a_done)
  nexus.await %b_done
  nexus.await %c_done
  nexus.return
}
```

A/B 可独立推进，C 可以提前提交 descriptor，但 V1 要等 A 成功完成才参与 Group admission。CPU submission 顺序不自动形成额外完成顺序；需要序列化就显式添加依赖。

状态区分：`SUBMITTED → WAIT_DEPS → WAIT_ADMISSION → ACTIVE → DRAINING → COMPLETE`。错误路径另设 `FAULT_DRAINING/FAILED`。

pending 区域必须有容量与背压，但依赖未满足者不能占 L2 reservation 或 Tile slot。不得因为 pending 队首等待依赖，就停止检查其他已满足依赖的 descriptor。V1 可在依赖就绪集合中按到达顺序做 admission；内存不足导致的队首等待属于可测量策略，而不是 ready-action 固有语义。

外部 context completion 记录要在消费者不再引用之前保持有效；不能直接把可复用的物理 context slot 当永久 event handle。

## 5. Group 前端：登记而非执行

### 5.1 每 context 一份状态，共享一套前端

每实例保存 `submission_pc`、不可变参数实例、循环/控制状态、事件命名空间、reservation handle 和计数器。共享前端选择一个具有提交资格的 context，并登记它的下一条动作。

提交资格必须同时满足：已经 admission、控制条件可确定、依赖生产者已登记或外部依赖已兑现、有该 context 的窗口配额、有共享空槽位、有可用事件/参数记录。

成功登记的原子效果：写入 descriptor；绑定输出 event 的生产者；增加 queued 计数；提交 PC 前进。失败不得留下半个 descriptor 或推进 PC。

### 5.2 暂停分成局部和全局

| 条件                                   | 正确行为                                |
| -------------------------------------- | --------------------------------------- |
| C0 本地配额满，C1 有配额且共享表有空位 | 改选 C1                                 |
| C0 的真实控制分支/地址参数尚不可确定   | 仅 C0 等待；其他 context 可登记         |
| 所有 context 都不可提交                | 前端 idle，不阻塞后端                   |
| 共享 action 表全满                     | 前端全局背压；切换 context 无法创造槽位 |
| 已登记 dispatch 等待其输入 DMA         | 后端保存等待；本身不要求前端 PC 停住    |

普通 action 依赖应写入 descriptor，不降成阻塞前端的 `await`。但真实控制流、区域切换及需要运行时数据确定参数的动作可使提交 PC 暂停。

### 5.3 交错登记但不形成混合 FIFO

```text
slot  owner  action       dependencies       state
0     C0     dispatch A   a_ready            WAITING
1     C0     store A      output_a_ready     WAITING
2     C1     prefetch B   none               READY
3     C1     dispatch B   b_ready            WAITING
```

后端允许发射 slot 2，不必等待 slot 0。每个 descriptor 有稳定的 owner、event、buffer 和参数标签，不能依赖“前端当前选中了哪个 context”来解释已登记动作。

### 5.4 登记不是完成

定义四个时点：`REGISTER`（登记）、`ISSUE`（目标适配器接收命令）、`ENGINE_START`（执行端实际开始）、`COMPLETE`（完成契约满足）。后三者可能相隔较长时间。

提交 PC 到末尾只设置 `submission_closed`。只有所有 action、在途操作、Tile tasks、必要的 DMA 写回和资源清理都结束后，才能报告 context completion。

## 6. Group 后端：有限 ready-action

### 6.1 可发射谓词

```text
eligible(action) =
    action 已登记且未发射
    AND 所有依赖成功完成
    AND owner 仍处于允许执行状态
    AND 所需 buffer 生命周期与 ownership 正确
    AND 对应 reservation 覆盖本动作
    AND 目标 adapter/queue 有 credit
    AND 所需 inflight 元数据可用
    AND SPMD epoch 约束允许
```

依赖 `PENDING` 表示等待；依赖 `ERROR` 表示错误传播，不能当作成功，也不能永久等待一个不会再成功的事件。

V1 从 eligible 集合公平轮转选择一个动作。暂不规定“所有 store 永远优先”或“所有 prefetch 尽早发射”。内存压力感知与关键路径优先级作为后续策略实验，不改变正确性谓词。

### 6.2 发射的资源事务

在目标 adapter 接收命令的同一个 commit 中：建立 inflight 记录、扣除 credit、保存参数、使 action 从窗口消失。任何一步不能完成则本次不发射，不得部分持有其他资源等待。

每类 engine 有独立有限命令接收路径，某个 adapter 背压不能锁住所有其他类型的 eligible action。V1 的共享发射宽度仍为一，不等于所有目标共用一个会无限期卡住的输出寄存器。

`RELEASE/JOIN` 等本地动作也有明确 commit 和完成事件，但不必占长期 engine inflight 槽位。

### 6.3 为什么仍保留提交 PC

大 context 可能超出窗口，循环与参数生成也需要顺序控制。因此前端持续供应后续动作；后端按事件发射已供应动作。如果一整个动作区域已全部登记，后端执行它时不需要提交 PC 再切回该 context。

## 7. Memory admission：本设计的资源安全基础

### 7.1 不是“看看现在够不够”，而是可兑现的预留

对资源池或 bank `b`：

```text
0 <= used[c,b] <= reserved[c,b]
sum_c reserved[c,b] <= capacity[b]
```

`reserved` 已包含 `used`，两者不能相加后再次扣容量。未使用的承诺额度不能重复许给其他任务。

V1 admission 至少检查：L2 静态 layout 可放置、对齐/bank/pool 约束、active context slot、event 预算及最小控制窗口保证。Tile 的实际 L1 空间由 Tile 在接受本地任务时检查；Group 不代替其决定物理地址。

若一个请求本身超过目标 Group 总能力，应报告不可满足的请求，而非无限 WAIT_ADMISSION。暂时不足则等待其他已准入任务释放。

### 7.2 预算必须适用于允许的重排

预算包含输入、输出、中间值、workspace 以及被允许同时存在的预取版本。可以复用物理空间，但必须由完整数据/控制/内存复用依赖保证生命周期不重叠。

例如 `B_input` 复用 `A_input` 的空间，要存在 `release(A_input) → prefetch(B_input)`；不能仅依靠 B 在源代码中写得靠后。外部异步内存系统也要求释放晚于所有访问；这支持生命周期契约，不证明本架构性能。[R2]

### 7.3 V1 默认与后续优化

**默认：整 context 静态资源预留，额度在 context 终止且安全排空后归还。** Context 内的 `RELEASE` 结束该 buffer 的访问租约，允许编译器计划内复用，但不会自动把保证给 context 的额度借给别的 context。

这是保守实现选择 D-03，不是要求未来永远这么做。必须分别记录 reserved occupancy 与 live occupancy，不能把预留的空洞伪报为有效数据驻留。

**后续：静态最后使用点提前归还额度。** 编译器证明此物理预留块不再被本 context 后续使用，且所有访问完成后，才减小剩余 reservation。普通 `input_released` 不能自动推出“这个 context 以后永远不再需要这份容量”。分阶段 admission 需要新的完整资源证明，V1 不做临时借用。

### 7.4 已解决与仍需硬件保证的边界

拓扑登记 + 可兑现的 memory admission，能够解决本轮讨论的“生产者被挡在登记之外”与“输入占满、输出无空间”两类问题。**不再把这些已由契约处理的问题当成 ready-action 的固有缺陷，也不要求整个 region 同时驻留。**

仍必须保证控制协议正确：完成通路不依赖 action 表有空位；inflight 转移不丢记录；仲裁公平；已经 issue 的宏动作不等待未来尚未登记的 Group 动作才能完成；故障不伪造成功。它们属于执行契约，不是增加通用死锁搜索器的理由。

## 8. 事件、Tile 任务与完成语义

| 事件              | 成功含义                                                    | 不能据此推断                                     |
| ----------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| `a_ready/b_ready` | 对应 prefetch 目的数据已经可供声明的消费者访问              | 另一份输入也已就绪                               |
| `input_released`  | 本 dispatch 的所有预期消费者都不会再访问该输入范围          | 仅第一个 Tile load 已完成就可释放全部输入        |
| `output_ready`    | 声明输出范围已经完成写入，并对 Group store 可见             | HBM 写回已完成，或 Tile context 已可释放         |
| `store_done`      | 写回完成并达到所声明设备/主机可见性边界                     | 任何未结束的其他消费者都已结束                   |
| `grid_done`       | 所有逻辑任务完成且本 grid 的 Tile 执行状态可安全退出        | 若 Group store 尚未完成，则 CPU 已可读取最终输出 |
| `context_done`    | 提交关闭、队列/在途工作排空、必要终结事件完成、资源清理结束 | 单纯 PC 到末尾                                   |

多 task dispatch 的里程碑必须按完整 task domain 聚合，包括尚未实际投递的任务。task 数为零时明确规定里程碑为真，不允许计数器下溢。可靠传输中的重复里程碑视为协议错误；不能重复扣 remaining。

事件用实例化 handle，例如 `(context_instance, region_generation, event_slot)`；物理槽位复用前必须排除旧引用和迟到完成。所有信号携带可验证 owner/tag。

V1 `tile_context_queue_id` 是逻辑投递队列/亲和提示，**不是物理硬件 slot ID**。Group 发布 program、task domain 和资源需求；task 分配端保证唯一领取，Tile program 用自己的 task 索引计算坐标。Group 不保存每 task 的执行 PC。

## 9. Tile 层与 SPMD epoch

Tile local admission 对一个任务原子取得所需 context/L1/局部元数据；失败不留下部分占用。接受后的任务必须能通过本地执行、已具备的 Group 数据及有限服务延迟完成，不依赖尚未准入的另一个 Tile 任务提供完成所需资源。

Tile 调度器从各 context 当前指令中检查 eligibility，而不只是 `state == READY`。当前 MMA 无 credit 时可以选择另一个 context 的 load/store。Context 内顺序发射不要求上一条异步操作完成后才能发下一条；数据 hazard 由编译器依赖和本地 scoreboard/等待契约处理。

V1 不增加一般跨 task barrier。如果某 collective 要求所有 task 同时驻留，必须显式 gang 资源保证或拆成完成同步的阶段，不能隐含使用 oversubscribed barrier。

建议的保守 epoch 策略：一个 Group 同一时间只允许一个兼容 program/ABI key 的执行 epoch；多个 context 的同程序 grid 可共享该 epoch。更换 key 必须等待原 epoch 的已接收与待完成 tasks 全部退出。prefetch/store 不因计算 epoch 不匹配而一概被禁止。此策略的串行化影响需要单独统计，不能归因给 ready-action 本身。

## 10. 端到端语义示例

以下是说明性伪 IR，具体 ODS、type 和 ABI 由实现指南约束后再落地。

```mlir
nexus.program @run(%A, %B, %C) {
  %done = nexus.submit_context.async @gemm_context(%A, %B, %C)
  nexus.await %done
  nexus.return
}

nest.context @gemm_context(%A, %B, %C)
    attributes {resource_plan = @gemm_l2_plan} {
  %ar = nest.prefetch.async %A into %l2_a
  %br = nest.prefetch.async %B into %l2_b

  %gd, %ir, %or = nest.dispatch.spmd.async @gemm_tile
      ins(%l2_a, %l2_b) outs(%l2_out)
      task_domain(%domain) tile_context_queue(%queue)
      depends_on(%ar, %br)

  nest.release %l2_a depends_on(%ir)
  nest.release %l2_b depends_on(%ir)
  %sd = nest.store.async %l2_out to %C depends_on(%or)
  nest.release %l2_out depends_on(%sd)
  nest.close_submission terminal_events(%gd, %sd)
}

nest.tile.program @gemm_tile {
  // Tile 本地获得逻辑 task 索引和已分配的 L1 frame。
  %ta = tile.load.async %A_tile
  %tb = tile.load.async %B_tile
  tile.await %ta, %tb
  %acc = tile.mma ...
  tile.signal input_released
  %ts = tile.store.async %acc to %group_output
  tile.await %ts
  tile.signal output_ready
  tile.return
}
```

这里 `tile.store` 指 L1/计算结果到 Group 声明输出范围的写入，`nest.store` 指后续向 Device 可见目标的写回，二者不能混同。`input_released` 在本例保守地放在计算之后；优化提前信号必须证明不再从 Group 输入读取。

若同一 context 内还有一条独立分支，前端可继续登记它；等待 A 的 dispatch 不阻止后端发射该分支已就绪的动作。

## 11. 编译器输出契约

编译器必须产生：context 间依赖；action DAG 的数据/控制/内存复用边；实际实例的拓扑登记顺序；静态资源计划及对齐/bank 约束；事件与参数生存期；task domain 与本地需求；SPMD epoch key；终结事件和错误策略。

“源代码顺序”不是依赖边；“event ID 已分配”也不等于生产者已经登记。前端需要区分事件的 UNBOUND、PENDING、SUCCESS、ERROR 状态。

结构化循环可由前端控制，但 V1 每次迭代/region 使用新的事件实例，且仅在前一代动作和引用安全排空后复用槽位。跨迭代重叠需要编译器显式展开为多个已预算版本，不自动推测。

本轮没有指定真实源码仓库；建议先在独立 JSON/解释器模型中冻结语义，再接入 MLIR。候选 pass 名可为 `build-action-deps`、`plan-resources`、`topological-registration`、`lower-to-group-descriptors`，它们是本项目待实现接口而非上游现成 pass。

## 12. FPGA 实现路线与方案取舍

| 方案                   | 优势                     | 代价/局限                                | 本提案位置             |
| ---------------------- | ------------------------ | ---------------------------------------- | ---------------------- |
| context-head 选择      | 状态与验证范围小         | 同 context 独立动作可被挡住              | 基线                   |
| 小窗口 ready-action    | 扩大可见性，保持资源边界 | action/event/inflight 管理与仲裁         | V1 主方案              |
| 大窗口/全局 action DAG | 更大候选集合             | 存储端口、唤醒扇出、布线、策略与验证复杂 | 后续上界实验，不是默认 |

Gemmini 的局部解耦控制器提供结构参考，但其具体 ROB/控制语义不能直接视为本设计的实现。[R3] FPGA BRAM 的端口与同步读约束意味着不能把软件中“遍历整张表”当成免费组合逻辑。[R4]

先实现有限宽度扫描、ready/valid 元数据和可流水化仲裁，测量发射需求后再决定是否加宽。控制表位数仅是成本的一部分；例如 32 条、每条 256 bit 的控制记录为 1 KiB，但不包含参数、事件、inflight 与多端口复制，不能据此声称硬件很小。

不预设频率与功耗。最终比较 `cycles / achieved_frequency`；功耗分析使用有代表性的活动和明确的器件、约束、工具版本，功耗估计与板上测量分开报告。[R5]

## 13. 验证与性能验收

优先顺序：语义模型 → 有限资源周期模型 → RTL 对拍 → 综合/P&R → 功耗估计。配套指南给出任务划分、接口、断言和测试矩阵。

至少比较三种模式：S0 每 context 最早未发射动作；S1 同等资源的小窗口 ready-action；S2 更大但仍有限窗口。S0/S1 采用相同 memory admission、相同数据容量、相同发射带宽和延迟；若旧实现使用多套 sequencer，另列 Legacy 配置，不能混作公平基线。

必须报告：makespan、单请求/尾延迟、稳态完成间隔、engine busy、reserved/live occupancy、各类 stall、admission 等待、窗口/inflight 峰值、吞吐与频率。资源空闲时可合法运行的动作被源顺序挡住才记为 ordering stall；需要离线未来信息的指标不得伪装成硬件可测量计数器。

没有实测前不承诺 S1 必然优于 S0；规则流水已经饱和时，扩大候选集合可能只增加控制开销。

## 14. 风险、未决项与退出条件

主要风险是：整 context 预留过于保守；有限扫描增加唤醒到发射延迟；epoch 策略限制混合程序并发；event/完成更新端口不足；compiler 资源计划漏掉重排生命周期；错误清理提前释放仍被 DMA 引用的内存。

若 S1 在相同资源下没有可测量收益，可保留接口而关闭同 context 内重排。若收益明显但 Fmax 降低，应先缩小/分层/流水化窗口，而非直接扩大状态机。若 admission 才是主瓶颈，先优化静态计划与可证明的最后使用点归还，不用无约束预取“填满内存”。

签出 V1 的条件：全部强制正/负测试通过；随机用例可复现；没有无限缓冲等理想化假设混入结果；RTL 对拍通过；源配置与工具版本可重现；所有未实现项明确列出。

## 15. 结论

**Device 管提交，Group 管动作级选择，Tile 管任务级执行。**

**拓扑登记和真实 memory admission 是正确性前提，不是 ready-action 的竞争对手。** 前端为多个 context 交错供应动作；表满则仅阻止新增登记；后端继续完成可执行工作。Ready-action 不替代 NEST 的任务身份、资源所有权和生命周期。

下一步按配套指南先冻结可执行语义与验收用例，再实现最小 RTL；不要先把所有微架构可能性放进第一版。

## 参考资料与证据边界

以下公开资料仅支持相应通用机制；本提案中的分层选择、无死锁条件、参数和测试要求是本次设计推导，不是这些资料对本架构的背书。访问日期：2026-09-12。在线文档会演变，实现时锁定工具版本。

- **[R1] NVIDIA, CUDA Graphs** — 节点依赖约束执行，满足依赖后由系统调度；仅用作依赖语义类比。https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- **[R2] NVIDIA, Stream-Ordered Memory Allocator** — 异步内存访问、释放和跨流依赖的生命周期契约。https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html
- **[R3] UC Berkeley, Gemmini 官方仓库 README** — Decoupled Access/Execute、控制器与依赖管理。https://github.com/ucb-bar/gemmini
- **[R4] AMD UG573, Block RAM Summary** — FPGA Block RAM 的端口和同步存取约束。https://docs.amd.com/r/en-US/ug573-ultrascale-memory-resources/Block-RAM-Summary
- **[R5] AMD UG907, Vector (SAIF) Based Power Analysis** — 有代表性的切换活动与实现阶段的功耗分析；器件支持须核对。https://docs.amd.com/r/en-US/ug907-vivado-power-analysis-optimization/Vector-SAIF-Based-Power-Analysis

依赖图类比补充：CUDA Graphs 将工作描述与实际调度分开，但本文没有推断其内部采用本提案的硬件队列、PC 或 admission 实现。[R1]
