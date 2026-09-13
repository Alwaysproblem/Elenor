# NEST Ready-Action 实施计划与循环记录

## 1. 本轮目标与边界

依据 `02_NEST_Ready_Action_Pipeline_Validator_Implementation_Review.md` 实施。Device 作为 CPU 软件控制模型独立，Group/Tile 作为未来 FPGA/RTL 的硬件行为模型独立；本轮交付是可执行 Python/xDSL 周期模型与证据，不声称已有 RTL、Fmax 或功耗结果。

- CPU：解释 `nexus.program`，有限 pending、submit dependencies、await、external completion。
- 硬件 Group：执行已 lowering 的 context descriptor，独立 active/context/action/event/inflight/adapter 资源、L2 admission、共享 S0/S1/S2。
- 硬件 Tile：物理 context/L1 admission、eligible-head 调度、引擎 credit 与 task 生命周期。
- CPU 不能读写 Group/Tile PC、选择物理 Tile context 或依赖 sequencer 内部状态。边界 adapter 只传提交与完成消息；物理 slot 在 adapter/Group 侧分配。
- 保留当前真实 allocation、phase、release、rollback 与 fault/reset 不变量。保留静态最后使用点 final-free，不引入未证明的 reservation 借用。

## 2. 共享接口先冻结

### 2.1 编译层

`ExecGroupAction` 新增 `dependencies/reads/writes` 和 `output_events`；普通 depends_on 不再 lower 成 WAIT。显式 `nest.await` 保留提交 fence；barrier 等待当前 context 全部前驱终结，再允许后继登记；return 必须 drain。编译层按实际访问集补 RAW/WAR/WAW，不给独立对象加人为全序。

`ExecDeviceOp.dependencies` 来自 `nexus.submit_context.async ... depends_on(...)`；仍用现有 xDSL，不建立第二套 JSON 执行 IR。

### 2.2 CPU/硬件消息边界

新增独立 `CpuDeviceController` 与 `GroupPortAdapter`。CPU 仅依赖 protocol：`try_submit(request, cycle) -> bool`、`poll_completions(cycle) -> completions`；request 有稳定 request_id、低层 task/参数引用、可选 Group context-table slot affinity，completion 有 request_id、成功/失败、reason、cycle。Python task 引用只代表已装载可执行对象，不是宣称冻结了物理 ABI 指针。Affinity 不是 Group ID，也不是物理 Tile context。

CPU `DeviceConfig` 与 `GroupSchedulerConfig` 分开。`device_context_count` 限制已被硬件端口接收的 outstanding requests；WAIT_DEPS 只消耗 CPU pending 容量，不消耗该额度。不再通过 max() 扩充 Tile 物理 contexts，也不再自动把 Device slot 映射成 Tile context_id。`context_count` 仅控制真实 Tile context 数。

### 2.3 Group 周期模型

完成先可见 → 对已登记动作最多 ISSUE 一条 → 最多 REGISTER 一条 → drain/retire。新登记最早下一周期发射。S0 每 context 最早未发射动作；S1/S2 同一后端有限扫描全部可见候选；S2 的窗口规模显式配置。两个阶段独立 RR。表满不阻塞完成。

配置字段已在 `GroupSchedulerConfig` 定义：policy、active_context_capacity、action_capacity、context_action_quota、scan_width、event_capacity、inflight_capacity、prefetch_capacity、store_capacity、dispatch_capacity、epoch_policy。默认 mixed，不默认串行 mixed-program 工作。

### 2.4 Tile admission 接口

Tile-owned `TileAdmission` 保存 tile、logical_task_id、context_id、l1_plan、prepare_cycles、l1_handles、bound。

- `ComputeTile.plan_admission(program, grid, role_event_id, logical_task_id, context_id=None)`：无副作用选择物理 context 与规划 L1；返回 plan、AdmissionFailure 或 None（无 context）。Tile 根据自己的 fidelity 决定是否物化 L1。
- `ComputeTile.commit_admission(admission, program, cycle)`：commit L1 并准备对应 SlotFrame，填充 handles；Group 仍组织全体 Tile 的提交边界及 L2 pins。
- `ComputeTile.abort_admission(admission, cycle)`：撤销本 Tile 已 bind/frame/handles；Group 清理自己的 L2 pins 与 grid bookkeeping。

非法/永久不可满足 fault；临时 L1 空间不足不 fault、不残留占用，由 Group 后端重试其他 eligible 动作。

## 3. 实施循环

每轮均执行“实现 → 定向运行 → 检查不变量/真实 trace → 修正”。并发修改期间不运行 formatter/lint/test；集成后统一运行，避免验证读取半套契约。

| 循环    | 内容                                                        | 验收                                                                   |
| ------- | ----------------------------------------------------------- | ---------------------------------------------------------------------- |
| P0      | 保存 Legacy、冻结 CPU/硬件边界、依赖/fence/错误语义         | 不把 Legacy 多发射当公平 S0；保存配置和输出                            |
| P1      | action deps/effects、finite metadata、typed CPU submit deps | 拒绝未绑定依赖；接收失败零副作用                                       |
| P2      | Group 共享 S0；register/issue/completion 分离               | 1 register+1 issue/cycle，窗口满仍可完成                               |
| P3      | S1/S2、RR/scan/quota、adapter 独立背压                      | 独立分支可越过阻塞，真依赖不可越过                                     |
| P4-CPU  | 独立 CPU controller + 消息 port、pending deps/错误传播      | WAIT_DEPS 不占 Group/L2；B 独立不被 C→A 依赖挡住                       |
| P4-Tile | eligible-head RR、Tile-owned admission、临时 L1 背压        | 当前指令无 credit 时其他 context 推进；late failure rollback           |
| P5      | 可选 same-program epoch 与资源/队列需求实验                 | epoch 只门控 dispatch；以 P3 实测决定是否需要扩展 task-domain/复用布局 |

P5 按已审报告的进入条件执行，不把通用 task stealing、动态 reservation 借用、结构化循环、MPMD 或 RTL 综合悄悄加入本轮。若实验显示需要新的逻辑队列/物理复用，则先给出完整身份/生命周期契约；不得把未实现接口作为完成交付。

## 4. 所有权与并发切片

- Main：xDSL parser/verifier/lowering、execution DTO/config、最终集成、案例、测试、文档与统一验证。
- Group agent：`tile_group.py`、`tile_group_sequencer.py`、Group scheduler 新模块、`runtime/event_table.py`；finite Group admission/adapter/event、epoch、phase/retirement。
- Tile agent：`tile.py`、`memory/l1_slot_frame.py`、必要的 `engines.py`；不修改 Group 文件，按上面接口供 Group 使用。
- CPU agent：独立 `device.py` controller、`runtime/group_port.py` 协议适配器、`simulator.py` 协同推进。无调用者的旧 `runtime/device_runtime.py` 固定延迟 wrapper 及导出已删除，不保留同名兼容层。
- Metrics agent：`pmu.py`、`report.py`、`trace.py`、`cli.py`；消费 Group snapshot `scheduler` 和 CPU snapshot `device`，不通过 trace 离线猜硬件 PMU。

共享文件的后续修正由 Main 统一集成；子 agent 不运行全库验证。

## 5. 实施结果与验收证据

### 5.1 完成状态

| 阶段    | 结果                                                                                                                                |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| P0      | 已冻结 CPU/硬件边界、显式 fence、错误/终结语义；保存 Legacy 命令、输出与源码 hash                                                   |
| P1      | xDSL submit/prefetch dependencies、SSA 身份校验、L2/global/Gather 访问 hazard、有限事件与接收资源已落地                             |
| P2      | 共享 S0、独立登记/发射/完成、次周期可见性已落地并核对 trace                                                                         |
| P3      | S1/S2 有限扫描与 RR 已落地；同资源比较独立分支和内存主导案例                                                                        |
| P4-CPU  | CPU 控制器和 message port 独立；依赖 pending 不占硬件额度；错误不准入依赖任务；CPU completion 资源死端明确 fault                    |
| P4-Tile | Tile-owned plan/commit/abort、eligible-head RR、暂时 L1 背压、真实 owner/generation 与 rollback 已落地                              |
| P5      | 可选 same_program epoch 已实现并测量；没有默认开启。逻辑 task queue / reservation-layout 复用仍按报告的条件阶段保留限制，未伪报实现 |

P5 的本轮决策：独立分支仅需当前两个 Tile contexts，窗口从 16 增至 32 未带来进一步收益；已测用例未提供引入 task stealing 或复用 lease 的收益证据。因此保留 1:1 gang task mapping、静态零任务拒绝、静态最后使用点 final-free，不加动态借用/阶段准入。此结论只覆盖已测场景，不声称后续 workload 永远不需要这些功能。

### 5.2 实际验证循环

1. **Legacy 冻结**：修改前 pow-dual-context=55,160、l2-admission-wait=29,052、nest-s04-single=38,337 cycles。Legacy 多 sequencer 不是公平 S0。
2. **独立切片集成**：Group、Tile、CPU、metrics 并发实施；主线程负责 IR/配置/测试与接口集成。
3. **首轮定位**：全量回归 310 passed、30 failed。修复 allocation-free timing_only release、warm-run PMU 累积、错误清理等真实源码问题；没有删掉内存安全检查。
4. **合同迁移**：旧 CPU trace/slot、固定首发周期、临时 L1 必须 fault、伪 SlotFrame generation gate 等不再是新合同。测试改为检查真实 admission、依赖、最终写回、rollback 和不泄漏；删除旧 action_index/DeviceRuntime 兼容路径。
5. **定向验证**：12 个 ready-action/CPU 边界测试通过，包含 W=1、global alias hazard、显式 barrier、epoch 不冻结 DMA、CPU 依赖错误、foreign SSA、事件预算复用、CPU completion 容量死端。
6. **全量验证**：`conda run -n elenor-validator python -m pytest pipeline_validator/tests -q --tb=short --durations=5`，**342 passed**。
7. **源码 gate**：对全部变更源文件运行 pre-commit，mypy、ruff、文件规范和 jupytext 通过。没有对无关文件运行自动格式化。

此外同一个 Simulator 连续运行的 cold/warm smoke：6,536 / 6,528 cycles，cold-load 32 / 0 cycles；第一次 SimResult 的 PMU 不被第二次运行改写。

### 5.3 最终 CLI 测量

均使用真实 `bash examples/run.sh ... --group-policy ... --json --memory-trace --trace-json ...`；完整 HardwareConfig、SimConfig 和 binding 参数保存在报告中，不只保留 CLI 参数摘要。

| 场景                     | 模式/差异                          | cycles | 结果                                           |
| ------------------------ | ---------------------------------- | -----: | ---------------------------------------------- |
| ready-action-branch      | S0，window=16/quota=8/scan=4       | 18,326 | 完成                                           |
| ready-action-branch      | S1，同资源                         | 17,339 | 完成；相对 S0 cycles 减少 5.39%                |
| ready-action-branch      | S2，window=32/quota=16/scan=8      | 17,339 | 完成；额外窗口无收益                           |
| ready-action-branch      | S1 + same_program epoch            | 18,031 | 完成；仅 dispatch 串行化，不阻止独立 prefetch  |
| device-dependency-submit | CPU outstanding=2，Tile contexts=2 | 13,250 | 完成                                           |
| device-dependency-submit | CPU outstanding=2，Tile contexts=1 | 14,563 | 完成；报告与 trace 均只有一个物理 context/Tile |
| l2-admission-wait        | S1                                 | 29,051 | 完成，保留提前 release 唤醒                    |
| pow-dual-context         | S0                                 | 55,158 | 完成                                           |
| pow-dual-context         | S1，同资源                         | 55,157 | 完成；没有显著收益                             |
| nest-n04-fifo-hol        | S1                                 | 43,511 | 完成；内存 FIFO HOL 仍是单独策略问题           |
| nest-n09-impossible      | S1                                 |      8 | 预期 exit 1，L2 永久容量 fault，不进入无限等待 |

CPU 依赖 trace：A cycle 0 submit/admit，C cycle 1 submit 但到 5,050 才 admit；A completion=5,049。独立 B cycle 2 已 admit。说明 pending C 没有抢占硬件额度阻塞 B。

独立分支的 L2 reserved peak=8,192 B、protocol-live peak=4,096 B；两条曲线不再把未填充的输出预留计为有效数据。所有 11 次调用均核对 register/issue 宽度、次周期可见性、容量峰值、最终控制/adapter credit 清空以及 trace/报告占用峰值一致性。

`dependency_wait_candidate_checks` 是候选检查次数，放在 event counter 中，不能解释为唯一 Group stall cycles。`group_ordering_stall` 单独统计可见且资源允许的动作被 S0 head 策略挡住的周期。

### 5.4 可复现文件

- [验证总表、完整配置、源码 SHA-256、Legacy 输出](../../examples/artifacts/ready_action/run-77r5vlfp/verification.json)
- 同目录包含每次调用的 `.report.json` 与 `.trace.json`。
- 新可执行输入：`examples/scenarios/ready_action_branch.mlir`、`examples/scenarios/device_dependency_submit.mlir`。
- 当前接口合同：`pipeline_validator/IR_SPEC.md`；范围限制：`pipeline_validator/Limitation.md`。

### 5.5 明确未交付的硬件能力

本轮未实现 RTL、FPGA bitstream、真实 CPU IPC、CPU/FPGA CDC 或 PCIe/AXI 延迟、tensor 数值验证、通用 oversubscribed task queue、复用 reservation lease 或 MPMD。CPU 与硬件行为模型已经通过独立接口分开，将来替换 FPGA 端口/时钟模型不需要把 CPU PC 放进 Group/Tile。
