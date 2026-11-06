# W2 Context 与 UCE（取指、调度、寄存器、配置、依赖、退休）详细设计

> 本组件设计已通过用户 review，推荐方案采用情况与参数冻结任务见[实施总计划](../plan.md)；跨包合同见[总体设计](../Coral_Nexus_Tile_Design.md)，r5 指[需求规格](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。[设计验证记录](../Design_Validation.md)不代表组件已实现。

## 1. 定位、目标和 First Silicon cutline

本文是 Coral 派生 Nexus Tile 中 UCE 与 Context 管理的中文详细设计评审稿，覆盖：Context 身份与生命周期、取指与分支重定向、硬件调度与发射、GPR/FPR/CSR/向量目的依赖（scoreboard）、配置快照、scalar 侧退休与任务终止。UCE 按用户决定 B-08 是**改造后的 Coral scalar 流水线**（含标量执行、退休与 scoreboard），不是纯控制 sequencer（需求 §4.3，[../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md) §4.3）。

目标（对应 r5 §§5–8）：

1. 支持 ≥2 个硬件 Context 同时驻留；`context_count` 物理值由后续规格冻结（validator 仿真默认 1、上限 8，`SimConfig.context_count`）。
2. Context 内按序发射（SCH-01），发射 bundle 只取单 Context 程序顺序的连续前缀（§7.5）；A 阻塞时 B 补位（SCH-02）；等待 region 的 Context 的后续指令不越过其头部 START 声明（MAT-20/MAT-21）。
3. 退休按完整 ContextKey（tile＋槽位＋generation）＋操作序号匹配，不按 PC（RET-01/RET-02）；无全局队首 HOL（RET-03）。
4. 局部重定向（fetch epoch）与任务代际（task_generation）、run 代际严格区分（CTX-05/CTX-06/INV-15）。
5. 正常调度、Context park（WFI）与 fail-stop 终止三套机制分离（ISA-04、CTX-07、SCH-11/FLT-01）。
6. 本轮 BF16 I/O 收窄只约束张量数据通路；UCE 的标量整数控制、地址/索引计算与地址 CSR 照常保留，不因 BF16 决策删除或降级（本轮范围确认，见总体设计）。

本组件方案已选；首版提供 C1/C2 配置，四 lane、复制 RF 与私有 ROB。机械深度／位宽／端口预算由实施计划的前置合同与配置任务冻结，不将未实测参数称为硅片规模结论。

## 2. 职责、非职责和 ownership

单一执行 owner 原则（总体设计）：一份可变状态只有一个管理方。

| 状态/机制                                          | Owner | 说明                                                                                                                                                                                 |
| -------------------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Context 槽位表、task_generation                    | W2    | 槽位生命周期 FREE→ADMITTED→…→TERMINAL（§6.2）；复用条件 CTX-15                                                                                                                       |
| 每 Context PC、fetch epoch、取指缓冲               | W2    | FET-01/FET-02；指令存储端口仲裁与本地指令存储本体归 W5/W4                                                                                                                            |
| 每 Context GPR/FPR、scoreboard                     | W2    | STA-01～STA-07；VRF 物理存储归 W3，W2 只持有向量目的 busy 摘要                                                                                                                       |
| Context-local CSR 与配置版本                       | W2    | CFG-01～CFG-05；Tile-global 特权写清单 CFG-06 非空（profile/reset/program invalidate/debug 属管理侧特权路径，普通 Context 禁止，§3.6）                                               |
| scalar ROB（每 Context 静态私有分区，推荐见 §3.4） | W2    | RET-01/RET-03/RET-06；匹配键改造见 §3.4                                                                                                                                              |
| 任务终止判定（RET＋本地事件全完成）                | W2    | RET-08；判定达成后仅向 W6 发内部 EXEC_DRAINED 通知（逻辑名），不对外发布成功终态；可靠 TASK_COMPLETION 归 W6；本地 Arena/view/Frame 退休归 W4（RTL 侧），父 L2/R 释放与聚合归 Python |
| 事件 wait 记录                                     | W6    | 唯一写归 W6；W2 只持 WAIT 状态位与只读摘要，不写 wait 记录本体                                                                                                                       |
| region park 等待记录                               | W3    | MAT-12"每 Context 至多一个"的记录本体归 W3；W2 只持 park 投影位并在头部停靠                                                                                                          |
| 停顿归因与 per-Context PMU 计数                    | W2    | SCH-10；进入全局 PMU manifest（W7）                                                                                                                                                  |
| VME 区域 lease/实例选择/路由                       | W3    | MAT-12/MAT-19/MAT-22；W2 只发起 START/END 声明并在未授予时 park                                                                                                                      |
| LSU/MFE 事务、store 完成信号                       | W4    | storeComplete 响应携带 `(ContextKey, txn)` 由 W4 合同保证（W4 文档）                                                                                                                 |
| 程序驻留与冷装载                                   | W5    | RES-09～RES-14；W2 以 W5 的 INSTALL→READY 真实握手门控首条取指；validator 的 ACCEPT 倒计时只是模型口径（RES-12），不作为 RTL 机制引用                                                |
| 事件表、复位/故障状态机、可靠 TASK_COMPLETION      | W6    | FLT-01/FLT-02；W2 提供归属与停发配合                                                                                                                                                 |
| 本地 Arena/view/Frame 退休                         | W4    | Group RETIRE 授权后执行本地 view/Arena/Frame 关闭与回收（§3.4 链）；TASK_COMPLETION 可靠保存配合 W6；父 L2 pin/R/Grid 记录归 Python Group（CTX-13/CTX-15 两条链不得互替）            |

## 3. 微架构和状态机

### 3.1 Coral 现状锚点（固定版本直接核对）

固定上游 f79522157fc0f1d7311f44e209fc36c9c8b7a039（[H01]～[H11] 见需求 §23；本节行号均为该 commit）：

| 现状（已核对）                                                                                                                                                                                                                 | 多 Context 化结论                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- |
| `RetirementBuffer.scala` L28：`storeComplete = Input(Valid(UInt(programCounterBits)))`，L587-589 `storeReady := storeComplete.bits === bufferEntry.addr`——scalar store 完成仅按 PC 匹配                                        | 同映像两个 Context 同 PC 并发在途 store 会错配（RET-01、X-27）         |
| 同文件 L494-495：非 RVV 故障按 `bufferEntry.addr === faultRetire.bits.mepc` 匹配，仅 RVV 故障用 `rob_tag`（L492-493）                                                                                                          | 故障会归错 Context；匹配键须含 Context／ROB 身份                       |
| 同文件 L467-476：scalar/float 长延迟写回完成按**寄存器号**匹配（`x.bits.addr === bufferEntry.idx`、`addr+base === idx`）；向量/VME 写回按 `rob_tag`（L482-483、L423、L573）                                                    | 同号寄存器跨 Context 串写完成；scalar 侧也须带 ROB 身份                |
| 同文件 L58-66 注释：completed 阶段已提交架构副作用；L106-110 断言 bundle 内"首拒之后不得再接受"（连续前缀）                                                                                                                    | 前缀性质已有硬件先例，保留并按 Context 化                              |
| `scalar/Decode.scala` L425-462（DispatchV2）：32 位平坦 scoreboard 位图＋bundle 内 scan；L660-668 `coreIdle = regd==0 && fscoreboard==0 && rvvIdle && !lsuActive`                                                              | 位图按 Context 复制/加维；coreIdle 是 WFI 静止输入，须拆分             |
| `scalar/UncachedFetch.scala` L39-104 FetchReorderBuffer 按 txid 匹配＋全局 `flush`；L463-538 单一 PC、`ongoingBranchOrFlush` 丢弃结果、`flushTx`；L577 branch/iflush 冲刷整个 instructionBuffer                                | 单流水全局冲刷；须改为每 Context epoch＋定向丢弃                       |
| `scalar/SCore.scala` L147 `dispatch.io.halted := csr.halted \|\| csr.wfi \|\| debug…`；L215-222 CSR 仲裁仅 dispatch/DM 两路；L301-335 每 lane 写回仲裁并断言 ≤1；L525-548 vstart/vxrm/vxsat/frm 单实例路由                     | 单 Context 假设贯穿 CSR/写回/halt；CoreAxi 直接用 core WFI 关钟（H06） |
| `rvv_backend_rob.sv` L127-148 单一 `multi_fifo` uop FIFO，`clear=trap_flush_rvv` 全清（L143/L181/L224/L269）；L293-295 从队首按序退休，lane i 依赖 lane i-1；L309/353 条目携带 `vector_csr` 快照                               | RVV ROB 是全局 FIFO＋全局冲刷；须按 Context 加逻辑队首/定向取消        |
| `rvv_backend_dispatch.sv` L165-168/L332/L365-411：dispatch 时读出 `vector_csr`（vstart/vxrm/frm/lmul/sew）随 uop 传递（H14）                                                                                                   | 配置快照在 dispatch 时刻的既有先例，直接扩展                           |
| `Parameters.scala` L77 `instructionLanes=4`、L100 `rvvVlen=128`、L119 `useRetirementBuffer = enableVerification`（mini/full 两态）、L131-142 `scalarRegCount=floatRegCount=32`、`retirementBufferSize=16`、`retirementLanes=8` | 规模不冻结；跨语言配置一致性是 INV-11 硬要求                           |

### 3.2 Context 身份与槽位表

身份按整合合同定义（逻辑名，物理编码待评审冻结）：**TaskKey = (root_launch_generation, GridInstanceId, task_id)**，物理 Tile 只作路由单列、不进聚合身份；**ContextKey = (tile_id, ctx_slot, ctx_generation)**，其中 `ctx_generation` 与 `arena_alloc_gen`、`program_epoch`、`profile_gen`、`run_gen` 是**互不替代的独立代际域**；操作与事务身份在 ContextKey 之上加 `op_seq`（父指令）、`subop_id`（向量 uop/访存分片）、`txn_id`（数据通路事务）。总线不必逐位携带完整字段，允许在途表索引恢复，但映射在操作结束前有效（r5 §6.1 约束）。W2 槽位表每项至少含：

- `ctx_generation`（槽复用代际，复用时递增；旧回包按 ContextKey 拒收并计数，CTX-09/INV-15）
- TaskKey（来自 W1 提交消息）、入口 PC、program 驻留句柄（W5，含 program_epoch 校验）
- `fetch_epoch`（本 Context 重定向计数，见 §3.5）
- `cfg_version`（本 Context 配置代际，见 §3.6）
- 状态 FSM、每 Context 私有 ROB 分区的 head/free 指针与在途 credit（§3.4 推荐）
- WAIT 状态位＋W6 wait 记录的只读摘要（记录本体 W6 唯一写）；region park 投影位（等待记录本体归 W3，MAT-12"每 Context 至多一个"）
- 停顿归因寄存器（SCH-10 唯一类别）与退休/发射计数

代际域边界（CTX-05/CTX-06）：`fetch_epoch` 只判取指回包新旧，不承担任务隔离；`ctx_generation` 只在槽位复用时递增并隔离旧任务的一切迟到结果；`run_gen` 覆盖程序重装与复位（同域递增）；`program_epoch`（W5 驻留失效）、`profile_gen`、`arena_alloc_gen` 各管各的资源代。禁止任何两者互相替代。

准入握手（W1/Group 主导，W2 只执行本地段；物理 opcode 名待评审，本文用逻辑名）：Group 有界 ticket 的无副作用 PLAN → 父 L2 pin/R 预留（Python）→ **RTL 本地资源 hold**（W2/W4 预留槽位、Frame、L1 空间，**不绑定可执行 UCE、不取指**）→ Group 确认父/本地全部持有后发最终 **COMMIT 授权** → RTL 原子安装 Task 绑定并 ACK **accepted**。授权前失败：补偿释放并 ACK，票据保持到释放证据；最终授权后不 ROLLBACK，走 CANCEL/drain。程序安装（W5）可与 hold 并行；**READY 本身永不授权未 accepted 的 Task 执行**。无跨 Tile gang（ADM-03）。validator 的 ACCEPT/FRAME_BIND 倒计时只是模型口径（§6.2 映射），不作为 RTL 机制。

### 3.3 调度与发射

每 Context 状态机映射 validator `_UCEContextState`（EMPTY/ACCEPT/FRAME_BIND/READY/WAIT_EVENT/WAIT_STREAM/WAIT_ENGINE_QUEUE/DONE/FAULT，[../../../pipeline_validator/tile.py](../../../pipeline_validator/tile.py) L55-64）加 W2 内部细分：WAIT_RESOURCE 拆 `wait_engine_queue`、`wait_stream`、`wait_region`（新类别，对齐 MAT-12 park）、`wait_fetch`、`wait_operand`、`wait_dest`；RUNNING 作为"本周期被选中"属性而非驻留状态（§6.2）。回绕到 validator 观测口径时合并为 §6.2 的表。

发射选择（推荐策略，非指纹，PAR-07）：

1. 每 Tile 每周期至多选择 1 个 Context，发射 1 个 bundle；bundle 全部 lane 来自该 Context 且为程序顺序连续前缀（§7.5）。已选保留 Coral 4 lane；不混发多个 Context。
2. 轮转：从上次选中槽位的下一个 Context 起扫描，取第一个 eligible 的头；**阻塞头不使指针前移**（validator `_select_context` 语义，tile.py L635-653）；本轮不加 aging/固定量子，须证明 RR 满足 SCH-03 有界服务（延伸到取指端口 FET-07 与 LSU/MFE 选择 MEM-02）。
3. `eligible(c)` 按 §7.1 分级预计算（槽态+取指头可用 → 源依赖+目的资源 → engine 队列容量探测 → 合法性），分级重试不得在检查与接受之间转移资源所有权。
4. 补位语义：A 的头等待事件/引擎队列/region 时不选 A，但 A 已接受工作照常执行；B 独立就绪即获服务（SCH-02）。region 等待的 Context：START 声明指令位于头部即 park（不进共享向量队列之前申请，MAT-20），其后续指令按 SCH-01 不得越过；授予后由 W3 的授予信号唤醒 A 在原头部继续（事件驱动，不做逐 tick 轮询，对齐 总体设计 的 L1 gate 原则与 MAT-12）。
5. 同周期竞争（新唤醒、资源释放、新准入）确定优先级： 完成回写/flags 落账 → 可退休处理 → 新发射授予 → 状态推进；已选此逻辑序，具体阶段由 W2-01／W2-04 固定并验证（SCH-08）。

发射 bundle 部分接受：接受 lane 数 k 后，仅前 k 条推进 next-PC/seq/scoreboard/发射计数；未接受 lane 的 payload（含操作数读出与配置快照）保持稳定直至接受或按协议取消（SCH-05/SCH-06）；Coral 已有"首拒后不得再接受"断言（RetirementBuffer L106-110），按 Context 保留。

### 3.4 寄存器堆、scoreboard 与退休组织

RF 组织候选（不冻结，§8.1 允许任一）：

- **已选 A（CS1）**：GPR(32×XLEN)/FPR(32) 每 Context 全复制；scoreboard 为每 Context 一份 32 位位图（DispatchV2 位图的按 Context 实例化）。隔离结构直接（STA-02），代价为每 Context 存储与写口路由；面积/时序尚无实现证据。
- **未采用 B**：共享分 bank 存储以 `{ctx, regnum}` 寻址，scoreboard 按 `(ctx,reg)` 键。是否节省面积取决于端口/mux/实现库；需 bank 冲突显式背压（STA-07）与无跨 Context 假依赖证明。
- VRF 物理与读出归 W3；W2 维护每 Context 的向量目的 busy 摘要（向量 scoreboard），uop 级跟踪在 RVV ROB（RET-02 的 RVV 侧逻辑队首同样按 Context，见 §7 SV 改造）。

ROB 组织选择：

- **已选 1（CS1）**：每 Context **静态私有分区**＋独立 head/free 指针与在途 credit。回收判定只依赖本分区，无跨 Context tail 推进问题，隔离与 credit 守恒证明最直白；代价是分区容量静态划分（某 Context 满不侵害他人，也不能互相借用）。
- **未采用 2（以后另行评审）**：共享 free-list＋每 Context 私用链表（动态借还）。容量利用更好，但 free-list 仲裁、借还与回绕隔离需要额外证明。
- **否决**：单一共享环形结构＋每 Context 逻辑队首——已退休条目的物理回收仍要等全局 tail 越过其他 Context 的更老条目，重新引入跨 Context 回收 HOL，RET-03 不成立；且"共享省面积"未经 PPA 证实，不作断言。

匹配键改造（RET-01 核心，全部替换 Coral 的 PC/寄存器号匹配）：

| 完成来源                | Coral 现键             | 新键                                              | 依据          |
| ----------------------- | ---------------------- | ------------------------------------------------- | ------------- |
| scalar store 完成       | PC（L28/L587-589）     | `(ContextKey, op_seq)`＋txn_id                    | RET-01、X-27  |
| 非 RVV 故障             | mepc==addr（L494-495） | `(ContextKey, op_seq)`                            | RET-01        |
| scalar/float 长延迟写回 | 寄存器号（L467-476）   | `(ContextKey, op_seq)`                            | STA-03/INV-05 |
| 向量/VME 写回、RVV 故障 | rob_tag（L482/L573）   | `(ContextKey, op_seq/subop_id)`，全局唯一命名空间 | RET-02        |

`op_seq`/`subop_id` 位宽按最大在途数＋最大响应存活期配置，或在受控排空后回绕（CTX-06）；分区 ROB 的 slot 复用自带 slot 代际，回绕串扰按 `(ContextKey, op_seq)` 整体判别。

副作用时刻（§8.4 基线）：寄存器写在 completed 阶段即提交（Coral L58-66 现状保留）；外部 store 效果在数据通路完成点结算，退休只释放跟踪资源；异常模型为任务级 fail-stop（VB-11/FLT-01），不做事务回滚，也不宣称可恢复精确 trap（D-09/D-17 为增强）。向量父指令 RET-04：全部子 uop 完成或错误结算才终态，故障不得使父项永久占用跟踪资源。

任务终止与退休链（RET-08；Group 驱动退休，W2/RTL 是本地执行 owner）：`RET` 退休 → 等该 Task 全部本地 launch 事件完成（对齐 validator `task_retirement_wait` 口径，tile.py L1298-1318）→ W2 仅向 W6 发**内部通知 EXEC_DRAINED（逻辑名）**，不是 Task 成功 → RTL 确认本地无旧在途、VME 区域已结束后报告 **RETIRE_READY** → Group 发 **RETIRE 授权** → W4 执行本地 view/Arena/Frame 关闭与回收 → RTL 可靠保存 **TASK_COMPLETION** 并交 Group（不等待父 R 释放；Python 不再重复执行 L1 retire）→ Group 清理剩余父 L2 pin、归还 R 并结算 Grid 记录，再走 root 自己的 L2 release/Arena 退休。输入纯读 pin 可先按 W6 §3.3 的 Grid 输入聚合解除，退休不重复回收，且输入早释放不归还 R／L1／Frame。授权后不回滚；退役控制/ACK 通道保留 credit 与 terminal record；槽位复用条件：旧义务全部闭合且记录与新代隔离（保守实现可等完成握手，CTX-15）。

### 3.5 取指与分支

- 每 Context 独立 fetch PC、取指缓冲与分支恢复状态（FET-01）；取指请求/响应携带 `(ContextKey, fetch_epoch, fetch_reqid)`（FET-02）。Coral FetchReorderBuffer 的 txid 机制（UncachedFetch L39-104）保留，键按此扩展。
- 重定向只作用于本 Context：分支/跳转解析、本 Context 的 LSU flush → `fetch_epoch++`，丢弃该 Context 过期在途响应（丢弃必须完成握手并归还事务信用，CTX-09）；不改写其他 Context 的取指缓冲（FET-05，替代 Coral 的全局 `flushTx`/instructionBuffer 全冲刷）。
- 旧回包处理：`(ContextKey, fetch_epoch, fetch_reqid)` 不匹配（含 ContextKey 中 `ctx_generation` 已换代）→ 拒收＋计数（可观察，INV-15 的"处理方式可观察"）＋归还信用，不得写新任务状态。
- 取指 miss/长延迟不得无条件阻塞其他 Context（FET-03）：指令存储端口按 Context 轮转＋有界服务（FET-07），饥饿计数进 PMU；共享端口带宽与仲裁实现归 W4/W5 界面，W2 提出需求。
- 首条取指门控：仅在 Task 已 accepted（COMMIT 授权完成）且 W5 程序 INSTALL→READY 真实握手完成后解锁；执行中 text 不得被覆盖（RES-08/FET-06 归 W5 合同，W2 检测取指 fault 上报）。validator 把冷装载代价计入 ACCEPT 倒计时只是模型口径（RES-12），不作为 RTL 冷装载机制引用。
- Context 切换产生有限气泡即可，无全核 flush（FET-04）；park 不冲刷取指状态。

### 3.6 配置快照与 CSR 顺序

- 每 Context 一份配置影子：`vtype/vl/vstart`（架构态）、`frm`、`vxrm/vxsat/fflags`（带 owner 的累加器）、矩阵 `mset*` 字段；Tile 只读能力/时间基准 CSR 共享不复制（§6.4）。
- 快照机制：SV 域已有 RVV 先例（H14）——dispatch 时刻把 `vector_csr` 快照随 uop 带走（rvv_backend_dispatch L332 等）；**本轮已选方案**：scalar/CSR 侧把同一原则推广为"dispatch 时刻快照＋架构态按 Context 内程序序在退休点提交"，Coral 单 Context 现状不含此机制，不得写成保留既有行为。后执行 Context 的 `vset*/mset*` 不改已接受操作（CFG-02），同 Context 配置写与相关操作严格有序、不穿透在途操作（CFG-03）。
- `mset*` 字段虽取自 GPR，仍属有限位域＋快照机制管辖（MAT-10 的运行期检查归 W3，W2 只保证快照时机一致）；配置转移越界由 W3 判定并回报故障。
- 异步 flags（fflags/vxsat）归属发起操作的 Context（CFG-04）：每 Context 一份架构累加器；异常更新作为带 `(ContextKey, op_seq)` 的 **delta 记录在所属 ROB 条目上**（早到的年轻 delta 挂条目等待，不直写累加器），架构累加器只在**该条目自己的退休点**按本 Context 程序序合并；软件清零是清零指令位置上的写、在其退休点生效。由此"older FP→clear→younger FP、年轻者提前完成"时，older delta 先合并、clear 随后覆写、young delta 在其条目退休时再合并，不会被清零抹掉；禁止"先到先合并"。
- **CSR 读值与配置可见性互锁**：退休点合并/提交意味着**提前执行读不到新值**——`csrr fflags/vxsat/vl/vtype` 等架构态读须等本 Context 更老的未合并 delta/未提交配置全部结算（本轮不实现 forwarding）；`vset*/mset*` 未提交时，后续依赖该配置的向量/矩阵 op 的快照须等其提交。Coral 先例：DispatchV2 `canDispatch` 的 `isCsr ⇒ retirement_buffer_empty`（Decode.scala L707-708，单 Context 用全 ROB 排空换正确性）与 RVV `configInvalid` 互锁（L524-533，挡 vset 后的向量 LSU dispatch）。**已选实现**：把"全 ROB 空"改为**每 Context** 的 CSR/pending-config gate——只在"本 Context 存在未结算的 flags delta 或未提交配置写"时挡住 CSR 读与依赖快照的后续 op。该屏障是本 Context 程序序必要串行化，不违反 CFG-05（该条只禁无关 Context 引起的全核串行化）。不采用以全面局部排空代替 ordered-delta 的备选方案；保留本条 CSR 读／pending-config 必需互锁。
- Context-local CSR 访问不得因无关 Context 在途而全核串行化（CFG-05）；Tile-global 特权清单（CFG-06）**非空**：profile 切换、reset、program invalidate、debug 控制属管理侧特权路径（W6/管理面持有，小 debug FIFO 与特权导出控制归 W6），普通 Context 禁止触发；只读能力/时间基准 CSR 共享。

### 3.7 等待、WFI 与终止的分离

- 事件等待（WAIT/WAITALL，EVT-01～05）：只挂起本 Context 后续发射，已接受工作与返回处理继续（EVT-04）；等待检查具备无丢唤醒语义（先到/同周期/后到，EVT-01）；事件是持久状态（EVT-02），event_generation 防旧唤醒（EVT-03）。**wait 记录本体由 W6 唯一写**；W2 只持 WAIT 状态位与只读 eligibility 摘要，头部 park、非冲刷。
- WFI/park（ISA-04）：WFI 退休 ⇒ 本 Context 进 PARKED；WFI **不新增任何隐含 drain 或自发完成合同**——只是 local park 提示，本 Context 已接受在途工作照常自然完成（不是 WFI 触发的排空），**不等价任何 event wait，也不由它产生 event 完成**；不产生全 Tile 关钟。Coral 中 `dispatch.halted <= csr.wfi`（SCore L147）与 CoreAxi 引用 core WFI 关钟的路径必须改为"每 Context park 输出＋聚合 quiescence 状态"两件事，后者由 W6/boundary 消费（H06 风险行）。
- 正常终止（CTX-07）：先停新发射（RET 之后无新指令）、继续处理已接受工作、本地事件全完成、DONE。
- fail-stop（SCH-11/FLT-01 基线）：任一 Context FAULT ⇒ 归属 `(task, ctx, 操作)` ⇒ 本 Tile UCE 停止发射（validator tile.py L315-317 同口径），引擎继续计时排空，交 W6 复位状态机。FLT-02a 的 Context 局部隔离是 SHOULD 级增强，CS1 不实现也不宣称。

## 4. 接口、descriptor、寄存器和协议

消息边界化、有界、可序列化（W1 总合同）；下表只列 W2 侧逻辑通道。

| 对端          | 方向 | 内容（逻辑字段；物理 opcode 名与编码待评审，勿自造固定位）                                                                                                 | 背压/错误                                                                                                                                 |
| ------------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| W1 边界       | 入   | 准入握手本地段：hold 请求（槽位/Frame/L1，不绑可执行 UCE、不取指）、COMMIT 授权、CANCEL；TaskKey＋context_slot 钉住（CTX-14）、入口 PC、各代际             | 钉住被占 ⇒ 该 Tile 该 Task `WAIT_SLOT`，他 Tile 继续（CTX-14）；越界静态钉住在加载期拒绝；授权前失败补偿释放并 ACK，授权后走 CANCEL/drain |
| W1 边界       | 入   | freeze/reset（复位域 GROUP）、代际更新                                                                                                                     | freeze 期间完成回包仍处理，不新发射（对齐 tile.py L300-313）                                                                              |
| W1 边界/Group | 出   | accepted ACK、RETIRE_READY、per-Context 进度采样（终态经 W6 的可靠 TASK_COMPLETION，W2 不提前发成功）                                                      | 记录受背压时稳定保存（CTX-11）；退役控制/ACK 保留 credit 与 terminal record                                                               |
| W3 RVV/VME    | 出   | uop 发射（`(ContextKey, op_seq/subop_id, 配置快照)`）、START/END 声明、flush/cancel(定向)                                                                  | 实例入口队列满只挡该实例（MAT-22）；定向 cancel 替代 RVV 全 FIFO 冲刷                                                                     |
| W3 RVV/VME    | 入   | 向量/VME 写回与 fpexp（`(ContextKey, op_seq/subop_id, last_uop)`）、region 授予/`region_done`、配置转移错误                                                | 写回匹配新键；region 授予事件驱动唤醒 park 的 Context（MAT-12；等待记录 W3 唯一写）                                                       |
| W4 LSU/存储   | 出   | 访存请求（`(ContextKey, op_seq, txn_id)`）、地址窗口校验需求                                                                                               | 目的地资源不足为独立归因类                                                                                                                |
| W4 LSU/存储   | 入   | load 返回、storeComplete（`(ContextKey, txn_id)`）、访问 fault                                                                                             | storeComplete 键由 W4 合同携带；fault 按 `(ContextKey, op_seq)` 归属                                                                      |
| W5 程序       | 入   | INSTALL→READY 握手、驻留句柄（program_epoch）、epoch 失效                                                                                                  | 未 accepted 且未 READY 不解锁首取指；热启动不改 resident text（RES-10）                                                                   |
| W5 程序       | 出   | 取指请求（`(ContextKey, fetch_epoch, fetch_reqid)`，经共享指令存储端口仲裁）                                                                               | 响应过期拒收＋信用归还＋计数（FET-02/CTX-09）                                                                                             |
| W6 事件       | 双向 | 事件完成通知（runtime_id→ContextKey）、wait 记录（W6 唯一写，W2 只读摘要）、EXEC_DRAINED 内部通知与可靠 TASK_COMPLETION（W6 持有）、故障上报、复位排空配合 | completion queue 满背压不丢事件（EVT-09）；控制通道独立于数据通道（INV-16）                                                               |
| W7 验证       | 出   | per-Context 发射/退休流、uce_launch_attempt/accepted 对应物、停顿归因、PMU                                                                                 | W7 只做 Host 侧格式/checker/观察；小 debug FIFO 与特权导出控制归 W6；观测不改变行为（总体设计）                                           |

接口采样遵循 W1 分相合同：P1 相位双边沿前 valid&&ready 快照，P3 相位只提交该接受集；C++ 侧不回调 Python、不执行 P4；cycle/P4 由 Python 唯一推进（W1 文档为权威）。

三个可分离时刻（§5.2/SCH-09）：UCE 接受（发射握手）、engine 接受（前端队列握手）、engine 完成——分别可观察、分别计数，对齐 validator `uce_launch_attempt`/`uce_launch_accepted`（tile.py L1011-1057）与完成通知触发本地事件。

## 5. 数据流、控制流和时序路径

1. **准入**：按 §3.2 握手执行本地段：hold（不绑可执行 UCE、不取指）→ COMMIT 授权 → 原子安装 Task 绑定并 ACK accepted；W5 程序安装并行进行。安装后槽位初始化写全量执行状态（入口 PC、栈、配置、事件、清 scoreboard/ROB 分区视图），旧 occupant 残留不得泄漏（CTX-02）；validator 的 ACCEPT/FRAME_BIND 倒计时只是模型口径。
2. **取指**：accepted＋程序 READY 后经端口仲裁发 `(ContextKey, fetch_epoch, fetch_reqid)` 请求 → 响应按键匹配写入本 Context 取指缓冲；miss 由存储侧（W4/W5）延迟，其他 Context 不受无条件阻塞。
3. **发射**：调度器选中 Context → 头部 bundle 依 scoreboard/配置快照/队列探测分级检查 → 握手接受 k 条 → 推进 PC/op_seq/scoreboard；未接受部分保持稳定。
4. **执行/完成**：ALU/BRU 按所选实现的真实完成点结算；LSU/长延迟/RVV/VME 异步完成，响应携带身份键回到退休结构；写回允许发往前端未选中的 Context（STA-06，改造 SCore 每 lane 写回仲裁 L301-335 为带 ctx 路由）。
5. **退休**：各 Context 从自己分区队首按序退休；store/故障/长延迟写回按 `(ContextKey, op_seq)` 匹配；A 队首未完成不挡 B 退休（RET-03）。退休计数与"指令完成""任务完成"口径分开（RET-07）。
6. **终止与退休**：RET 退休 → 等本地事件全完成（task_retirement_wait 对应计数）→ EXEC_DRAINED 内部通知 W6 → RETIRE_READY → Group RETIRE 授权 → W4 本地回收 → 可靠 TASK_COMPLETION 交 Group（§3.4 链条全文）；槽位复用按 CTX-15＋旧义务闭合与记录隔离后进行（ctx_generation++）。
7. **分支**：解析重定向 → `fetch_epoch++` → 本 Context 过期取指丢弃；更早合法在途访存/计算继续（FET-05，T-07）。
8. **故障**：归因 `(task, ctx, 操作)` → 停全 Tile 发射 → 引擎排空 → W6 复位状态机；同周期多故障 first-fault 摘要归 W6。
9. **槽位复用**：上一 Task 终态且退役控制/ACK 通道 credit 与 terminal record 已按新代隔离（保守实现等完成握手）＋本地 launch 事件全完成＋ingress 无残留（CTX-15）→ 复用；旧通知/重复响应按 ContextKey 拒收并完成握手（CTX-09）。

错误表（W2 判定者，非穷尽）：未知事件 wait、重复完成事件名（ISA-09）、取指 fault、scoreboard 违例（写口冲突/未标记写回）、ROB 满且归因、epoch/代际不匹配（拒收非故障）、配置转移非法（W3 判定回传）、region 协议违规（W3 判定回传，如嵌套 START）。故障停发与归因走 §3.7 fail-stop 路径。

## 6. 配置、PPA、性能模型和 PMU

| 参数                  | 建议值/策略                                                                                                                                | 状态                                                 |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| `context_count`       | 首版 C1=1、C2=2；更多 Context 不自动列为 supported                                                                                         | 目标配置已选；状态深度仍需资源证明                   |
| 发射 bundle 宽度      | 保留 Coral 4 lane，单 Context 连续前缀                                                                                                     | 已选；进入 target 指纹                               |
| ROB 组织              | 每 Context 静态私有分区＋独立 head/free/credit                                                                                             | 已选候选 1；深度由 W2-01 冻结                        |
| GPR/FPR 组织          | 每 Context 全复制                                                                                                                          | 已选 A；写回端口预算仍须验证                         |
| 取指缓冲/ROB/seq 位宽 | 覆盖最大在途×最大响应存活期；否则受控排空后回绕（CTX-06）                                                                                  | 由后续规格冻结                                       |
| 停顿归因优先序        | 既有 wait_event > engine_queue > stream 相对顺序保留；新增 wait_region/wait_fetch/wait_operand/wait_dest 按实际 eligibility 拦截点唯一归因 | W2-01 与 W7-01 冻结完整枚举表，禁止 UNKNOWN fallback |

性能观测：per-Context 发射/退休/停顿计数、`ctx_switch`、取指饥饿计数、拒收旧回包计数、`task_retirement_wait` 对应计数。比较口径：RTL 的 per-Context **原生指令序**对拍**独立 ISA oracle**（golden model，W7 建）；validator 只对拍 **abstract contract 节点**（launch/wait/signal/RET 等合同节点序列）与跨 Context 依赖偏序——不做 raw 指令序等价、不比较 PC，也不要求全局 interleave 逐周期一致（总体设计 时间权威条款）。不承诺利用率估算、无 PPA 宣称。

## 7. RTL/软件实现建议

目标 RTL 目录 `hardware/coral_nexus/` 待 W0 创建；本轮仅设计批准同步与计划，不依赖历史 hardware 工程。

Chisel 侧（scalar 域）：

1. `Parameters.scala`：增加 `contextCount` 等参数；保持 `useRetirementBuffer = enableVerification` 两态在对比时显式对齐（mini/full RB 差异，L119 与需求 §3.3）。
2. `UncachedFetch.scala`：PC/epoch/取指缓冲按 Context 实例化；FetchReorderBuffer 键扩 `(ContextKey, fetch_epoch, fetch_reqid)`；`flushTx`/instructionBuffer 冲刷改为按 Context 定向（替代 L577 全冲刷）。
3. `Decode.scala`（DispatchV2）：scoreboard 向量化（每 Context 位图）；`canDispatch` 的 `isCsr ⇒ retirement_buffer_empty`（L707-708）改为每 Context CSR/pending-config gate（§3.6）；`rvvConfigInterlock`/`configInvalid`（L524-533）按 Context 化；`coreIdle` 拆为 per-Context idle＋Tile quiescence 聚合。
4. `RetirementBuffer.scala`：条目加 ContextKey＋op_seq（分区 slot 自带代际）；替换 L28/L467-476/L494-495/L587-589 四处匹配键；按每 Context 私有分区组织队首与回收（§3.4 候选 1）；保留前缀断言。
5. `SCore.scala`：CSR 按 Context 实例化＋Tile-global 只读共享；写回仲裁加 ctx 路由；`dispatch.halted` 的 wfi 项改为 park 语义。
6. `CoreAxi.scala`：删除"core WFI ⇒ 全核关钟"直连，改为 park 聚合＋quiescence 输出（与 W6 联合评审）。

SV 侧（RVV 域）：

1. `rvv_backend_rob.sv`：uop FIFO 条目带 ctx；`trap_flush_rvv` 的全 FIFO `clear`（L143/L181/L224）改为按 Context 定向取消或退休屏障，否则一个 Context 的 trap 冲掉其他 Context 的合法 uop（RET-06）；退休组织与 scalar 侧一致选每 Context 私用分区（§3.4 候选 1/2，与 W3 联合定案）。
2. `RvvFrontEnd.sv`：`vset*` 配置态按 Context；dispatch RS 条目在 `vector_csr` 快照（H14 先例）之上加 ctx 归属。
3. `rvv_backend_define.svh` 与 Chisel 参数同步 context 数/深度（INV-11；VLEN 教训同源）。

落地顺序建议：身份与匹配键（RET-01）→ scoreboard/RF → 取指 epoch → 配置快照 → park/fail-stop 分离；每步保持单 Context 退化配置可运行（N=1 时所有 ctx 字段退化为常量，MAT-22 同款退化原则）。

## 8. 验证、bring-up 和验收标准

下表把本任务点名的验收场景映射到需求 T 用例、具体观测与不变量（执行载体归 W7；RTL/validator 对拍口径见其文档）：

| 场景                  | T 用例              | 观测/不变量                                                                                                                                                                                                                                                                                   |
| --------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 同映像同 PC stores    | T-02                | 两个 Context 同映像、同 PC store 并发在途：storeComplete 只按 `(ContextKey, op_seq/txn_id)` 命中各自条目；不变量：任一时刻不存在两个活条目共享 `(ContextKey, op_seq)`；负向：按 PC 匹配的基线行为注入可见错配（动机证据，Coral L28/L587-589）                                                 |
| 同号寄存器            | T-02/T-06           | A 写 x5 在途，B 读 x5 不被 A 假依赖阻塞（发射记录证明）；forwarding 命中含 owner（STA-03）；不同 Context 对同一逻辑寄存器号的 scoreboard 位图互不可见                                                                                                                                         |
| 旧 fetch 回包         | FET-02/T-07 变体    | 重定向后带旧 epoch 的迟到响应被拒收＋信用归还＋计数递增；其他 Context 取指缓冲位不变；epoch 单调                                                                                                                                                                                              |
| 按序前缀部分接受      | T-11/§7.5           | 4 lane 接受前 2：op_seq/接受计数推进 2，PC 按两条指令实际长度推进，scoreboard 只更新已接受指令的目的项；剩余 lane 重新呈现不变形（SCH-05）；无重复/跳过发射（SCH-06 的 valid/ready 边界注入）                                                                                                 |
| A 退休头阻塞 B 可退休 | T-09/RET-03         | A 头等 store 完成，B 完成即从 B 队首退休（退休流时间戳证明）；各分区/逻辑队首无全局队首等待                                                                                                                                                                                                   |
| 回绕/满队列           | T-12/CTX-06         | seq 回绕仅在该 Context 无活操作时（或位宽证明无别名）；ROB/scoreboard 满 ⇒ 发射停且归因类唯一（SCH-10）                                                                                                                                                                                       |
| 待 region 不越过      | T-28/T-62/MAT-20/21 | A 在 START 头 park 期间其年轻指令零发射记录；B 补位运行；授予后 A 在原头继续；嵌套 START 判协议故障（W3 回传）                                                                                                                                                                                |
| A 在途时切 B          | T-05                | B 发射期间 A 的合法写回照常落账（STA-06）；A 结果按身份返回（CTX-04）                                                                                                                                                                                                                         |
| flags 有序合并        | T-06/CFG-04         | older FP→clear→younger FP，年轻者提前完成：三条全退休后架构 fflags 恰为 younger delta（older 先合并、clear 覆写、young 条目退休时合并）；观测：逐条目 delta 记录＋最终 CSR 读值；不变量：delta 只在其 `(ContextKey, op_seq)` 条目退休点合并，年轻 delta 不因 older 清零丢失                   |
| CSR 读/快照可见性互锁 | T-06 变体/CFG-01~04 | 本 Context 有未合并 flags delta 或未提交 vset\* 在途时：`csrr fflags`/依赖新配置的向量 op 不发射（发射流证明），结算后放行且读到/快照到新值；观测：gate 挡下的发射记录＋读值；不变量：架构态读值＝该读之前全部同 Context 更老写的结果序合并；其他 Context 的 CSR 读不被本 gate 阻塞（CFG-05） |
| WFI≠事件等待/park     | T-27/ISA-04         | 全 Context park 时其后端在途工作自然完成、完成事件仍被处理（SCH-07）；WFI 不新增 drain/自发完成合同（无由 WFI 产生的 event 完成、无隐含 store completion）；门控不依赖被 park Context 再运行（INV-08）                                                                                        |
| fail-stop             | T-25/SCH-11         | 故障归属三元组正确；本 Tile 发射停止且引擎排空计数继续；与正常 RET 终止路径可区分（两条独立观测）                                                                                                                                                                                             |
| 槽位复用/旧注入       | T-08                | 复用后旧代响应/通知全部拒收且信用守恒；DONE→复用满足 CTX-15 三条件                                                                                                                                                                                                                            |
| 联仿对拍              | T-39                | RTL per-Context 原生指令序对独立 ISA oracle；validator 只对拍 abstract contract 节点序列＋跨 Context 依赖偏序（不比较 raw 指令序/PC）；`uce_launch_attempt/accepted` 三时刻计数一致（SCH-09）                                                                                                 |

实现 agent guide（给低能力编码模型的执行指南，不是再造降阶模拟器）：①读——先读本文锚点的 Coral 源与 validator 轨迹（`uce_issue`/`uce_launch_attempt`/`uce_launch_accepted`/退休流，tile.py L680-708/L1011-1057）；②实现不变量——匹配键唯一性（`(ContextKey, op_seq)` 无双活）、scoreboard 按 Context 不相交、前缀推进一致性（op_seq 增量等于接受指令数、PC 增量等于已接受指令长度之和）、epoch 单调、credit 守恒（拒收也归还）、WAIT/region park 只挡本 Context；③证——ISA oracle 逐指令对拍＋validator 合同节点对拍；停止条件：全部 Context 终态＋在途队列为空＋在途占用清零且可用信用恢复初值＋退役 ACK 闭合。

## 9. 风险、取舍和后续细化方向

1. 首版已选 C1/C2、四 lane、私有分区 ROB 与复制 RF；深度／端口与面积代价仍须实际配置和验证，不作省面积断言。
2. RVV 定向取消采用按 owner 标记并排空／隔离已接受项，不整队列 clear；W3 实现，W2 集成 scalar/顶层接口。CS1 仍为 Group fail-stop，不能据此宣称 Context 局部故障恢复。
3. wait_region 等新增停顿类别由 W2-01／W7-01 注册进唯一枚举表，按实际拦截点固定优先序并做重叠原因负例。
4. 已选 fflags/vxsat ordered-delta 合并＋CSR 读／pending-config 局部互锁；mset 快照归 W2，MAT-10 合法转移检查归 W3，W5 封印同一张表。
5. 写回端口预算：全复制 RF＋N Context 写回路由可能成为时序瓶颈，需在实现规格中给出端口数与仲裁表。
6. 取指端口与冷装载争用本地指令存储（RES-17/T-58/T-66）由 W4/W5 主责，W2 只提有界服务需求；若仲裁失效表现为 FET-07 饥饿计数。
7. validator 不建模取指/寄存器/译码（程序为 insts 列表＋label 跳转，tile.py L716/L733-740）；validator 对拍只用合同节点/依赖（§8），RTL 指令序对独立 ISA oracle；仅当需要把 RTL 合同节点对回 validator 节点时，才由 W7 定义两者的锚点映射。
8. FLT-02a（Context 局部故障隔离）明确不在 CS1：本文按 fail-stop 书写；若未来升级，退休结构与调度需重开设计。

[H01]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/Core.scala
[H11]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/verilog/rvv/design/RvvFrontEnd.sv
