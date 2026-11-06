# W6 事件／故障／取消／复位／时钟门控 实施计划（r1）

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

状态：**实施计划（本轮只写计划，不执行任何 build/test/formatter，不创建任何 hardware/ Python 目标代码）**。
W1 权威接口对齐（按 W1-01 计划草案）：协议冻结产物由 W1-01 单点生成（`protocol/cn_protocol_v1.yaml` + gen 产物 + `PROTOCOL_FREEZE.json`）；本包以 `hardware/coral_nexus/protocol/w6_event_facility.contract.yaml` 与 `protocol/w6_terminal_reset.contract.yaml` **fragment** 输入 W1-01（gen_protocol.py 读取、hash 以 fragment 自身为准），不复制内容、不写 W1 文件。新增消息 `GEN_WRAP_QUIESCE`（Tile 只查不超时的 ContextKey 级旧代排空查询）已由 W1-01 登记（W1-01 持 channel/message_seq/幂等键）。libcn_cosim_stub.so 仅单元，不得用于 CS1 判定（W1-03 草案）。位宽/CSR 地址由 W1-01 gen 产物消费，W2/W3/W5 禁手写数值（W1-01 csr_name_map/生成 helper）。message_seq（边界层）与指令 op_seq（归因层）分域，`root_launch_generation == grid.launch_generation` wire 只存一份。

授权：用户已 review 批准 `Coral_Nexus_Tile_Design.md` + W0–W7；本计划按 `../plan.md`（下称"合同"，含其 §2/§3：权限、ID、DAG、已选方案 9、10、未来命令）与 `design/coral_nexus_tile/components/W6_Events_Fault_Reset.md`（下称 W6 设计）展开 W6-01…W6-05，并只写本文件，不改源码、设计文档、其他包计划（`../plan.md` 亦即 `design/coral_nexus_tile/plan.md`，若父级暂未落盘，以主会话下发的同名合同内容为准）。

## 0. 阅读顺序约定

1. 合同（权限、ID、DAG、已选方案 9、10、合同 §未来命令）；
2. W6 设计（逐条带文件证据）；
3. 本计划 §2（文件证据 / 待创建清单）、§3（依赖 DAG）、§4（共享合同）、§4.4（CoreAxi 改造请求）、§5（任务）。
4. 实现轮次只读本计划 + 对应设计段 + 引用的 validator 行号；先读后改，缺接口时回 owner 报证据，不自行改合同。

## 1. 已选方案摘要、非目标与本包 owner

**已选方案（合同 §已选方案 10，全部为推荐默认，非已实测参数）：**

- 事件是持久可观察状态（EVT-02 终态 sticky，PENDING→{DONE|ERROR|TIMEOUT|RESET} 终态唯一不回退）；WAIT/WAITALL 每条**一次成功唤醒**、原子 check-register、**同拍完成优先**（完成先于 park 生效，同拍不 park 且不延迟唤醒）；
- WAIT 单事件 / WAITALL 全 AND（不设 OR/WAIT_ANY）；每 Context 一条记录（WAIT 单操作数，W6 设计 §3.2）；
- Group（Python）是 **reset 顺序与 poison 唯一权威**，Watchdog 归 Group，Tile 不设任何可影响功能状态的计时器——**超时永不等于排空成功**（RST-10、FLT-14）；
- 已接受外写不可凭 generation/ timeout 撤销（CTX-16、FLT-04）；失败隔离 ⇒ poison 拒绝新 launch 直至显式静止 reset（RST-08）；
- 关钟期间 frontier 状态、返回路径、故障路径、WAIT 匹配唤醒**必须存活**（CLK-04）；门控**不吞同拍到达**（CLK-05）；门控边界排除同拍新到达（TIM-03）；
- 正常模式零保留 trace、零 HBM log 写；debug FIFO 小而可丢、满则丢＋计数＋窗口不完整标；特权导出默认关（OBS-14/15/16）；关闭/取消/复位先停新采集、日志请求，已接受 HBM 日志写保留 owner/credit 至真实终态或隔离（OBS-17）；
- 可靠终态/事件通路与 debug 可丢通路**逻辑独立**（FLT-13），物理端口可共享但须给出隔离证明；
- fail-stop 基线：本包不实现局部恢复与 FLT-02a（W6 设计 §1）。

### 1.1 非目标（不在本包范围）

- Group 侧事件表/聚合 FSM/复位 FSM/FaultRing/poison/credit 的 Python 实现（继承 validator，仅做"RTL Tile 接入的共享接线"，由 W1 持）；
- L2/HBM 事务取消的**物理路径**（W4）；L2 执行 owner 在 Python；
- 矩阵区域语义与计算本身（W3）；多 VME 实例（W3）；
- WAIT_OR、WAIT_ANY、任何自动重试或有损重传机制；
- 时钟树/门控单元的实际功耗、面积、Fmax 估算（W6 设计 §6 未承诺）；
- Trace/重放 checker 格式（W7/OBS-11）；`Ctx{N}:<name>` manifest 映射的物理层实现（W5 manifest 侧）。

### 1.2 本包 owner 表（唯一写者，其他包只读或发请求）

| 状态/模块                                                             | 唯一 owner                                                                                                | 说明                                                                                                             |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 本地事件表（per-Context frontier 槽、full key、两计数）               | W6 (RTL)                                                                                                  | full key = (物理 Tile, ContextKey, local_event_id, event_generation, sequence)；`ctx{N}:<name>` 仅 manifest 命名 |
| WAIT / WAITALL 等待记录（每 Context 1 条，W6 唯一 writer）            | W6 (RTL)                                                                                                  | atomic check-register，同拍完成优先                                                                              |
| 完成通知/阶段信号/可靠终态发送端                                      | W6 (RTL)                                                                                                  | PHASE_SIGNAL 不去重、EVENT_NOTIFY 全键接收、TASK_COMPLETION 保存/发送                                            |
| first-fault 锁存 + 故障摘要 + 复位端点 FSM + 取消/超时排空（Tile 段） | W6 (RTL)                                                                                                  | always-on，Host 确认后可清；复位段被动执行                                                                       |
| 时钟门控控制块（per-Context front-end / per-unit / Tile 级）          | W6 (RTL)                                                                                                  | W2 提供 Context park 状态；CoreAxi 改造按 §4.4                                                                   |
| 小 debug FIFO + 特权导出控制 + 限额 + 关闭排空                        | W6 (RTL)                                                                                                  | 满丢＋溢出计数；HBM 路径默认关、特权开                                                                           |
| 事件表 per-Context 计数（stale/wrong-owner/duplicate/…）              | W6 (RTL)                                                                                                  | 命名对齐 validator `event_table.py`                                                                              |
| 事件表 Group 侧 / FaultRing / ResetDomain / StreamQueue 信用          | Python（validator 行为）                                                                                  | 本包不改语义；接 RTL 侧共享接线走 W1-04/W1-05                                                                    |
| Watchdog（max_drain_cycles / run / profile 超时）                     | Python Group                                                                                              | Tile 不设计时器；`max_drain_cycles` 当前 config 无此键（见 §5.3.6）                                              |
| POST_EVENT Tile→Group                                                 | W6（**manifest 显式授权才产生**，见 §4.1）                                                                | 无授权的 post 在 Loader/译码拒绝；无任意事件冒写                                                                 |
| region 等待记录 / region_done 语义链                                  | W3                                                                                                        | 本包只做 event_kind=region_done 映射与计数                                                                       |
| CoreAxi.scala 改动（WFI/cg/门控接线）                                 | W2                                                                                                        | 本包只**提供请求**（§4.4），不编辑该文件                                                                         |
| debug 记录格式 schema / checker                                       | W7（格式）                                                                                                | 字段语义 W6 定、位编码 W7 schema 定（W6 设计 §3.4 / W7 设计）                                                    |
| RTL 语言选择                                                          | **手写 SystemVerilog，编译产物在 `out/coral_nexus/`（本包源码 owner=本包，`rtl/` 不用 Chisel 生成双源）** | W6-04 gate_ctl / W6-02 等模块单一 SV 源；编译产物（Verilator 模型）出 out/；上游 Chisel 文件仍归 W1/W2 补丁域    |

## 2. 文件证据与创建清单

### 2.1 已存在证据（当前仓库根 `/home/yongxiy/Desktop/coral-npu-nexus`，HEAD 814bdf9 "reviewed design"）

| 项                                                                       | 路径:行                                                                                                                                                                                                              | 事实                                                                                                                       |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| EventStatus 枚举（4 终态 + PENDING）                                     | `pipeline_validator/runtime/event_table.py:11-22`                                                                                                                                                                    | EVT-16 基线                                                                                                                |
| EventEntry 字段（含 producer_bound/consumer_refs/future_uses）           | `.../event_table.py:29-46`                                                                                                                                                                                           | 表项字段全集                                                                                                               |
| reserve_quota / preflight / reserve_outputs / bind_producer              | `.../event_table.py:167-298`                                                                                                                                                                                         | frontier 配额与即时分配、producer 唯一                                                                                     |
| consume / cancel / abandon_future_uses / release_owner / \_maybe_reclaim | `.../event_table.py:299-396, 498-546`                                                                                                                                                                                | 两计数结算与回收条件                                                                                                       |
| signal 的 stale/wrong-owner/duplicate 分类计数                           | `.../event_table.py:401-455`                                                                                                                                                                                         | EVT-16 拒绝计数                                                                                                            |
| wait 只看 DONE 成功、无失败出口                                          | `.../event_table.py:457-484`                                                                                                                                                                                         | 当前 validator 不处理失败终态                                                                                              |
| reset 仅 PENDING→RESET + clear                                           | `.../event_table.py:548-567`                                                                                                                                                                                         | RST-09 复位语义                                                                                                            |
| WAIT / WAITALL 参数与 probe                                              | `pipeline_validator/tile.py:569-588`                                                                                                                                                                                 | WAIT 单操作数（args[0]），WAITALL 全 AND，无 WAIT_ANY（EVT-01 基线）                                                       |
| \_issue_wait 原子 check-register（ready 即不 park）                      | `.../tile.py:978-992`                                                                                                                                                                                                | EVT-01 无丢唤醒                                                                                                            |
| \_complete_context（全事件终态才 DONE）                                  | `.../tile.py:1298-1318`                                                                                                                                                                                              | task_retirement_wait                                                                                                       |
| \_fault_context（清 wait_refs，故障即时 FAIL）                           | `.../tile.py:1320-1324`                                                                                                                                                                                              | FLT-08/EVT-05                                                                                                              |
| 复位期间 frozen tile step（完成仍 drain，不新发）                        | `.../tile.py:300-313`                                                                                                                                                                                                | RST-07                                                                                                                     |
| SIGNAL_PHASE 发射时同步产生（Tile 不去重）                               | `.../tile.py:747-756`                                                                                                                                                                                                | EVT-14/15                                                                                                                  |
| Group `notify_event` → `t.uce.notify_event` 转发                         | `pipeline_validator/tile_group.py:1553-1570`                                                                                                                                                                         | Tile 侧外部事件信箱模型                                                                                                    |
| `_on_phase_signal` 聚合（stale→忽略、unknown/duplicate→协议故障+计数）   | `.../tile_group.py:1391-1477`                                                                                                                                                                                        | EVT-13/14、T-45                                                                                                            |
| `_unpin_grid_readers` 仅读 pin                                           | `.../tile_group.py:1010-1025`                                                                                                                                                                                        | ARN-04b                                                                                                                    |
| FaultCode / FaultDomain / FaultRecord 结构                               | `pipeline_validator/runtime/fault_ring.py:16-66`                                                                                                                                                                     | FLT-11/12、RST-01                                                                                                          |
| FaultRing 覆盖写、无 first-fault-wins                                    | `.../fault_ring.py:68-88`                                                                                                                                                                                            | FLT-10 注：模型只作 fault 计数参考，不作硬件锁存替代                                                                       |
| ResetDomain / ResetState 全相位                                          | `.../reset_domain.py:11-233`                                                                                                                                                                                         | RST-07 固定顺序                                                                                                            |
| `_outstanding_zero` 逐计数判据                                           | `.../reset_domain.py:167-219`                                                                                                                                                                                        | DRAIN 判据                                                                                                                 |
| `trigger_fault`(tile_id 符号决定 FaultDomain) / `on_scheduler_fault`     | `pipeline_validator/tile_group.py:2364-2375, 2354-2360`                                                                                                                                                              | 复位入口                                                                                                                   |
| poison 门（`poisoned_reason` 拒新启动/准入）                             | `.../tile_group.py:1992-1993, 2108-2110, 2235-2267`                                                                                                                                                                  | RST-08                                                                                                                     |
| `_event_resource_requirements`（Group 级 event frontier / event_uses）   | `pipeline_validator/compiler/resources.py:339-396`                                                                                                                                                                   | Group 级证明，本包用作"Tile 本地事件 frontier"推导的参考                                                                   |
| `budget.event_frontier` / `max_live_grids` 校验                          | `pipeline_validator/execution_verifier.py:1968-2031`                                                                                                                                                                 | 预算核对模式参考                                                                                                           |
| WAIT/WAITALL 编码处                                                      | `.../compiler/lowering.py:966-970`                                                                                                                                                                                   | 单/多操作数选择                                                                                                            |
| `CoreAxi.scala` 时门控 enable 含 `!core.io.wfi`（[H06] 已核实）          | 上游 pin f795221…L115                                                                                                                                                                                                | `cg.io.enable := irq_reg \|\| timer_irq_reg \|\| software_irq_reg \|\| (!csr.io.cg && !core.io.wfi) \|\| dm.io.haltreq(0)` |
| `ClockGate.sv` 通用 RTL 模型（ICG latch 语义）                           | 同 pin，`hdl/verilog/ClockGate.sv:44-48`                                                                                                                                                                             | 门控单元可复用，无独立时钟域                                                                                               |
| Group 现有事件行为 / poison / 复位测试基线（已可运行）                   | `pipeline_validator/tests/test_runtime.py:891-1010`（TestEventSequence / TestFaultReset）、`.../test_l2_sharing.py:394-461`、`.../test_runtime.py:3346,3408,3467`（TestCycleCapPoison / Drain / SuccessExitClosure） | 行为 oracle（部分）                                                                                                        |
| 现有 CLI                                                                 | `pipeline_validator/cli.py:213-322` / `__init__.py`                                                                                                                                                                  | **当前可执行**；本计划只引用其语义，不新增接口                                                                             |

### 2.2 待创建（本包；目标根 `hardware/coral_nexus/`（不存在，W0-01/02 建目录后落位））

> 与 W1-01 计划草案已对齐（其 §7.2 第 7 项，joint-seal 口径）：W1-01 `gen_protocol.py` 读取本包两份 fragment（只读、不复制内容，hash 以 fragment 自身为准），`freeze_check.py` 校验 path+hash；全部 contract 齐后 joint seal 进 `PROTOCOL_FREEZE.json`（不可逐条追加）。本包不改 W1 文件正文。W6 fragment 不设位编码，仅登记 `GEN_WRAP_QUIESCE` 等跨通道语义（channel/message_seq/幂等键由 W1-01 生成）。

| 路径                                                                                                                             | 产物                                                                                                                                                          | 用途                                                                                                | owner |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----- |
| `hardware/coral_nexus/protocol/w6_event_facility.contract.yaml`                                                                  | 本地事件/等待/终态/first-fault **接口合同 fragment**（逻辑名，不含位编码）                                                                                    | W1-01 `gen_protocol.py` 读取并由 `freeze_check.py` 校验 path+hash；W2/W3/W4/W5 经生成的 helper 消费 | W6    |
| `hardware/coral_nexus/protocol/w6_terminal_reset.contract.yaml`                                                                  | 可靠终态/取消/复位段/计数接口合同 fragment                                                                                                                    | 同上                                                                                                | W6    |
| `hardware/coral_nexus/rtl/w6/event_table.sv`（手写 SystemVerilog）+ `hardware/coral_nexus/rtl/w6/event_table_tb`（Verilator tb） | per-Context 槽表 + full key 查找 + 计数器                                                                                                                     | W6-02/03 的基础；本包自检                                                                           | W6    |
| `hardware/coral_nexus/rtl/w6/wait_record.sv` + tb                                                                                | WAIT/WAITALL 记录 + atomic check-register + 唤醒                                                                                                              | W6-02                                                                                               | W6    |
| `hardware/coral_nexus/rtl/w6/completion_bus.sv` + tb                                                                             | 引擎/MFE/VME/Group 完成通知汇入仲裁                                                                                                                           | W6-02/05                                                                                            | W6    |
| `hardware/coral_nexus/rtl/w6/first_fault_latch.sv` + tb                                                                          | first-fault 锁存 + suppressed 计数                                                                                                                            | W6-03                                                                                               | W6    |
| `hardware/coral_nexus/rtl/w6/reset_endpoint.sv` + tb                                                                             | Tile 段 reset FSM + 取消/排空/复位/复位确认/credit 重建/RESUME                                                                                                | W6-03                                                                                               | W6    |
| `hardware/coral_nexus/rtl/w6/gate_ctl.sv` + tb                                                                                   | 分层时钟门控 + 唤醒存活性逻辑（WFI/IRQ/halt 语义）                                                                                                            | W6-04                                                                                               | W6    |
| `hardware/coral_nexus/rtl/w6/debug_fifo.sv` + tb                                                                                 | 小 FIFO + 溢出计数 + 导出/关闭                                                                                                                                | W6-04                                                                                               | W6    |
| `pipeline_validator/runtime/reset_domain.py`（行为适配，共享接线仍 W1）                                                          | RTL Tile 接入的复位驱动（Tile 段复位逻辑/证据上报）                                                                                                           | W1-04/05 消费                                                                                       | W6    |
| `pipeline_validator/tests/test_w6_event_facility.py` / `..._fault_reset.py`                                                      | 本包单元 pytest 入口【未来，随 RTL 落地新增；现有 `test_runtime.py` 的 TestEventSequence/TestFaultReset/TestCycleCap\* 是**当前可执行**基线，不是这些新文件】 | 事件表/等待/复位/取消/毒死门行为基线                                                                | W6    |

## 3. 任务依赖与完成条件

| 任务  | 主题                                                               | 硬前置                                                                                                                          | 并行窗口                                  | 共享合同（-01 产物）                                                                                                                                                                                    | 实施窗口    |
| ----- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| W6-01 | 本地事件/终态/复位接口与计数合同                                   | —（读 W6 设计 + `event_table.py` / `tile.py` `:569-992,1298-1324` + `reset_domain.py` / `fault_ring.py` / `ClockGate.sv` 即可） | 与 W1-03、W2-01、W3-01、W4-01、W5-01 并行 | 产出两份 **fragment**（`protocol/w6_event_facility.contract.yaml` + `protocol/w6_terminal_reset.contract.yaml`）交 W1-01（gen_protocol.py 读取、全部 contract 齐后 joint seal）；不与任何 -01 互等 DONE | CN-G1       |
| W6-02 | WAIT/WAITALL、SIGNAL_PHASE、可靠 TASK_COMPLETION/RETIRE_READY      | W6-01（合同片段冻结）；W1-03（C ABI）供联仿                                                                                     | 与 W2-04、W3-04、W4-04、W5-03/04 并行     | 取消/复位/first-fault 接口取自 W6-01 合同                                                                                                                                                               | CN-G2/CN-G3 |
| W6-03 | first-fault、取消/隔离、Tile 段复位端点、poison、Watchdog 行为适配 | W6-01；W6-02 的 RETIRE_READY/TASK_COMPLETION 先于本任务                                                                         | 与 W2-04、W4-04/05、W5-03 并行            | 契约取 W6-01/02                                                                                                                                                                                         | CN-G3/CN-G4 |
| W6-04 | 时钟门控、WFI/halt 语义、debug FIFO 与导出关闭排空                 | W6-03（first-fault、poison、关闭排空依赖）；W2 采纳 §4.4 CR-W6-W2-01/02                                                         | 与 W4-05、W5-03 并行                      | gate_ctl 对接 reset_endpoint（W6-01 合同）                                                                                                                                                              | CN-G4       |
| W6-05 | 竞争/故障/满/毒死/回绕/外写撤销 集成案例                           | W6-02 + W6-03 + W6-04；真实 CS1 场景走 W7-03 场景框架；one-Tile 真实 loop 取 W1-03/W1-04                                        | 最后                                      | 使用全部前续合同                                                                                                                                                                                        | CN-G5       |

**不引入实施循环**：W6-01 不依赖任何其他 -01 的 DONE；W6-02/03/04 通过共享合同（合同冻结后可读）相互引用，不互相 DONE-gate；W6-05 不以 W7-03 的 checker 全部落地为硬前置（可先在 W6 单元 `hardware/coral_nexus/verification/w6/` 跑）。**W1-05 4-Tile 最终集成不作为 W6-05 硬前置**：W6-05 以 W1-03/04 的 one-Tile 真实 loop 为证据（`--rtl-tile-mask 0x1`），4-Tile 汇总验证归 W1-05 与 W7-04 消费同一批场景文件；W6 不回头依赖 W2-05/W5-05。

**合同冻结门（CN-G1）**：W1-01 `gen_protocol.py` 读取本包两份 fragment 与其他包 fragment，**全部 contract 齐后 joint seal** 生成 `PROTOCOL_FREEZE.json`（新内容新 hash，**不可逐条追加**）；W6-02 起实施任务须先确认 `PROTOCOL_FREEZE.json` 存且涵盖本包两个 fragment hash（共享 freeze 产物，"齐了才 seal"，不互等别的 -01 DONE）。权威仍为 `hardware/coral_nexus/protocol/cn_protocol_v1.yaml` 等 W1-01 gen 产物；本包只写 fragment 文件。

## 4. 跨包共享合同（本包输出供其他包冻结的字段/状态/阶段 ACK）

以下"字段名"是**逻辑名，不是位编码**；位宽/编码由 W1-01 机器可读 ABI 冻结，本包不预设硅位宽（W6 设计 §6：`由后续规格冻结`）。

### 4.1 与 W1 的消息字段（EVENT/CTRL 通道）

| 消息                                                                      | 方向      | 字段（语义层）                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 备注                                                                                                                                             |
| ------------------------------------------------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `EVENT_NOTIFY(owner, generation, sequence, scope, status, error_code)`    | G→T       | 同 validator `event_table.py:400-455` 的 owner/generation/sequence 校验语义；Tile 侧收到后按 (ContextKey, local_event_id, event_generation, sequence) 查等待记录/事件槽                                                                                                                                                                                                                                                                                         | 完整键：物理 Tile ＋ ContextKey ＋ `local_event_id` ＋ `event_generation` ＋ `sequence`＋`scope`（§4.2）。外部组/L2 scope 事件才走 G→T           |
| `POST_EVENT(owner, generation, sequence, scope, status, error_code?)`     | T→G       | **已批准设计存在的逻辑接口（不 no-op、不静默禁用）**：语义仅允许"封印 manifest 显式授予该 Task 以 Group 事件 producer 身份"的 post——kernel 请求未授予 Group-scope 事件 producer 的 POST_EVENT 在 Loader/译码期拒绝，**禁止运行期冒写任意事件**；消息字段与 `EVENT_NOTIFY` 对称，走可靠通道，携带 owner/sequence/scope。若 CS1 语料不用它，它仍是合法接口而非禁用能力；启用用例由 W5-01 在封印 manifest 中显式授予                                               | W1-01 消费：该消息 owner/sequence 语义与 `EVENT_NOTIFY` 共用一套字段                                                                             |
| `PHASE_SIGNAL(TaskKey, phase∈{input_released,output_ready}, message_seq)` | T→G       | 物理物理physical Tile 仅是路由信息**不进聚合键**；一次 accepted SIGNAL_PHASE 发射→**一条 PHASE_SIGNAL**（一次指令→一次消息，逐次发射逐次发）；Software/Loader 已保证"每 Task 每方向至多一次"，**Tile 不新增 per-Task 状态抑制重复**——若 kernel 违反一次性声明（重复 SIGNAL_PHASE），重复逐条发出，由 Group 以 live duplicate 判协议故障（EVT-14；`_on_phase_signal` duplicate→协议故障计数恰 +1，`tile_group.py:1435-1445` 是 Group 侧权威，T-45 负例在此验证） | Tile 不做 per-(Context,phase,Task) 单次硬约束（该约束来自 manifest/编译），不做"第二次同方向才发"的 Tile 侧 gate                                 |
| `TERMINAL/RETIRE_READY/TASK_COMPLETION`                                   | G→T / T→G | `RETIRE_READY(ContextKey, TaskKey, drain_evidence_hash)`；`RETIRE`/`RESET_PHASE_REQ`（复位段）；`TASK_COMPLETION(TaskKey, status∈{DONE,ERROR,TIMEOUT,RESET}, first_fault_summary_ref)`                                                                                                                                                                                                                                                                          | 终态四类，无 success；状态只能前向；`TIMEOUT` 是合法 Task 终态之一，**但须在 W4 证实真实完成/隔离后才发布**（不是凭超时计时直接发布，见 §5.3.4） |

### 4.2 与 W2 的接口（本包请求，未写 CoreAxi）

见 §4.4 CR-W6-W2-01..03。核心语义：CoreAxi 删除 `!core.io.wfi` 全核关钟直连，替换为 W6 gate_ctl 的分层 enable（Context park / unit idle / Tile quiescence），**`io.wfi` 仅作为 Context park 提示**，不再进入 Tile 级关钟条件（INV-10、CLK-01/02）。

### 4.3 与 W3 的接口

- W3 的 `region_done` / 失败终态映射为本地事件 kind=`region_done`（MAT-15 链的终点；该 kind 是**本地事件表的一种完成语义**，不与 `OUTPUT_READY`/`STORE_DONE`/`TASK_COMPLETION` **互换**；W6 设计 §3.3 表格"产生条件（唯一来源）"仍适用，**不写成五类"互斥枚举"**——五类是不同语义，不是同一个事件的互斥状态）；
- W3 的 region 等待记录归 W3 唯一 writer；W6 不代写，只承接其 `region_done()` 事件与失败终态上报（MAT-16：失败终态唯一，绝不发布成功 region_done）。

### 4.4 需 W2 执行的 CoreAxi 实体改造请求（W6 不改、W2 执行）

**CR-W6-W2-01（CoreAxi WFI 语义变更）**：

- 上游证据（已核实，pin f795221…L69/L115-119/HDL ClockGate.sv:44-48）：`cg.io.enable := irq_reg \|\| ... \|\| (!csr.io.cg && !core.io.wfi) \|\| dm.io.haltreq(0)`，`io.wfi` 输出 `core.io.wfi`；
- 请求：删除 `!core.io.wfi` 全核关钟项；替换为 W6 gate_ctl 的 `tile_clk_en` 输入；`io.wfi` 改为**Context park 提示聚合**（各 Context `WAIT`/`WFI` 位）的**只读输出**，不参与关钟 enable；保留 `irq/timer_irq/software_irq/haltreq` 唤醒语义；
- 退出条件：改造后任一 Context park/WFI **不**导致其他 Context/在途操作/唤醒逻辑停钟（CLK-01/02/04/05、INV-10）；T-27 断言通过。

**CR-W6-W2-02（CoreAxi reset 域分级）**：first-fault 摘要/事件终态/可靠通道状态放 always-on 或 reset-safe 域；Core 级 `csr.io.reset/dm.ndmreset` 只触发 Core 复位；fault 摘要须在 Tile 级复位（RST-09）后仍可查，直至 Host/Group 确认（`trigger_fault`/FaultRing 语义已由 W1 验证）。**`csr.cg` 按 W2-01 计划草案建议降级为"每 Context park 投影输入之一"，聚合消费归 W6-04 gate_ctl；cg 编码归 W1-01（W2/W6 不自造）。**

### 4.5 与 W4 的共享接口（cancel 链、debug 导出执行腿）

- 取消链（FLT-02）：本包发 `TaskCancel(TaskKey)` → W4 按其 §4.4（终态三态 `COMPLETED|ISOLATED|FAULTED`）逐外写/事务结算，**不因超时把已接受外写当成功**；已接受外写不可凭 generation 或 timeout 撤销（CTX-16、FLT-04）。任何 `CANCEL_REQUESTED`→终态 的迁移**必须有 W4 侧终态/隔离证据**；超时只触发"强制取消"（RST-08），不改变"真实完成或隔离确认"的退出条件。
- 取消只关**该 Task** 的新 issue；Group reset（RST-07）是独立于 cancel 的全 run 排空，不因一个 CANCEL_TASK 自动启动——Group 是顺序权威，Tile 段被动执行（§5.3.2）。
- 隔离失败 ⇒ **保持未闭合**（不返回 ISOLATED 成功态）：W4 侧事务留 pending/隔离未确认，poison 成立；**不是**"ISOLATED 后释放资源"，而是资源保持占位直至真零证据。
- debug 导出（OBS-14/15/16/17）：W6 owns 小 FIFO/特权导出控制；**执行腿**（HBM/外存事务、地址范围检查、credit/优先级）在 W4；W6 只做采集/过滤/关闭/溢出计数，经 W4 通道发事务，受 W4 取消结算约束。

## 5. 逐任务：输入 / 产物 / 步骤 / 数据结构 / 验证 / 退出条件 / T 追踪

> 每条任务必须可**独立验证**：正例看期望行为，负例看错误行为被拒绝/计数，退出条件读"计数器+状态断言"，不读"文件存在"。
> 每条新命令：标注 **【当前可执行】**（现有 pipeline_validator CLI/pytest）或 **【未来，待 W1-02/W0-02 创建】**（`bash hardware/coral_nexus/scripts/cn.sh ...`）。

### 5.0 单周期提交顺序（W6 内部权威，跨包引用）

单 Cycle 内 W6 各状态按固定先后决策，形成**全 Tile 唯一线性化点**（EVT-10）：

```
1. 读上周期事件终态快照 + 采集本周期完成通知（completion_bus 入队）
2. WAIT/WAITALL check-register（原子、同拍完成优先，§5.2.1/§5.2.2）
3. 应用本周期引擎/MFE/VME/Group 完成通知（写事件表，终态 sticky）
4. 应用 UCE 故障/取消输入（first_fault latch、Task fault 化、停 issue）
5. 应用 Group CTRL 命令（CANCEL_TASK / RETIRE / RESET / POISON / FAULT_ACK）
6. 应用复位端点状态推进（仅被动执行 Group 置的 stage）
```

线性化点：步骤 1–3 在 Group 故障命令到达之前的决策，事件终态/已发布 OUTPUT_READY **不撤销**（EVT-06、FLT-04）；步骤 4 之后未终态的任务/事件以 ERROR/RESET 结束，**不发布成功**（FLT-14、MAT-16）。

---

### 5.1 W6-01 本地事件/终态/复位接口（合同与基础，CN-G1）

- **优先级**：最高（CN-G1；总计划 §2/§3 任务表）。**硬前置**：W6 设计文档 + 本节引用的 validator 行号 + 上游 `ClockGate.sv` pin（W0-01 锁定）。**并行窗口**：与 W1-03、W2-01、W3-01、W4-01、W5-01 共同进入合同冻结（joint seal）。
- **输入**：W6 设计 §3.1/§3.2/§3.4/§3.5/§4/§6；`event_table.py:11-567`、`tile.py:569-992,1298-1324`、`reset_domain.py:11-233`、`fault_ring.py:16-88`；`ClockGate.sv`（上游 pin）。**不做**：位编码预设、POST_EVENT 启用裁断。

**产物**（全部新建，落 `hardware/coral_nexus/protocol/` 与 `rtl/w6/`，非本包已有文件；与 W1-01 计划草案已对齐（全部 contract 齐后 joint seal））：

1. `protocol/w6_event_facility.contract.yaml`——含：
   - 事件**完整键**：`ContextKey=(物理 tile, context slot, context generation)` + `local_event_id`（Context 局部唯一, manifest 层映射 `ctx{N}:<name>`，不用于硬件匹配）+ `event_generation`（区分同一事件的使用周期）+ `sequence`（同一事件的使用周期）+ `scope∈{task-private,tile-shared,group(L2),external}` + `kind`（登记时确定、发布端不可改写）；
   - 事件条目逻辑字段与状态机 `PENDING→{DONE|ERROR|TIMEOUT|RESET}`（终态 sticky，不回 PENDING，EVT-16）、`producer_id`（登记时确定、唯一定序）、`error_code`、`consumer_refs`、`future_uses`（编译器证明的剩余未来引用）；
   - 回收谓词 `terminal ∧ consumer_refs==0 ∧ future_uses==0`（对齐 `event_table.py:498-502`）；
   - per-Context **frontier 槽数**（编译器 manifest 提供的 live frontier 上限）＋ Tile 全局槽数上限，满表只背压不丢（EVT-17）；
   - 计数器集（`stale_sequence / wrong_owner / duplicate_signal / capacity_reject / protocol_error / unknown_event_notify`），各计**精确递增条件**与饱和 sticky；
   - WAIT/WAITALL 操作数/记录字段（§5.2.1）与"同拍完成优先"合同；
   - `SIGNAL_PHASE` 发射语义（一次 accepted 指令→一条消息、Tile 不抑制重复）与事件 kind 语义表（`STORE_DONE/TASK_COMPLETION/VME_REGION_DONE` 为本地事件 kind；`INPUT_RELEASED/OUTPUT_READY` 是 PHASE_SIGNAL 消息而非本地事件 kind；五类是**不同语义、不互为替代**，判定条件按 W6 设计 §3.3 表"产生条件（唯一来源）"），并明确"不新增互斥判定要求"。
2. `protocol/w6_terminal_reset.contract.yaml`——含：
   - `first_fault_latch` 记录字段（code/source/severity/task/Context/command/event/program/descriptor/group/tile/queue/slot/patch/engine/offending_addr/aux0/aux1/cycle，FLT-11 全集；first-fault-wins 仲裁规则：同拍比较 source_id→engine_id→ctx_slot→op_seq 的直接字段确定序，不用哈希）；
   - 复位端点阶段与阶段次序（FAULT/RESET_CMD→FREEZE_ISSUE→CANCEL_UNISSUED→DRAIN→MARK_EVENTS→INVALIDATE_RESIDENCY→REBUILD_CREDIT→RESUME/DONE），`is_quiescent_snapshot` 证据条目集（engine accepted=0、engine 队列空、L1 MSHR 空、传输静止、无未决 debug 导出事务、无未发 PHASE/POST_EVENT；**DRAIN 判据不含 PENDING=0**——PENDING 由随后的 MARK_EVENTS 置 RESET，`reset_domain.py:77-102,167-219` 同源），`max_drain_cycles` 上限由 Python Group 独占（Tile 不自行计时超时，§5.3.6）；
   - Task 取消/终态映射：`CANCEL_REQUESTED`（W4 标记）→真实完成或隔离确认后结算为 `COMPLETED|ISOLATED|FAULTED`；**隔离失败时不返回 ISOLATED、不释放**，保持未闭合占位并 poison；Task 终态 ∈ `{DONE,ERROR,TIMEOUT,RESET}`，**`TIMEOUT` 是合法终态（须先有 W4 真实完成/安全隔离证据才发布）；排空判定永不因 timeout 提前成功**（RST-08/10）；
   - `CANCEL_TASK(TaskKey, generation)` 完整 key 匹配规则（**live-未知与 stale 严格区分**：live binding 无此 TaskKey→计数+幂等 ACK；TaskKey 已知但 ctx_generation 冲突→协议错误进故障/复位链，不任意丢）、旧代/重复命令幂等、失败隔离 ⇒ poison ⇒ 拒新启动直至显式静止 reset（RST-08）；
   - `PHASE_SIGNAL`（一次 accepted 指令→一条消息、Tile 不抑制重复）、`EVENT_NOTIFY`（全键）、`POST_EVENT`（**仅 manifest 显式授予 Group-scope producer 身份的 Task 可用**；无授权在 Loader/译码拒绝；不 no-op；字段与 EVENT_NOTIFY 对称、走可靠通道）的发送方行为、每 Context 前向序列；
   - `TASK_COMPLETION` 四态（DONE/ERROR/TIMEOUT/RESET）与 first-fault 引用、状态机（PENDING 不可跳过 RETIRE_READY）；
   - `GEN_WRAP_QUIESCE`（W1-01 新增消息）：**Tile 侧发起方（本包）向 Group 查询"ContextKey 级旧代消息（在途完成/迟到 ACK/事件队列/协议重放缓存）是否全部闭合"**；**响应方=Group（Python，W1-02/W1-04 驱动）**；**仅 Group 回答"全空"后，Tile 才被授权对位型（ctx_generation/event_generation/sequence）做受控复用**——仅 Tile 本地静止不构成复用授权（W6 设计 §3.1 禁止"位型完全相同的古老副本在途时复用"）；查询幂等、Tile 不超时、响应可验证（排空位图＋证据序号）。channel/message_seq/幂等键归 W1-01。

**机械步骤**（实现轮可勾选）：

1. 读 §2.1 引用行号；把 W6 设计 §3.1"条目回收谓词"逐字段比对 `event_table.py:_maybe_reclaim`，差异（如 `reset()` 只扫 PENDING）写入合同 YAML 注释而非直接覆盖 validator 行为。
   1a. 消费 W2-01 产物 `hardware/coral_nexus/rtl/scalar/doc/rob_tag.yaml` 的 **OwnerTag** 类型并在 `w6_terminal_reset.contract.yaml` 显式声明"first_fault 归属字段 (task, ctx, op) 与 OwnerTag 同源（引用而非复制）"，同时读 `doc/ctx_ctrl.yaml`（WAIT/WAITALL/SIGNAL_PHASE/RET/EXEC_DRAINED/park-request 的 op/operand/side-effect/顺序语义）作为 W6-02 的接口输入；二者字段语义若有出入，按 owner 任务 ID（W2-01）回最小证据请求（不引用聊天通道）。
2. 按本节逐字段列 YAML，先**逻辑名**，字段位宽与 ABI 编码**留给 W1-01**（合同 §W1-01"位宽/编码由 W1-01 根据最大在途/回绕证明产出"）；每字段标"输入来源:文件:行"。
3. 单独列 Tile→Group 的一组"计数器语义等式"（各计数器的递增条件 + 饱和 sticky 位）并验证每条可从 W6 设计 §8 或 validator 行号溯源。
4. **容量参数不是空待定**：写"推导脚本（未来）"入口 `hardware/coral_nexus/scripts/derive_w6_params.sh`（W6-01 内先产 Python 原型 `pipeline_validator/runtime/w6_param_derive.py`，输入=当前可运行的 `examples/*.ir` 编译产物 `budget.event_frontier`，输出=`per-Context frontier 槽` 建议值）——**这是实测/推证产物，非遗留待定项**；后续 W5-01 的 manifest frontier 若给出更大实证值，参数必须重派并重验。
5. **per-Context 事件槽数上界**（**运行期满表一律背压，绝不 fault**，对齐 EVT-17/W6 设计 §3.1 "分配失败＝该 Context 背压"）：
   - 实装合同（fragment 记）：
     a. **load/译码期拒绝**：manifest 声明的 per-Task "本地活跃事件数上界"（静态 launch/await 图的活跃界，见 c）**大于** per-Context 硬件槽数 ⇒ Loader/译码期拒绝该 Task（PERMANENT_CAPACITY 语义，`allocator.py:137-149` 同族），不留到运行期；
     b. **运行期满表只背压**：分配失败 ⇒ 该 Context 的 LAUNCH 在发射头停（StallReason 归 `WAIT_OPERAND`/`WAIT_EVENT`，W6 不新造枚举），**不 fault、不丢已完成终态、不先释放身份**；只要仍有 pending 槽在途，随完成即可释放；
     c. **上界的静态来源**：per-Context 上界取"该 kernel 静态 launch/await 图的**活跃界**"（每事件从"其 launch 可发射的最晚程序点"到"其最后一个 await 消费点"的活跃区间峰值，编译器可证明）——**当前 validator 的 `budget.event_frontier`/`event_records` 峰值只作 Group 级与辅助证据**，不是 per-Context 上界的静态证明；W5-01 manifest 承载该 per-kernel 静态活跃界，W6-01 步骤 7 合入；
     d. 零完成通道阻塞：满表不阻塞完成通知/信用回收（EVT-17），作为 U1-01-f 负例判据。
6. **WAITALL 操作数上限**：不空定。取（`lowering.py:966-970` 产出的 WAITALL `len(ins.args)` 峰值）与 W5-01 声明值取 max；超过即译码期拒绝（合同 §4.1 第 3 行）；运行期不判 fault（操作数超限在译码已拦）。
7. 与 W1-01 合入机器可读 ABI；W1-01 未完成前，本任务输出先于任何 W6-02 以上的实施进行（防互等循环：W6-01 不依赖其他 -01 的 DONE，可先发合同草案片段，主计划统一冻结）。

**正/负验证（独立可跑，退出条件 = 全部断言 PASS）**：

| 用例    | 类型         | 输入                                                                                                                                                  | 断言（状态/计数）                                                                                                                                                  | 当前命令【当前可执行】                                                                                                                                    |
| ------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| U1-01-a | 正           | 现 `pipeline_validator` 以 full_memory 跑一个含 WAIT/WAITALL 的多 Context 程序                                                                        | `result.pmu` 中 `task_retirement_wait>0` 且 `completed=True`                                                                                                       | `pytest -n0 pipeline_validator/tests/test_transformer_multicontext.py`（先 `export PYTEST_ADDOPTS="-n0"`）                                                |
| U1-01-b | 正+负        | `test_runtime.py` TestEventSequence（`test_stale_sequence_rejected` / `test_reset_marks_pending_reset` / `test_error_status_not_treated_as_success`） | stale→计数、reset 只改 PENDING、ERROR 不冒成功                                                                                                                     | `pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestEventSequence"`                                                                              |
| U1-01-c | 正+负        | TestFaultReset（`trigger_fault`、`drain`、隔离确认）                                                                                                  | 排空成功判据=真零计数，非超时；隔离失败→poison 语义链                                                                                                              | `pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestFaultReset"`                                                                                 |
| U1-01-d | 负（合同级） | 构造：`EventTable.reset()` 的终态保持性                                                                                                               | 终态条目 reset 后状态**不变**                                                                                                                                      | 同上                                                                                                                                                      |
| U1-01-e | 负           | 超时语义：`test_runtime.py` TestCycleCapPoison / TestCycleCapDrain                                                                                    | `max_cycles` 触发的 drain **不计为 completed**                                                                                                                     | `pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestCycleCap"`                                                                                   |
| U1-01-f | 负           | frontier 满表背压（运行期，非 fault）                                                                                                                 | per-Context 槽全 pending ⇒ 新 LAUNCH 停发射头（StallReason=WAIT_OPERAND/WAIT_EVENT）；不 fault 不丢已完成终态；完成通知/信用回收不被阻塞；一个槽完成后分配立即恢复 | 【未来】单元（满槽注入用测试专用参数驱动；场景源：`hardware/coral_nexus/scenarios/w6/u1-01f_full_table_backpressure.yaml`）；【未来】联仿同 U2-a 命令形式 |

**退出条件**：两份 fragment 经 W1-01 joint seal 进 `PROTOCOL_FREEZE.json`；`U1-01-a…f` 全 PASS（现有命令真实可运行 + 满槽负例有构造）；若 `derive_w6_params.py` 原型给出参数建议值，一并记录在 fragment 附注（值待 W5-01 manifest 落地后重验，不空挂）。

**T 追踪**：T-13/EVT-01~05/16/17、T-45、T-51、T-52 前置、EVT-15（Tile 侧不做运行期检查，前置检查归 Loader/编译器，本计划 §5.2 第 6 条）、`max_drain_cycles` 归 Python Group（§5.3.6）。

---

### 5.2 W6-02 WAIT/WAITALL、SIGNAL_PHASE、可靠终态

- **优先级**：P0（总计划 CN-G2）。**硬前置**：W6-01 合同冻结；W1-03 C ABI（联仿阶段）；W2-01 ContextKey/op_seq 身份定义（总计划 §2/§3 引用，不待 DONE）。
- **输入**：W6 设计 §3.2/§3.3/§5.5、§4 表；`tile.py:978-992`（\_issue_wait）、`:569-588`（probe）、`:1298-1318`（complete）、`fault_ring.py`、本计划 fragment YAML。

**产物**：`rtl/w6/wait_record.sv`（+ tb）、`rtl/w6/completion_bus.sv`（+ tb）、`rtl/w6/terminal_store.sv`（+ tb，可靠 TASK_COMPLETION/RETIRE_READY 保存/发送）、`runtime/event_table.py` 行为注释（若 W6-01 已写合同，无需改 Python 源）。

**数据结构 / 机械步骤**：

1. **Wait record（per-Context 唯一，等待记录唯一 writer=本模块）**：
   - 字段：`ContextKey`（slot+generation）、`kind∈{WAIT,WAITALL}`、每操作数 `local_event_id / event_generation / sequence / scope`（完整键 CAM，不含 `ctx{N}:<name>` 串）、armed 位；
   - **原子 check-register**：发射提交点（P1/P3 对应 Tick 内）一次性写入；check 读的是**事件表持久终态**，非单周期脉冲（EVT-02）；armed 位只整记录裁决后清（EVT-04）。
2. **WAIT/WAITALL 语义（无丢唤醒）**：
   - 事件先到 → check 直接命中终态不 park（`tile.py:978-992` 同口径）；
   - 同拍到：check 与 register 同一提交点，**完成广播优先**——命中则直接判决（DONE/ERROR/RESET/TIMEOUT），不 park 不出现"下一拍才唤醒"的歧义（W6 设计 §3.2 优先级固定，本包默认**不切换为 park+wake 同拍**，若 W2 提出等价实现须走合同 §4.5 报告）；
   - park 后到：完成广播按**完整键** CAM 匹配（ContextKey+slot+generation+event_generation+sequence+scope），**逐操作数命中记录终态**；WAITALL 全 DONE 才成功唤醒，任一操作数失败终态走失败出口（不等其余，§3.2）；
3. **失败出口（区别于 validator 基线）**：本包定义 WAITALL 或 WAIT 的操作数命中 `ERROR/RESET` 终态时 → 等待者**不永久等待**、Context 进入其故障/取消/复位既定状态，事件按 EVT-05 结算——**不伪装成功唤醒**；`RESET` 终态命中（复位 MARK_EVENTS）等待者按失败出口结算并停止（W6 设计 §3.2 末段 + validator `reset_domain.py:98-102`）。`TIMEOUT` 终态若出现（Group 侧注入/外部事件），W6 不引入"等待者自行计时"路径：TIMEOUT 命中等待记录按**失败出口结算并停止与 ERROR 同路径**；但**排空/复位完成的判据永不包括 timeout 本身**（`TIMEOUT` 只能作为一条 Task 终态在 W4 隔离/完成证据后发布，RST-08）。
4. **SIGNAL_PHASE**：发射点同步产生，携带完整 TaskKey；**一次 accepted SIGNAL_PHASE 指令→一条 PHASE_SIGNAL 消息（逐次发射逐次发）**；"每 Task 每方向至多一次"是**软件/Loader 静态保证**（EVT-15 前置在编译/装载核查），**Tile 不新增 per-(Context,phase,Task) 单次状态/抑制门**——若 kernel 违反一次性声明（重复 SIGNAL_PHASE 指令），重复逐条发出，由 **Group** 以 live duplicate 判协议故障（EVT-14；`_on_phase_signal` `tile_group.py:1435-1445` duplicate→协议故障计数恰 +1 是 Group 权威语义，T-45 负例在 Group 侧验证）。Tile 侧只保证"指令→消息"恒等映射（不合并、不丢、不伪造）。
5. **可靠终态（TASK_COMPLETION / RETIRE_READY）**：
   - `retire_ready(ContextKey, TaskKey)` 以 drain_evidence 预留（§4.2 上报），**非成功终态**（FLT-14；`W2 EXEC_DRAINED` 是内部通知，非对外成功）；
   - 本地资源归还（view/Arena/Frame）在 `RETIRE` 授权下由 W4 执行，**本包只保存/发送 TASK_COMPLETION**（W6 设计 §5.5）；
   - 终态状态集合 `{DONE,ERROR,TIMEOUT,RESET}`（无 success，`device.py:23-27` 的 SUCCESS 语义经 W1-01 枚举映射替换，不在本包重命名）；**`TIMEOUT` 是合法终态之一**，但须以 W4 证实"真实完成或安全隔离"为前提才发布（不是凭本地计时器直接发布）；
   - **背压**：通道满 → 稳定保存（身份保留，先不释放发送游标，EVT-09/CTX-11），不丢、不变形（T-23 语义）。本包只写"保存端不变、状态可查、无丢失计数器"，不做"成功即释放"。
6. **INPUT_RELEASED 的早释放路径**（总体设计 §4.4.1；合同/总体设计 L2 早 release 用户裁定）：
   - `INPUT_RELEASED/OUTPUT_READY` 是 **PHASE_SIGNAL 消息语义，不是本地事件 kind**——本包不把二者登记为 Tile 本地事件表条目，也不给它们分配 local_event_id（W6 设计 §3.3 表"产生条件（唯一来源）"，**五类完成是不同语义，不是互斥枚举**）；
   - Tile 侧发射点只判定"`SIGNAL_PHASE(input_released)` 指令本身是否被 accepted 并产生一条 PHASE_SIGNAL"（编译期"全部 L2 loads 已 await、其后无 L2 load"的静态前置由 Loader/编译器保证，**本包不运行期复核**；Group `_unpin_grid_readers` 仅在 `all_tasks` 收齐后调用，`tile_group.py:1446-1453`）；
   - 早释放**不归还 L1/Frame/UCE/R、不递减程序驻留引用**（本包不发任何本地资源归还信号）；已早解 pin 退休不重复 unpin（无重复发送路径）；
   - 与退休链解耦：`input_released`→`PHASE_SIGNAL` 发射后 Context 仍在跑，直到 W2 `EXEC_DRAINED`→本包 RETIRE_READY→Group `RETIRE`→本地资源归还→`TASK_COMPLETION`（W6 设计 §5.5、总体设计 §4.4.2）。
7. **`WAIT`→事件表写入 vs 完成通知的提交窗口**：`wait_record.sv` check 读表状态与 register 同拍；`completion_bus.sv` 在该拍收到的完成通知进入**"同拍完成集"**，既写入表（终态）又直接命中 check（无丢唤醒）；两写口（表写、armed 记录写）的相对次序在本模块内以"完成集→check"单点原子化。

**消费者**：W2（等待上下文状态）、Group `_on_phase_signal`（聚合）、W4（TaskCancel → 终态）、W7（计数器消费）。

**正/负验证**：

| 用例 | 类型  | 输入/场景                                                                                                    | 断言（状态/计数/字节）                                                                                                                                                   | 命令                                                                                                                                                                                                                                                                                                                                     |
| ---- | ----- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| U2-a | 正    | WAIT 先到：事件先于 WAIT 终态，WAIT 不 park（Context 立即 eligible）                                         | 表终态命中、无 armed 记录、zero wake 延迟                                                                                                                                | 单元【未来】`hardware/coral_nexus/rtl/w6/` tb（Verilator，驱动与内部状态可见口）；联仿【未来】`bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/scenarios/w6/u2a_wait_before_event.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>` |
| U2-b | 正    | 同拍：完成与 WAIT 同拍到达，check 即命中                                                                     | `wake==1` 当拍出现，不延迟，不双唤醒                                                                                                                                     | 同上（u2b_same_cycle.yaml）                                                                                                                                                                                                                                                                                                              |
| U2-c | 正    | 后到：armed 记录 + 完成广播完整键匹配                                                                        | 唤醒延迟有界（≤1 cycle，TIM-03），逐操作数记录                                                                                                                           | 同上（u2c_after.yaml）                                                                                                                                                                                                                                                                                                                   |
| U2-d | 正+负 | WAITALL：A 完成、B 未完成 → 记录保持 armed；B 完成 → 恰一次唤醒；任一操作数 ERROR → **失败出口**，不成功唤醒 | 唤醒恰一次、失败出口成立、无残留 armed                                                                                                                                   | 同上（u2d_waitall.yaml）                                                                                                                                                                                                                                                                                                                 |
| U2-e | 负    | 完整键负例：同 owner/gen/seq 的**无关事件 C**（不同 local_event_id / 错 scope）命中 A 的操作数 → **不命中**  | stale/错 owner/错 scope/错 sequence 分别计数且状态不变（`event_table.py:441-448`）                                                                                       | 同上（u2e_key_mismatch.yaml）                                                                                                                                                                                                                                                                                                            |
| U2-f | 负    | 重复 SIGNAL_PHASE 指令（违反软件一次性声明）                                                                 | Tile 每条 accepted 指令各发一条 PHASE_SIGNAL（逐条）——**负例的判读在 Group 侧：`tile_signal_duplicate` 协议故障计数恰 +n-1（`tile_group.py:1435-1445`），Tile 无抑制门** | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_runtime.py -k "tile_signal_duplicate"`（已核实存在：`test_runtime.py:1883 test_duplicate_signal_faults_without_advancing_phase`）                                                                                                                                                |
| U2-g | 正    | PHASE_SIGNAL 每方向只发一次（合法 kernel）                                                                   | 每 (Task, phase) 恰 1 条消息、`phase_aggregate` 只触发一次（`test_transformer_multicontext.py:129-144` 已断言 phase_aggregate 与 `_out`/`inrel` 的偏序）                 | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_transformer_multicontext.py`（**已核实存在该覆盖**：`phase_aggregate` 事件与 inrel/out 偏序检查）                                                                                                                                                                                |
| U2-h | 正    | TASK_COMPLETION 通道满：保存不丢、身份稳定、终态唯一                                                         | 状态寄存器/记录字节不变、无丢失                                                                                                                                          | 单元【未来】（u2h_backpressure.yaml）                                                                                                                                                                                                                                                                                                    |
| U2-i | 负    | RETIRE_READY 先发后取消/复位介入                                                                             | 状态机次序不倒退、不重复结算、不发布成功终态                                                                                                                             | 同上（u2i_retire_vs_cancel.yaml）                                                                                                                                                                                                                                                                                                        |

**退出条件**：U2-a…i PASS；`wait_record.sv` / `terminal_store.sv` 通过单元；`cn.sh unit --suite w6`（【未来】）注册的条目全绿；T-13 全断言通过。one-Tile 真实 loop（`cn.sh cosim --rtl-tile-mask 0x1`）覆盖 RET→task_retirement_wait→EXEC_DRAINED→TASK_COMPLETION 全链一例（【未来】），不要求 4-Tile mask（W1-05 汇合）。

**T 追踪**：T-13（wait/事件/错误语义）、T-13 边界（同拍/回绕/无丢）、T-45（Tile"指令→消息"恒等 + Group 侧 duplicate/漏发判读）、T-23 保存端、T-14 早释放 Tile 侧行为、T-27（park→唤醒存活）。

---

### 5.3 W6-03 first-fault、取消/隔离、Tile 段复位端点、poison

- **优先级**：P0。**硬前置**：W6-01 合同、W6-02 产物。
- **输入**：W6 设计 §3.4/§3.5/§5.3/§5.5、`reset_domain.py:11-233`、`fault_ring.py:16-115`、`tile_group.py:2364-2375,2354-2360,1992-1993,2108-2110,2235-2267`、需求号（RST-07/08/09、FLT-01/02/02a、MAT-16、INV-15、CTX-16、EVT-16）。

**产物**：`rtl/w6/first_fault_latch.sv`（+ tb）、`rtl/w6/reset_endpoint.sv`（+ tb）、`runtime/reset_domain.py` 行为适配（仅加"Tile 段执行/证据上报"，**不重命名/不改现有字段**；共享接线仍 W1-04）。

**数据结构 / 关键决策（已选方案，本包默认固化）**：

1. **first-fault latch（每 Tile 一个，always-on/reset-safe，FLT-10）**：
   - 触发条件：复位**未激活且未完成**（对齐 validator `reset_domain.is_active or is_done` 抑制，`tile_group.py:2354-2360,1431-1432`）；
   - 仲裁（**本包定死，覆盖 validator "FaultRing 覆盖写但不作为 first-fault-wins"的历史事实**）：first-fault-wins（首个进入 latch，后续只计数到 `suppressed_fault_count`）；同拍多 fault 仲裁按**直接字段确定序**：`source_id`→`engine_id`→`ctx_slot`→`op_seq`（逐字段字典序；**不用指令身份哈希**——避免哈希冲突与无谓硬件计算，结果确定性可复现），`w6_terminal_reset.contract.yaml` 给出该表；
   - 保留至 Host 确认（FLT-11；经 CTRL 端 `fault_register` 读 + 显式 clear，RST-09）。**复位不清**；清除只走显式 `FAULT_ACK`/特权 `fault_clear` 路径。

2. **Tile 段复位端点 FSM（W6 设计 §3.5、RST-07 固定次序；Tile 是本地事件/本地资源段唯一 writer）**：
   - 状态次序与 W6 设计 §3.5 一致：`FAULT/RESET_CMD→FREEZE_ISSUE→CANCEL_UNISSUED→DRAIN→MARK_EVENTS→INVALIDATE_RESIDENCY→REBUILD_CREDIT→RESUME/DONE`；
   - **Group 是顺序与阶段推进唯一权威**；Tile 段对 Group 的阶段命令**被动执行并 ACK**，不自动推进状态；`RESUME` 由 **Group** 决定。本端点不自作主张触发"其他 Context 继续"分支（FLT-02a 不实现）；
   - 阶段可观察点对齐 `reset_domain.py:11-49`（`ResetState` 枚举与阶段推进一一映射，供 W7 联仿对拍）。
   - **fault→domain 映射（已核实，勿删真实事实）**：`trigger_fault`（`tile_group.py:2371`）函数本身保留"`tile_id>=0 → FaultDomain.TILE，否则 GROUP`"的分支；**生产故障路径** `on_scheduler_fault`（`tile_group.py:2354-2360`）不传 tile_id ⇒ 落 `GROUP`；`_on_phase_signal` 的 `trigger_fault(ADDRESS_FAULT, tile_id=-1)`（`:1432/1444/1460`）同样落 GROUP。CS1 生产故障域仍 = GROUP 全域；**"函数支持 TILE 参数"不等于"生产路径局部恢复"**，FLT-02a 仍不实现，TILE 域仍是测试/注入用途（RST-01）。RTL 侧 FAULT_REPORT 的 tile_id 字段仅用于**归因记录/诊断**，不得改变 Group 复位域选择。

3. **DRAIN 证据（is_quiescent_snapshot）——只门控"真实义务闭合"，不门控"PENDING 事件"**：
   - 字段＝各引擎 accepted 计数、engine 队列空、L1 MSHR 空、传输静止（W4 侧闭包静止）、无未决 debug 导出事务、无未发（未 accepted）PHASE_SIGNAL/POST_EVENT；
   - **PENDING 事件不进 DRAIN 判据**：未发/取消后的 PENDING 本地事件由后续 `MARK_EVENTS` 置 RESET（W6 设计 §3.5 第 4 阶段语义，`reset_domain.py:98-102` 同源）；把 PENDING=0 写进 DRAIN 判据会造成"等所有事件先终态才 DRAIN，而终态又来自 MARK_EVENTS"的**死锁**——本合同明确**排除**该写法；
   - **"outstanding 真零"才完成 DRAIN**，超时只触发 Group 侧"强制取消"升级（RST-08），**超时本身永不判排空成功**。

4. **MARK_EVENTS（Tile 本地事件表是本地段唯一 writer，W6 实改表并 ACK）**：
   - **顺序**：Group 下发 `RESET_PHASE_REQ(MARK_EVENTS)` → Tile 实际**扫描并写自己的本地事件表**（PENDING→RESET；已终态保持）→ ACK(`MARK_EVENTS_DONE`, 含 PENDING→RESET 计数) → Group 才推进 `RESET_DOMAIN/INVALIDATE_RESIDENCY`（对齐 `reset_domain.py:98-124` 阶段序）；
   - **Group 只写它自己的 Group 侧事件表**（`event_table.py:548-553` 不变）；**Tile 本地表由 Tile 写**（本包 `reset_endpoint`），不是"Tile 只上报 PENDING=0 证据、由 Group 写"——那是双 owner 混淆，合同明确**否决**该写法；
   - Tile **不**自作主张把事件标 ERROR（基线无该路径，W6 设计 §3.4；`MARK_EVENTS` 语义固定 PENDING→RESET）。

5. **取消/隔离（RST-08、CTX-16、FLT-04、FLT-05）**：
   - `CANCEL_TASK(TaskKey, generation)` 只停**该 Task 所属 Context**的新发射（§3.5）；**"live 但未知"与"stale"严格区分**：TaskKey/ContextKey 与当前任何绑定**均不匹配→stale，计数并幂等 ACK**；TaskKey 存在但 `ctx_generation`/TaskKey 序号与 live Task 冲突→**协议错误**（按 RST-08 语义进入对应故障/复位链，不能任意丢；W6 设计 §3.5 "旧代命令忽略并计数"仅指已退休/旧 generation，不包含"未知的 live cancel"）；
   - 已接受外写**不可撤销**（CTX-16）：cancel 不触发"外写回滚"；W4 逐外写结算为 `COMPLETED|ISOLATED|FAULTED` 三态之一（本计划 §4.5）；**隔离失败 ⇒ 不返回 ISOLATED**，占位直至真零证据（§4.5 修正）；
   - Task 终态集合 `{DONE,ERROR,TIMEOUT,RESET}`：**`TIMEOUT` 是合法 Task 终态（可发布），但必须先有 W4 证实真实完成或安全隔离的终态证据**（不是凭 Tile/Group 计时器直接发布）；排空判定**不因 timeout 提前成功**（RST-08）；
   - `CANCEL_REQUESTED`（W4 §4.4 标记）只在真实完成或隔离确认后才可结算为 `COMPLETED|ISOLATED|FAULTED`，不存在"超时→成功结算"捷径；
   - **隔离失败 ⇒ poison**：poison 状态由 **Group 独占拥有**（`tile_group.py:1992-1993,2108-2110`），Tile 端点收到 POISON 后拒绝一切新启动直至"显式静止 + 证据清零 + 显式恢复"（对齐 `reset_domain.py:2235-2247`）。

6. **Watchdog 归 Group（Tile 不设时钟）**：`max_drain_cycles`、`max_cycles`、`profile_command_timeout_cycles` 全部由 Python Group 持有（§2.1 引用），Tile 端点不计时、无"自判超时"路径（本包合同冻结"Tile 不含任何可改变功能状态的计时器"）。`HardwareConfig` 当前无 `max_drain_cycles` 键，读值来自 reset_domain.py:89 的 getattr 默认 100——本包不改该值，作为既有事实写进合同附注。

7. **`VME_REGION_DONE` 路径**：W3 上报失败终态（`FAILED`），本包映射为 `kind=region_done, status∈{ERROR,RESET}`（**绝不发布成功 region_done**，MAT-16），不产生本地 `OUTPUT_READY`（W6 设计 §3.3 表"产生条件（唯一来源）"，**五类是不同语义、不是互斥枚举**）。

**正/负验证**：

| 用例 | 类型  | 场景                                                                                                                                     | 断言（状态/计数/字节）                                                                                                                                                                                | 命令                                                                                                                                                                                                                                                                                                                                                           |
| ---- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| U3-a | 负    | 同拍注入两个不同源故障                                                                                                                   | first-fault 保存**源序确定序最高优先级**的那个，另一进入 `suppressed_fault_count`，确定性可复现（同 input → 同 state/字节）                                                                           | 【未来】单元（场景文件：`hardware/coral_nexus/scenarios/w6/u3a_multi_fault.yaml`，W7 schema 冻结前作为素材）；联仿【未来】`bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/scenarios/w6/u3a_multi_fault.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>` |
| U3-b | 正    | fault 触发 → FREEZE_ISSUE → CANCEL_UNISSUED → DRAIN → MARK_EVENTS(Tile 实改本地表) → INVALIDATE_RESIDENCY → REBUILD_CREDIT → RESUME/DONE | 每阶段一个可观察信号，与 `ResetState` 一一对应；阶段次序不可乱；MARK_EVENTS ACK 带 PENDING→RESET 计数                                                                                                 | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestFaultReset or TestCycleCap or TestSuccessExitClosure"` + 【未来】cn.sh cosim（u3b_full_reset_drain.yaml，全链 one-Tile RTL）                                                                                                                                                        |
| U3-c | 负+正 | DRAIN 判据 = 真零计数（**PENDING 事件不等**：有 PENDING 事件但无未决义务时 DRAIN 判据仍可为真，PENDING 由后续 MARK_EVENTS 清）           | 任一在途（HBM outstanding / stage busy / NoC 非静止 / L1-L2 MSHR / 未决 debug 导出）→ 不上报 DONE                                                                                                     | 同上（u3c_drain_pending_ok.yaml + u3c2_drain_outstanding.yaml 两场景）                                                                                                                                                                                                                                                                                         |
| U3-d | 正+负 | `CANCEL_TASK` 旧代/stale vs live-未知                                                                                                    | 完整 TaskKey+generation 不匹配（旧代/退休 Task）→ stale 计数、幂等 ACK、无终态误发；TaskKey 存在但 ctx_generation 冲突 → 协议错误（RST-08 语义），不能任意丢                                          | 【未来】单元 + 【未来】`cn.sh cosim --scenario hardware/coral_nexus/scenarios/w6/u3d_cancel_stale_vs_live.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>`                                                                                                                             |
| U3-e | 负    | 已接受外写在 `CANCEL_REQUESTED` 后的终态路径                                                                                             | W4 侧三终态 `COMPLETED/ISOLATED/FAULTED` 证据到达前，`CANCEL_REQUESTED` 只能停在 pending 计数；**超时强制取消不产生"成功外写"伪终态**；隔离失败 ⇒ 资源保持未闭合 + poison                             | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestCycleCap or TestSuccessExitClosure"`（外写结算语义同源）+ 【未来】单元（u3e_iso_pending.yaml）                                                                                                                                                                                      |
| U3-f | 正    | 隔离失败 → poison                                                                                                                        | `poisoned_reason` 非空；新启动/准入全拒（`tile_group.py:1992-1993,2108-2110`）；恢复仅"显式静止 + 证据清零 + 显式 reset()"                                                                            | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestCycleCapPoison"`；【未来】`cn.sh cosim --scenario hardware/coral_nexus/scenarios/w6/u3f_poison_then_reset.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>`                                                  |
| U3-g | 正    | `FaultRing` 覆盖写（模型行为）                                                                                                           | **合同只采其"计数参考"**；RTL latch 语义覆盖此行为                                                                                                                                                    | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestFaultReset"`                                                                                                                                                                                                                                                                        |
| U3-h | 负+正 | VME region 失败                                                                                                                          | 绝不发布成功 `region_done`；失败终态唯一且不回退                                                                                                                                                      | 【未来】单元（W3 场景侧归 W3；文件：`hardware/coral_nexus/scenarios/w6/u3h_region_fail.yaml`）；联仿命令同 U3-a 形式                                                                                                                                                                                                                                           |
| U3-i | 负    | fault-injection 预期"安全失败"路径                                                                                                       | 注入一个 fault ⇒ fault_record_idx≥0、`reset_domain` 走到 `DONE`、Task 终态唯一非成功、零泄漏——**注入场景本身 PASS**（注入是受控行为，不是 checker 应 fail 的缺陷；缺陷 mutation 才导致 checker fail） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_runtime.py -k "TestFaultReset"`；【未来】单元（u3i_fault_injection_safe_expected.yaml）                                                                                                                                                                                                                |

**退出条件**：U3-a…i 通过（含 U3-i "fault-injection 预期安全失败"是 PASS-case 而非 checker-fail case）；`first_fault_latch/reset_endpoint` 单元通过；`is_quiescent_snapshot` 字段被 W1（Python 侧驱动）消费验证；MARK_EVENTS Tile 侧实改（PENDING→RESET 计数进 ACK）作为 U3-b 一部分验证。

**T 追踪**：T-25、T-26、T-52、FLT-01/02/02a、RST-07/08/09/10、CTX-16、INV-15（外写保留）、MAT-16。

---

### 5.4 W6-04 时钟门控、WFI/halt 语义、debug FIFO 与导出关闭排空

- **优先级**：P0（总计划 CN-G4）。**硬前置**：W6-01/02/03 合同与产物；W2 采纳 §4.4 CR-W6-W2-01/02。
- **输入**：W6 设计 §3.6、§4"门控控制（W6→全 Tile 单元）"、§5.4、上游 `CoreAxi.scala`（pin）+ `ClockGate.sv`（pin）、`tile.py:300-313`、总计划 §已选方案 与 W2 设计 CFG-06（Tile-global 特权清单：profile 切换、reset、program invalidate、debug 控制属管理侧，普通 Context 禁止）。

**产物**：`rtl/w6/gate_ctl.sv`（+ tb）、`rtl/w6/debug_fifo.sv`（+ tb）、合同 YAML 中追加 debug 导出/门控字段（`w6_terminal_reset.contract.yaml` 更新 hash）。

**数据结构 / 关键决策**：

1. **分层门控（CLK-01/02/03、INV-10）**：
   - `Context park`：该 Context WAIT/WFI/停靠 → 只允许关该 Context front-end 取指/译码时钟，**排除 RF/VRF 回写端口**（CLK-06），**不影响在途操作/其他 Context/唤醒逻辑**；
   - `unit idle`：engine/LSU/队列粒度，由各自 accepted/occupied 计数驱动（W3/W4 提供状态）；**MFE/RVV idle 计数不参与"Context park"门**（避免误关）；
   - `Tile quiescence`：全部 Context park ∧ 在途表全空 ∧ 完成通道无待处理项（CLK-03，严格条件）。
2. **CoreAxi 改造请求**：见 §4.4 CR-W6-W2-01/02——`io.wfi` 不再直连关钟，Tile 级 enable 由本模块给出，`haltreq`/`irq/timer/software_irq` 保留唤醒语义；`csr.cg` 不再单 Context 决定全核。
3. **唤醒存活性（CLK-04/05、TIM-03）**：
   - 必要响应/事件接收/唤醒逻辑在关钟期间可服务或可靠恢复：门控边界**排除同拍新到达**——"门控判定→时钟真正关断之间到达的完成"按**未门控**处理（不丢唤醒、不漏写回、不重复完成，T-27/T-36 判定）；
   - 复位/取消命令强制解除目标单元门控（`force_wake` 端口）：freeze/cancel 命令到达时不依赖被门控单元已醒。
4. **debug FIFO 与特权导出（OBS-14/15/16/17、FLT-13、T-51/T-67）**：
   - 正常模式默认关：无片上历史 memory、无 HBM 日志 ring，仅保留 PMU 计数/first-fault 摘要/溢出 sticky（OBS-14）；
   - 满则**丢新记录**（丢弃不阻塞 issue/state_free/唤醒/终态/故障处理，OBS-15）＋ `dropped_count` 饱和计数＋`窗口不完整` sticky；
   - **深度不预设工程默认值**（不再写"默认 8"）：实际深度由三项实证冻结——(a) 记录带宽/突发分析（正常模式满速导出路径单拍产率 × 导出延迟）(b) 终态与导出的隔离证明（FIFO 深度不影响终态/唤醒/`state_free`，深度=1 边界测试 U4-d1 覆盖）(c) W6-01 `derive_w6_params.sh` 脚本按 cs1 配置的导出预算推证后冻结（进 configs 的 `cs1*\*.yaml` 与 target manifest；`由后续规格冻结` 按 W6 设计 §6 语义，即"冻结需要证据，而非无限期空挂"）；
   - **特权导出**（默认关）：`DEBUG_ENABLE/限额/捕获配置` 走 CTRL 特权清单（CFG-06），目标 ring/范围/权限检查在 W4 执行（§4.5），经 W4 通道发事务；满/写失败→停采/丢诊断，不扩张片上缓冲（OBS-16）；
   - **关闭/取消/复位排空（OBS-17）**：先停新采集、日志请求；已接受 HBM 日志写保留 owner/addr/credit，**须真实完成或隔离确认**（W4 终态三态），排空/隔离确认之前状态可查询为 `pending`/`poison`，**不能只清 FIFO 冒充完成**、"超时不清空"。
5. **debug halt（RST-04/06/11）**：`halt` 停在指令边界/launch 边界，不破坏信用与 descriptor 一致性（`tile.py:300-313` 冻结后不新发、完成照常处理的同口径）；RST-11 的"停点"语义由本包与 W2（Core）共同裁定，本包只承担"门控/事件/复位路径在 halted 期间不改变"。

**正/负验证**（本包独立可跑部分）：

| 用例  | 类型 | 场景                                                         | 断言                                                                                                                                                        | 命令                                                      |
| ----- | ---- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| U4-a  | 正   | 单 Context park（WAIT）+ 其他 Context 正常 issue             | Tile 整钟未关（`tile_clk_en==1`），`ctxN_front_en==0` 仅该 Context；在途写回正常处理                                                                        | 【未来】单元 + 【当前可执行】等价 pytest 集成场景留 W7-04 |
| U4-b  | 负   | WFI 单 Context 置位后立即有其他 Context 发射                 | 全核（Tile）关钟**不发生**；不依赖 `!core.io.wfi` 项                                                                                                        | 【未来】单元                                              |
| U4-c  | 正   | 全 Context park + 后端在途 + 门控边界到达事件                | CLK-05：唤醒完成无丢失/重复；gated 周期计入统计（PWR-04/05）                                                                                                | 【未来】单元                                              |
| U4-d1 | 负   | debug FIFO 深=1 注入（连续两条记录，第一条占用、第二条必撞） | 第 2 条被丢：`dropped_count`=1、overflow sticky、窗口不完整标——**FIFO 深度参数化仅覆盖正确性边界（深度=1），实际深度由 W6-01 推证冻结**（不占配置工程默认） | 【未来】单元（u4d1_depth1.yaml）                          |
| U4-d2 | 负   | debug FIFO 满注入                                            | `dropped_count` 计数、overflow sticky 置位、issue/唤醒/终态**不受影响**（无功能阻塞）                                                                       | 【未来】单元（u4d2_saturating.yaml）                      |
| U4-e  | 正   | debug 关闭/复位排空                                          | 已接受 HBM 日志写保留 credit/owner 至真实终态/隔离；未排空时状态查 `pending`，非"清 FIFO 即完成"                                                            | 【未来】单元（外存事务腿在 W4）                           |
| U4-f  | 负   | 特权未开时 debug 导出尝试                                    | 无 HBM 日志 DMA；无 ring 写；计数器有溢出标记                                                                                                               | 【未来】单元                                              |

**退出条件**：U4-a…f（含 U4-d1/U4-d2）通过；`gate_ctl/debug_fifo` 单元通过；合同 §4.4 命令被 W2 计入其 CoreAxi 待办（主计划 CN-G4 前核对）。

**T 追踪**：T-26、T-27、T-51、T-67、CLK-01~06、INV-10、OBS-14~18、PWR-04/05、T-36、RST-09（复位后 debug 默认关）、T-54（控制通道独立，通道隔离判据在本包合同 + W7-03 检查）。

---

### 5.5 W6-05 竞争/故障/满/毒死/回绕/外写撤销 集成

- **优先级**：P0（CN-G5 出口）。**硬前置**：W6-02、W6-03、W6-04 全部产物；W7-03 checker 框架**不作 DONE 前置**——本包以"注入矩阵 + RTL 单元/one-Tile 真实 loop 的边界日志"为自判证据，checker verdict 落 W7-03 后由 W7-04 出口门消费（W7 是 checker/出口 owner）。
- **输入**：全部前述合同 + 线性化点定义（§5.0）+ `fault_ring.py/reset_domain.py/event_table.py` 基线 + `tile_group.py:1391-1477`（Group 侧）+ `CoreAxi.scala`（W2 改造后）。

**目标**：把 §5.0 单周期线性化点下的**具体边界状态**逐一注入并判读。本包负责 RTL 侧行为正确与计数可观察；场景脚本/checker 归 W7-03，本包提供场景源素材（文件名与判读字段写入本节，正式 schema 依据 W7-03 冻结的 scenario 格式落 `hardware/coral_nexus/scenarios/w6/`，W7 schema 冻结前这些文件只是**计划级素材**，不属于"可执行命令"的一部分）。

**注入矩阵 / 判读（每行给出：注入、字节/状态判据、终态）**：

| 冲突                                       | 具体注入                                                                         | 期望终态/判据（状态/字节/计数）                                                                                                                                                                                                                                                                                                                                                                              | 归属                      |
| ------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| 同拍竞争（EVT-10）                         | completion / fault / cancel / reset 同拍到达（多组合：同 Context vs 异 Context） | 表终态与 Group 观察一致；先到合法终态不撤销；同拍未终态者 reset/error；单一线性化点可对拍 W1 boundary log                                                                                                                                                                                                                                                                                                    | W6 RTL + W7 checker       |
| 满表（EVT-17 / EVT-09）                    | per-Context frontier 满（分配超限） / 终态记录槽满 / 通道满                      | 分配失败=Context 背压**不 fault**；完成通知/信用回收不被满阻塞；Task 记录保存不丢；**无"满表即错误"**。frontier 满表下新 LAUNCH 的 stall 归因 = WAIT_OPERAND / WAIT_EVENT（StallReason 现有枚举，W6 不新造）                                                                                                                                                                                                 | W6                        |
| 满日志（OBS-15/16）                        | debug FIFO 满 / HBM ring 满                                                      | drop + overflow + `窗口不完整` sticky；不阻塞 issue/state_free/唤醒/终态/故障处理                                                                                                                                                                                                                                                                                                                            | W6                        |
| 旧代（CTX-06 / INV-15）                    | 旧 generation 完成/ACK/signal                                                    | 按完整键拒绝/忽略/计数；**不写新代状态**；迟到回包完成握手并归还事务信用                                                                                                                                                                                                                                                                                                                                     | W6                        |
| 回绕（CTX-06）                             | 计数器/序号回绕窗口                                                              | 位型复用前受控排空：W6 发 `GEN_WRAP_QUIESCE(ContextKey)`、**Group（W1-01 生成的通道）响应"该 ContextKey 无旧代在途/队列/ACK 缓存"并授权复用，Tile 才允许位型回绕**；仅"Tile 本地静止"不足以答复复用授权——**旧事件身份义务（在途完成/迟到 ACK/EventTable 旧条目）必须一并闭合**；同位型古老副本在途时**必须阻止回绕**（或保留未回绕外层 epoch）；同位型旧副本尚可到达时禁止复用该序号/代际（T-45 负例同口径） | W6 查询 + W1-01 响应/授权 |
| 外部永阻（RST-08，"**超时不是排空成功**"） | 外写/外存事务永不响应（W4 侧无响应）                                             | DRAIN 阻塞至 `max_drain_cycles`→Group 强制 cancel→仍无真实终态→**poison**；Task 终态非成功；`is_quiescent` **不置 1**；显式静止恢复仍要求真零证据（`reset_domain.py:2235-2247`）；`result.completed=False`、`group.poisoned_reason` 非空、`reset_domain.is_done=False`                                                                                                                                       | W6（证据）+ Group（判据） |
| 隔离失败（RST-08/MAT-16）                  | 隔离确认失败/回滚尝试                                                            | 终态**不返回 ISOLATED**（不释放该事务/view/lease），保持未闭合占位 + poison 成立；不可凭 generation 撤销；显式恢复仍需真零证据                                                                                                                                                                                                                                                                               | W6 + W4                   |

**负例设计（必须，先失败后通过）**：以上矩阵每一行**注入前必须先在 baseline (无注入) 上通过正常终态一致性**，再注入冲突；判读器（W7-03 checker）按 W6 设计 §8 断言列表逐条判定，**不允许"看起来能跑"跳过注入**。

**正/负验证（本包独立可跑）**：

| 用例 | 类型 | 场景                                                                                            | 断言                                                                                                                                                                                   | 命令                                                                                                                                                                                                                                                                                                                                                                                       |
| ---- | ---- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| U5-a | 正   | 多 Context 正常流 + 全部前续 U1-U4 正例重跑全绿                                                 | 基线不退化                                                                                                                                                                             | 【当前可执行】`pytest -n0 pipeline_validator/tests/`（重点 multicontext/l2_sharing/runtime）                                                                                                                                                                                                                                                                                               |
| U5-b | 负   | 注入矩阵每行（每行一个 scenario 文件，命名 `hardware/coral_nexus/scenarios/w6/u5b_<row>.yaml`） | 表格判据逐项成立；合法故障注入 = 受控行为（场景 PASS），仅**缺陷 mutation**（如去同拍 bypass、去 event_generation、提前释放、超时判成功）使 checker fail（W7-03 输出 checker verdict） | 【未来】`bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/scenarios/w6/u5b_<row>.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>`；【未来】`bash hardware/coral_nexus/scripts/cn.sh unit --suite w6 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>` |
| U5-c | 负   | 外部永阻 + 强制取消（scenario：`u5c_external_permanent_block.yaml`）                            | 排空**不成功**（`is_quiescent==0` 持续）、poison 成立、终态不成功                                                                                                                      | 【未来】`bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/scenarios/w6/u5c_external_permanent_block.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>`                                                                                                                                  |
| U5-d | 负   | first-fault 重复注入                                                                            | suppressed 计数准确、无覆盖（直接字段确定序：source_id→engine_id→ctx_slot→op_seq）                                                                                                     | 【未来】单元（复用 U3-a 场景 `u3a_multi_fault.yaml`）                                                                                                                                                                                                                                                                                                                                      |

**退出条件**：U5-a…d 每行注入矩阵（先无注入 baseline 通过，再注入后按判据**不通过→判读错误被检出**）+ 本包单元全部 PASS；外部永阻行的退出口径是"永不判成功排空"：该行在 DRAIN 超时后只能停在 `poison`（`reset_domain.is_done=False`、`completed=False`、`poisoned_reason` 非空），不允许"等更久就成功"；CN-G5 出口读 W7-03/W7-04 的 checker verdict（本包不替代 checker，checker verdict 落 W7-03 后 W6-05 才算"集成验证完成"）。

**T 追踪**：T-13/14/15/23/24/25/26/27/45/51/52/54/67 + CTX-06/09/11/16 + INV-15/16 + OBS-15/16/17 + RST-07/08/09/10/11 + MAT-16。

---

## 6. 给低能力模型的手册（面向实现 agent）

1. **只做本包的清单**：W6 状态/事件/复位接口是唯一写者（§1.2）；不要去改 `CoreAxi.scala`（W2）、`tile_group.py` 共享接线（W1）、`memory/transfer.py`/`arena.py`（W4）、`compiled_program.py`/`package/`（W5）、checker/schema（W7）。跨包需求走合同 fragment YAML + §4.4 请求。
2. **先读哪些**：`../plan.md` §2/§3（权限/ID/DAG）；W6 设计 §3/§4/§5 全文；本计划 §2.1（证据行号）、§4（跨包）、§5（单任务）、§5.0（线性化点）。
3. **不能做什么**：
   - **不预设位宽/位编码**（W1-01 冻结后再消费；合同 cn_protocol 生成物是权威）；
   - 不实现 FLT-02a 局部恢复、不加 WAIT_ANY、不加超时自判、不把 `max_drain_cycles` 在 Tile 侧复制；
   - 不把 FaultRing 的 **覆盖写**当 first-fault-wins 硬件语义（validator 只作计数参考，FLT-10 注）；
   - 不在 Tile 发"成功即释放"信号；POST_EVENT 仅 manifest 授权（§4.1）；
   - **接口缺失时不自行改合同**：先在本包合同 fragment YAML 加注释并**发最小证据请求给该字段 owner**（引用 W6 设计 §：行号 + 期望字段 + 请求理由；W1-01 field-requests 路径 `protocol/field_requests/w6.yaml`，W1-01 计划草案定义），拿到冻结合同后继续；不使用任何 `agent://…` 通道或聊天回退作为持久合同引用。
4. **先失败后通过的负例设计**：每个 U-xx 用例**先构造能触发的错误输入**（如 stale sequence / 同拍 fault / frontier 满表 / 旧代取消）确认**拒绝/计数/终态非成功/不阻塞**；再叠加正常流确认不误伤。**fault-injection 的"安全失败"（fault→干净 drain→终态唯一）是 PASS-case**；只有**缺陷 mutation**（如去掉同拍 bypass、去掉 event_generation 键）才应使对应 checker fail——两类不可混。判读以**计数器+状态寄存器+边界日志字节**为证据，不以"跑通了/trace 变长/非空"为证据。
5. **何时可标 done**：本条任务的 U 清单全部 PASS 且负例有检出（无"跳过"）、退出条件段落所列前置合同 hash 已 joint-seal（CN-G1 由主计划统一核对）、无未闭合"待实现 TODO"。

## 7. 风险与需上报事项（合同冲突、需 parent 裁决/整合）

均不自行实施，仅报告；每条给"最小证据 + 请求理由"。

| #   | 风险/冲突                                                             | 证据                                                                                                                                           | 建议                                                                                                                                                                     | 需谁裁决                                    |
| --- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| R1  | CoreAxi `!core.io.wfi` 全核关钟直连                                   | 上游 pin f795221…CoreAxi.scala L115                                                                                                            | CR-W6-W2-01（§4.4）：删除该直连，Tile enable 归 W6 gate_ctl；WFI 仅作 Context park 提示                                                                                  | W2（已接受）                                |
| R2  | reset_domain.py 的 `max_drain_cycles` 在 `HardwareConfig` 无对应键    | `config.py:486`（max_cycles）、reset_domain.py:89 `getattr(cfg,'max_drain_cycles',100)`                                                        | W1-04 把该值变成显式 config 字段并实测基线；W6 不改该语义                                                                                                                | W1-04 / parent                              |
| R3  | `FaultRing` 是覆盖写、无 first-fault-wins                             | `fault_ring.py:81-88`                                                                                                                          | RTL 按 FLT-10 first-fault-wins 硬锁存（直接字段确定序）+ `suppressed_fault_count`；validator FaultRing 仅作计数参考；差异写入 fragment YAML                              | parent（合同层确认）                        |
| R4  | `POST_EVENT` 在批准设计表中存在                                       | W6 设计 §4 表"W6→Group POST_EVENT"                                                                                                             | 保留为逻辑接口；语义限"`manifest 显式授予 Group-scope producer 的 Task`"；无授权 kernel 在 Loader/译码拒绝；实现走可靠通道（§4.1）。**不 no-op、不静默禁用**；已告 W1-01 | W1-01（授权字段），W5-01（manifest 端授予） |
| R5  | 现有 pytest 默认 `-n auto` 并行                                       | `pytest.ini`                                                                                                                                   | 本包验证命令统一 `pytest -n0`，并给 `export PYTEST_ADDOPTS="-n0"` 提示                                                                                                   | W6 自身，无需裁决                           |
| R6  | `budget.event_frontier` 是 Group 级字段，不能当 per-Context 上界      | `execution_verifier.py:2013-2031`、`profiles.py:632-635`                                                                                       | per-Context 上界按 W6-01 步骤 5c 的静态 launch/await 图活跃界（W5-01 manifest 承载），runtime 峰值仅辅证                                                                 | W5-01（合同字段），parent 知会              |
| R7  | `max_drain_cycles` 具体值（模型 100）是 getattr 默认而非 config 键    | reset_domain.py:89                                                                                                                             | 建议显式设键并从实测基线派生，不空指定                                                                                                                                   | W1-04                                       |
| R8  | `csr.io.cg` 语义（单 Context 写位 vs 全核门控）当前未重核上游 CoreCSR | 合同只引用 `CoreAxi.scala` L115 表达式                                                                                                         | W2-01 在 CoreCSR 侧确认"Context 写 cg 不影响全核"，与 CR-W6-W2-01 同改（W2-01 计划草案已建议降级为 park 投影输入）                                                       | W2（已接受）                                |
| R9  | fault→reset domain：生产路径仅 GROUP                                  | `tile_group.py:2371`（函数支持 tile_id≥0→TILE 分支）、`:2354-2360`（on_scheduler_fault 不传 tile_id ⇒ GROUP）、`:1432/1444/1460`（tile_id=-1） | 保留函数事实；CS1 生产 fault 域固定 GROUP；TILE 域仅测试/注入；FAULT_REPORT 的 tile_id 只作归因，不改域选择                                                              | parent 知会（已核，RST-01 一致）            |

## 8. 本包与 parent 的整合说明

1. 本计划**只**创建 §2.2 清单里明确列出的文件；`hardware/coral_nexus/` 目录树不存在，文件落位由 W0-01/02 创建目录后执行——**不要提前创建脚手架**。
2. W6-01 输出两份合同 fragment 给 W1-01 joint-seal（`gen_protocol.py` 读取、`PROTOCOL_FREEZE.json` 全部 contract 齐后 seal、不可逐条追加）；若 W1-01 认为字段/语义有误，回最小文件:行证据；不接受"无注释直接覆盖"或"另起语义"。W6-01 产物本身不等其他 -01 DONE（joint seal 只看 fragment 齐）。
3. §4.4 CR-W6-W2-01/02 是**请求**而非合同变更：W2 已接受（W2-04 计划草案）；若发现语义冲突（例如 `csr.cg` 上下文语义不可用），在其计划回最小证据，parent 统一裁决。本包不修改任何上游/共享文件。
4. `max_drain_cycles` 等参数由主计划 W1/W2/W5 统一登记，本包不写配置数值。
5. 本计划不含永久指向 `agent://…` 或聊天通道的引用；所有跨包请求以 fragment/field_requests 文件 + owner 任务 ID（W1-01/W2-01/W5-01 等）为准。

---

附：本计划所有 `bash hardware/coral_nexus/scripts/cn.sh ...` 命令、`hardware/coral_nexus/rtl/`、`hardware/coral_nexus/scenarios/w6/` 路径均为**未来产物**（`../plan.md` §2/§3：先锁命令 I/O，不谎称现在存在）；【当前可执行】命令只引用现有 pipeline_validator pytest。
