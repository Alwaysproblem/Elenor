# W1 边界与联仿实施计划（W1-01..W1-05）

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

> 状态：实施计划（父计划 `../plan.md` 的分包文件；共享合同要点见父计划 §2/§3）。
> 上游设计：[W1_Boundary_Cosimulation.md](../components/W1_Boundary_Cosimulation.md)（下称 W1 设计）、[总体设计 §4/§5](../Coral_Nexus_Tile_Design.md)、[需求规格 r5](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。
> 本轮只写计划：不建目录、不改代码、不运行 build/test/formatter。所有 `hardware/` 与 `pipeline_validator/cosim/` 路径均为**待创建目标**。validator 编译产物当前仍 schema 3／ABI v3（`compiled_program.py:894,1089`），本计划不改该域。

## 1. 已选方案摘要（链接总计划；未实测参数不伪称冻结）

1. **替换边界**：整 Tile 换 RTL（UCE/寄存器/ROB/RVV/VME/LSU/MFE/L1/MSHR/Frame/本地事件/程序安装全在 RTL），CPU Host/Device/Group/L2/cache/MSHR/NoC/HBM/页授权留 Python（W1 设计 §1）。首个集成 1 RTL Tile smoke；最终 CS1 = 1 Group × 4 RTL Tile（cs1_c2）；mixed 仅调试。
2. **唯一时间权威**：Python `Simulator.tick` 单调整数 cycle；RTL 每 tick 由 Python 驱动恰一次，C++ 侧不回调 Python、不执行 P4（W1 设计 §3.1/§7）。
3. **五相位**：P0 drive→P1 双向 pre-edge `valid&&ready` 唯一接受点→P2 唯一上升沿→P3 仅提交 P1 接受集（staged 输出只作下 tick 候选）→P4 Python 外层恰一次。相位合同由 W1-01 机器可读冻结，W7 直接引用。
4. **通道**：CTRL/EVENT/PROGRAM/MEMORY/DEBUG 独立预算；**STREAM 缺口已由 parent 裁定**：Tile 的 `STREAM_POP/PUSH/ACQUIRE/RELEASE/PUSH_EOS`（`tile.py:544-834`，`tile_group.py:1349` 直接绑 Group 侧 `StreamQueue`）在 W1 设计 §3.2 无承载消息——消息化为 **MEMORY 通道的独立逻辑子队列/子预算**（不新增第六物理通道；Group `StreamQueue` 唯一 writer，§7.2/§13.1）。
5. **准入**：保留"每 Tile 每 tick **至多一个**新提交决策"（`tile_group.py:1229-1246`），拆为多 tick 消息事务：PLAN→PARENT_PREP→HOLD_REQ→LOCAL_HOLD→FINAL_COMMIT→ACCEPTED（§3.4/§3.5），最终授权后不 ROLLBACK；Group 串行逐 ticket 管理，ticket 并发只受有界准入事务槽约束、hold 不超卖（IF-13），**不禁止 Tile 侧多 ticket 并发**。
6. **退休**：EXEC_DRAINED→RETIRE_READY→RETIRE→本地回收→TASK_COMPLETION→Group 关剩余父 pin/R；**无任何"等父资源结清"前置**（禁环，§10.1 项 2）；TASK_COMPLETION 不等父 R free；Python 不再执行 L1 retire。
7. **输入早释放**：`PHASE_SIGNAL(input_released, TaskKey)` 经 EVENT 通道，Group `all_tasks` 聚合后解除纯读 pin；安全退休清理剩余 pin（不重复释放）；不归还 L1/Frame/UCE/R/程序引用。
8. **gate 双唤醒**：L1 issue gate 打开时 Group 自行唤醒 Route 登记与 Task 准入两条等待路径（事件驱动，不逐 tick 轮询），`GATE_NOTIFY` 另行通知 Tile 侧。
9. **C ABI**：单线程 ctypes 调 Verilator 共享库 `libcn_cosim.so`；exports＝`bp_abi_info/bp_begin/bp_layout/bp_drive/bp_tick/bp_collect/bp_status/bp_destroy`（最终命名，无别名/无 CMake；`bp_begin` 复制 cfg 不留 Python 指针；`bp_drive`=独立 P0；`bp_tick`=仅 P1-P3；`bp_collect`=P3 后 P4 前、调用者预分配缓冲拷必要有效 beat；每 tick 零分配）；不做 socket/共享内存/DPI 备选；无有损重试。
10. **身份**：TaskKey=(root_launch_generation, GridInstanceId, task_id)。已核实 `execution_ir.py:288-298` 的 `GridInstanceId` 本身含 `launch_generation`，且 `tile_group_sequencer.py:76-82` 用同一 `context_launch_generation` 填充，二者当前恒等。冻结：wire 只带 grid 的 `launch_generation` 一份，Group 侧断言 `root_launch_generation == grid.launch_generation`（不一致即协议错误），不重复发两份。

## 2. 本包唯一 owner 表

| 资产                                                                                                                 | 唯一 owner                      | 边界                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 逻辑消息/身份/相位/预算/编码合同（机器可读权威）                                                                     | W1-01                           | 各包以 request 文件提交字段，W1-01 单点合并；禁止他人自造编码                                                                                                                               |
| `pipeline_validator/cosim/`（driver/adapter/port/服务适配/coordinator）                                              | W1-02..W1-05                    | 消费各包语义设计，不发明新语义                                                                                                                                                              |
| 共享 Python 文件接线：`simulator.py`、`tile_group.py`、`device.py`、`runtime/group_port.py`、`cli.py`（cosim flags） | W1（合同 §"未来代码集成owner"） | 语义级改动（scheduler 准入序、reset_domain 行为、profile_controller、event_table、program_table、transfer.py/arena.py）仍归原 owner 包；W1 只做接线与消息化改造，冲突上报                   |
| `__main__.py`                                                                                                        | 现状保留（不改，W1 不触碰）     | **仅 5 行 shim 不扩大**（parent 指示）：cosim 新 flags 一律加在 `cli.py` 的 argparse，`python -m pipeline_validator` 继续走 `cli.main`；cosim 场景入口归 `cli.py` + `cosim/run.py`（W1-05） |
| `hardware/coral_nexus/sim/cabi/*`、`hardware/coral_nexus/sim/unit/*`、`hardware/coral_nexus/protocol/*`              | W1-01/W1-03                     | RTL 侧 Tile core 文件归 W2-W6                                                                                                                                                               |

## 3. 当前文件证据（已存在）与待创建清单（分开）

### 3.1 已存在代码证据（2026-11-05 HEAD 814bdf9，行号以当前仓库为准）

| 主题                                | 证据                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 主循环                              | `simulator.py:99-247,307-331`（model 路径 CPU→session→`group.step`→harvest；standalone 路径 `group.step` L165）                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Group 单 tick 顺序                  | `tile_group.py:1740-1818`（NoC→transfer→collective→stream→tiles→profile→scheduler→admission→reset FSM→PMU→退休）                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 准入                                | `tile_group.py:1159-1227`（`dispatch_role`，L1173 dispatch_capacity、L1191 L1 gate check）、`_step_task_admission:1229-1282`、`_commit_route_task:1284-1350`（commit→pin→program 登记→ensure_resident→load_program→abort 补偿 L1341-1345）、`_activate_admitted_context:933-967`（first action at cycle+1）                                                                                                                                                                                                                                                      |
| 退休链                              | `_step_tiles:1669-1738`（terminal→in-flight 检查→invalidate_view→retire_arena→Frame release→unpin→lease→route.retired→grid 完成 notify）；`retire_context_arena:1523-1548`；`_retire_sequencer:1820-1831`                                                                                                                                                                                                                                                                                                                                                        |
| 输入早释放                          | `_on_phase_signal:1391-1477`（stale/duplicate/invalid 计数与 fault）、`_pin_grid_l2:970-1008`、`_unpin_grid_readers:1010-1025`、`_unpin_task_l2:1352-1360`（幂等）；`PhaseSignal`（`execution_ir.py:310-314`）                                                                                                                                                                                                                                                                                                                                                   |
| Tile Python 执行（被关闭路径）      | `tile.py:300-353`（UCE.step）、`plan_admission:1541-1570`、`commit_admission:1572-1596`、`abort_admission:1598-1631`、`rollback_program:1667-1679`、`load_program:1633-1651`、`drain_context_terminals:1681`                                                                                                                                                                                                                                                                                                                                                     |
| Tile↔Group 对象耦合（跨界须改造点） | `tile.py:1462-1527`（直接持 l1_allocator/l1_cache/l1_mshr/streams）、`tile_group.py:262-276`（构造 ComputeTile）、`277-278`（precise_write_recorder 注射）、`366-368`（byte_store.register_cache l1）、`449`（`tiles[pool].l1_allocator` 生成校验）、`1690`（`transfer_manager.has_inflight_access(admission.l1_handles)`）；`profile_controller.py:911-958`（`_tiles/_pool/_caches/_mshrs` 直取 `tile.mfe.l1_cache/l1_mshr`）；`reset_domain.py:138-219`（`tile.uce._clear_engine_queues`、`tile.mfe.begin_reset_drain`、engine accepted_count、l1_mshr.stats） |
| CPU/Device 协议                     | `device.py:23-27`（`DeviceCompletionStatus` 只有 SUCCESS/ERROR）、`30-46`（LaunchRequest 携带 Python `task` 对象）、`108-127`（DevicePort 协议）                                                                                                                                                                                                                                                                                                                                                                                                                 |
| GroupPort                           | `runtime/group_port.py:184-186`（重复 request_id 异常）、`355-377`（唤醒令牌 `(admission_version, _slot_version)`）、`_retire_slots:384-418`（SUCCESS/ERROR→DeviceCompletion）                                                                                                                                                                                                                                                                                                                                                                                   |
| 事件                                | `runtime/event_table.py:400-460`（signal 全键校验/陈旧计数）、`362-375`（回收）；外部事件进 Tile 的现路径 `tile.py:355-359`（`notify_event` 对 `ev_dma_` 前缀写 `_external_events_done`）、`tile_group.py:1566-1570`（组 DMA 完成广播给 active context 的 Tile）、`1836-1839`（root 退休时 `retire_external_events`）                                                                                                                                                                                                                                            |
| STREAM（缺口主体）                  | `tile.py:544-570,620-634,795-845`（STREAM ops 直接操作 `tile.streams`）、`stream_queue.py`（credit 不变量 `available+leased+valid_in_fifo+popped_not_released==depth`、`reset()` 代际++）、`tile_group.py:1348-1349,1790-1793`（bind/unbind）                                                                                                                                                                                                                                                                                                                    |
| 程序驻留（被替代）                  | `runtime/program_table.py:79-100`（纯解析、同步置 TILE_RESIDENT）、`tile_group.py:1311-1318`（register+ensure_resident 计入 prepare_cycles）、`L1350`（program_cold_load PMU 直加）                                                                                                                                                                                                                                                                                                                                                                              |
| transfer/L2                         | `memory/transfer.py:208-256`（`MemoryTransaction`：issuer/src/dst/legs/generation 字段，`tile_id` 标 tile-local）、`tile_group.py:370-457`（transaction reference、generation 校验、L1 allocator 校验点 `449`）                                                                                                                                                                                                                                                                                                                                                  |
| CLI                                 | `pipeline_validator/cli.py:204-323`（argparse 实际所在；`--sim-override fidelity`、`--context-mode` 等）                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 运行命令                            | `python -m pytest pipeline_validator/tests/ -v`、`python -m pipeline_validator ...`（README.md:244,261；当前可执行）                                                                                                                                                                                                                                                                                                                                                                                                                                             |

### 3.2 待创建文件清单（全部为新目标）

| 文件                                                                                                                                                                                                                                | 任务              | 摘要                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `hardware/coral_nexus/protocol/cn_protocol_v1.yaml`                                                                                                                                                                                 | W1-01             | 消息目录/身份/相位/幂等键/stale 规则（机器可读权威）                                                                                        |
| `hardware/coral_nexus/protocol/channel_budgets.yaml`                                                                                                                                                                                | W1-01             | 通道 credit/事务槽/终态引用/PROGRAM 字节窗（来源实测推导登记）                                                                              |
| `hardware/coral_nexus/protocol/lifetime_bounds.yaml`                                                                                                                                                                                | W1-01             | 每序号/ID 域在途上限、驻留上界、复用规则（ABA 论证输入）                                                                                    |
| `hardware/coral_nexus/protocol/csr_name_map.yaml`                                                                                                                                                                                   | W1-01             | 新增声明/配置 CSR 的命名域 + 与上游已占地址的冲突清单                                                                                       |
| `hardware/coral_nexus/protocol/extract_upstream_csr.py`、`csr_allocator.py`、`csr_lint.py`                                                                                                                                          | W1-01             | 上游占用表抓取（f795221 Csr.scala 现读实证）；固定排序位宽推导与分配（不随文件遍历漂移，变更重 seal）；collision/lint 校验                  |
| `hardware/coral_nexus/protocol/field_requests/{w2,w3,w4,w5,w7}.yaml`；W6 的 local event/reset fragment（`w6_event_facility.contract.yaml`+`w6_terminal_reset.contract.yaml`，路径与内容以 PlanW6 计划为准，W1-01 单点合并、读不改） | 各包 -01          | W1-01 单点合并进 `cn_protocol_v1.yaml`；fragment 语义权威归 W6-01；W1 不复制、不改 fragment                                                 |
| `hardware/coral_nexus/protocol/gen_protocol.py`、`derive_widths.py`、`freeze_check.py`                                                                                                                                              | W1-01             | 权威→代码生成器；位宽推导器；联合冻结校验器                                                                                                 |
| `hardware/coral_nexus/protocol/generated/{cn_messages.h,cn_offsets.svh,cn_layout.json}`                                                                                                                                             | W1-01             | 生成产物（禁手写数值）                                                                                                                      |
| `hardware/coral_nexus/protocol/PROTOCOL_FREEZE.json`                                                                                                                                                                                | W1-01             | 联合冻结记录（各 -01 产物 hash；共同 freeze 产物不要求互等 DONE，由 freeze_check 汇聚）                                                     |
| `hardware/coral_nexus/sim/cabi/{cn_abi.h,cn_abi.cc,BUILD.bazel}`                                                                                                                                                                    | W1-03             | Verilator 共享库 `libcn_cosim.so` C ABI 封装（bp_begin/bp_drive/bp_layout/bp_tick/bp_collect/bp_status/bp_destroy；`bp_tick` 不隐式 drive） |
| `hardware/coral_nexus/sim/unit/cn_protocol_stub.sv`                                                                                                                                                                                 | W1-03             | **仅单元级**协议 stub Tile（边界状态机+故障注入），永不用于 CS1 判定                                                                        |
| `pipeline_validator/cosim/__init__.py`、`wire.py`、`abi.py`                                                                                                                                                                         | W1-01/W1-02/W1-03 | 生成 pack/unpack（wire.py），ctypes 绑定与 ABI hash 校验                                                                                    |
| `pipeline_validator/cosim/{tile_port.py,python_backend.py,rtl_backend.py}`                                                                                                                                                          | W1-02             | TilePort 协议 + Python Tile 适配 + RTL Tile 端点                                                                                            |
| `pipeline_validator/cosim/{memory_service,program_service,event_service}.py`                                                                                                                                                        | W1-02             | MEMORY/PROGRAM/EVENT 外侧服务适配器                                                                                                         |
| `pipeline_validator/cosim/{admission,retirement,caps.py,reset_phase.py}.py`                                                                                                                                                         | W1-04             | 准入协调器、退休协调器、caps/复位相位                                                                                                       |
| `pipeline_validator/cosim/incomplete.py`                                                                                                                                                                                            | W1-04/W1-05       | 崩溃/超时/失恒终态落盘（`INCOMPLETE*` verdict＋残留 pending set 明细；非假 drain）                                                          |
| `pipeline_validator/cosim/gate_wake.py`                                                                                                                                                                                             | W1-04             | gate 双唤醒登记（profile gate OPEN_ISSUE → 两路径重试）                                                                                     |
| `bash hardware/coral_nexus/scripts/cn.sh`（**总合同唯一宿主入口路径**；W1-05 提供 cosim/unit 的 dispatch 处理函数，不另建 `sim/cn.sh`）                                                                                             | W1-05             | cosim/sim-dispatch 入口（W0-02 的 `cn.sh` dispatch 落到此）                                                                                 |
| `hardware/coral_nexus/verif/w1/…`（pytest 文件，flat 于 `pipeline_validator/tests/` 亦可）                                                                                                                                          | 各                | 见各任务验证条目                                                                                                                            |

## 4. 冻结合同概览（跨包要点，正文在各任务）

- **seq 分域**：`message_seq`（每 tile×channel×dir 的边界传输层序号，重复检测/ACK 对应）≠ `instruction op_seq`（W2 内部，随指令）≠ `txn_id`（memory/program 事务）。跨边界消息头转发 op_seq 供对拍但**不参与**传输层重放判重；`subop_id` 只在 RTL 内部，不上边界线。**subop 不在线的恢复前提（parent 要求明确）**：每个跨边界事务由全键唯一的 `txn_id`（＋分片 `seg_idx/seg_last`）标识，Group 侧 accepted-txn 表从 accept 时刻记账到终态/补偿闭合，迟到响应、分片、credit 归还全部经该表结算且可查（`ledgers.py` + `services.py` 状态查询），无 `subop_id` 也能无歧义恢复归属；该表为 W7 对拍与 checker 的可查询账本。
- **预算分立**（IF-09/IF-10）：五个账本（channel credit、准入/事务槽、终态引用、PROGRAM 字节窗、NoC flit credit）互不挪借，逐条守恒式冻结于 `channel_budgets.yaml`。
- **无对象跨边界**：`ExecTileGroupTask`/`MemoryOwner`/`ResolvedMemoryView`/`ArenaPlan.pool_token` 等一律在 Python 内解析；线上只有 TaskKey/ContextKey/身份+地址+范围+权限+分配代（IF-11）。
- **staged 握手**：P1 唯一接受点；P1 接受集（含 tile→G 输出 payload 快照）在 P2 前冻结；P2/P3 不得新增接受；post-edge ready 禁配 pre-edge payload（W1 设计 §3.1/§18.3）。

## 5. 任务依赖表（硬前置；共同 freeze 产物不要求另一 -01 先 DONE）

| 任务                | 硬前置                                                                                                                                                                                                        | 并行窗口                                                                                         | 说明                                                            |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| W1-01               | 无任务级硬前置（读 r5/W1 设计与当前源码；参数从实测源码推导登记，`derive_widths.py` 以"来源=当前代码常量"运行即冻结非空初值；W0-04 最终 config 落地后由 `freeze_check` 复核，冲突→升级 parent，不得静默改值） | 各 -01 互不阻塞，经 `PROTOCOL_FREEZE.json` 汇聚                                                  | 字段请求以 `protocol/field_requests/*` 文件汇聚，W1-01 单点合并 |
| W1-02               | W1-01（wire.py 生成物 + 冻结 hash）                                                                                                                                                                           | 与 W1-03 并行                                                                                    |                                                                 |
| W1-03               | W1-01；Verilator/容器（W0-02 提供运行载体——**仅容器与工具链可用性，无需 W0 判定完成**，W0-02 的环境构建先于本任务运行时）                                                                                     | 与 W1-02 并行                                                                                    |                                                                 |
| W1-04               | W1-02、W1-03                                                                                                                                                                                                  |                                                                                                  | 协调器单元验证先于 4 Tile                                       |
| W1-05(1 Tile smoke) | W1-02/03/04 + W2-05/W4-05/W5-04/W6-02 提供的最小 RTL 件                                                                                                                                                       |                                                                                                  |                                                                 |
| W1-05(final 4 Tile) | 上述 + 4 Tile RTL 就绪（W2/W3/W4/W5/W6 各自 -05/-06 的 RTL 就绪**接口/端点**，以其 manifest 为准）                                                                                                            | W7-04 场景执行消费本包 harness，不互相等待 DONE：W1-05 交付"可运行 harness+证据文件"，W7-04 引用 |                                                                 |
| （禁循环）          | W1-05 不作为 W2-05/W5-05/W6-05 的前置：这些包用 W1-03/04 的 oneTile 真实 loop + 本包端点验证自己的 Tile RTL                                                                                                   |                                                                                                  |                                                                 |

## 6. producer→consumer 总表

| 产物/字段                                                                                           | 生产者（任务）                                | 消费者（任务）                                                                                                                                                                                                                           | 落点/验证                                                       |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `cn_protocol_v1.yaml`、`PROTOCOL_FREEZE.json`、generated `cn_messages.h`/`cn_offsets.svh`/`wire.py` | W1-01                                         | W2-01（准入握手/CANCEL/EXEC_DRAINED 接口）、W3-01（EVENT 声明字段、MEMORY）、W4-01/02/04（MEM 语义、GATE_NOTIFY、页授权检查点）、W5-01（PROGRAM 通道字段）、W6-01（EVENT/FAULT/RESET 消息）、W7-01（边界日志 protocol 枚举、phase 枚举） | 各包对其引用字段在 `field_requests/*` 登记；W1-01 合并后 freeze |
| `message_seq` 域位宽与复用规则                                                                      | W1-01（`lifetime_bounds.yaml`）               | W6-03（迟到旧包隔离）、W7-01（seq 断档→窗口不完整）                                                                                                                                                                                      | `derive_widths` 单元负例：slot 过小拒绝                         |
| 相位模型（P0..P4 语义、accept 集规则）                                                              | W1-01                                         | W7-01（trace phase 字段语义唯一）、W2-03（发射相位采样）、W4-02                                                                                                                                                                          | W1-03 stub 负例（post-edge ready 错配）                         |
| `TilePort` Protocol                                                                                 | W1-02                                         | W2-01（Tile 侧端点接口对齐）、W6-01                                                                                                                                                                                                      | tile_port.py Protocol 检查（mypy/单元双端驱动）                 |
| `python_backend.py`（Python Tile 适配）                                                             | W1-02                                         | TileGroup（mixed）、W7-04（对拍基线）                                                                                                                                                                                                    | 现有 pytest 全绿为回归门                                        |
| `memory_service.py`（MEM_REQ→TransferManager/L2）                                                   | W1-02                                         | W4-02/03（L1 legs 切除后外侧腿）、W3-04（gather/scatter 服务段）                                                                                                                                                                         | 单元：ByteStore 字节一致性负例                                  |
| `program_service.py`（INSTALL/READ/ACK/READY）                                                      | W1-02                                         | W5-04（真实安装、READY 时序）、W2-03（首取指晚于 READY+accepted）                                                                                                                                                                        | 单元：hash 不符=FAULT；READY 不授权执行负例                     |
| `event_service.py`（EVENT_NOTIFY→Tile、`_on_phase_signal` 保留）                                    | W1-02                                         | W6-02（等待唤醒）、W3-03（region 等待）                                                                                                                                                                                                  | 单元：陈旧 owner/generation 拒收计数                            |
| `caps.py`（QUERY_CAPS/CAPS_RESP）                                                                   | W1-02（字段 W1-01 冻结）                      | W2-01/W5-01（能力协商）                                                                                                                                                                                                                  | 单元：能力缺失→任务拒绝                                         |
| `cn_abi.h/cn_abi.cc`、`cn_protocol_stub.sv`、`abi.py`                                               | W1-03                                         | W1-04（真实 tick 序列）、W6-03（复位相位经 ABI）、W7-02/05（重放驱动复用）                                                                                                                                                               | stub 单元套件；CS 1 判定永不采信 stub                           |
| `admission.py` 协调器 + tile_group.py 拆分                                                          | W1-04                                         | W2-01（hold/FINAL 执行侧）、W4-01（L1 hold 记账）、W7-03（T-20/T-21 checker）                                                                                                                                                            | 单元：补偿集合完整、票据保持到释放证据                          |
| `retirement.py` + `_step_tiles` 拆分 + TERMINAL_ACK                                                 | W1-04                                         | W2-04（EXEC_DRAINED 归属）、W6-02（可靠终态）、W4-02（MFE in-flight 结清）                                                                                                                                                               | 单元：早退休负例判协议错误                                      |
| `gate_wake.py` + GATE_NOTIFY                                                                        | W1-04                                         | W4-05（L2/L1 gate 行为侧）、W3-02（RTL 侧 hold 状态机）                                                                                                                                                                                  | 单元：gate 关闭注入 OPEN_ISSUE→恰一次重试                       |
| STREAM 消息族（parent 已定：MEMORY 通道独立逻辑子队列/子预算）                                      | W1-01（消息/预算冻结）、W4-01（payload 布局） | W2-01（Tile 内 STREAM 指令）、W6-02（流等待）                                                                                                                                                                                            | 单元：credit 不变量、EOS/ERROR 规则                             |
| `hardware/coral_nexus/scripts/cn.sh` 的 `cosim`/`unit --suite w1` dispatch                          | W1-05                                         | W7-04（场景执行）、W7-05（replay/bench）                                                                                                                                                                                                 | cs1_c1 回归 + cs1_c2 4 Tile 门                                  |
| run 级 evidence/INCOMPLETE（`bp_status`+Python 捕获）                                               | W1-03/W1-05                                   | W7-06（出口）、W0-05（容器复现引用）                                                                                                                                                                                                     | 崩溃注入测试＝INCOMPLETE 而非假 drain                           |

---

## 7. W1-01 消息/身份/预算/编码合同（P0；硬前置：无任务级；输出给全部包）

### 7.1 输入

`r5 §11.2/§13/§18`、W1 设计全文、W2-W7 组件设计的接口节（字段级出入以对应组件文为准，W1-01 负责登记差异）、当前源码（§3.1）、字段请求文件（`protocol/field_requests/{w2,w3,w4,w5,w6,w7}.yaml` 与 W6 的两份 fragment 契约文件）。
**冻结规则**：可先写**草案**（`protocol/drafts/*` 或带 `status=draft` 的字段）供各包 review；**joint seal 时缺任何必需包的 field_request/fragment＝阻断 jointfreeze（不冻结）**，不得以 `w1-inferred` 顶替冻结；`PROTOCOL_FREEZE.json` **一次 seal、不可追加修改**，后改内容须以**新 hash** 产生新 seal 文件（旧 seal 归档）。joint gate 不要求各 -01 互等 DONE：各 -01 以自身完整产物先 DONE，后续 implementation 任务等 joint seal。

### 7.2 产物与冻结内容

1. `cn_protocol_v1.yaml`：
   - **通道与子通道**：CTRL/EVENT/PROGRAM/MEMORY/STREAM/DEBUG；每通道每方向独立 valid/ready 端口与 FIFO；STREAM 语义见项 4。
   - **身份表**：
     - `TaskKey`：wire 形式＝`launch_generation`（=root_launch_generation，断言相等）+`dispatch_ordinal`+`task_id`+`context_handle`（Python 侧 `context_name/device_slot` 的小整数句柄；映射表进 run manifest）。
     - `ContextKey`：`tile_id`（=physical tile，单列路由字段）、`ctx_slot`、`ctx_generation`。
     - 分域代际：`run_generation`、`program_epoch`、`profile_l1/l2_gen`、`arena_alloc_gen`、`fetch_epoch`（fetch_epoch 仅 W2 内部，不上边界线，随 `FIELD_VISIBILITY: hidden` 登记）。
     - 序号与去重（advisor 修正）：`message_seq`（每 tile×channel×dir 传输层，自 0 连续；其连续/重复检查是**传输层丢失/重放检测**，与语义去重分开）；语义去重键＝**完整事务域 `txn_id`＋`message_type/阶段`**（同一 admission 事务的 HOLD、FINAL、各阶段 ACK 是**不同合法消息**，单 `txn_id` 不构成去重命中）；ACK 必须引用其对应请求的 `(txn_id, message_type)`；`txn_id`+分片（`seg_idx/seg_last`）只区分**同类型消息的分片重传**。验证条目：同票 HOLD→FINAL 不得被误去重（§10.3 正例）。`message_seq` 永不参与语义去重（重复检测键见上），但传输层 seq 断档/重放→`INCOMPLETE`（传输层故障，非 Tile 语义 fault）。**冻结禁项：不通过"位宽足够大"逃避回绕约束**（CTX-06/INV-15），每个序号/ID 域的回绕前提＝受控排空完成、无同位型旧包存活；`message_seq` 在回绕前必须经 `GEN_WRAP_QUIESCE` 查询确认全部旧代包已 ACK/已消费。禁止把通道序号与父指令 op_seq 混用一个计数。
   - **消息目录与幂等键**：按 W1 设计 §4.2 表逐条登记字段（每字段标 owner 包/来源设计节），新增/补全消息（W1 设计无、代码路径必需）：
     a. `TERMINAL_ACK(txn_id, status)`——MEMORY 终态引用归还（IF-05；对应 `transfer_manager.acknowledge` 语义，`transfer.py:1690-1722`）。
     b. `EVENT_NOTIFY`（G→T）：现路径 `tile_group.py:1566-1570` 组 DMA 完成广播给 active context 的 Tile（`tile.uce.notify_event`→`tile.py:355-359` `_external_events_done`），以及 `_step_tiles` 退休链中 `route.sequencer.notify_event`。字段与 `event_table.signal` 全键对齐（owner/generation/sequence/status/`err_code`，**含 status=RESET 的迟到处理**对齐 `event_table.py:433-441`）。
     c. `EVENT_RETIRE(list)`（G→T）：对应 `retire_external_events`（`tile_group.py:1836-1839`），root 退休后 Tile 侧外部事件清账。
     d. `STREAM*_`族（T→G REQ / G→T RESP）：ACQUIRE/POP/PUSH/RELEASE/PUSH*EOS/EOS/ERROR，字段对齐`stream*queue.py`（token_id/producer_id/queue_generation/seq/first_fault_index；DMA-13 语义）。**parent 已定**：放 MEMORY 通道的**独立逻辑子队列/子预算**（不新增第六物理通道）；Group `StreamQueue` 唯一 writer；`QUERY_CAPS`带`stream_supported`，不支持的配置在加载期拒绝、不为未用 stream 新增范围（§13.1）。
e. `QUERY_CAPS`/`CAPS_RESP`：Tile 能力协商（context_count、VME 实例数（**不进路由身份**，MAT-19）、program 槽容量、stream 队列支持（含 `stream_supported`，不支持的配置加载期拒绝）、ABI 兼容域）。W2-01 request 缺失时以 `Cs1TileCaps.from_hardware`草案登记（joint seal 前必须收齐 W2-01 真请求或确认以草案冻结，不得静默顶替）。
f.`RUN_BEGIN(run_generation)`（G→T 广播）＋ACK：对应 `begin_launch`（`tile_group.py:1990-2048`）只推进 `run_generation += 1`，**不同步 bump program epoch**——`begin_launch` 明确保留 residency（program_residency 不重置），`program_table.py:90-98`按`epoch` 判 warm、`invalidate_group/invalidate_tile`（`L123-128/L113`）时才递增程序代；RUN_BEGIN 不得把既有 warm 程序冷装化（护 RES-10/T-58）。Tile 侧在 RUN_BEGIN 确认前不得接受任何 `TASK\*\*`；program epoch 的更新只随 `PROGRAM_INSTALL`(新程序)/`PROGRAM_INVALIDATE`消息发生，不随 RUN_BEGIN。
g.`POST_EVENT`（T→G，受限）：**仅 sealed manifest 授权的 Group-event producer 任务**可发；full key（owner/generation/sequence/scope）＋一次终态；未经授权＝Loader 拒绝＋RTL 协议 fault（不 no-op，parent 已定）；Tile 本地 event 不自动广播（§13.8）。
   - **相位模型**：P0..P4 语义、accept 集（P1 pre-edge 双向 `valid&&ready`）、P3 staged 规则、`abort`/`drain`/`quiesce` 阶段命令字段（W6-03 所需字段；`TileResetCommand.phase` 对齐 `reset_domain.py:23+` `ResetState` 枚举）以 phase 枚举值引用而非重新发明。
   - **编码**：wire 布局由 `gen_protocol.py` 产出（`cn_messages.h` 供 C ABI、`cn_offsets.svh` 供 RTL stub、`wire.py` 供 Python pack/unpack）；**不冻结硅片 opcode/位编码**（r5 §13.1），线 opcode 只为联仿消息定序，声明为 `protocol_only=true`。位宽由 `derive_widths.py` 从 `lifetime_bounds.yaml` + W1 设计 §6 预算推导，禁止在本计划凭空指定；每个推导出的位宽必须回写来源（当前代码常量或 config 字段名），来源缺失＝推导失败。
2. `channel_budgets.yaml`：逐账本（channel credit/准入+MEM 事务槽/终态引用/PROGRAM 字节窗/NoC flit credit 分立）给出守恒式与容量来源；容量数值取当前 `HardwareConfig`/`GroupSchedulerConfig` 常量实测记录（如 `dispatch_capacity/default 16`、`event_capacity/4096`、`max_drain_cycles/100`），W0-04 的 cs1_c1/c2 config 落地后由 `freeze_check.py` 复核；缺来源＝冻结失败，不落"空待定"占位（合同禁项：物理编码/容量须有实证冻结任务而非空待定）。
3. `lifetime_bounds.yaml`：每序号/ID 域的窗口上界与回绕规则，来源＝实测代码推导（`event_table.signal` 的 stale 判据、`dispatch_capacity` 在途上限、`context_pending_capacity` 等）；`derive_widths.py` 输出 `w<bit>` 位宽。**位宽推导公式（parent 修正）：`2^w > max_in_flight + rate × lifetime`，同一域统一以**消息条数**计（lifetime 以 tick 为单位时先乘该域每 tick 到达率上界，禁止把 cycle 数与条数直接相加——单位混加视为推导错误）**；回绕前必须全部旧义务闭合（受控排空、无同位型旧消息/ACK/占位存活，CTX-06），**不用位宽逃避 ABA**；推导失败＝该域扩宽任务立项（来源登记），**不写空待定**；`message_seq` 等每 tick 恰一窗口的窄域回绕由 `GEN_WRAP_QUIESCE` 前置确认。
4. **`csr_name_map.yaml`**（CS1 需要，但**不在本计划捏造硬件 opcode**；CSR 只做"逻辑名+所属指令字段域"登记，位编码随 W2/W3 的 ISA 冻结）：
   - 上游已占（**实证读自固定上游 pin 文件** `hdl/chisel/src/coralnpu/scalar/Csr.scala` f795221 `CsrAddress` 枚举，脚本 `extract_upstream_csr.py` 落 `protocol/upstream_csr_occupied.json`，禁止手抄数字）：MCONTEXT0..7=0x7c0..0x7c7、MPC=0x7e0、MSP=0x7e1、MTYPE=0xc23 等。
   - 新增声明：`nexus.mctx.*`（W2-01 配置 dispatch snapshot）、`nexus.msnap.*`、`nexus.mregion.*`（START/END/FIRST/CONTINUE/FINAL pending metadata 绑定，W3-01）、`nexus.missue.*`（W3-01/02）等仅登记**名称/宽/访问级/所属 Context**；位编码、CSR 地址在 W2-01/W3-01 的 `-02` 段经 `csr_allocator.py` 冻结：以 RISC-V priv/custom 窗口为骨架，在 `0x7c8..0x7db`（MCONTEXT 区后空档）等未占窗口首次适配，避开上游 `0x7c0-0x7c7/0x7e0-0x7e1/0xc23` 与标准段；分配记录进 `csr_name_map.yaml` 供汇编器生成，**W2/W3/W5 手写数值被 `csr_lint.py`（检查生成物 hash 一致性而非 source-text grep，输出 diff→fail）阻断**。
   - 冲突核查：`csr_lint.py` 读 `upstream_csr_occupied.json` + 新分配表，任何重叠/错位（如把 R/W 位序弄反）→ FAIL 并指名占用者。
5. `gen_protocol.py` 输出：C 头（`cn_messages.h`：消息 enum + 每消息定长 wire struct + `sizeof`/offset 断言）、`cn_offsets.svh`（SystemVerilog 局部参数）、`pipeline_validator/cosim/wire.py`（pack/unpack，Python 侧唯一权威）；全部从 YAML 生成，手改生成文件→`freeze_check` FAIL。
6. `freeze_check.py`+`PROTOCOL_FREEZE.json`：登记每个 -01 包的产物路径+hash；**共同 freeze 产物不要求另一 -01 先 DONE**（jointfreeze gate 非互等）。**冻结规则**：`PROTOCOL_FREEZE.json` 是一次性 seal 产物——全部输入（各包 field_request/contract fragment 落盘＋hash）收齐一次后 seal，seal 后**不可追加修改**；seal 前的登记状态叫 `collected`（W1-01 可先完成自身 producer 产物，但未收齐不得叫 frozen）；任何输入变更→重跑 `freeze_check.py` 产新 hash 的新 seal（`protocol_seq+1`，旧 seal 归档不可复用）。`csr_allocator.py` 分配序固定（RISC-V custom 窗口升序首适配），不随文件遍历漂移；上游已占地址（MCONTEXT0..7/0x7c0..0x7c7、MPC/0x7e0、MSP/0x7e1、MTYPE/0xc23）在 `upstream_csr_occupied.json` 保留并参与 collision 检查，版本变更→重 seal。
7. `derive_widths.py`：`lifetime_bounds.yaml` → 每字段位宽 + 溢出守恒式（单位一致：条数/tick 速率×tick 寿命，§7.2 项 3）；**拒绝规则：仅"位宽不足以覆盖 max_in_flight+rate×lifetime 窗口"的域拒绝生成**（不搞 slot<2 一刀切——c1 单 slot 合法）；负例：把条数与 cycle 数直接相加的公式输入→推导拒绝。`message_seq` 回绕由受控排空+`GEN_WRAP_QUIESCE` 规则而非位宽"够大"解决（合同禁项：不通过"位宽足够大"逃避 ABA）。

### 7.3 消费者/顺序

| 产物                                      | 消费者任务                                                | 消费动作                      |
| ----------------------------------------- | --------------------------------------------------------- | ----------------------------- |
| `cn_protocol_v1.yaml`/冻结 hash           | W1-02/03/04/05；外部包按 §6 表                            | import 生成 wire.py；对拍哈希 |
| `channel_budgets.yaml`                    | W1-03（stub FIFO 深度）、W1-04（协调器账本）、W4-05/W6-02 | 容量注入                      |
| `lifetime_bounds.yaml`+`derive_widths.py` | W1-03 stub 布线、W4/W5 硬件 RTL 参数                      | 位宽参数注入，禁止手写        |
| `csr_name_map.yaml`                       | W3-01（声明 CSR 逻辑名）、W5-01（汇编 helper）、W2-01     | 名称引用，无自造数值          |

### 7.4 步骤（机械可执行）

1. 从 r5 §13.1 表 + W1 设计 §4.2 抄录**逻辑字段**到 `cn_protocol_v1.yaml`（逐字段标 `owner` 包与 `source` 设计节）。
2. 写 `channel_budgets.yaml`：从当前代码常量（§3.1 行号）逐项记录初值＋来源文件:行号（`dispatch_capacity=16 tile_group.py:1173`、`event_capacity=4096 config.py:458`、`max_drain_cycles=100 reset_domain.py:89` 等）。
3. 写 `lifetime_bounds.yaml`：逐序号域给窗口上界（当前代码在途上限实测值）。
4. 写 `csr_name_map.yaml`：`extract_upstream_csr.py` 自固定上游 pin URL（`https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/scalar/Csr.scala`，只读 URL 即可，无需 W0-01 DONE）抓取占用表；登记新逻辑名（只名称，无数值编码）；父已确认：上游已占 MCONTEXT0..7/0x7c0..0x7c7、MPC/0x7e0、MSP/0x7e1、MTYPE/0xc23 **不可默认 0x7c0 空闲**；冲突清单落 `upstream_csr_occupied.json`。
5. 写 `derive_widths.py` + 运行一次生成位宽（以"来源＝当前代码常量"登记）；位宽全部由工具输出，禁止计划正文写死数字；每字段标 `source` 文件/常量名（实测非编造）。
6. 写 `gen_protocol.py` 三份生成物；`freeze_check.py`；`PROTOCOL_FREEZE.json` 为**一次 seal**：待全部必需包 field_request/fragment 收齐后 seal（缺者＝阻断，**不冻结**）；各 -01 以自身完整产物先 DONE，joint seal 可晚于个别 -01 的 DONE；seal 后任何输入变更→新 hash 新 seal，旧 seal 归档不可追加。
7. 字段请求汇聚：读 `field_requests/*`（如已存在）合入；W1 自身新增消息（STREAM/TERMINAL_ACK/EVENT_NOTIFY/EVENT_RETIRE/QUERY_CAPS/RUN_BEGIN、以及 W6-01 fragment 请求的 `GEN_WRAP_QUIESCE(ContextKey)→QUIESCE_RESP`，Tile 只查不超时、不自行清空）：先以 `status=draft` 登记并上报 parent 审；**parent 批准后**才把条目标 `status=frozen`（这些消息的语义/消费者在 W2/W3/W6 组件设计有落点，不算 w1-inferred 无主字段）。
8. W6 fragment 合并方式：`gen_protocol.py` 读取（不复制）W6 计划中的 local event/reset 两份 fragment（`protocol/w6_event_facility.contract.yaml`+`w6_terminal_reset.contract.yaml`，最终路径以 PlanW6 计划落盘为准）；fragment 内容/hash 由 W6 维护，W1-01 不改 fragment、seal 后变更走新 seal（§7.1），冲突经消息与 W6/main 裁决。

### 7.5 验证（全部【未来命令】；`test_w1_protocol.py` 尚未创建，不得标注"当前可执行"）

- **正例**：`wire.py` pack→unpack 往返逐字段相等；`derive_widths.py` 覆盖 `lifetime_bounds` 全部条目且无缺 `source`；`gen_protocol.py` 幂等（两次运行字节相同）；`freeze_check.py` seal 一致性通过。
- **独立正例（跨语言对拍）**：`cn_protocol_stub.sv`（W1-03 早期产出即可）消费 `cn_offsets.svh` 解码 1 条 TASK_PLAN wire blob，与 Python pack 结果逐 byte 相等（fixture 落盘 hex+比较脚本）。
- **负例**：a) 手改生成文件→`freeze_check` hash 不匹配 FAIL；b) `derive_widths` 对"位宽覆盖不了 max_in_flight+rate×lifetime"的域→拒绝生成；对"条数与 cycle 数混加"的公式输入→拒绝（单位一致检查）；c) `stream` 族消息缺 `queue_generation` 字段→schema 校验拒绝（完整性自检，防手写漂移）；d) `csr_lint` 检测到新生成 CSR 与上游地址冲突→FAIL 并指名占用者；e) `gen_protocol` 读 YAML 缺失 `source` 字段→FAIL（阻断后续包手抄数值）；f) joint seal 时缺任一必需包 field_request/fragment→阻断（不冻结，不 w1-inferred 顶替）。
- **退出条件**：上述(在依赖就绪的范围内)全部 PASS 且 joint seal 产新 hash；缺包阻断状态显式记录为 `collected-not-frozen`，不标 DONE。失败即本任务 FAIL，不得带病冻结。

### 7.6 单元驱动与真实 C ABI/CS1 命令的区分

- W1-01 的验证全部为**单元级协议**（无 RTL、无 C ABI 调用）；`cn.sh unit --suite w1`【未来命令】只执行这些协议单元。真实 C ABI 链路（W1-03）和真实 CS1 命令（W1-05）的验收各自独立，不共享验证器，也不允许"stub 通过=协议通过"的混账。
- stub Tile（`cn_protocol_stub.sv`）只在 W1-03 单元使用，**永不**出现在 `cosim`/`gate` 命令输出；`cn.sh cosim` 对 stub 硬编码拒绝（防混账）。

### 7.7 不变量（W1-01 冻结后对全链生效）

- `message_seq` 严格递增、不复用（回绕经 `lifetime_bounds` 排空规则）；op_seq 不得当传输层序号使用；`txn_id` 分域不复用。
- 通道预算互不挪借：任何消息只允许消耗其 channel 的 credit，TERMINAL_ACK 归还仅针对其 txn 对应账目。
- 无对象跨边界：`encode()` 拒绝一切非 POD（list/dict/Python 对象）字段。
- 编码禁止：不产生硅片 opcode；`csr_name_map` 只登名称/访问级，数值地址全部来自 `csr_allocator` 生成物。

## 8. W1-02 Python TilePort 及外部服务边界（P0；硬前置：W1-01）

### 8.1 输入

W1-01 冻结 hash；`pipeline_validator/`（`tile.py`、`tile_group.py`、`runtime/*`、`memory/*`）现有接口（§3.1 证据）；W2/W3/W4/W5/W6 组件设计的接口节。

### 8.2 产物（文件/模块/API）

新 `pipeline_validator/cosim/`：

| 文件                                                                                               | 内容                                                                                                                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wire.py`【W1-01 生成物，W1-02 只消费】                                                            | 消息 enum/字段 pack/unpack                                                                                                                                                                                                                                                                                                 |
| `tile_port.py`                                                                                     | `Protocol TilePort`：`tile_id`、`backend_kind∈{python,rtl}`、`send(msg)`、`poll_inbound() -> bool`、`outbox_ready() -> bool`、`step(cycle)`、`reset()`、`snapshot()`                                                                                                                                                       |
| `python_backend.py`                                                                                | `PythonTileBackend(TilePort)`：**保留现 ComputeTile 语义**的适配层——`send(msg)` 仅入队/记录意图（**绝不调 `UCE.step`、绝不 advance**，防双 tick）；`step(cycle)`（P4 内每 Tile 恰一次）才驱动 `TileUCE.step/plan_admission/commit_admission/load_program/_on_phase_signal`（行号同 §3.1）；用于纯 Python Tile 路径回归不破 |
| `rtl_backend.py`                                                                                   | `RtlTileBackend(TilePort)`：驱动消息队列/credit/`message_seq`，对接 C ABI 收发的 TilePort 实现侧；Tile 侧语义（FSM 细节）归 W2-W6 各自的 RTL，本文件只承载**边界状态机+消息编解码**（无计算路径，其 step 为边界 tick 的 P0-P3 参与，不是执行 owner）                                                                       |
| `memory_service.py`                                                                                | MEM_REQ→`TransferManager` 分发与响应；页授权检查点接入 W4-01（GAT-20）；终态记账                                                                                                                                                                                                                                           |
| `program_service.py`                                                                               | PROGRAM 消息执行：INSTALL 头→READ_REQ 分段→ByteStore 提供真实 image 字节（W5-04 冻结后接 `program_table` 驻留）；INSTALL_ACK/READY                                                                                                                                                                                         |
| `event_service.py`                                                                                 | EVENT 消息执行：POST_EVENT/PHASE_SIGNAL→事件表（signal 语义）、EVENT_NOTIFY/EVENT_RETIRE 分发回 Tile、RESOURCE_UPDATE                                                                                                                                                                                                      |
| `caps.py`                                                                                          | QUERY_CAPS/CAPS_RESP，能力协商                                                                                                                                                                                                                                                                                             |
| `reset_phase.py`                                                                                   | TILE_RESET 相位命令桥（`ResetState` 枚举映射，语义归 W6-03）                                                                                                                                                                                                                                                               |
| 共享接线：`simulator.py`/`tile_group.py`（tick 拆分）/`device.py`/`runtime/group_port.py`/`cli.py` | 唯一集成点归 W1；语义级改动归原 owner 包，冲突上报（见 §8.2 切分表）                                                                                                                                                                                                                                                       |

需切分的现 `tile_group.py` 函数（拆分后语义等价、纯 Python 路径回归不破）：

| 现函数                                                                           | 现行号                  | CS1 拆分                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TileGroup.__init__`                                                             | 198-368                 | Tile 构造工厂按 `backend_kind`（python→`ComputeTile` 现状；rtl→`RtlTileBackend`，不再构造 `ComputeTile`）；`byte_store.register_cache` 仅 python Tile；`transfer_manager` 仍构造但 tile-local legs 的执行 owner 只剩 Python Tile 侧                                                     |
| `dispatch_role`                                                                  | 1159-1227               | 保留全部校验；Tile 无关部分不变；`can_dispatch_role:1154` 的 `t.can_accept_context` 改经 TilePort（rtl Tile 用 `CONTEXT_ACCEPT` 消息/RESOURCE_UPDATE 缓存，不直调 UCE）                                                                                                                 |
| `_step_task_admission`+`_commit_route_task`                                      | 1229-1350               | **此整块移交 W1-04 `admission.py` 协调器**（异步 request 经 `TilePort.send()` 队列；`TilePort` 不提供同步 `plan_admission/commit` 直调，见 §8.5）；`route.admissions[tile_id]`（现 `TileAdmission.l1_handles`，W4-01 消费）对 python Tile 保留、对 rtl Tile 由 coordinator 票据记录替代 |
| `_step_tiles`                                                                    | 1669-1738               | 拆 `_step_python_tiles`（保留现行为）+`_service_rtl_inbound`（消费 tile→G 的 MEMORY/CTRL/EVENT/PROGRAM/STREAM 消息）+`_finish_task_retirement`（父侧 L2 pin/R lease/route 记账，对 `TilePort.backend_kind==rtl` 的 L1 部分改为经 `retirement.py`）                                      |
| `release_context_memory`                                                         | 1049-1125               | 保留语义；对 rtl Tile 的 `l1_allocator/l1_frames` 访问改为 TilePort `retire`/`cancel` 消息，Tile 本地清理由 Tile 侧 RTL 完成；Python 侧仅释放父 L2 pin/lease                                                                                                                            |
| `reset`、`_reset_freezes_new_work`、`trigger_fault`、`on_scheduler_fault`        | 2235-,1848-,2364-,2354- | 保留；Tile 内 UCE/engine 访问改为 `TilePort.request_cancel/reset`（W6-03 语义，W1 只接通信）                                                                                                                                                                                            |
| `validate_transaction_generation`、`_acquire/_release_l2_transaction_references` | 417-457,370-414         | generation 部分（run/profile）保留；`tiles[pool].l1_allocator.assert_live(handle)`（L449）改为按分配代 + TilePort `RESOURCE_COMMIT_IND` 账本对账（RTL Tile 的 L1 handle 校验移交 Tile 侧，Python 仅做父账本核对）                                                                       |
| `_finish_sequencer_retirement`                                                   | 1833-1846               | `tile.retire_external_events` 改 TilePort `EVENT_RETIRE` 广播                                                                                                                                                                                                                           |

### 8.3 独立验证（当前 vs 未来命令分开）

- **当前可执行（不依赖任何新文件）**：W1-02 开始前先落**行为等价基线**——`python -m pytest pipeline_validator/tests/ -q`（当前通过集），并记录纯 Python 4 样例（现有 transformer prefill 语料 2 个 + `examples/scenarios/l2_admission_wait.mlir` 等）的 cycles/PMU/credit_invariant/trace 偏序 hash 到 `hardware/coral_nexus/protocol/baseline_python.json`【未来产物；当前只登记命令与预期字段，不谎称已写】。
- **改造后等价**：`python -m pytest pipeline_validator/tests/ -q` 通过 + `baseline_python.json` 字段逐项 equality（cycles、每 tile PMU 命名计数、credit_invariant、trace 事件偏序）。**退出条件**：等价成立 → 纯 Python Tile 路径行为未变。
- `TileProtocolStub`（**仅单元**，未来文件 `pipeline_validator/tests/test_w1_port.py`）：Python 侧 `TilePort` Protocol 的结构一致性（mypy/手工实例化 stub 走 pack/unpack round-trip，不调 C ABI、无 Verilator）；与"真实 C ABI 驱动 stub Tile"（W1-03）显式分层：stub 端点永远标 `"backend_kind": "stub"`，任何 CS1 场景引用 stub 报告即 FAIL。
- 单元消息一致性负例：`PythonTileBackend` 收到 STALE op_seq 的 TASK_PLAN（来自旧 launch_generation）→拒绝并计数，不更新状态（对齐 `_on_phase_signal` stale 分支与 event_table 行为）。
- MEMORY 服务正确性（以真实排空/字节/verdict 行为断言，不以 wiring/echo/非空通过）：Tile 发 `MEM_REQ`（L2 地址/范围/权限）→ `memory_service.py` 校验（W4-01 语义）→ TransferManager/ByteStore 真实读/写 → 输入 bytes 与 oracle 相等断言 → MEM_RSP 带 `write visibility scope` → MEM_TERMINAL_ACK 归还终态引用、`unclosed` 清零。**负例**：伪造已释放 view 的 txn_id→拒绝且账本不变；credit 归还伪造→守恒断言失败。
- **PythonBackend 执行时序**：`PythonTileBackend.send()` **不调用 UCE.step**（否则双 tick）；advance 全部发生在 `step(cycle)`（P4 内恰一次）；该属性有单元断言（send 后 `pmu.total` 不变、step 后 +1）。

### 8.4 消费者/顺序

`memory_service/program_service/event_service/caps/reset_phase` 被 W1-04 协调器调用；`wire.py` 在所有方向编解码。**关键顺序依赖（防环）**：W3-04/W5-04 各自的 -03/04 任务消费本包的 `program_service/event_service` 端点定义并自证语义，随后 W1-05 汇合；不允许"W1-05 先行"式反向等待。

### 8.5 不变量

- **执行 owner**：被替换 Tile 的唯一执行 owner 是 RTL（`rtl_backend` 无任何 UCE/engine 计算路径）；`PythonTileBackend` 是**未替换 Tile 的合法执行 owner**（`step(cycle)` 内驱动 ComputeTile 一步），但其 `send()` 不得 advance（双 tick 防），执行与 advance 都只在 `step()`。
- 对象永不跨界：`send()` 类型检查（memoryview/TaskKey dataclass）外其他 Python 对象→TypeError。
- 通道 credit 守恒在 `tile_port` 层逐 tick 断言（`credits + in_flight + occupancy == capacity`）。
- **异步准入消息接口**：准入口经 `TilePort.send()`/入站轮询（异步 request），`TilePort` 不提供同步偷增的 `plan_admission/cancel` 直调（TileGroup 现直调只在 `PythonTileBackend.step`/`PythonGroupController` 的父侧保留）；API 表（§8.2/§10.2）与此一致。

## 9. W1-03 C ABI/共享库/五相位（P0；硬前置：W1-01；W0-02 提供容器与 Verilator 工具链）

### 9.1 产物（文件精确签名/所有权/生命周期）

`hardware/coral_nexus/sim/cabi/cn_abi.h`（**最终权威命名**，与早前私信 `cn_wire.h/CnWire.scala/libcn_sim/bp_end/FREEZE_INDEX` 命名不一致处一律以本文件/本节为准；低级实现模型只读本节，禁止两套命名并存）：

```c
/* ---- cn_abi.h 最终权威（整块替换旧草案；无别名、无 CMake、命名不可微调）----
 * 生成产物：hardware/coral_nexus/protocol/generated/{cn_messages.h, cn_offsets.svh, cn_layout.json}
 *           （cn_wire.h / CnWire.scala 命名作废，不双轨。）
 * Python:   pipeline_validator/cosim/{abi.py, wire.py}
 * 库：      libcn_cosim.so（stub 版 libcn_cosim_stub.so 仅 cn.sh unit 使用）
 * teardown：bp_destroy（bp_end 命名作废）。
 * 冻结：    PROTOCOL_FREEZE.json（FREEZE_INDEX.json 命名作废）。 */

#include <stdint.h>
#include <stddef.h>

#define CN_ABI_VERSION 1u

struct CnSim;  /* 不透明句柄：C 侧唯一拥有。 */

/* 每 tile×channel×direction 一个端口 slot；slot_id 布局由
 * protocol/generated/cn_layout.json 生成（不手写编号）。 */
struct CnSlot {
  uint32_t slot_id;
  const uint8_t *bytes;  /* 调用方内存；bp_drive 内即拷贝/消费 */
  uint32_t len;          /* 字节长度；超出该 slot 容量 → CN_ERR_RANGE */
};

/* bp_drive 输入批（调用者预分配；C 侧同调用内拷贝完即返回，不保留指针） */
struct CnIoVec {
  struct CnSlot *slots;
  uint32_t slot_count;
};

/* bp_collect 输出接收缓冲（**调用者预分配**；C 侧拷 necessary accepted beats in） */
struct CnCollect {
  uint8_t *buf;
  uint32_t buf_len;
};

/* 全部 entry 的最终清单（parent 总合同 bp_begin/bp_drive/bp_tick/bp_collect 保留；
 * 本清单完整，无“可选/等价 entry”，实现不得删略或另造别名）： */

/* (0) ABI/协议身份：返回 CN_ABI_VERSION，并在 out_hash 填
 *     **128 字节 hex**＝sha256(cn_protocol_v1.yaml)64 hex ＋
 *     sha256(cn_layout.json)64 hex（两摘要直接拼接，无分隔符、无 NUL 于中段，
 *     末尾 NUL 由调用方缓冲保证——调用方给 129 字节；C 侧只写前 128+NUL=129）。
 *     布局以本注释为准（旧 96/48 起算的写法作废，同步 ctypes 解析）。
 *     必选 entry。 */
uint32_t bp_abi_info(uint8_t out_hash[129]);

/* (1) 会话开始：**复制 cfg 整体内容**（深拷贝：字符串/数组全部复制），
 *     返回后不保留任何 Python 指针；调用者 cfg 可立即释放。
 *     内部完成 Verilator 对象/所有共享 slab 的 malloc（此后每 tick 零分配）。
 *     必选 entry；返回 NULL = 参数非法/依赖缺失（errno 保留）。 */
struct CnSim *bp_begin(const struct CnBeginCfg *cfg);

/* (2) 布局导出：bp_begin 后**恰一次**；填 caller 预分配的 slabs 数组，
 *     述每个 slot 的 offset/len（cn_layout.json 生成，禁手写）。
 *     必选 entry。 */
int bp_layout(struct CnSim *sim, struct CnSlotLayout *slabs, uint32_t count);

/* (3) P0 单独入口：每 tick 恰一次，**先于 bp_tick**；写共享 slab+置输入引脚，
 *     不 eval 不改变模型状态。**缺 bp_drive 即调 bp_tick = CN_ERR_STATE**
 *     （P1 输入必须是上一 bp_drive 冻结内容，bp_tick 不隐式 drive）。
 *     必选 entry。 */
int bp_drive(struct CnSim *sim, const struct CnIoVec *vec);

/* (4) P1→P3：eval/采样 pre-edge 唯一握手判定点→恰一次上升沿→
 *     eval/仅提交 P1 接受集（staged 输出仅登记为下 tick 候选）。
 *     不做 drive、不执行 P4。
 *     必选 entry。 */
int bp_tick(struct CnSim *sim);

/* (5) **P3 后、P4 前**恰一次：把必要有效 accepted beats 从 C 侧缓冲
 *     拷入调用者预分配的 collect 缓冲（len 不足 = CN_ERR_RANGE 不截断）。
 *     必选 entry。 */
int bp_collect(struct CnSim *sim, const struct CnCollect *out);

/* (6) 非阻塞状态导出（只读 POD）：fatal_code/相位计数/ledger 计数/
 *     message_seq_gap_seen/accepted_but_unreturned_credit；任意时刻可读。
 *     必选 entry。 */
int bp_status(struct CnSim *sim, struct CnStatus *out);

/* (7) teardown：释放全部 C 侧 malloc（C 侧唯一拥有）。必选 entry。 */
void bp_destroy(struct CnSim *sim);
```

错误码（`int` 返回，`0`=成功）：`CN_ERR_ARGS`（参数非法）、`CN_ERR_STATE`（调用序错误：漏 bp_drive、重复 collect 等）、`CN_ERR_RANGE`（collect 缓冲不足/slot 超 len）、`CN_ERR_ABI`（hash/布局校验失败）、`CN_ERR_FATAL_DUT`（可捕获的 DUT fatal；不可捕获 crash 则进程直接退出，无封窗记录→§11.1 项 5 判 INCOMPLETE）。

所有权与生命周期（**明确单线程、Python 唯一时间权威**；与 §9.1 最终签名一致）：

- `struct CnSim`、Verilator 对象、C 侧共享 slab（drive/collect IO）由 **C 侧 `bp_begin` malloc、`bp_destroy` free**；Python 侧**只持指针引用、禁 free**。
- `bp_drive` 的 `CnIoVec`：C 侧**在该调用内即拷贝/消费**，返回后 Python 可释放/复用；C 侧不留 Python 指针。
- `bp_begin` 的 `CnBeginCfg`：`bp_begin` **整体深拷贝 cfg**（含字符串/数组），返回后不保留任何 Python 指针。
- `bp_collect` 的 `CnCollect`：调用者预分配；C 侧**P3 后、P4 前**恰一次把必要有效 accepted beats 拷入；len 不足＝`CN_ERR_RANGE` 不截断。
- **无回调**：C 侧不回调 Python、不执行 P4、无 socket/共享内存/DPI 备选；不提供超时自动重试；无别名/兼容入口。
- 共享 Python 文件 owner：本包不动 `pipeline_validator/` 源码；`abi.py`（ctypes 绑定包装）只 import `cosim/wire.py` 与 frozen `cn_layout.json`。

### 9.2 Verilator 共享库封装（cn_abi.cc）步骤

1. `VnCosim*` 由 Verilator 4 Tile top（W2/W3/W4/W6 的 Tile RTL，W0-01 submodule f795221 已锁）生成；混编路径 `hardware/coral_nexus/sim/cabi/BUILD.bazel`；单元 stub top `cn_protocol_stub.sv` 与真 Tile top 同接口替换（stub **仅单元**，不得用于 CS1 判定，§7.6）。
2. `bp_begin` 步骤：`Verilated::commandArgs`；`contextp = new VerilatedContext`（eval 调用一律 `contextp->eval()`/`top->eval()` 与 Verilator pin 版实际 API 为准——**实现时以生成头文件为准，计划不锁写调用形式**；`tracefile` 按需（`sim.trace` 开启时）打开 VCD；`trace` 关闭默认；`randReset` 关闭（**确定性**，全模块显式复位）；返回句柄。**`bp_begin` 复制 `CnBeginCfg` 内容后不再持有任何 Python 指针**（调用者 cfg 可立即释放）。`bp_layout` 随后恰一次，Python 侧缓存 slab 视图（此后每 tick 不再 alloc）。
3. `bp_drive`（P0，Python 每 tick 恰一次，**先于 bp_tick**）：将 drive 集（通道消息+本 tick 到期响应、credit/ready 信号）写入共享 slab 并置输入引脚（不改变模型状态，仅输入引脚）；**缺 bp_drive 即调 bp_tick = 状态错误**（P1 输入必须是已冻结的 drive 内容）。
4. `bp_tick` 只做 P1-P3（对齐 W1 设计 §3.1）：
   (a) **P1**：eval → 采样 valid/ready/payload，**唯一握手判定点**（post-edge ready 不得配 pre-edge payload）；`payload/身份快照` 复制进 pending result（**仅复制 P1 已握手接受集**，不复制未握手 staged 数据）；
   (b) **P2**：`clk=0→1` 恰一次；**每 tick 至多一次 clk 上升沿**（VER-18）；
   (c) **P3**：eval 后只**提交 P1 接受集**到输出队列（staged 输出仅作下 tick 候选登记、**不作本 tick accepted**），`clk=1→0` 收尾。
   (d) `bp_collect`（**P3 后、P4 前，由 Python 调**）：将必要有效 beat 从 C 侧缓冲**拷入调用者 `bp_collect` 预分配缓冲**（调用者给 `len`，不足即错误码、不截断，不每 tick 分配）；`bp_status` 任意时刻可读，不阻塞。
5. 全部 C 侧函数**禁回调 Python**；**不保证 Verilator fatal 可 catch 永不 abort 进程**（以实际 pin 版本行为 probe 记录）：可 catch 的断言/fatal → 填 `CnStatus.fatal_code` 返回码；不可 catch 的真 crash → 无封窗记录，由 §11.1 项 5 判 INCOMPLETE。**协议故障不等于 C 侧 crash**：Tile 侧协议 fault 仍走 Group fail-stop 排空（`run` 循环继续 tick 至 drain 完成），不一概停止 tick；
6. 编译产物：`libcn_cosim.so`（默认）+ `libcn_cosim_stub.so`（W1-03 单元专用）。**编译命令**【未来命令，待 W0-02 落地】：`bash hardware/coral_nexus/scripts/cn.sh build --config cs1_c1 --out out/coral_nexus/w1-03`；仅 BUILD.bazel 入口，**无 CMake 兼容层、无别名入口**。

### 9.3 五相位 Python 驱动（W1-02 已建 driver、W1-03 打通真 ABI）

- W1-02 的 `rtl_backend.py` 驱动循环在 W1-03 完成后可指向 `libcn_cosim.so`：`pipeline_validator/cosim/abi.py` 加载 .so，`ctypes` 结构体（`CnIoVec/CnCollect/CnStatus`＋`CnBeginCfg`/`CnSlotLayout`）与 `cn_abi.h` 一致性以 `cn_abi_info()` 返回的 hash + `bp_layout` 实测（`cn_layout.json` 哈希校验，防手解漂移）。
- P4 归 `simulator.py`/`tile_group.py`（parent 指示共享接线，W1-02）；本任务只保障 C ABI 侧 P0-P3 相位正确，P4 不在 C 侧出现。

### 9.4 独立验证 stub 单元（当前与未来命令分开；`cn.sh unit --suite w1`【未来命令】）

- `test_cn_abi_phase.py`（未来文件，`pipeline_validator/tests/`）：加载 `libcn_cosim_stub.so`，驱动 `cn_protocol_stub.sv`（**单元级** stub，`backend_kind=stub`，与 CS1 判定隔离）：
  1. 正例：`bp_begin→bp_layout→(bp_drive→bp_tick→bp_collect)×N`，stub 侧 echo 消息在 P3 恰一次提交；随机 ready 拉低（seed 固定）→ `valid&&!ready` 期间 payload 稳定、接受集恰一次、credit 守恒（IF-09）。
  2. 负例：`bp_drive` 与 `bp_tick` 分离语义——漏调 `bp_drive` 就调 `bp_tick` → `CN_ERR_STATE`（P1 必须输入冻结，`bp_tick` 不隐式 drive）；手工构造 `post-edge ready` 与 `pre-edge payload` 错配的向量 →断言拒绝，错误码（`CnStatus.proto_err`，schema 常量 `CN_EP_PAYLOAD_READY_MISMATCH`）。
  3. 负例：`bp_tick` 之间不 collect、连续两次 tick → `staged` 输出不丢（P3 候选下 tick 才可 accepted），接受集无重复提交；未 collect 直接下一 tick drive 的接受集不丢（buffered pending result），连两次 collect → `CN_ERR_STATE`。
  4. `bp_collect` 缓冲不足 → `CN_ERR_RANGE`（不截断）；单 tick 双升沿探针：stub 内嵌 `posedge_count` 计数器经 DEBUG 通道读出，断言每 tick 恰 +1。
- `test_cn_abi_crash.py`：stub 配置 `inject_fatal_at_tick=N`（stub 参数，非生产代码路径）。**两分支各测**：(a) **可捕获 fatal**→ `bp_tick` 返回 `CN_ERR_FATAL_DUT`、`bp_status.fatal_code` 非 0 → Python 侧停 tick、落盘 `INCOMPLETE_FATAL`＋残留 pending set（**不伪造 drain**，不继续 tick 造假完成；协议 fault 仍走 Group fail-stop 排空，不一概停止）；(b) **进程级真 crash**（stub `abort()` 注入，不可 catch 场景）→ Python 进程被终结、无封窗记录 → 外层 wrapper（`cn.sh`）依据"无 run_end.json 封窗"判 `INCOMPLETE`（证据＝已落盘 trace 前缀＋INCOMPLETE 标记），**不得宣称已排空或从初态重跑即等同**（W0 §4 行）。
- 真实 Tile 接入点（G1 阶段）：`bp_begin` 读 `cn_layout.json` 里 Tile top 端口映射（W2/W4 Tile 顶层引脚列表）；**本任务先交付 stub 驱动打通，真 Tile smoke 归 W1-05**（防环：W1-05 用到 1 Tile 真实 loop 时仅依赖本任务交付的 stub 驱动 + W2-05 已就绪的 Tile RTL，不反向依赖 W2-05 DONE）。

### 9.5 消费者/退出

- 消费者：W1-04（admission/retirement 协调器）、W7-02/06（重放/出口，通过 `bp_status` 导出计数）、W6-03（复位相位）。
- 退出条件：stub 单元正负例全 PASS + `libcn_cosim_stub.so` ABI hash 与 `cn_abi_info()` 匹配 + 崩溃注入负例落盘 INCOMPLETE 证据文件。

## 10. W1-04 准入退休/早释放/gate 唤醒（P0；硬前置：W1-02、W1-03）

### 10.1 产物

1. `pipeline_validator/cosim/admission.py`：准入协调器（§3.4 FSM 逐项消息化）：
   - 状态：`PLAN_SENT→HOLD_REQ_SENT→HOLD_ACKED→FINAL_COMMIT_SENT→ACCEPTED（或 COMPENSATE）`；
   - `PLAN`（无副作用版本票据，ADM-01）→`PARENT_PREP`（父 L2 pin 预留+R 配额扣减，账本立即入账——对齐 `_pin_grid_l2:970-1008`/`_task_leases:1253-1256`，占资源防 backing 回收，对调用者未可见仅指未 accepted）→`HOLD_REQ`（Tile §3.5 本地 hold：不 bind 可执行 UCE、不取指）→`FINAL_COMMIT`（父+本地全持有后授权）→Tile 原子安装→`ACK(accepted)` 对外宣布点（IF-13）。
   - **补偿**：授权前任一步失败→`COMPENSATE`（父：unpin 父 L2/退还 R/释放事件预留；Tile：释放 hold 并 ACK；**票据保持到各补偿 ACK 收齐**才释放）；无跨 Tile gang（ADM-03）。
   - **ROLLBACK/CANCEL**：最终授权之后不再 ROLLBACK；取消走 CANCEL_TASK/drain（ADM-04）；两分支（窗口内/外）皆可测（T-20）。
2. `pipeline_validator/cosim/retirement.py`：退休协调器（**无"等父资源结清"前置——禁止环**）：
   - 链序（固定）：Tile 内 `EXEC_DRAINED`（W2 内部）→Tile 发 `RETIRE_READY`（非 Task success，仅本地旧在-flight/VME 闭合）→Coordinator 发 `RETIRE` 授权（**不等待父 R free/pin 结清**——Group 若等 TASK_COMPLETION 而 Tile 等 R free 即为被禁环）→Tile 本地回收（view/Arena/Frame，RTL 内）→Tile 发 `TASK_COMPLETION`（可靠终态 DONE/ERROR/TIMEOUT/RESET＋错误定位；**不等父 R free**）→Coordinator 收到后清**剩余父资源**：`_unpin_task_l2`（幂等，`1352-1360`，早释已解的不重复解）+`_task_leases.remove`（`1704-1711` 现逻辑）+Route retired→root 侧 `retire_context_arena`（`1523-1548`）→sequencer 完成；**Python 不再执行 L1 retire**。
   - 每方条件闭合、不制造环：Tile 侧 RETIRE_READY 条件＝本地 in-flight/MFE/VME/事件义务闭合；Group 侧 RETIRE 授权条件＝收到 RETIRE_READY（父侧不做本地资源等待）；root/Grid 收尾条件＝全 participant TASK_COMPLETION＋自身 L2 view release＋Arena 退休（`retire_context_arena` 全条件，L1514-1548）后发布 root 完成。
   - **输入早释放**：TASK 途中 Tile 可提前发 `PHASE_SIGNAL(input_released, TaskKey)`（≤Grid all_tasks 聚合）；Coordinator 收齐 all_tasks 后 `_unpin_grid_readers`（`1010-1025`）；安全退休时 `_unpin_task_l2` 清理**剩余** pin（幂等，不要求等别人聚合）；已早解 pin 不重复解。**早释放归还范围仅纯读 L2 pin**：不归还 L1/Frame/UCE/R/程序引用，不等于 nest.release。
   - Tile 侧 TERMINAL_ACK 后才可复用终态引用槽（终态引用账本）。
3. `pipeline_validator/cosim/gate_wake.py`：L1 gate 双唤醒：
   - **现状**（实证）：`_step_task_admission:1231`/`dispatch_role:1191` 每 tick 轮询 L1 gate；唤醒令牌 `(admission_version, _slot_version)`（`runtime/group_port.py:355-377`）**不含 L1 gate**（`admission_version` 元组在 `tile_group.py:2070-2078` 无 L1 gate）。注意：`_step_task_admission` 由 `TileGroup.step`（`tile_group.py:1769`）直接调用，**不在** `scheduler.step` 内部，别误称。
   - **CS1（parent 已定）**：`OPEN_ISSUE`（`profile_controller.py:713-715` 置 gate 开）时 Group **自行唤醒**：Route 登记（`dispatch_role` 的 BACKPRESSURE 路径）与 Task 提交（hold 请求）两条等待路径的受阻候选重试（事件驱动 ADM-06/ADM-10，非逐 tick 轮询）；`GATE_NOTIFY`（G→T）另行通知 Tile 侧 hold/issue 状态机。
   - **实现（唯一方案，无旁路）**：直接改 `pipeline_validator/group_scheduler.py`——**仅限 gate 事件唤醒的共享接线**：受阻 BACKPRESSURE 记录打 `wait_gate_epoch` 标记后暂离扫描队列，`gate_wake.register(level)`（`profile_controller._profile_gates` 翻转/`OPEN_ISSUE` 一步内触发）将对应记录推回调度队列并广播 `GATE_NOTIFY`。**边界**：`group_scheduler` 本身的调度语义/准入行为/WAIT 归因的改造仍归相关 owner（W2-04 等）一起进行——W1 不独自重写 scheduler 行为，每处改动附最小 diff 与等效性测试供 parent 复核（本计划的 scheduler 改动点仅 §10.1 项 3 列述的唤醒路径），消费语义的并行改动由 owner 包提 diff；纯 Python 基线经由 §8.3 等价门逐项验证。
4. `pipeline_validator/cosim/caps.py`、`reset_phase.py`：能力协商与 Tile 端复位相位（配合 W6-03）。

### 10.2 步骤（机械可执行）

1. 在 `admission.py` 定义 ` AdmissionTicket`（TaskKey/票号/资源集合/COMPENSATE 完成集）；从 `tile_group.py:1229-1350` 抽取纯函数 `_plan_ticket/_parent_prep/_tile_hold/_final_commit`，保留"无资源变更即无消息"性质（P0 基线零消息）。
2. TilePort/ticket 侧：每 Tile 的 ticket 队列**有界但允许多 ticket 并发**（`channel_budgets.yaml` 的准入事务槽上限；Group 串行管理逐 ticket 推进，**不等于禁止所有并发**），hold 占用计数实时入账、FINAL 授权不超卖（IF-13：父资源+Tile 本地资源的串行有界 ticket）。
3. 消息映射：TASK_PLAN/TASK_COMMIT(=HOLD_REQ)/COMMIT_FINAL/TICKET_ROLLBACK/CANCEL/PROFILE_PREPARE/PROFILE_COMMIT/GATE_NOTIFY/RETIRE/STATUS ↔ Tile 侧 ACK/RETIRE_READY/TASK_COMPLETION/FAULT_REPORT/RESOURCE_COMMIT_IND（W1 设计 §4.2 全表逐字段落 `cn_protocol_v1.yaml`，W1-01 已含）。
4. 早释放：Tile→G `PHASE_SIGNAL(input_released, TaskKey)` 每方向每 Task ≤1 次；`_on_phase_signal` 现有聚合/duplicate/stale 计数逻辑保留（`1391-1477`，不改语义，仅改为 TILE→G 消息触发），`_unpin_grid_readers` 后按已订规则归账。
5. `retirement.py` 状态机：`ACTIVE→RETIRE_READY→RETIRE 授权→LOCAL_RELEASED→TASK_COMPLETION→PARENT_CLOSED`（**无"等父资源结清"状态——禁止环**，§10.1 项 2）；每次转移产生 W7 可读事件（经 `event_service.py` 落 trace）。
6. 共享接线：`simulator.py`/`tile_group.py` 中 `_step_tiles`（`1669-1738`）与退休链重定向为本包协调器；`runtime/group_port.py:355-377` 增 `gate_epoch` 唤醒令牌字段（保持旧调用兼容：无 cosim 时等价旧元组，cosim 时多一维；`admission_version` 本体的 L1 gate 缺口见 `tile_group.py:2070-2078`）。
7. `config.py` 共享接线：新增显式字段 `max_drain_cycles: int = 100`（沿现模型 baseline，`HardwareConfig`/`MemoryTargetConfig` 风格挂靠），`runtime/reset_domain.py:89` 改读该字段（去掉 getattr fallback——显式字段存在后 fallback 仅为防御残留，负例验证显式值生效，如改为 3 时 drain 在 3 步后强制 cancel）；watchdog 逻辑与协议字段经 `freeze_check.py` 校验。
8. POST_EVENT 权威条目：`event_service.py`/`cn_protocol_v1.yaml` 落 sealed-manifest 授权检查（producer 任务 grant 表），未授权 POST_EVENT→Loader 拒绝＋RTL 协议 fault（不 no-op）；Tile 本地 event 不自动广播（§13.8）。
9. TILE_RESET/复位相位：`reset_phase.py` 执行 Group 唯一排序契约——DRAIN_SAFE 不要求 Tile pending events=0（Tile 本地 `PENDING→RESET` 由 W6/RTL 执行并 ACK），Python 只写 Group 侧事件表（§13.9）。

### 10.3 可独立验证（【未来】`cn.sh unit --suite w1`：`test_w1_admission.py`/`test_w1_phase_signal.py`/`test_w1_gate_wake.py` 尚未创建，不得标"当前可执行"）

- **正例**（`test_w1_admission.py`，Python 侧 + stub TilePort）：
  1. 冷启动全链：PLAN→HOLD→FINAL→accepted 时序逐相位断言（ACK(hold) 后 Task 不 bind/不取指；FINAL 授权前无执行；READY 永不授权执行）；**去重键验证**：同一 admission 事务的 HOLD 与 FINAL（同 `txn_id`、不同 `message_type`）不得被误去重——FINAL 必须独立执行（advisor 修正）；ACK 引用对应请求的 `(txn_id, message_type)` 配对唯一可恢复。
  2. 准入 hold→PARENT_PREP 资源：`_pin_grid_l2` 后 l2_sram live pin 计数增加且 backing 回收被阻止（复用 `test_l2_sharing.py` 既有断言形式）。
  3. 补偿完整性：HOLD_REQ 后 CANCEL→账本恢复基线：`_unwind_grid_l2_pins`/`_unpin_task_l2` 后 l2_sram 终态全零（复用 `unclosed_l2_objects`）。
  4. **早释放**：`test_w1_phase_signal.py`：T 液化后发 `PHASE_SIGNAL(input_released)`×4 Tile→聚合恰一次→纯读 pin 全解除；仅部分 Tile 发→不解除；重复发→FAULT（`_on_phase_signal` 现语义保留）；某 Task 安全退休在聚合前发生→仅清自身剩余 pin（幂等）；`_unpin_task_l2` 后已早解 pin 不重复解（幂等）。注入"input_released 且 Task 无 L2 load"→资源账不变。
  5. gate 双唤醒：`test_w1_gate_wake.py`：gate 关→Route 登记 BACKPRESSURE；`OPEN_ISSUE`→`gate_wake` 唤醒→Route 登记恰一次成功（T-21/T-22 语义）；**负例**：gate 恒关+无 OPEN_ISSUE+反复 tick→断言无重试探针（无轮询）。
  6. 退休链：EXEC_DRAINED 前不发 RETIRE_READY；TASK_COMPLETION 带 TERMINAL_ACK 前终态引用账本不减；`Task free 后 second Task` 得 `WAIT_CONTEXT_LIMIT`（T-41）。
- **stub 驱动部分**（W1-03/04 共产，`backend_kind=stub`）：同套测试的 stub-Tile 版（消息真实经 C ABI，非 monkey-patch）；对拍以**消息时序/排空/背压/字节行为**为准：断言相同序列在同一背压与取消注入下产生相同接受集、相同补偿终态、相同 `unclosed` 清零，而非"消息 echo 相等/非空"。
- **未来 CS1 证据**：本任务不宣称 CS1 级 PASS——CS1 级（真 RTL Tile 1/4 Tile）的 T-18..22/T-39..46/T-54/T-57/T-64 真实执行归 `cn.sh cosim` 与 W7-04，本任务交付可运行的协调器与 unit 证据，W7-04 消费。
- **退出**：Python 侧单元全绿 + stub 驱动行为对拍全绿 + 共享文件（group_scheduler.py gate 唤醒、tile_group.py 拆分）接线 diff 供 parent 复核（无"仅命令壳/help"即完成的情况——W1-04 交付协调器实现与可运行验证，非占位）。

## 11. W1-05 4 Tile 与 mixed 集成（P0；硬前置：W1-02/03/04 + W2/W4/W5/W6 各自 RTL 端点就绪）

### 11.1 产物

1. `hardware/coral_nexus/scripts/cn.sh` 的 `cosim` 分派（**总合同唯一宿主入口路径**，W0-02 交付 wrapper、W1-05 交付 `cosim` dispatch 处理函数，不另建 `sim/cn.sh`）：实现 `cosim --scenario <path> --backend rtl --rtl-tile-mask 0xf --config <path> --out <path>`【未来命令】。**交付＝可运行 harness**：完整参数校验（缺 config/场景/schema 版本不符→非零退出＋指名缺失项）、真实执行驱动（`bp_*` 全序列＋P4 外层）、证据落盘管道；**不是"交命令壳+--help"即完成**——无真实 RTL Tile 可接入时本任务保持 IN_PROGRESS，交付以至少一条 `one_tile_rtl` evidence_tier 的实际成功 run 为最低完成线（scenario 消费按 W7-01 schema）。
2. `pipeline_validator/cosim/run.py`：`run_cosim(scenario: Path) -> 0/1/2/3` 退出码（0 全 PASS/1 checker FAIL/2 INCOMPLETE/3 异常退出），供 `cli.py --cosim-scenario` 与 W7-04 调用；**退出码 2/3 永不代表 PASS**（合同：整体失败任务仍失败、超时不是 PASS）。
3. 混合 Tile 门（仅调试）：`python_backend` Tile 与 `rtl_backend` Tile 在同 Group 并存（mixed），仅用于 bring-up 观测，**不代最终门**；mixed 门输出必须带 `evidence: mixed_debug_only` 字段，W7-06 拒绝其作为 CS1 final PASS 证据。
4. 4 RTL Tile 最终门：`cs1_c1`（单硬件 Context）与 `cs1_c2`（双硬件 Context）两个显式配置（W0-04 config 产物）× 全 4 Tile RTL mask 0xf，P0-P4 全链路；**最终门须实际读证据**（`bp_status` 导出的每相位/每通道守恒计数 + trace seq 完整性 + credit_invariant），不检查文件存在。
5. **crash=INCOMPLETE（parent 已定的两态区分）**：run 级 verdict 三态 `PASS/INCOMPLETE/FAULT`。
   - **协议/运行时 fault→仍走 Group fail-stop 排空**：Tile 侧 `FAULT_REPORT`/terminal ERROR 等协议级故障**照常进入既有 fail-stop drain**（`reset_domain` FSM、W6 语义），排空完成且账本清零的 faulted run verdict＝`FAULT`（不是 INCOMPLETE）；**不一概停止 tick**。
   - **真实 crash/不可恢复 C 侧 fatal→INCOMPLETE**：`bp_*` 函数返回 fatal status（`bp_abort` 信号量，**不能保证 Verilator fatal 可 catch 永不 abort 进程**——进程真死时无封窗记录）；run 结束时无 `run_end.json` 封窗（或封窗校验失败）、进程退出码<0、`credit_invariant=False`、`unclosed_l2_objects()` 非空、`bp_status.message_seq_gap_seen`、`bp_status.accepted_but_unreturned_credit>0` 任一成立→`INCOMPLETE`（`incomplete.py` 落盘残留 pending set 明细），非假"drain 完成"；
   - verdict 判据表：`FAULT` 必须附封窗内完整 fail-stop 链证据（RETIRE_READY/TASK_COMPLETION/reset FSM DONE + `unclosed` 全零）；否则只能 `INCOMPLETE`。

### 11.2 步骤

1. cs1_c1 回归（单硬件 Context）：`cs1_c1.yaml`（W0-04 产物，1 Tile RTL 已有）→1 Group×4 Tile 全 RTL→执行 `test_transformer_baseline` 同规模语料的 Python 侧对拍（`python -m pytest pipeline_validator/tests/test_transformer_workloads.py -q`）。
2. cs1_c2（双 Context）：context_count=2，4 Tile RTL；P1/PN 口径（需求 §16.3）：P1=改造后同硬件仅 1 Context 活跃，PN=同硬件多 Context 活跃；两者都使用 cs1_c2 target/同 RTL hash，仅 workload active context 数不同；**cs1_c1 是退化配置功能回归，不得当 P1 与 PN 比较宣扬多 Context 收益**；P0 原核数值等价单列控制/装载差异。
3. 4 Tile 最终门：全 4 Tile RTL；P0-P4；运行 `l2_admission_wait.mlir`+ transformer prefill pipeline 语料（保持 mandatory 能力集：softmax/norm/recurrence/gather 不缩水为 matmul-only smoke）；`assert_l2_closed`/`unclosed_l2_objects`/`credit_invariants_hold` 三项终态检查；早释放、退休链、gate 唤醒、复位（W6-05）、debug（W6-04）在 4 Tile 环境下各跑一条正/负例。
4. mixed：仅 `--rtl-tile-mask` 子集（0x1/0x3/0x5 等部分 RTL）用于 bring-up；每条 mixed run 输出带 `evidence: mixed_debug_only`。
5. 证据落盘：每次 run（unit/stub/1-RTL/4-RTL/mixed/crash）写 `out/coral_nexus/w1/<run_id>/report.json`，含：`evidence_tier ∈ {stub, unit, one_tile_rtl, four_tile_rtl, mixed_debug_only}`、cycles、PMU 峰值、BP 状态快照、credit/账本守恒布尔、消息序完整性布尔、`drained`/`drain_reason`、`unclosed` 对象清单、python 侧对拍哈希（仅 4-RTL tier 要求）。

### 11.3 独立验证（cs1 命令；非 unit）

- `test_w1_integration_gate.py`（pytest 侧入口桥，实际执行经 `cn.sh`）：对 4-RTL tier 报告逐字段断言（`bp_status.fatal_code==0 && credit_ok && message_seq_complete && unclosed==[] && l2_closed && evidence_tier=="four_tile_rtl"`）。
- 崩溃注入门（stub 已覆盖，本级不重复）：4-RTL tier 不做 crash 注入，仅以 `bp_status` 终态与退出码判定。
- mixed 层负例：mixed run 的 report 若被 W7 checker 当为 final PASS 证据→FAIL（checker 侧 W7-03 义务，W1-05 交付字段供其读取）。

## 12. 逐 T 场景与需求族映射（W1 主责行）

| T                                           | W1 内落点                                                                                                                | 备注                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------- |
| T-18/19（profile 切档）                     | W1-02（PROFILE_PREPARE/COMMIT 消息）＋W4-05（profile controller 行为侧归 W4）                                            | 本包只交付消息通道  |
| T-20（提交故障两分支）                      | W1-04（admission 补偿/ROLLBACK 窗口）                                                                                    | 现场注入归 W7-03    |
| T-21（WAIT_CAPACITY 唤醒）                  | W1-04（gate_wake＋RESOURCE_UPDATE），仅经 release/version 事件触发                                                       |                     |
| T-22（consumer 依赖未准入 producer）        | W1-04（等待者零持有不变量＋C5 有界假设）；C5 条件落 `cn_protocol_v1.yaml` 无环断言                                       |                     |
| T-39/40（同任务偏序/乱序对拍）              | W1-05（4-RTL tier evidence）＋W7-02/03 oracle                                                                            | 本包不写数值 oracle |
| T-41（R=1 lease 归因）                      | W1-04（retirement/账本）                                                                                                 |                     |
| T-42/43（占满单 Tile/碎片/槽/空池五类归因） | W1-04（admission wait reason 原样转发五类，`allocator.py:152-157`；空池→PERMANENT 不入等待队列）                         |                     |
| T-44（WAIT_SLOT 定向）                      | W1-04 + TilePort resource_update                                                                                         |                     |
| T-45（phase 重复/旧代/未知/漏发）           | W1-04（`_on_phase_signal` 现语义保留为消息侧入口＋W4/W6 侧语义确认；重复＝协议故障、旧代忽略+计数、每 phase 聚合恰一次） |                     |
| T-46（Prepare/Commit ACK 机制）             | W1-04（profile member ACK）＋W4-05                                                                                       |                     |
| T-54（控制通道独立/无 HOL）                 | W1-01（channel_budgets 独立账本）、W1-03（stub 深度注入）、W1-05                                                         |                     |
| T-57（L2 引用闭合）                         | W1-04（retirement+早释放）＋W4（TransferManager/arena 行为侧）                                                           |                     |
| T-64（owner 唯一）                          | W1-02（`PythonTileBackend` 保留、`RtlTileBackend` 零计算路径，禁 Python 后门）、W1-05（4-RTL tier 断言）                 |                     |
| T-49（cold/warm）                           | 消息通道＋真实装载归 W5-04；W1-02 交付 PROGRAM_RESIDENT 状态位通道                                                       |                     |
| T-58（冷热端到端）                          | W5-04+本包 `program_service.py` 消费                                                                                     |                     |
| T-59/60                                     | CS2/CS3 阶段（合同：明确 N2 子项延后，不把 L2 条目整体延期；本包不交付）                                                 |                     |

需求族其他主锚：IF-04/05/06/09/10/11/12/13（本包）、ADM-01..12、ARN-01..09、EVT-01..16、GAT-15..20 的**检查点落位**在 W4-01；VER-25/T-64 的"对象跨边界禁令"机制在本包。

## 13. 风险与已定项（parent 裁决已落，无 pending 项）

1. **STREAM 消息族**（parent 已定）：STREAM 是现有可选语义（`stream_queue.py`，DMA-13/ISA-08）的消息化，放 **MEMORY 通道的独立逻辑子队列/子预算**，不新增第六物理通道；Group 侧 `StreamQueue` 仍是唯一 writer；Tile 发消息、Group 服务并回应。**核实 RTL 有真实可用 stream 路径**（Tile 内 STREAM\_\* 指令→MEMORY 子队列消息），且不为未用 stream 的配置新增范围：`QUERY_CAPS` 报告 `stream_supported`，不支持时 stream 配置在加载期拒绝（不运行时补合同）。
2. **gate 唤醒**（parent 已定）：**直接改 `pipeline_validator/group_scheduler.py` 事件唤醒**；不做 cosim 旁路第二 scheduler（无 `blocked_route_retry` 队列方案）。**scope 边界**：W1 只做 gate 事件唤醒的**共享接线**（唤起路径与 `wait_gate_epoch` 标记），调度语义/准入行为/WAIT 归因改造仍归相关 owner 包一起进行、各自附最小 diff 与等效测试；W1 不独自重写 scheduler。行为：L1 gate 关闭时的 BACKPRESSURE 记录不再每 tick 重扫，`OPEN_ISSUE` 时事件唤醒登记与提交重试；纯 Python 基线的等价门（§8.3）逐项验证。
3. **`report.py`**（parent 已定）：由 **W7 唯一修改**，消费 W1 交付的只读 ledger/snapshot 字段（W1-04 `retirement.py`/`incomplete.py` 提供 `l1_ledger_snapshot()` 等只读字段，report.py 行 `660-676` 的 L1 快照语义迁移由 W7-01/W7-06 执行；W1 不改 report.py）。
4. **死代码**：`_pending_activations`、`uce_dispatch_per_cycle`、`stream_token_overhead_cycles`、`stream_fence_cycles`、`StallReason` 12 成员与 `primary_stall_owner()` 无 caller 等现象记录为现有 validator 事实（memory 记录一致）；**本计划不额外删除与 cutover 无关的 PMU 死配置**（不超范围），其清理归各包 owner 在各自任务的"删死代码"原则下自证不破坏基线。
5. `DeviceLaunchRequest` 保持 Python 对象入参（CPU 内部）；跨界仅 TaskKey+descriptor/身份字段——**不修改 device.py 数据模型**，改动点仅在 `runtime/group_port.py`（gate_epoch 唤醒令牌字段）与 `cli.py` cosim flags。
6. `reset_domain.py` 的 Tile 侧直调（`_cancel_unissued_engine_requests` 等，`reset_domain.py:138-165`）在 4-RTL 配置下必须全部经 TilePort 消息（W6-03 行为侧）；W1 只保证 TilePort 暴露所需方法且经 `group.step`/`_step_tiles` 拆分后的调用点一致，不重写 ResetDomain FSM。
7. **W6 local event/reset fragment 合并**（parent 已定）：W1-01 合并 W6 计划中的本地 event/reset fragment 进入 `cn_protocol_v1.yaml`；**不追加已 seal 文件**（seal 后变更走新 hash 新 seal，§7.1/§7.2 项 6）。
8. **POST_EVENT 裁定**（parent 已定，写进 `cn_protocol_v1.yaml` POST_EVENT 权威条目）：保留已批准设计的 POST_EVENT 接口，只允许**sealed manifest 授予的 Group-event producer 任务**发出；事件以 full key（owner/generation/sequence/scope）＋一次终态接受；**未经授权的 POST_EVENT 由 Loader/RTL 双侧拒绝**（Loader 编译/装载期拒绝＋RTL 协议 fault），**不静默 no-op、不禁用**；局部（Tile 本地）event 不自动广播 POST_EVENT——只有程序显式发出且其 producer 任务被 manifest 授权才上边界线（对应 `event_table.signal` 的 owner/generation/sequence 校验语义，`event_table.py:400-460`）。
9. **复位排序裁定**（parent 已定，落 `reset_phase.py` 契约与 W1-04 步骤 7）：Group reset 唯一排序权威；**DRAIN（DRAIN_SAFE）不要求 pending events=0**——事件 MARK_EVENTS 在 DRAIN_SAFE 之后（`reset_domain.py:86-102` 现序即如此），Tile 侧本地 `PENDING→RESET` 由 **W6/RTL 自行执行**（本地事件表归 RTL），Python Group 只写 Group 侧事件表（`event_table.reset()` 在 MARK_EVENTS，`reset_domain.py:98-101`）；Tile 侧不等待、不重复执行 Group 侧复位动作。
10. **`max_drain_cycles` 显式化**（parent 已定）：现值仅 `getattr(cfg, "max_drain_cycles", 100)` 隐式默认（`reset_domain.py:89`），无 config 键；W1 在 `config.py` 共享接线中新增显式字段 `max_drain_cycles: int = 100`（**初值沿现模型作实验 baseline，不做任意 derived"最佳"**），`reset_domain.py:89` 改读该字段；watchdog 权威仍在 Group，字段进协议 freeze/checks（`freeze_check.py` 校验 config↔协议字段一致）。
11. `channel_budgets.yaml` 的 `max_drain_cycles` 引用自 `config.py` 显式字段（不再引用 getattr 默认值）。

## 14. 给能力有限模型的实施手册

1. **先读**：父计划 `../../plan.md`（总体 DAG/合同）→ `design/coral_nexus_tile/implementation/W1_Boundary_Cosimulation_Plan.md`（本文件）→ `design/coral_nexus_tile/components/W1_Boundary_Cosimulation.md` §2-§5 → r5 §11.2/§13/§18。**不要读** `BOA_Gemmini_Integration_Design.md`（旧路线）。本计划不依赖 `local://` 或聊天记录路径。
2. **接口缺失**：不自行改共享合同。找到 owner（§6 producer 列）登记最小证据（需要什么字段、现路径哪个函数缺它、行号），经 `protocol/field_requests/*` 草案或消息给 owner；**草案不算冻结**——joint seal 前缺必需包请求＝阻断 jointfreeze（§7.1），继续其余不依赖该字段的步骤即可。
3. **测试顺序**：先写**负例**再写正例（§7.5/§8.3/§9.4/§10.3 已列）；负例必须先失败后通过（e.g. 无 gate 唤醒时 gate_wake 测试先 FAIL，注入唤醒后 PASS）。
4. **禁止**：实现脚手架/mock 代替真实路径；`TODO: implement`；把 `bp_status`/账本守恒/`drained` 写成常量；在 `cn.sh cosim` 输出里出现 `cn_protocol_stub` 的 evidence_tier；把 1-Tile 当 P1、cs1_c1 当 P1 与 cs1_c2(PN) 比较。
5. **何时可标 done**（逐任务）：§7.5/§8.3/§9.5/§10.3/§11.2 各退出条件全部满足且证据文件落盘；任一负例未被实现为可执行断言＝未 done。

## 15. 未来命令汇总（均【待创建/未创建】，不得声称已执行）

| 命令                                                                                                                                                            | 实现任务 | 成功非零退出含义                                                 | 证据产物                                                |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------- | ------------------------------------------------------- |
| `bash hardware/coral_nexus/scripts/cn.sh unit --suite w1 --config <path> --out <path>`                                                                          | W1-03/04 | 1=协议/协调器断言失败（负例设计缺陷或实现错误），2=stub 构建失败 | `report.json`（含每个测试 verdict+hash）                |
| `bash hardware/coral_nexus/scripts/cn.sh cosim --scenario <path> --backend rtl --rtl-tile-mask 0xf --config <path> --out <path>`                                | W1-05    | 同上,2=INCOMPLETE(非假 drain)                                    | `report.json`（`evidence_tier` 字段）                   |
| `python -m pytest pipeline_validator/tests/test_w1_protocol.py pipeline_validator/tests/test_w1_port.py -q`【未来：文件创建于 W1-01/02 后方可执行；当前不存在】 | W1-01/02 | pytest 惯例                                                      | 无文件，仅退出码                                        |
| `bash hardware/coral_nexus/scripts/cn.sh sources verify`【W0】                                                                                                  | （消费） | 协调器引用的冻结 hash 失效                                       | sources manifest                                        |
| `bash hardware/coral_nexus/scripts/cn.sh gate --stage G4 --out <path>`【W0/parent，消费本包 G4 证据】                                                           | （消费） | 4-RTL tier 证据缺失→非零                                         | gate 报告引用 `out/coral_nexus/w1/<run_id>/report.json` |

退出码统一：0＝PASS；1＝断言/checker 失败（run 完成但 verdict 非 PASS）；2＝INCOMPLETE（crash/超时/staged 不完整/账本失恒）；3＝崩溃/被杀（Python 侧未捕获异常、进程异常退出）。
