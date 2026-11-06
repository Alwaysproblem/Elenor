# W1 Tile 替换边界与联合仿真合同（CS1 Boundary / Cosimulation）

> 本组件设计已通过用户 review，推荐方案采用情况与参数冻结任务见[实施总计划](../plan.md)；跨包合同见[总体设计](../Coral_Nexus_Tile_Design.md)，r5 指[需求规格](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。[设计验证记录](../Design_Validation.md)不代表组件已实现。

## 1. 定位、目标和 First Silicon cutline

本组件定义 CS1 阶段"整 Tile 换成 RTL、其余留 Python"的**替换边界合同**：消息集合、状态 owner、时间推进与准入在边界上的表达。边界是完整 Tile 替换（UCE/寄存器/ROB/RVV/VME/LSU/MFE/L1/cache/MSHR/Frame/本地事件/程序 SRAM 全在 RTL 侧），不是任何单个 engine 的代理接口；RTL 范围只到 Tile 内部，Group/CPU Host/L2/cache/HBM/NoC/页授权保持 Python（r5 §18.4.1、VER-11、IF-12）。

输入依据：r5 §11.2（分层准入）、§13（Group—Tile 接口与任务 ABI）、§18（联仿计划）及当前源码（见各节证据链接，行号以当前仓库为准）。上游固定版本 f79522157fc0f1d7311f44e209fc36c9c8b7a039、compiler d715556c3853719e414182c0bf5f4a10f248f1d9。

非目标：不冻结 CSR 地址/opcode/总线/消息二进制布局（r5 §13.1 明示逻辑合同优先）；不做跨 Tile gang 提交或共同 activation（ADM-03）；不做数值 oracle（VER-16）；不引入 Gemmini/共享 AM；不定义 CS2/CS3 的 Group 级边界。

其他组件可直接引用本文件的通道、相位与身份字段：W2 消费 CTRL 任务生命周期与 EXEC_DRAINED（内部通知 W6）、RETIRE/TASK_COMPLETION；W3 消费 EVENT/MEMORY（region 等待唯一 writer 在 W3）；W4 消费 MEMORY 与页授权检查点（GAT-20 CS1 由 Python 授权权威经消息配合）、debug 外存事务等取消结算；W5 消费 PROGRAM 通道；W6 消费 EVENT/FAULT/RESET（事件等待记录唯一 writer，小 debug FIFO/特权导出 owner）；W7 只做 Host 格式/checker/观察，消费边界日志（VER-22）。

## 2. 职责、非职责和 ownership

### 2.1 owner 表（CS1，对应 r5 §18.2.1）

| 状态/服务                                                                                                                                     | CS1 执行 owner                | 证据与说明                                                                                                                                                                                                                                                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 被换 Tile 的 UCE/Context FSM、L1 Arena commit、Slot Frame、本地事件等待、RVV/VME/LSU/MFE 时序、L1/cache/MSHR、程序 SRAM 安装                  | RTL Tile                      | 现 Python 实现 `../../../pipeline_validator/tile.py`（`_UCEContextState` L55-58、`bind_context` L255-290、`step` L300-353）、`../../../pipeline_validator/memory/l1_slot_frame.py`（`FrameState`/`prepare`/`bind` L100-231）；CS1 关闭这些 Python 执行路径（VER-25，T-64）                                                     |
| root 准入、Grid Route 登记、逐 Tile 提交协调、L2 Arena/claim/backing、事件表、Profile 控制器（含 member bus）、NoC/传输/HBM、复位域、PMU 汇总 | Python Group                  | `../../../pipeline_validator/tile_group.py`（`try_admit_context_task` L2097-2202、`dispatch_role` L1159-1227、`_step_task_admission` L1229-1282）、`../../../pipeline_validator/runtime/event_table.py`（`signal` L400-460）、`../../../pipeline_validator/memory/profile_controller.py`（member bus 每 tick 恰一项 L756-769） |
| 程序映像供应                                                                                                                                  | Python 供字节，RTL 安装并确认 | 现 `../../../pipeline_validator/runtime/program_table.py` `ensure_resident` L79-100 为纯解析模型（同步置 `TILE_RESIDENT`、冷热差只在 `runtime/full_memory`）；CS1 改为真实装载端口 + `PROGRAM_INSTALL_ACK` 确认，Python 不代写 SRAM、不代发安装成功（r5 §13.1 PROGRAM_READY 行自述"现有 validator 没有此消息"）                |
| 数值/协议记分板、trace/重放文件                                                                                                               | 仿真 Host（只读影子）         | 不得替 RTL 发计算输出、完成或推进 credit（r5 §18.2.1 表、VER-25）                                                                                                                                                                                                                                                              |

### 2.2 关闭的 Python 路径（CS1 起逐项关闭并在报告列出，VER-24）

- 替换 Tile 的 `ComputeTile.step`/engine 时序、`plan_admission`/`commit_admission`/`abort_admission` 的 Python 执行（保留 Group 侧调用点改为消息交互；`../../../pipeline_validator/tile.py` L1541-1631）。
- `tile.uce._phase_signal_callback = self._on_phase_signal` 对象回调（`../../../pipeline_validator/tile_group.py` L1347）→ 改为 EVENT 通道 `PHASE_SIGNAL` 消息。
- `DeviceLaunchRequest.task` 携带 `ExecTileGroupTask` Python 对象与 binding 映射（`../../../pipeline_validator/device.py` L30-46）→ 边界上只传 TaskKey + descriptor 引用（按 id/hash 索引封印产物）；对象引用仅存在于 Python 内部（IF-11）。
- `MemoryOwner`/`ResolvedMemoryView` 等对象跨 Tile 边界 → MEMORY 通道传身份/地址/范围/权限（r5 §18.2.1 末段）。
- 程序驻留解析计费（`program_cold_load` PMU 直加，`../../../pipeline_validator/tile_group.py` L1350）→ 由真实安装端口的可观测时序替代。

### 2.3 身份与代际字段（逻辑字段，不冻结编码）

| 字段                 | 组成                                                                                                                                                                                                                                                             | 现源码对应                                                                                                                                                                                                                             |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TaskKey（逻辑身份）  | root launch_generation + GridInstanceId(context_name, device_slot, dispatch_ordinal) + logical task_id；physical tile_id 为**单列路由字段**，不入逻辑身份                                                                                                        | `TaskIdentity(grid, task_id)`、`GridInstanceId`（`../../../pipeline_validator/execution_ir.py` L288-306）；lease key `(binding_id, launch_generation, tile_id)`（tile_group.py L1253）                                                 |
| 版本票据             | 线上携带并重验 `(pool_id, pool_epoch, pool_version, extent_pool_version, profile_mode, profile_generation)`；Python 内部的 `ArenaPlan.pool_token`（object，arena.py L61-71）**不上线**，由 Group 侧以 pool_id+pool_epoch 映射回本地 token 后再做 `is` 池身份校验 | `ArenaPlan` 由 `plan_arena` 产出（`../../../pipeline_validator/memory/arena.py` L328-337），`commit_arena`/`rollback` 对同 generation 的版本漂移同样拒绝（L342-359）；pool_token 为对象不能过 C-ABI（IF-11），故线协议只认可序列化身份 |
| ContextKey           | tile_id + context slot(ctx_id) + context generation（槽位复用代）                                                                                                                                                                                                | context generation 与 Arena allocation generation **分域**：后者仅随 Arena commit 产生（`l1_slot_frame.py` L123-125），作为资源/lease 消息字段携带，不并入 ContextKey；Context 复用时槽位代递增以隔离旧事务                            |
| 指令／消息／事务序列 | `op_seq` 标识 Context 内父指令，`subop_id` 标识其展开；`message_seq` 按 (tile, 通道, 方向) 递增；`txn_id` 关联多消息事务                                                                                                                                         | 三类序列不混用；事件自己的 sequence 仍按 `event_table.py` 校验；具体 wire 表见 W1-01                                                                                                                                                   |
| epoch                | program epoch + profile generation + run_generation                                                                                                                                                                                                              | program_table `epoch`、`profile_controller.generations`、`group_port._run_generation`                                                                                                                                                  |
| VME 实例号           | 不进边界身份，仅 RTL 内部/验证可见，软件不指定                                                                                                                                                                                                                   | MAT-19/r5 §9.4                                                                                                                                                                                                                         |

## 3. 微架构和状态机

### 3.1 单周期五相位（本节为 adapter 新合同，非现有 RTL/Python 行为；总体设计 统一）

一个 Python `Simulator` tick = 一个 RTL core clock cycle（VER-18）。当前 Python 同步调用序（`../../../pipeline_validator/simulator.py` L307-331 CPU→Group→harvest；`tile_group.py` `step` L1748-1779 NoC→transfer→collective→stream→tiles→profile→scheduler→task admission→reset FSM）改为按相位消息化。**只改接口与相位，不改合同**：ADM/ARN/IF/VER 各 ID 的语义、§11.2 分层与补偿规则原样保留；相位相对旧模型的移动（NoC/transfer 从"Tile 之前"移到 P4a、RTL Tile 在 P2 沿步进、Python Tile 在 P4b 步进）不构成周期级对拍义务（VER-04/VER-20 只比偏序，不比周期数）。

| 相位                    | 动作                                                                                                                                                                                                                                                                                                                                                                                                              | 规则                                                                                                                                                                                                    |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0 drive                | host 将本 tick 到期消息（含上 tick P4 形成的响应）与各接收队列的 ready/credit 信号驱动到端口；无到期消息保持 idle/ready                                                                                                                                                                                                                                                                                           | 驱动前检查该通道 credit>0；ready 由 P4 计算的队列容量决定并保持到 P1 采样点                                                                                                                             |
| P1 eval-low 采样        | drive 后组合稳定，**唯一握手判定点**，两个方向同点判定：接受集 = {该稳定点 valid&&ready 的 (消息,方向)}，payload/身份快照在此冻结                                                                                                                                                                                                                                                                                 | §18.3 断言：valid&&!ready 期间 payload 稳定；本点之外的 ready 变化（P2/P3 观测）只能作为下 tick P1 的条件，禁止 post-edge ready 配 pre-edge payload                                                     |
| P2 单上升沿             | clk 0→1 一次                                                                                                                                                                                                                                                                                                                                                                                                      | P1 采样值随沿进寄存器；被替换 RTL Tile 在此唯一步进；RTL 内部多时钟/CDC 由 RTL 自检取证（VER-18）                                                                                                       |
| P3 settle+staged commit | eval 后组合稳定，**只把 P1 已握手接受的 pre-edge payload/身份快照提交进 Python 队列**（恰好一次，VER-25）；P2 之后新出现的 valid/payload 仅登记为下 tick P1 的候选（staged），不得因 P3 首次看见就当作本 tick accepted——否则会消费未握手终态/credit、漏拍或重复。P3 另可被动观测状态/debug instrumentation，但功能消息一律以 P1 接受集为准                                                                        | Python 消费 Tile 可见状态的唯一时刻                                                                                                                                                                     |
| P4 Group control        | Python Group 每 tick 恰好一次步进，子序固定：(a) memory tick——NoC router、transfer manager、group transfers、collectives、stream queues 各恰好一次（与 Python Tile 共享同一 memory tick）；(b) 未被替换的 Python Tile 在此步进一次（mixed 配置；被替换 Tile 不再走 `_step_tiles`，已在 P2 步进）；(c) profile controller、scheduler ISSUE/REGISTER、task admission、reset FSM、sequencer 退休；形成下 tick 驱动集 | P4 新形成的消息最早下 tick P0 驱动、P1 握手，**新接受 request 最早下周期返回，无同 tick 递归**（对齐现行为 `_activate_admitted_context`"first group action issues at cycle+1"，tile_group.py L933-947） |

约定：本文其余各序列中"Tile 输出 X 于 P3 提交"均指 X 在某 tick P1 完成握手、于该 tick P3 进入 Python 队列；"Group 消息 Y 于 P0 驱动"均指 Y 由上 tick P4 形成、本 tick P0 驱动并参与 P1 握手。

### 3.2 逻辑通道与预算（独立信用池，IF-09 精神；物理编码待 review）

| 通道    | 方向                      | 承载                                                                                                                                                                                                                                                                                                                                                                                        | 与现状差异                                                                                                                                                                                                                                                                                                                                       |
| ------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| CTRL    | G↔T                       | G→T：QUERY_CAPS、TASK_PLAN/COMMIT(hold 请求)/COMMIT_FINAL(最终授权)/ROLLBACK/CANCEL、PROFILE_PREPARE/COMMIT、GATE_NOTIFY、RETIRE、STATUS、TILE_RESET；T→G：ACK、STATUS_RESP、RETIRE_READY、TASK_COMPLETION、FAULT_REPORT、RESOURCE_COMMIT_IND                                                                                                                                               | 现 `DevicePort` 协议（device.py L108-127）只有 Python 对象入参+布尔背压，无 ACK/credit/授权消息；profile member bus 每 tick 一项保持                                                                                                                                                                                                             |
| EVENT   | T→G（G→T 唤醒）           | POST_EVENT(owner,generation,sequence,scope,status)、PHASE_SIGNAL、RESOURCE_UPDATE(T→G，报告本地容量/slot/gate 的 ACK 状态)；G→T EVENT_NOTIFY 唤醒有界等待记录                                                                                                                                                                                                                               | 现 `signal()` 校验 owner/generation/sequence、重复/陈旧计数（event_table.py L400-460）语义保留；等待唤醒事件驱动，不逐 tick 轮询（§5.4）                                                                                                                                                                                                         |
| PROGRAM | 双向                      | G→T：PROGRAM_INSTALL(只带身份+位置的头)、PROGRAM_READ_RESP(带身份的程序字节响应)、PROGRAM_READY(id/version/hash/epoch/local handle)、PROGRAM_INVALIDATE；T→G：PROGRAM_READ_REQ(按 INSTALL 头拉取字节)、PROGRAM_INSTALL_ACK(实测 hash/长度+本地句柄，校验失败=fault)、INVALIDATE_ACK(带 sequence/epoch，先失效 descriptor 缓存再 ACK)；PROGRAM_REGISTER(Host→Group) 为 Python 内部登记不过界 | 现无此消息（r5 §13.1）；`ensure_resident` 解析模型被真实端口替代；热启动=**零 Tile 外部 program-text 字节**（不发 READ_REQ、不改 resident text），但 PROGRAM_READY/ACK 等控制消息仍在本通道合法流动（RES-07/RES-10、T-58）；ack/control 不依赖字节 blob FIFO（无队头阻塞）；descriptor patch 随 Task 提交路径（W2，RES-07）；语义以 W5 §4.1 为准 |
| MEMORY  | T→G 请求 / G→T 响应       | MEM_REQ(身份/地址/范围/权限/generation/写数据)、MEM_RSP(数据/状态)、MEM_TERMINAL_ACK(接收方完成终态消费后的引用确认)                                                                                                                                                                                                                                                                        | 现 Tile 直接持 `MemoryOwner`/view 对象；数据面 flit 仍走 Python NoC VC0-3（`../../../pipeline_validator/memory/noc.py` L22-29），本通道是 Tile 端口语义层                                                                                                                                                                                        |
| DEBUG   | T→Host 诊断流，可丢、限额 | 采样/FIFO 状态，不进功能结果；开启/关闭/限额配置和必要 ACK 仍走可靠 CTRL（VER-26、T-67）                                                                                                                                                                                                                                                                                                    | 现无独立 debug 通道                                                                                                                                                                                                                                                                                                                              |

### 3.3 通道发送 FSM（每通道每方向一实例）

`IDLE →(有到期消息且credit>0) DRIVE →(P1 valid&&ready) ACCEPTED →(对端 P3 消费) DONE`；`DRIVE →(P1 !ready) DRIVE`（**保持 valid 与 payload 不变直至 ready**，§18.3）。本地 C-ABI transport 可靠不丢帧，故**无 ACK 超时自动重发、无 last-response 缓存**；恰一次语义由"已接受重复检测 + 有界事务状态"保证（同 op_seq 在事务活跃期再到达=重复，不重复执行；显式 STATUS 查询幂等，IF-04）。CPU 侧 `GroupPortAdapter` 现行语义（重复 request_id 即异常，group_port.py L185-186）**原样保留**，不随边界扩展；未来若换有损网络传输的重传属另立 review。信用守恒不变量：`credits + in_flight + fifo_occupancy == capacity`（IF-10，与 noc.py L57-97 同式）。

### 3.4 Group 准入协调器 FSM（IF-13：父资源 + Tile 本地资源的串行有界 ticket）

每 Tile 每 ticket 一实例，Group 侧串行管理；对应当前 Python 在 `_commit_route_task`（`../../../pipeline_validator/tile_group.py` L1284-1350：`commit_admission` → `_pin_grid_l2` → 程序登记 → `load_program`，失败 `_unpin_task_l2`+`abort_admission` L1341-1345）一次调用内完成的父/本地交错，CS1 拆为多 tick 消息事务，子序与补偿集合逐一对应：

```text
PLAN（版本票据，无副作用，ADM-01）
→ PARENT_PREP（父 L2 pin 预留（对应 _pin_grid_l2 L970-1008：l2_sram.pin 立即记账并阻止 backing 回收）与 R 配额校验/扣减（lease key 检查 L1253-1256）、事件/控制预算核对；预留立即进入 Group 内部占用账本并参与容量与 backing 回收判定——否则 PREP→hold 多周期窗口内 backing 可被 final-free 或 R 被重复分配；"不可见"仅指对调用者/CPU 侧尚未 accepted，不产生对外完成承诺）
→ HOLD_REQ（TASK_COMMIT 于 P4 形成，语义=请求本地 hold）
→ LOCAL_HOLD（Tile §3.5 hold L1 资源：不 bind 可执行 UCE、不取指；ACK(hold) 经 P1 握手、P3 提交）
→ FINAL_COMMIT（Group 确认父+本地资源全部持有后发出最终 COMMIT 授权，下 tick P0 驱动）
→ ACCEPTED（Tile 原子安装 Task 绑定并 ACK(accepted)；Group 收到后对外宣布 accepted）
→ 执行（executable 条件，§3.4 末段）
→ 退休（§5.1 第 7 步）
```

授权之前任一步失败/超时 → `COMPENSATE`（父：unpin 父 L2、退还 R 配额、释放事件预留；Tile：释放 hold 并 ACK；**票据保持到释放证据（各补偿 ACK）收齐**才释放）→ 补偿 ACK。**最终授权之后不再 ROLLBACK**，取消走 CANCEL_TASK/drain；程序安装可与准入并行进行，READY 永不授权未 accepted 的 Task 执行；无跨 Tile gang（ADM-03）。

ticket 取得记录（ledger，补偿与 VER-15 零泄漏检查的依据）：每取得一项记 `{kind, 身份, generation, epoch, acquired_tick, released_tick}`，kind ∈ {L1_ARENA, L1_FRAME, UCE_SLOT, L2_PIN, R_LEASE, EVENT_BUDGET}；L2_PIN/R_LEASE 由 Group 侧记录，L1 三项经 RESOURCE_COMMIT_IND 由 Tile 报告镜像，双方记录必须一致（W7 对拍项）。

**执行授权与唯一 commit 决定**：本地 hold 不授权 UCE 执行（IF-13）。Tile 首条发射条件 = 最终 COMMIT 后 ACK(accepted) 完成 ∧ PROGRAM_READY 已提交 ∧ Frame/Context 就绪 ∧ 无 gate/fault 禁止（r5 §13.1 ACTIVATE 行）——仅有 ACK(hold) 或仅有 READY 都不得执行。可见 commit 决定唯一：Group 以收到 ACK(accepted)（P3 提交、P4 记分）为 Task accepted 的唯一对外宣布点。

### 3.5 Tile 本地 commit FSM（多周期，局部）

`VALIDATE（重验 PLAN 版本票据，见 §5.3）→ ACQUIRE_L1(Arena+Frame，hold：不 bind 可执行 UCE、不取指) → ACK(hold_ready) →（等待最终 COMMIT 授权于 P1 握手）→ BIND_INSTALL（原子安装 Task 绑定）→ ACK(accepted) → executable`；授权前任一步失败 → `COMPENSATE`（release Frame → invalidate views → retire Arena，对应 `abort_admission` `../../../pipeline_validator/tile.py` L1598-1631）→ ACK(WAIT/FAULT/COMPENSATED)；授权后失败/取消不再回滚，走 CANCEL_TASK/drain（ADM-04、§11.2.5）。

## 4. 接口、descriptor、寄存器和协议

### 4.1 身份/通用字段（所有消息必带）

`abi_ver, channel, dir, op_seq, txn_id?, epoch{program, profile_l1, profile_l2, run_gen}, TaskKey|ContextKey, opcode, status/reason`；内存类另带 `addr/range/perm, data`（IF-11：有界、可序列化、不传对象/callback）。物理位宽/编码由后续规格冻结。

### 4.2 消息最小集与关键字段

| 消息                                                        | 方向            | 关键字段                                                                                                                                                                                        | 语义要点                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TASK_PLAN                                                   | G→T             | TaskKey、版本票据（§2.3）、R/L1 需求、预估放置                                                                                                                                                  | 纯函数票据，不占资源不执行（ADM-01）；COMMIT 时重验，漂移即 STALE_VERSION 重规划                                                                                                                                                                                                             |
| TASK_COMMIT（hold 请求）                                    | G→T             | TaskKey、PLAN 票据引用、descriptor 引用(id/hash/长度)、R、L1 需求、placement、事件依赖、program 引用                                                                                            | 由 §3.4 协调器在 PARENT_PREP 之后发出；语义=请求本地 hold（不 bind、不取指）；每 Tile 独立，无跨 Tile barrier（ADM-03）                                                                                                                                                                      |
| TASK_COMMIT_FINAL（最终授权）                               | G→T             | TaskKey、txn_id                                                                                                                                                                                 | 前提=父 pin/R 已确认 ∧ ACK(hold) 已提交；Tile 收后原子安装绑定并 ACK(accepted)；此后不再 ROLLBACK，取消走 CANCEL_TASK/drain                                                                                                                                                                  |
| ACK                                                         | T→G             | op_seq、outcome(HOLD_READY/ACCEPTED/WAIT(reason)/FAULT/STALE/COMPENSATED)                                                                                                                       | WAIT 原因枚举沿用 `AdmissionWaitReason`（`../../../pipeline_validator/memory/allocator.py` L152-157）；STATUS 查询幂等                                                                                                                                                                       |
| RETIRE_READY / RETIRE / TASK_COMPLETION                     | T→G / G→T / T→G | RETIRE_READY=本地旧 in-flight/VME 已结束（非 Task success）；RETIRE=Group 授权关闭本地 view/Arena/Frame；TASK_COMPLETION=本地回收完成后可靠保存的终态(DONE/ERROR/TIMEOUT/RESET)+错误定位(IF-06) | 现 `_step_tiles` 退休链（tile_group.py L1669-1722）拆分：本地回收归 RTL、父资源/记录归 Group；TASK_COMPLETION 不等父 R free；Group 不再执行 L1 retire；现 `DeviceCompletionStatus` 仅 SUCCESS/ERROR（device.py L23-27），边界按 §13.1 全枚举映射                                             |
| GATE_NOTIFY                                                 | G→T             | level(l1/l2)、closed/open、profile generation                                                                                                                                                   | 现 admission_version 元组不含 L1 gate（tile_group.py L2070-2078），L1 gate 仅在 `dispatch_role` L1191 与 `_step_task_admission` L1231 每 tick 轮询；CS1：Group 在 `OPEN_ISSUE` 时**自行唤醒**受阻 Route/Task 准入重试（事件驱动），GATE_NOTIFY 供 Tile 侧 hold/issue 状态机（ADM-06/ADM-10） |
| RESOURCE_UPDATE                                             | T→G             | 本地容量/slot/gate 相关 ACK 状态与版本                                                                                                                                                          | 方向为 Tile→Group 报告；Group 侧重试探针按版本变化 coalesce（§5.4）；对应现唤醒令牌 `(admission_version, _slot_version)`（`../../../pipeline_validator/runtime/group_port.py` L355-377）                                                                                                     |
| PROGRAM_INSTALL / READ_REQ / READY / INSTALLED / INVALIDATE | G↔T             | INSTALL 只带身份+位置的头；READ_REQ 按 INSTALL 头拉取字节；READY 带 id/version/hash/epoch/local handle；INSTALLED 带实测 hash/长度                                                              | 首次取指前必须已 READY 且 Task accepted；READY 不必先于 Route 登记（r5 §13.1）且**永不单独授权未 accepted Task 执行**；INSTALLED 与登记不符即 fault（CMP-18）；INVALIDATE 引用归零后先失效 descriptor 缓存（带 sequence/epoch）再 ACK（RES-14）；warm=零 Tile 外部 program-text 字节（§3.2） |
| MEM_REQ/RSP/TERMINAL_ACK                                    | T↔G             | txn_id、身份、地址/范围/权限、数据、状态                                                                                                                                                        | 响应声明实际 write visibility scope；TERMINAL_ACK 确认终态消费，不代替目标写提交；页授权检查点走本通道（GAT-20）                                                                                                                                                                             |
| POST_EVENT / PHASE_SIGNAL / EVENT_NOTIFY                    | T→G / G→T       | owner/generation/sequence/scope                                                                                                                                                                 | Group 去重、聚合、判陈旧（EVT-13；现 `_on_phase_signal` tile_group.py L1391+）；事件等待记录的**唯一 writer 是 W6**（W2 只维护 WAIT 状态/只读 eligibility，W3 只写 region 等待）                                                                                                             |

### 4.3 背压、credit 与重复处理

- 接受仅发生在 P1 的 valid&&ready，两个方向同点判定；P2 后新出现的 valid 只能 staged 供下 tick P1（§3.1）；ready 可随接收方容量变化（§18.3）。
- 每通道有限 credit：发送方 P1 接受即扣减，接收方 P3 消费（或 ACK 穿越）后归还；重复/不匹配的 credit 归还属错误（IF-10）。
- 无自动重发：本地可靠 transport 不丢帧；同 op_seq 在事务活跃期再到达=重复检测（不重复执行、无双重资源），终态后显式 STATUS 查询幂等（IF-04）；CPU 侧 `GroupPortAdapter` 现行语义（重复 request_id 即异常，group_port.py L185-186）原样保留；有损网络重传需求出现时另立 review。
- 终态稳定：TASK_COMPLETION 在 Tile 侧可靠保存、被 Group 消费前保持可读，发布背压下不丢终态（IF-05）；退役控制/ACK 持有保留 credit 与 terminal record。

## 5. 数据流、控制流和时序路径

### 5.1 冷启动任务端到端（每步标注相位与"最早生效"）

1. CPU `try_submit`（P4，占 pending 元数据，ADM-01）→ root 准入 commit（L2 Arena/事件预算/槽位，原子或回滚）→ `ACTIVE`。
2. `DISPATCH_ROLE` 登记 Route（占 dispatch credit，不占 Tile 资源，ADM-12）；L1 gate 关闭时返回 BACKPRESSURE 等待（现 tile_group.py L1191-1192）。
3. 协调器（§3.4，P4）：队首候选经版本票据重验 → `PARENT_PREP`（父 L2 pin 预备+R 配额校验+事件预算核对；SAME 队首优先、每 Tile 每 tick 至多一个候选的现合同保留，L1229-1246）→ 形成 TASK_COMMIT（hold 请求）于 P4；L1 gate 关闭时 Route 登记与提交均 WAIT（§5.4）。
4. P0（下 tick）驱动 TASK_COMMIT（hold 请求）→ P1 接受 → §3.5 hold → ACK(hold_ready)（P1/P3）→ Group P4 确认父+本地全部持有 → FINAL_COMMIT（下 tick P0）→ Tile 原子安装绑定 → ACK(accepted)（P1/P3）→ Group 对外宣布 accepted：**accepted ≠ installed ≠ executable**；descriptor patch 字段随本提交路径由 Tile 解析（W5 §4.1，warm 不改 text）。
5. 程序安装与准入**并行**：PROGRAM_INSTALL 头（P0 驱动）→ Tile PROGRAM_READ_REQ 拉取字节（多 tick）→ PROGRAM_INSTALL_ACK（实测 hash/长度+本地句柄）→ Group 记账后发 PROGRAM_READY；首次取指晚于 READY 且晚于 accepted（T-58）；同程序安装中的 Task 共享完成点（r5 §13.1）。
6. 执行期：MEM_REQ/RSP/TERMINAL_ACK、POST_EVENT/PHASE_SIGNAL 按 §4.3；本地等待记录由 EVENT_NOTIFY 唤醒。输入早释放按总体设计 §4.4.1：Tile 经 EVENT 通道发送 Task 级 `input_released`，Group 收齐 `all_tasks` 后解除尚存纯读 L2 pin，再由按 buffer 的 release 与引用闭合决定物理回收；该路径可早于 TASK_COMPLETION，不归还 R／L1／Frame。
7. 终态链（Group 驱动退休，r5 CTX/ARN 合同）：W2 `EXEC_DRAINED` 内部通知 W6 → RTL 本地旧 in-flight/VME 结束、view 在途清零后上报 `RETIRE_READY`（非 Task success）→ Group 发 `RETIRE` 授权 → RTL 关闭本地 view/Arena/Frame 完成本地回收 → 可靠保存并上报 `TASK_COMPLETION`（不等父 R free）→ Group 清理剩余父 L2 pin、归还 R 并结算 Grid 记录 → root 自身 L2 view 全 release、Arena 退休后发布 root 完成（ADM-11）。Group **不再执行 L1 retire**（现 `_step_tiles` L1669-1722 的本地顺序转为 RTL 内部合同）；Context 复用须旧义务闭合且终态记录与新代隔离（保守可等完成握手）。

### 5.2 背压与重复

- 随机 ready 拉低（VER-07）：断言 payload 稳定、无消息丢失、credit 守恒。
- 同 op_seq 重复到达（活跃期注入）：重复检测、恰好一次执行、无双重资源提交；显式 STATUS 幂等；负例：终态后无 STATUS 的重复提交判协议错误。

### 5.3 过期 ticket 与取消跨 commit

- 过期（generation）：generation/epoch 不匹配的消息（含迟到的 PLAN/COMMIT ACK、旧代 PHASE/终态）→ 接收方丢弃并回 STALE，不更新新代状态（对齐 event_table.py `signal` 的 owner/generation/sequence 校验与 stale 计数；EVT-13）。
- 过期（版本票据，同 generation）：PLAN/COMMIT 线上携带并重验 `(pool_id, pool_epoch, pool_version, extent_pool_version, profile_mode, profile_generation)`（pool_token 为 Python 对象不上线，见 §2.3）；同 generation 下池/extent/profile 版本漂移同样判 stale（`commit_arena`/`rollback` 对版本不匹配的 ArenaPlan 一律拒绝，arena.py L342-359）→ 回 STALE_VERSION、丢弃票据重新 PLAN（ADM-02 的"不占资源版本票据"机制），非 fault；仅 INVALID_REQUEST/PERMANENT_CAPACITY 才终止。
- 取消：ROLLBACK_COMMIT 有效窗口 = 最终 COMMIT 授权发出之前（§3.4；授权后不再 ROLLBACK，走 CANCEL_TASK/drain）；与 commit FSM 竞争时按到达序在 Tile 控制队列串行化：窗口内→协调器与 Tile 双向 COMPENSATE、票据保持到释放证据（各补偿 ACK）收齐、父资源（pin/R/事件预算）全额返还；窗口外→CANCEL_TASK 排空；ACK 携带实际路径。两分支都必须可测（T-20）。
- 失败补偿：一律发生在授权之前（对应现 `abort_admission` 的 `rollback_program` 路径 tile.py L1604-1605 在 CS1 只可能出现在 hold 阶段）；已执行工作不回滚，走 fail-stop/排空（ADM-04、§11.2.5）。

### 5.4 L1 gate 唤醒（差异项，重点）

现状：L1 gate 关闭由 `_step_task_admission` 每 tick 主动检查（tile_group.py L1231）+ `dispatch_role` 提交时检查（L1191），唤醒令牌不含 L1 gate（L2070-2078）——即模型按 tick 轮询。CS1：`OPEN_ISSUE`（profile_controller.py L713-715 置 gate 开）于 P4c 步进时，Group **自行唤醒**受阻的 Route 登记与 Task 准入重试（事件驱动，非轮询，ADM-06/ADM-10）；`GATE_NOTIFY`（G→T）另行通知 Tile 侧 hold/issue 状态机。重试探针按**版本变化 coalesce**：每次相关版本（池/槽位/gate）变化至多重试一轮，不与 gate 翻转次数 1:1；coalesce 断言进 W7。

## 6. 配置、PPA、性能模型和 PMU

- 三类预算**分立记账、不得互扣**：(1) 通道 credit——每通道每方向的消息流控（§3.3）；(2) 事务容量——同时在飞的准入 ticket/事务数（对应 `context_pending_capacity`、`dispatch_capacity` 等现有有界量）；(3) 终态 ACK 引用——TASK_COMPLETION/补偿 ACK 被消费前的保留引用计数（Context 复用与零泄漏依据）。三者数值全部待 review、由后续规格冻结，本文不设初值。
- 每 tick host 开销 = P0 驱动 + eval（P1）+ 上升沿 + eval（P3）+ clk 回低 settle + P4 Python step；单线程确定性优先于吞吐（VER-06 记录 seed/binary/SHA/manifest）。
- 性能以 RTL cycle 证据为准；本组件不产生 PPA 结论，不做利用率估算（VB-14、D-29）。
- 通道 credit 不复用 NoC VC0-3 数据面信用（IF-09：合并通道须重做死锁分析）；MEMORY 语义层与 flit 层关系见 §9.4。

## 7. RTL/软件实现建议

- **Verilator host 绑定 transport（已选基线）**：目标目录 `hardware/coral_nexus/`（待创建；当前无 `hardware/`，不依赖旧 Gemmini／BOA）。Verilator 以 C++ harness 编译为共享库，暴露窄 C ABI（`bp_begin/bp_drive/bp_tick/bp_collect`），消息为定长 POD 字节缓冲+显式长度；Python 经 ctypes 绑定。**Python 主 loop 负责编排**：P0 前 `bp_drive` 装载本 tick 驱动集，P4 由 Python 执行 Group 步进；`bp_tick` 只做 C++ 侧一个周期的物理序列——P1 eval（采样 pre-edge valid&&ready）→ P2 clk 0→1（每 tick 恰一次上升沿）→ P3 eval（采样 staged 输出）→ clk 回低 + settle eval；`bp_collect` 取回 P1 接受集与 P3 输出。C++ 不回调 Python、不执行 P4（IF-11）。优点：boring、单线程可复现、无跨进程时序抖动。Verilator 版本与波形/日志开关由 W0 基线环境固定。
- socket/JSON、共享内存 ring、DPI 回调 Python 不在本轮实现；有损传输或其他并发载体以后另立合同，不为此预留兼容 fallback。
- "响应最早下 tick 生效"由 Python 编排顺序天然保证（P4 形成的消息只进下 tick `bp_drive`）；相位与 总体设计 §5.1一致（P1 唯一握手）。
- 日志：Host 被动捕获边界完整窗口（方向/身份/相位时序/完整 payload 或可取回 blob），摘要不可重建数据，缺口明确拒绝作重放证据（VER-22、T-67）。

## 8. 验证、bring-up 和验收标准

| 验证项                                | 方法（含负例）                                                                                                                                                                                                                                                                                                                                                            | 需求/用例                      |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 序列正确                              | 冷启动全序列断言：hold/accepted/installed/READY/executable 分离；首取指晚于 PROGRAM_READY 且晚于 accepted；负例一：程序先 READY 时 Task 不得提前执行；负例二：仅有 ACK(hold) 无最终授权不得 bind/取指；PREPARED pin/R 即时入账（准备期占用非零且阻止 backing final-free/重复分配），对调用者 accepted 只在 ACK(accepted) 后；断言失败补偿后账本恢复基线，而非准备期零占用 | RES-13/19、T-58、IF-13、ADM-02 |
| 背压与 staged 边界                    | 随机 ready 拉低 + payload 稳定断言 + 无丢失；负例一：P3 观测的 ready 变化作用于同 tick P1（错配）；负例二：P2 后新出现的 valid 被 P3 直接当 accepted（消费未握手终态/credit）——两者都必须被判为协议错误                                                                                                                                                                   | §18.3、VER-07                  |
| 重复请求                              | 同 op_seq 活跃期重复注入：重复检测、恰好一次执行、无双重资源；显式 STATUS 幂等；负例：终态后无 STATUS 的重复提交、伪造不匹配 credit 归还，均判协议错误                                                                                                                                                                                                                    | IF-04/IF-05/IF-10              |
| 过期 ticket（generation）             | 旧 generation COMMIT/PHASE/终态迟到：STALE、零状态变更；负例：旧代回包写新代存储必须报错                                                                                                                                                                                                                                                                                  | EVT-13、§18.3                  |
| 过期 ticket（同 generation 版本漂移） | 在 PLAN 与 COMMIT 之间注入池/extent/profile 版本变化（如另一 root 的 backing final-free 推进 pool_version）：COMMIT 重验失败回 STALE_VERSION 并重规划，资源零变化；负例：忽略版本仍 commit 必须被判协议错误                                                                                                                                                               | ADM-02、arena.py L342-359      |
| 取消跨 commit                         | ROLLBACK 与 §3.5 各相位竞争（含恰在最终授权同 tick），两分支均测；断言：窗口内协调器+Tile 双向补偿恰好逆序撤销 ledger 全部取得项、票据保持到释放证据收齐、父资源（pin/R/事件预算）全额返还、等待者无半提交残留；最终授权后 ROLLBACK 被拒并转 CANCEL_TASK                                                                                                                  | ADM-02/04、IF-13、T-20、T-52   |
| 父/本地账本一致                       | RESOURCE_COMMIT_IND 与 Group ledger 逐项对拍（L1_ARENA/L1_FRAME/UCE_SLOT/L2_PIN/R_LEASE/EVENT_BUDGET 计数一致）；终态 ledger 清空（零泄漏）                                                                                                                                                                                                                               | VER-15、VER-25                 |
| credit 守恒                           | 每通道每 tick 断言 `credits+in_flight+occupancy==capacity`；复位后本地初值重建                                                                                                                                                                                                                                                                                            | IF-10、VER-15                  |
| L1 gate 唤醒                          | gate 关闭期提交/登记全部 WAIT_CONTROL_RESOURCE；OPEN_ISSUE 后有界周期内出现重试探针；断言重试探针按版本变化 coalesce（每次版本变化至多一轮，不与 gate 翻转 1:1）且无逐 tick 轮询                                                                                                                                                                                          | ADM-06/ADM-10                  |
| 退休链                                | EXEC_DRAINED→RETIRE_READY→RETIRE→本地回收→TASK_COMPLETION→Group 关父记录→root 发布的相位序断言；负例一：本地回收未完成即发 TASK_COMPLETION 判协议错误；负例二：TASK_COMPLETION 前等待父 R free（死锁）；Context 复用须旧义务闭合、旧代记录隔离                                                                                                                            | ADM-11、ARN、T-52、VER-15      |
| owner 唯一                            | 禁用被替换 Python 路径（含 Python L1 retire）后全链路仍通过；每状态转移恰一次                                                                                                                                                                                                                                                                                             | VER-25、T-64                   |
| 故障/复位                             | 边界注入 fault/cancel/reset（含背压），fail-stop、复位零泄漏、旧代响应隔离                                                                                                                                                                                                                                                                                                | VER-23、T-51、T-52             |
| 一致性                                | 与纯 Python 在同语料上按每 Context 顺序+跨 Context 偏序对拍（不比周期数）                                                                                                                                                                                                                                                                                                 | VER-04/VER-20/21、T-39、T-40   |

未运行项：本轮无任何构建/仿真执行；上述均为设计验收条件（VER 系列与 T-39/40/58/64/67 的执行属后续阶段）。

## 9. 风险、取舍和后续细化方向

1. 相位模型 P0-P4 是新 adapter 合同（非现有 RTL/Python 行为），需 总体设计 统一并与 W0 环境锁定 Verilator 调用约定。
2. 消息物理布局、通道深度/credit/超时、PROGRAM 分段 MTU：待 review，由后续规格冻结。
3. TILE_RESET 作用域与在途事务处理：生产路径只用 GROUP 域复位（RST-01），Tile 级复位仅在调试场景，语义未定（r5 §13.1）。
4. MEMORY 通道与 Python NoC VC0-3 数据面的credit关系（端口语义层 vs flit 层）需 W4 联合裁决，避免双份账本。
5. 物理消息名/opcode/位编码未冻结（总体设计 合同：不得自造固定位编码）；有损网络下的重传/ACK 缓存属未来另立 review，本设计只覆盖本地可靠 transport。
6. 多 Tile 替换时 host 每 tick 驱动开销线性增长；如成为瓶颈，评估批量驱动 API（不改变相位语义，另行 review）。
7. PROGRAM 通道分段传输并发上限（有限多程序驻留合并，RES-19）依赖 W5 的驻留容量模型；descriptor patch 已按 W5 §4.1 移入 Task 提交路径（RES-07 分层），W2 需在 Frame/Context 就绪时序中吸收其解析周期。
