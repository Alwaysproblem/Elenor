# W6 事件、故障、取消、复位与时钟门控（Events / Fault / Cancel / Reset / Clock-gating）

> 本组件为设计评审稿，跨包合同与用户裁决见[总体设计](../Coral_Nexus_Tile_Design.md)；r5 指[需求规格](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。本轮检查见[设计验证记录](../Design_Validation.md)，不代表组件已实现。

## 1. 定位、目标和 First Silicon cutline

本组件负责改造后 Tile 内的**本地事件设施、等待/唤醒、阶段信号发送、故障检测与 first-fault 记录、取消/超时排空、Tile 侧复位端点以及时钟门控控制**。它是"任务如何终止、终止如何被外界可靠观察"的唯一 Tile 内权威；矩阵区域与计算本身归 W3，L1 数据面与传输取消接口归 W4，L2 执行 owner 为 Python，Group 侧聚合/复位 FSM/故障环/poison 保持 Python（总体设计 §2.3）。

目标按优先级：

1. 事件是**持久可观察状态**（EVT-02），WAIT/WAITALL 具备原子 check-register 无丢唤醒语义（EVT-01），终态唯一且不回退（EVT-16、FLT-14）。
2. 五类进度/完成语义严格区分：`INPUT_RELEASED`、`OUTPUT_READY`、`STORE_DONE`、`TASK_COMPLETION` 与 VME 的 `region_done` 不可互相替代；只允许依据各自明确的前置条件推导（§12.1、MAT-15）。
3. 故障基线 **fail-stop**：任一 Context 故障→本 Tile UCE 停发→Group 复位状态机排空整个 run（FLT-01、SCH-11、RST-07）；本组件**不实现**局部恢复（FLT-02a 是 SHOULD，D-17）。
4. 取消/超时只停止新发射；已接受工作真实完成或隔离确认，失败链绝不产生成功终态（FLT-04/05、RST-08、MAT-16）。
5. 可靠终态路径（first-fault 摘要＋事件终态）与可丢 debug 历史**逻辑独立**：可靠消息有独立预算且可证明不被 debug 背压阻塞；物理端口可以共享，但共享必须给出隔离证明（FLT-13、OBS-14～OBS-18）。
6. 时钟门控分层：Context park ≠ 单元 idle ≠ Tile quiescence；一个 Context 的等待/WFI 绝不关掉其他 Context、在途操作或唤醒逻辑所需时钟（CLK-01～06、INV-10；Coral `CoreAxi` 把 core WFI 直接作为全核关钟条件，见 §3.2 [H06] 行，此事实由 r5 记录，本轮未重新核对上游源码）。

不覆盖：Group 侧事件表与聚合 FSM 的 Python 实现（继承 validator）、L2/HBM 事务取消的物理路径（W4）、debug FIFO/HBM ring 的导出格式（W7/OBS-16）、多 VME 实例调度（W3）。

## 2. 职责、非职责和 ownership

| 状态/机制                                                     | CS1 owner                                                 | 说明                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tile 本地事件表（frontier 配额）                              | RTL（本组件）                                             | 硬件键＝ContextKey（物理 Tile, context slot, context generation）＋local_event_id＋sequence；`ctx{N}:<name>` 只是 validator 模型命名（EVT-16），经 manifest 映射，不作硬件键                                                                                                                                            |
| 每 Context 等待记录（wait record）                            | RTL（本组件，唯一 owner；W2 只持 Context 状态与只读摘要） | WAIT 单事件 / WAITALL 全事件 AND（**不设 OR/WAIT_ANY**）；发射时原子 check-register                                                                                                                                                                                                                                     |
| 完成广播/唤醒网络                                             | RTL（本组件）                                             | 引擎/LSU/MFE/VME 完成通知按身份匹配等待记录与事件表                                                                                                                                                                                                                                                                     |
| 阶段信号发送（Tile 侧）                                       | RTL（本组件）                                             | `SIGNAL_PHASE` 发射时同步产生、携带 Task 身份、Tile 侧不去重（EVT-14）                                                                                                                                                                                                                                                  |
| first-fault 锁存与故障摘要                                    | RTL（本组件，always-on/reset-safe）                       | 每 Tile 一个，复位保留至 Host 确认（FLT-10、RST-09）                                                                                                                                                                                                                                                                    |
| Tile 侧复位端点 FSM                                           | RTL（本组件）                                             | 执行 RST-07 中 Tile 本地段：停发、取消未发射 engine 队列、排空证据、事件置 RESET、驻留失效 ACK、信用重建                                                                                                                                                                                                                |
| 取消/超时排空（Tile 内）                                      | RTL（本组件＋W3/W4 钩子）                                 | 取消只停新发；接受计数清零与隔离确认是唯一排空证据                                                                                                                                                                                                                                                                      |
| 时钟门控控制块                                                | RTL（本组件）                                             | 分层关钟条件与唤醒路径存活；门控边界排除同周期新到达（CLK-05、TIM-03）                                                                                                                                                                                                                                                  |
| Group 事件表、阶段聚合、复位 FSM、故障环、poison、stream 信用 | Python（继承 validator）                                  | `_on_phase_signal`、`ResetDomain`、`FaultRing` 等见 §3 证据                                                                                                                                                                                                                                                             |
| 统一完成/退役链                                               | RTL 本组件＋W2/W4＋Python Group（见 §5.5）                | W2 EXEC_DRAINED 内部通知→W6 判 RETIRE_READY（非 Task success）→Group 授权 RETIRE→RTL 闭合本地 view/Arena/Frame→W6 可靠保存 TASK_COMPLETION→Python 清理剩余父 L2 pin、归还 R 并结算 Grid 记录→root 自己的 L2 release/Arena 退休→发布 root 完成；输入纯读 pin 另按 §3.3 提前解除；不得等父 R 先 free 再发 TASK_COMPLETION |

## 3. 微架构和状态机

### 3.1 事件身份与本地事件表

事件条目逻辑字段（物理总线不必逐位携带，可经在途表索引间接恢复，§6.1 末行）：

- **身份**：`(ContextKey＝(物理 Tile, context slot, context generation), local_event_id, event_generation, sequence)`。硬件比较用 ContextKey＋local_event_id＋event_generation＋sequence；validator 的 `ctx{N}:<name>` 字符串是模型层命名（EVT-16），只在 manifest 中声明映射，不得作硬件匹配键。`event_generation` 区分同一事件的使用周期，旧 signal 不得唤醒新 wait（EVT-03、§6.1 `event_generation` 行）。
- **kind**：记录本地 launch 完成的来源与承诺（计算、传输、区域等），由登记时确定，发布端不可改写。`INPUT_RELEASED/OUTPUT_READY` 是 `PHASE_SIGNAL` 消息，不自动分配一个 Tile 本地完成事件；`TASK_COMPLETION` 是外部终态报告，Task 不能等待自己退休后才产生的事件。§3.3 按语义并列表述，不是新增统一事件枚举。
- **scope**：task-private / tile-shared / group(L2) / external（§12.3）。`OUTPUT_READY` 的 scope 必须与实际 LSU/MFE/SPM/cache/L2 路径一致（EVT-07、EVT-18）。
- **状态**：`PENDING → {DONE|ERROR|TIMEOUT|RESET}`，终态 sticky，不回 PENDING（EVT-16）。
- **两计数**：`consumer_refs`（已登记未结算的依赖引用）与 `future_uses`（编译器证明的剩余未来引用）。二者归零且已终态才可回收槽位——直接对齐 validator `_maybe_reclaim`（`event_table.py:498-502`）。

容量按**编译器证明的 live frontier 配额**计账而非全历史（EVT-17）：受管 launch 只预留 frontier 上限，输出事件即时分配，即时回收；RTL 用每 Context 有限槽数＋"分配失败＝该 Context 背压"实现，不需要全历史表。4096 是现有 Group EventTable 的模型默认值，不是 Tile 本地表容量；本地容量与槽位位宽 `由后续规格冻结`。

证据：`../../../pipeline_validator/runtime/event_table.py` `EventStatus`(:11-22)、`EventEntry`(:29-46)、`reserve_quota`(:167-198)、`preflight_action`(:248-268)、`reserve_outputs`(:270-281)、`add_consumer`(:299-314)、`consume_dependencies`(:316-341)、`cancel_dependencies`(:343-360)、`abandon_future_uses`(:362-375)、`cancel_outputs`(:377-396)、`signal`(:401-455，stale sequence/wrong owner/duplicate 分别计数且不改状态)、`reset`(:548-553，只把 PENDING→RESET)。

### 3.2 WAIT/WAITALL：原子 check-register

每 Context 一条等待记录（bounded：WAIT 恒 1 个操作数，WAITALL 为 N 个操作数全 AND，N 上限`由后续规格冻结`；**不设 OR/WAIT_ANY 模式**——validator WAIT 只取 `ins.args[0]`，`tile.py:576-577`），记录内容＝逐事件身份引用＋armed 位。语义分三段全覆盖（EVT-01）：

1. **事件先到**：事件表/完成广播已把条目置终态；WAIT 发射时 check 阶段直接命中终态，Context 不进入 park（validator：`_issue_wait` 先 `_wait_refs_ready` 为真则不置 WAIT_EVENT，`tile.py:978-992`）。
2. **同周期到**：check 与 register 在同一提交点完成；仲裁规则固定为**完成广播优先**——同周期到达的完成让 check 直接命中，不 park。该规则单向、可观察（不出现"park 后下一周期立即唤醒"与"不 park"两种交错都合法的歧义）。
3. **park 后到**：记录已 armed，完成广播按**完整事件键** CAM 匹配——`local_event_id/槽号＋ContextKey（物理 Tile, context slot, context generation）＋event_generation＋sequence`，并校验 scope 一致；`sequence` 只区分同一事件的使用周期，不同事件可同 ContextKey/gen/seq，故不得只用代际三元组作匹配键。匹配是**逐操作数**的：命中的操作数记录其终态（DONE/ERROR/TIMEOUT/RESET），记录整体的裁决为——WAITALL：全部操作数终态且全为 DONE 才成功唤醒，任一操作数失败终态即走失败出口（不等待其余操作数）；WAIT（单操作数）：该操作数终态即裁决，DONE 成功唤醒、失败终态走失败出口。armed 位只在整条记录裁决完成后清除，唤醒仅恢复该 Context 发射资格（INV-03/EVT-04；validator `_wait_refs_ready` 对 wait_all 取 `all()`，`tile.py:569-574`）。

关键不变量：**check 与 register 之间不存在可被完成事件穿越的窗口**——等待记录在 WAIT 指令发射提交点一次性写入（含全部操作数的引用），事件表又是持久状态（非单周期脉冲），因此"检查时未完成"必然意味着"记录已登记"，之后的完成必被其所属操作数匹配（EVT-02）。失败终态同样逐操作数记录并触发失败出口，等待者得到明确的失败传播而非永久等待，且**失败链不经由成功唤醒**（EVT-05、FLT-14；validator 基线中依赖者的等待最终由复位 MARK_EVENTS 置 RESET 结束，不做逐事件 ERROR 扇出，`reset_domain.py:98-102`）。

WAIT 指令本身不含 fence/store_done/释放语义（EVT-11）；`tile.await` 只挂起本 Context，不得升级为 Group/Device 栅栏（EVT-19）。Context 故障时其等待记录直接作废（validator `_fault_context` 清 `wait_refs`，`tile.py:1320-1324`），不得让故障 Context 的记录残留匹配后续事件。

### 3.3 五类完成的产生条件（不混用）

| 事件 kind         | 产生条件（唯一来源）                                                                                              | 显式不能推断                                  |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `INPUT_RELEASED`  | 该 Task 全部 L2 load 完成均已按 SSA 身份 await，且其后不再有 L2 load（EVT-15，编译器保证＋loader 核查）           | L2 空间已回收、计算结束、输出就绪             |
| `OUTPUT_READY`    | 该 Task 全部 L2 store 完成结算且其后不再有 L2 store；发布满足 release（EVT-06）                                   | 已写 HBM、buffer 可释放                       |
| `STORE_DONE`      | 单次写事务到达接口承诺的完成/可见点（W4 定义的点）                                                                | 持久化、所有消费者已用完                      |
| `TASK_COMPLETION` | RET 之后本 Task 全部本地事件完成＋退役链结算（§5.5）；状态＝DONE/ERROR/TIMEOUT/RESET 之一                         | 无类型 done 不能替代四态                      |
| `VME_REGION_DONE` | 仅 W3 区域链：compute_done ≤ state_committed ≤ output_visible ≤ state_free ≤ region_done ≤ 本地完成事件（MAT-15） | L1 buffer free、Task 退休、Group OUTPUT_READY |

`RESOURCE_RELEASED` 不是可等待事件：表现为 `arena_retire`/`task_lease_release`/`l2_extent_release` 追踪记录与准入唤醒（§12.1、INV-13）。RTL 本地事件表中不设该 kind，防止软件把资源回收误当同步原语。

每 Task 每方向（input/output）的阶段信号**至多一次**；Tile 侧发射期同步产生、不去重，去重/聚合/陈旧判定全部在 Group 侧以 `(root launch generation, GridInstanceId, phase, 逻辑 Task)` 为键完成（EVT-13/14；validator `_on_phase_signal`：stale→忽略计数、unknown→协议故障、live duplicate→协议故障且不二次计数、seen==expected 时聚合恰好一次，`tile_group.py:1391-1477`）。

**输入早释放的 Group 接收动作**：只有该 Grid 的 `all_tasks` 输入聚合达成，才沿阶段路径调用 `_unpin_grid_readers` 等价逻辑解除尚存纯读 L2 pin；单个未退休 Task 的 signal 不立即解 pin。读写 pin 仍须满足写阶段与 publish/release 条件，或由安全退休清理。随后 Group 按 buffer 执行 `nest.release` 撤销 owner view；只有 producer 所有权、全部 claim/view、pin 与 in-flight 均已关闭，L2 finalizer 才归还完整 backing（总体设计 §4.4.1、ARN-04a／04b／09）。

这条路径不等 `TASK_COMPLETION`，不归还 L1 Arena／Frame／UCE 槽位／R lease，也不递减程序驻留引用；Task 退休只清理仍存的 pin，防止二次 unpin。当前通知不携带可独立释放的 buffer／范围列表，没有同一 Task 内分批输入释放能力；不能把单 job last-source-read 或一个消费者通知当作共享源已可复用。

### 3.4 故障检测、first-fault 与可丢 debug

- 检测源：UCE 译码/执行错误、LSU/MFE 地址与权限错误、VME 协议故障（嵌套 START、缺 END 退出等，MAT-16）、Frame/descriptor 校验（FLT-06：trap→故障记录→Task ERROR，不静默退回重试）。
- 归属：故障记录逻辑上含 code/source/severity＋task/Context/command/event/program/descriptor 身份＋group/tile/queue/slot/patch/engine＋错误地址（FLT-11）。RTL 承载为**first-fault 摘要寄存器＋每 Tile 一个锁存**（always-on/reset-safe），静态文本映射由 Host manifest 恢复；**first-fault-wins**：复位未激活且未完成时才触发，后续故障只计数不覆盖（FLT-10）。validator 只在调用方抑制（`_on_phase_signal`/`on_scheduler_fault` 先查 `reset_domain.is_active or is_done` 再 `trigger_fault`，`tile_group.py:1431-1432,2354-2360`），故障环本身覆盖写（`fault_ring.py:81-88`）——RTL 不得继承"环覆盖"当语义，只能当 debug 历史模型。
- 可靠/可丢分离（**逻辑独立**，物理端口可共享但须证明隔离）：first-fault 摘要、事件终态、TASK_COMPLETION 走带独立可靠预算的控制通道/sideband（IF-09 VC0 语义，W1 承载），不依赖批量数据或 debug 排空（FLT-13）；扩展诊断走小 FIFO/HBM ring，满则丢弃/停采＋溢出计数＋窗口不完整标记，绝不阻塞 issue、state_free、唤醒、终态或故障处理，功能消息绝不走该路径（OBS-15～OBS-17）。若实现故障触发的 ERROR 事件，则故障记录须先于该事件可见（FLT-14 的 SHOULD 次序；validator 无此路径，事务故障基线走 PENDING→RESET）。
- 故障码集合：RTL 故障码空间覆盖 validator `FaultCode` 全部 11 成员的逻辑集合；"生产路径只构造 6 个"（FLT-12）是**当前 validator 覆盖事实，不是硬件支持上限**，不得据此裁剪错误源支持；编码位宽`由后续规格冻结`。

### 3.5 取消/超时与 Tile 侧复位端点 FSM

取消与超时只做一件事：**停止新发射**，且两个作用域严格分开——`CANCEL_TASK`（Group→Tile）目标**单个 Task**（按 TaskKey＋generation 匹配，只停该 Task 所属 Context 的新发）；Group RESET（RST-07/§3.5 端点 FSM）是 run 级排空。取消路径只同步清除**尚未接受**的 ingress 项；UCE 已接受/已排队的操作不得静默丢弃，必须以失败终态结算（基线事务故障走 PENDING→RESET；ERROR 事件为 SHOULD，见 §3.4）；已接受的 engine 作业/传输/写义务运行到真实完成或隔离确认；信用、地址、pin、Arena、gate 不提前复用（FLT-02）。`CANCEL_REQUESTED` 的传输只能因真实完成或隔离确认结束，超时永不判成功；隔离失败置 poison，拒绝一切新启动直至显式静止复位（RST-08；validator poison 门在 `tile_group.py:1992-1993,2108-2110`，显式复位 `reset()` 要求全部静止并重建 Profile，:2235-2267）。

Tile 侧复位端点执行 RST-07 的 Tile 本地段，阶段次序固定、可观察：

```
FAULT/RESET_CMD → FREEZE_ISSUE（停取指停发射；等待中记录作废）
  → CANCEL_UNISSUED（清 UCE→engine 未发射队列；MFE 进复位排空；取消 Profile 活动命令）
  → DRAIN（上报 outstanding 证据：各引擎 accepted 计数、engine 队列空、MSHR 空、传输静止）
  → MARK_EVENTS（本地 PENDING→RESET；终态保持）
  → INVALIDATE_RESIDENCY（按 W5 合同 ACK；descriptor cache 失效）
  → REBUILD_CREDIT（stream 信用按本地初值重建；generation 递增）
  → RESUME/DONE
```

Group 侧 drain 判据是**真实接受工作清零**而非冻结的 PC：accepted 计数、传输闭包静止、无 HBM outstanding、NoC 静止、Profile 已初始化且不在取消、L1/L2 MSHR 空闲（validator `_outstanding_zero`，`reset_domain.py:167-219`）。DRAIN 有界：超 `max_drain_cycles`（模型 100，`由后续规格冻结`）后强制取消在途传输＋取消活动 MFE 作业（RST-08、RST-10 的排空上限同机制）。STOP_QUEUE 至 DONE 前各层拒绝新工作，旧传输/装载/debug 写结算或隔离后才递增 generation、重建信用、复用资源（RST-09，RST-02：复位不能靠清计数忘掉旧事务）。

**fail-stop 边界**：复位排空中的 Profile 命令取消与 VME 实例级清理（区域按 owner/generation 隔离后释放，MAT-16）是排空步骤，**不是**"健康任务继续运行"的故障模式。同周期 completion/fault 竞争收敛到**单一线性化点**（EVT-10）：线性化点之前已合法终态的任务与事件保持其终态，已发布的 `OUTPUT_READY` 承诺不撤销（FLT-04、EVT-06）；线性化点之后未终态的任务以 RESET/ERROR 结束、不得发布成功（故障链无成功完成，FLT-14；region 绝不发布成功 region_done，MAT-16）。FLT-02a 局部隔离未获批准前不得在 RTL 中出现"其他 Context 继续"的分支。

### 3.6 时钟门控控制块

分层条件（CLK-01）：

- **Context park**：该 Context 处于 WAIT/WFI/停靠——只允许关该 Context 的前端取指/译码时钟；其 RF/VRF/配置仍可能接收旧操作回写，门控条件必须排除这些写回端口（CLK-06）。一个 Context 等待不得影响其他 Context 或在途操作时钟（CLK-02、INV-03）。
- **单元 idle**：engine/LSU/队列粒度，由各自 accepted/occupied 计数驱动（W3/W4 提供状态）。
- **Tile quiescence**：全部 Context park **且** LSU/MFE/RVV/VME 在途表空 **且** 完成通道无待处理项（CLK-03）。Coral `CoreAxi` 把 core WFI 直接作全核关钟条件（[H06]，pin <https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/CoreAxi.scala>，本轮未重新核对上游源码）；改造后 **WFI 只表达本 Context park 提示，绝不得直接进入 Tile 关钟条件**（INV-10）。

唤醒存活：必要响应、事件接收和唤醒逻辑在关钟期间可服务，或能可靠使目标时钟恢复（CLK-04）——等待记录匹配逻辑、完成广播接收、复位/取消命令译码放 always-on 域或带自解除门控；进入/退出门控与事件到达竞争时不丢唤醒、不漏写回、不重复完成（CLK-05）——门控条件采样必须排除同周期新到达事件（TIM-03），门控判定到时钟真正关断之间到达的完成按"未门控"处理。复位/取消路径不得依赖被门控单元已醒：freeze/cancel 命令强制解除目标单元门控。clock-gated 周期不得从任务延迟/能量统计中消失（PWR-05，计数归 W7）。

## 4. 接口、descriptor、寄存器和协议

| 接口（方向）                                                               | 内容                                                                                                                                                                                                     | 错误/背压                                                                        | 对端 owner      |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | --------------- |
| WAIT/WAITALL（UCE→W6；W2 只持 WAIT 状态与只读 eligibility 摘要，不写记录） | WAIT＝单事件名；WAITALL＝事件名列表全 AND；无 OR/WAIT_ANY（`tile.py:576-577`）；发射提交点原子 check-register                                                                                            | WAITALL 操作数越界＝译码期拒绝；记录随指令槽存在，无记录满路径                   | W2              |
| 完成通知（引擎/LSU/MFE/VME→W6）                                            | `(ContextKey=(tile_id,ctx_slot,ctx_generation), op_seq/subop_id/txn_id, event_generation, kind, status, error_code)`；ctx_generation 与 arena_alloc_gen/program_epoch/profile_gen/run_gen 分域，不得互替 | 源不存在/代际不符＝计数并按 INV-15 拒绝，不写新代状态                            | W3/W4           |
| 区域等待（W3 内部，W6 不代写）                                             | VME 区域停靠等待记录唯一 writer 是 W3（MAT-12 停靠）；region 完成后经 W3→W6 映射为本地事件（MAT-15 终态链）                                                                                              | W6 不创建/清除 W3 的区域等待记录；只承接映射                                     | W3              |
| SIGNAL_PHASE TX（W6→Group）                                                | 逻辑 TaskKey（root_launch_generation＋GridInstanceId＋task_id）＋phase；物理 Tile 仅为路由信息，不进聚合身份（EVT-13）；发射期产生，不去重                                                               | EVENT 通道背压时稳定保存（CTX-11），不丢不改                                     | W1/Python Group |
| POST_EVENT/RETIRE_READY/TASK_COMPLETION（W6→Group）                        | 终态事件／退役就绪（非成功）／任务终态，均带完整身份与故障摘要引用                                                                                                                                       | 通道满＝背压＋本地保存，不先释放身份再发送（EVT-09）                             | W1/Python Group |
| CANCEL_TASK（Group→W6）                                                    | 目标单个 TaskKey＋generation；只停该 Task 新发（§3.5）                                                                                                                                                   | 旧代命令忽略并计数；已接受操作失败结算不静默丢                                   | Python Group    |
| RETIRE 授权/RESET 命令（Group→W6）                                         | RETIRE：授权关闭本地 view/Arena/Frame（§5.5）；RESET：Group 域（RST-01 基线）                                                                                                                            | 复位激活期重复命令幂等；本地回收未完成不复 ACK                                   | Python Group    |
| 排空证据上报（W6→Group）                                                   | accepted 计数/队列/MSHR/传输静止快照＋RETIRE_READY 前置条件（无旧 in-flight、VME 区域已结束）                                                                                                            | 证据未齐不得上报 DONE/RETIRE_READY（RST-07 DRAIN 判据）                          | Python Group    |
| 驻留失效 ACK（W5↔W6）                                                      | reset 时 invalidate（RES-14）                                                                                                                                                                            | 引用未清先排空，不得提前 ACK                                                     | W5              |
| 故障记录与 debug 导出（W6 owns 小 debug FIFO 与特权导出控制）              | first-fault 摘要＋后续计数；扩展历史走 debug FIFO（可丢）；W7 只做 Host 侧格式/checker/观察；经 W4 执行的 debug 外存事务同样受取消结算约束                                                               | FIFO 满丢弃＋overflow 计数，不阻塞功能终态（OBS-15）；特权导出默认关闭（OBS-14） | W7/W4           |
| 门控控制（W6→全 Tile 单元）                                                | 分层 enable（Context/单元/Tile）＋强制解门控                                                                                                                                                             | 门控条件排除同周期到达；复位路径强制解除                                         | W2/W3/W4        |

控制/事件/故障消息与批量数据使用独立通道或等价机制（IF-09、INV-16）：数据通道信用耗尽时事件、故障、屏障消息仍有界前进，无 HOL（T-54）；信用守恒 `credits + inflight + occupancy == capacity`，复位后按本地初值重建（IF-10，物理 credit 归 W1/W4）。

## 5. 数据流、控制流和时序路径

### 5.1 正常完成路径

UCE 发射 LAUNCH/受控 drain → 引擎接受（`uce_launch_accepted` 对应点）→ 完成通知按身份写事件表（PENDING→DONE）→ 命中等待记录则唤醒 → 该 Context 的 RET 触发 `_complete_context` 等价检查：**全部本地事件完成后 Context 才 DONE**（validator `tile.py:1298-1318`，未齐计入 `task_retirement_wait`）；W2 的 EXEC_DRAINED 是**内部通知**（发给 W6 参与退役判定），不是对外事件、不是 Task 成功 → 终态沿 §5.5 统一退役链发布。

### 5.2 事件生命周期与引用结算

登记（frontier 配额内即时分配）→ 生产者绑定（唯一）→ 消费者登记（add_consumer）→ 接受后结算（consume：两计数递减）或未接受撤销（cancel_dependencies）→ 终态＋两计数归零 → 回收。故障关 Context 时 `abandon_future_uses` 等价清零未来引用，但已登记消费者未清不得 abandon（`event_table.py:362-375`）。旧 signal（过期 sequence/错 owner/重复）分别计数拒绝且不改状态（EVT-16）。generation/序号位宽须与最大在途数、最大响应存活期配套；回绕前提是受控排空且**排空范围覆盖传输/重放/ACK 队列与重复消息寿命**（CTX-06）——位型完全相同的古老副本（ABA）无法靠不等比较识别，无法闭合时阻止回绕或保留未回绕的外层 epoch；迟到回包按 INV-15 拒绝并可观察——可观察回绕/迟到用例见 §8"回绕与迟到消息"行。

### 5.3 故障路径（fail-stop）

检测→first-fault 锁存＋摘要→本 Tile UCE 停发（引擎继续走钟排空，SCH-11）→故障报告上 Group→Group 复位 FSM（§3.5）→未终态任务以 RESET/ERROR 结束（PENDING 事件被 MARK_EVENTS 置 RESET；已终态保持，EVT-05/FLT-08）。同周期 completion/fault 收敛到单一线性化点（EVT-10）；取消与正常完成竞争只产生一个一致终态，双方路径不重复释放（FLT-05）。

### 5.4 门控交互流

Context park 请求（WAIT/WFI/停靠）→门控控制块验证无同周期到达完成、无旧写回端口占用→关该 Context 前端时钟；完成广播/命令到达→强制或自然解除→唤醒。全部 Context park 不足以关 Tile：还需在途表空＋完成通道空（CLK-03）。

### 5.5 统一退役链与可靠 completion（Group 驱动、RTL 本地 owner）

目标设计（非 validator 现状）：W2 `EXEC_DRAINED` 内部通知到 W6 → W6 检查本地无旧 in-flight、VME 区域已结束等前置后向 Group 报 **RETIRE_READY（不是 Task success）** → Group 下发 **RETIRE 授权**关闭本地 view/Arena/Frame → RTL（W4 机制、W6 序）完成本地回收并**可靠保存 TASK_COMPLETION**（此后该 Context 槽在旧义务闭合、记录按 generation 与新代隔离的前提下可复用，保守实现可等完成握手，CTX-11/CTX-15）→ Python 收到 TASK_COMPLETION 后清理剩余父 L2 pin、归还 R 并结算 Grid 记录，再满足 root 自己的 L2 view release/Arena 退休后发布 root 完成。纯读输入 pin 可先按 §3.3 的 Grid 输入聚合解除，L2 backing 按引用闭合回收；二者不等待这条退休链，也不重复释放。**两个禁止**：不得等待父 R 先 free 才发 TASK_COMPLETION（不成环）；Python 不再执行 L1 retire（本地闭合由 RTL 完成）。退役控制/ACK 保留 credit 与 terminal record，确认后才归还。Task Arena（L1）与 root Arena（L2 预留）是两条不同链：后者先于 root 成功 completion 发布（validator `retire_context_arena` 即 root 侧证据，`tile_group.py:1523-1548`；逆序为不变量错误，:1815-1823）。准入侧（Group 有界 ticket→父 pin/R 预留→RTL 本地 hold→最终 COMMIT 授权→原子安装 ACK；最终授权后不 ROLLBACK，走 CANCEL/drain）由 W1 协调、W2/W4 执行本地段，本组件承接其取消/终态后果。每 Context 终态寄存器＋溢出队列大小 `由后续规格冻结`。

## 6. 配置、PPA、性能模型和 PMU

- 本地事件表：每 Context frontier 槽数（编译器 manifest 提供）＋全局上限；Group 模型默认 4096 不作为 Tile 配置；本地上限 `由后续规格冻结`。
- 等待记录：每 Context 1 条；WAIT 恒 1 操作数，WAITALL 操作数上限＝指令编码上限（编码归 W5/ISA 评审，`由后续规格冻结`）。
- `max_drain_cycles`（模型 100）、debug FIFO 深度、first-fault 摘要寄存器宽度：`由后续规格冻结`。
- 性能关注点：完成广播扇出（TIM-01 建议本地复制/树形分发）、CAM 匹配深度、事件表读端口数（每 Context check＋广播写同周期）；clock gating 的实际进入/退出周期与活动可观测（PWR-04），不承诺零周期零功耗。
- 时间权威：Python Simulator 一 tick＝一个 RTL core cycle；统一采样/提交相位（P1 双向 pre-edge valid&&ready 快照、P3 只提交本接受集、P4 由 Python 唯一执行）由 总体设计 定义，W6 内部 FSM 不自行前进时间、C++ 侧不回调 Python。

## 7. RTL/软件实现建议

1. 事件表用寄存器堆＋每 Context frontier 窗口（基线小容量），完成广播写口与 UCE check 读口分离；slot 复用以两计数归零为条件，不需要按名清刷。
2. 等待记录 CAM 匹配键用 ContextKey＋本地事件槽号（而非全名比较）；`ctx{N}:<name>` 仅在 manifest 层映射（§6.1 允许经在途表索引恢复身份）。
3. first-fault 锁存放 always-on 域，复位类型用"摘要保留、计数清零语义明确"的专用清除（Host 确认后），不复用功能复位。
4. 复位端点 FSM 每阶段一个可观察信号（freeze/cancel/drain/mark/invalidate/credit/resume），与 Python `ResetState` 一一对应以便联仿对拍（W0/W7）。
5. 门控控制块输出分层 enable，Tile 级条件在硬件上由"各单元静止"线与产生，不从单一全局寄存器直扇出（TIM-01）。

## 8. 验证、bring-up 和验收标准

断言均以可观察信号（事件表状态、等待记录、first-fault 摘要、FSM 阶段、PMU 计数、通道占用）表达；T 编号对应需求 §19 场景表。

| 场景                                                      | 断言（节选）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T-13 事件先到/同周期/后到＋重复＋错误＋WAITALL 组合       | 三种时序均无丢唤醒：先到不 park；同周期完成优先于 park 判定；park 后命中唤醒延迟有界。WAITALL(A,B)：A 完成后仍 park、armed 保持，B 到达且全 DONE 才成功唤醒；A 或 B 任一失败终态走失败出口、不产生成功唤醒。同 owner/gen/seq 的无关事件 C 完成不得命中 A 的操作数（完整键＋scope 校验）。重复 signal 只计数、状态不变、信用不双还；错误完成逐操作数记录并进失败路径（无成功发布）。偏序：唤醒不早于其裁决所依赖的全部事件终态可见。                                                                                                                        |
| T-14 多消费者共享输入与早释放                             | 负例：只收到部分 Task 的输入信号或仍有其他消费者 claim/view/pin 时不得物理复用；不能用事件表 `future_uses` 替代 L2 引用判据。正例：全部相关 Grid 的输入聚合完成、owner release 且 backing 引用闭合后，完整输入 backing 在原 Task 仍计算时回收并供另一 root 复用；原 Task 的 L1／R／程序引用仍保留，本地 staging 不被破坏；后续退休不重复 unpin。                                                                                                                                                                                                           |
| T-15 输出已 ready、外层 store 延迟；A 已发布消费后 B 失败 | `OUTPUT_READY`≠`STORE_DONE`≠HBM 可见按 scope 分层成立；A 的已发布承诺在 B 失败后仍有效；B 绝不发布成功 ready。                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| T-23 表满/持续背压                                        | 满表只产生背压：终态与事件不丢、不变形；身份先于发送保留（EVT-09）；完成与信用回收不被 action 表满阻塞（EVT-17）。                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| T-24 取消在 store 发出后、响应前                          | generation 不当撤销用；写义务真实结算后才复用资源；终态唯一（竞争只一个结果）。                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| T-25 一 Context 故障另一正常                              | 故障归属到 task/Context/操作；本 Tile 停发；Group 复位排空整个 run；**故障线性化点之后**未终态任务不得 DONE/发布成功，**之前**已合法 DONE 的任务与已发布 `OUTPUT_READY` 保持有效；等待者由 RESET 唤醒。                                                                                                                                                                                                                                                                                                                                                    |
| T-26 reset/debug halt 与在途竞争                          | 状态可判定（未执行/完成/失败）；无消失的已接受工作；halt 停在指令/launch 边界，不破信用与 descriptor 一致性（RST-11）。                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| T-27 全 Context 等待＋后端在途＋门控边界                  | 完成路径存活、可唤醒、继续运行；门控不吞同周期到达事件；gated 周期计入统计。                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| T-45 阶段信号重复/旧代/未知/漏发                          | 重复＝协议故障且聚合只计一次；旧代忽略并计数；未知 grid/task/phase＝协议故障；聚合恰在 seen==expected 时触发一次；**漏发**：期望集合未满时聚合事件不得触发（不得提前触发或由他人代办）；`input_released`/`output_ready` 的前置 await 缺失由编译器/Loader 静态拒绝，不留到运行期（EVT-15）。                                                                                                                                                                                                                                                                |
| T-49（W6 切片）复位使驻留失效                             | 复位端点 INVALIDATE_RESIDENCY 先于 RESUME；epoch/generation 递增后旧驻留不命中（细节归 W5）。                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| T-50（W6 切片）gather/scatter 权限与过期租约故障          | scope 页违例/过期租约＝可归属故障：写故障记录（code＋TaskKey/ContextKey/操作身份），走 fail-stop 路径（FLT-01）；未成功发布的输出不得发布成功 ready（FLT-04）；已接受分片在排空闭包（accepted 清零/隔离确认）后得到**唯一失败终态**，不静默消失、不重复终态（FLT-02/14）；地址算法、重叠段 `(i,j)` 提交序、hazard 编译期拒绝的细节归 W4（GAT-11～GAT-20）。                                                                                                                                                                                                |
| 回绕与迟到消息（CTX-06/03/09）                            | 小位宽强制回绕下旧包分两类：**身份可区分**（sequence/代际不同）的旧 signal/ACK＝stale sequence 计数拒绝（`event_table.py:441-443`），不唤醒新代等待记录（无误唤醒）、不二次递增信用/引用（无重复返还，IF-10/CTX-09）；**身份位型完全相同（ABA）**的古老副本无法靠不等比较识别（`signal` 的 stale 判据是"比较不等"，位型相同即通过）——因此复用该位型前必须先证明传输/重放/ACK 队列与重复消息寿命已全部闭合（受控排空），否则**阻止回绕**或保留未回绕的外层 epoch；**同位型旧副本尚可到达时禁止复用该序号/代际**。拒绝旧数据写回仍完成握手并归还该事务信用。 |
| T-51 同周期多故障/FIFO 溢出/事件状态                      | first-fault 可查询、后续只计数；debug 缺口可观察且不阻塞功能终态；每已接受命令终态唯一；基线事务故障走 PENDING→RESET；若实现故障触发的 ERROR 事件则故障记录先于该事件可见（SHOULD，不作必验事实）。                                                                                                                                                                                                                                                                                                                                                        |
| T-52 复位 FSM 时序/超时/隔离失败                          | 阶段次序固定可观察；取消确认前不前进；超时强制取消在途传输；隔离失败 poison 并拒绝新启动直至显式静止复位。                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| T-54 控制通道独立                                         | 数据信用耗尽时事件/故障/屏障有界前进；无 HOL；信用守恒检查通过。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

负向用例（必须有）：旧 generation 完成命中新等待记录（必须被拒）；强制回绕窗口内旧代 signal 唤醒新等待（必须拒绝并计数）；迟到 ACK 二次归还信用（必须判错误，信用守恒不破坏）；**同位型（ABA）旧副本尚可能在途时复用序号/代际（必须阻止回绕或保留外层 epoch，不得复用）**；同 owner/gen/seq 的无关事件 C 满足 A 的等待（必须不匹配，完整键＋scope 校验）；WAITALL 部分完成即唤醒（必须仍 park）；终态事件再次 signal（计数拒绝，不回 PENDING）；`RESOURCE_RELEASED` 被程序 WAIT（编译/loader 拒绝）；WFI 后其他 Context 仍在跑（Tile 不得关钟）；复位期间新 Task 提交（必须被拒并可观察）；abandon 时仍有登记消费者（不变量错误）。

低能力模型实现指南（面向实现 agent，读→实现不变量→证明，不是另造降阶模拟器）：**读**——`event_table.py` 全部状态转移＋`tile.py` `_issue_wait/_probe_wait_instruction/_wait_refs_ready/_complete_context/_fault_context`＋`reset_domain.py step/_outstanding_zero`＋`tile_group.py _on_phase_signal/trigger_fault/reset/poison`；**实现不变量**——(I1) check-register 原子性：armed⇒全部操作数引用已登记，且每个完成必匹配其所属操作数（完整键）或该操作数已终态；(I1a) WAITALL 成功唤醒⇒全部操作数 DONE，任一失败终态⇒失败出口，不经成功唤醒；(I2) 终态 sticky＋两计数归零才回收；(I3) 每个 signal 至多改变一个事件状态，拒绝路径只计数；(I4) 复位阶段偏序固定，DRAIN 证据真实计数；(I5) 故障线性化点后无新 DONE，此前合法终态不撤销；(I6) 门控条件不含"仅 Context park"；(I7) first-fault 锁存先到先得，Host 确认清除前保持；(I8) 序号/代际位型复用⇒同位型旧副本已证明不可达（排空闭合或外层 epoch 隔离）；**证明**——以不变量做 SVA/k-induction，重点 T-13/T-51 竞争窗口、别名与 ABA 位型复用反例；**停止条件**——全部 T 行断言在联仿中对拍通过＋不变量无反例，即停止加约束。

## 9. 风险、取舍和后续细化方向

1. 同周期"完成优先"仲裁对 UCE 关键路径的影响（CAM 匹配与 park 判定同拍）——待 review，可选"park+wake 同周期"等价实现，但必须二者择一固定。
2. Tile 侧复位端点与 Python Group FSM 的职责切分（端点只执行本地段、Group 持顺序权威）是否会在 CS1 联仿中引入双 FSM 漂移——待 review（与 W0/W1 联合）。
3. frontier 配额的 RTL 承载（每 Context 槽 vs 全局池）与编译器 manifest 字段格式——依赖 W5 manifest 冻结。
4. `VME_REGION_DONE` 映射到本地事件命名空间的确切形式（`ctx{N}:<name>`）与 manifest 映射——依赖 W3 接口定稿。
5. debug FIFO 与 first-fault 摘要的物理边界（寄存器数、always-on 域功耗）——`由后续规格冻结`。
6. WFI 语义改造对既有 Coral 软件流（FPGA 每次 Run 重写起始 PC）的兼容影响——本阶段无 FPGA，仅记录。
7. ABA 位型复用的机制选择（受控排空阻止回绕 vs 保留未回绕外层 epoch）、外层 epoch 位宽与寿命上界——待 review，依赖 W1 通道重放/ACK 寿命证据。

依赖：W2（UCE/Context 状态机、发射握手）、W3（VME region/实例清理钩子）、W4（传输取消/隔离确认、MSHR/accepted 计数）、W5（manifest 事件声明、驻留失效）、W1（CTRL/EVENT 通道与信用）、W0/W7（联仿对拍与观测）。接口先共审、单元可独立验证；集成依赖按总体设计 §7.3，不把双向接口误作循环开工顺序。

[H06]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/CoreAxi.scala
