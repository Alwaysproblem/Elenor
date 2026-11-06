# W4 内存子系统已评审设计（LSU／MFE／L1 bank SPM+cache／MSHR／gather-scatter／Profile／Arena）

> 本组件设计已通过用户 review，推荐方案采用情况与参数冻结任务见[实施总计划](../plan.md)；跨包合同见[总体设计](../Coral_Nexus_Tile_Design.md)，r5 指[需求规格](../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md)。[设计验证记录](../Design_Validation.md)不代表组件已实现。

状态：设计已获用户 review 通过；推荐机制用于实施计划。机械深度／位宽／预算由具体前置任务冻结，未开始实现。
范围: CS1 Tile 内存相关全部 owner。**Tile 内一切可变内存状态的唯一执行 owner 是本包 RTL**（LSU/MFE/L1 bank/SPM+cache/MSHR/Arena 账本/Frame/program SRAM 物理；本地事件归 W6）；Python 只保留 Tile 外部服务（Group 调度、L2/cache/MSHR 存储器服务、NoC、HBM）与父资源账本（R lease、L2 pin、页授权权威、root retirement），并提供只读 shadow/静态 plan 镜像，**不代执行、不代写 L1、不代发本地完成**（总体设计 §2；W1 全 Tile 联仿口径）。
需求基线: `../../../Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md` r5 §10（MEM-01…13、DMA-01…13、GAT-01…20、ADR-01…07）与 §11（PRO-01…17、ADM-01…12、ARN-01…09、FRM-01…04、RES-09…19）；RST-07/08 排空、FLT-02/04/05 取消合同；模块入口见 r5 §20.1 `scalar/Lsu.scala` 行、§20.1.1 "本地 L1 存储子系统"/"scope 授权服务" 新增行。

## 1. 定位、目标和 First Silicon cutline

把 Coral 基线（单 `LsuV3` SuperSlot + 队首输入、全核 ITCM/DTCM、`CoreAxi` 全核门控，r5 §3.2 [H05][H06][S08]）改造为**多 Context、bank 化、有代际与 profile 生命周期、可取消可排空**的 Tile 本地存储子系统；Tile 自主产生合法 load/store/gather/scatter/搬运请求但不无约束直达全局内存（r5 §10.1）。

- 一份可变状态一个 owner: 每 InFlightOp 归 LSU；L1 腿/本地信用归 L1Fabric；每 bank 归 L1Bank；L1 Arena 账本归 L1ArenaManager；profile 成员状态归 Profile Member；program SRAM 物理归 TileProgramMemory。Python 外部腿（L2/HBM/NoC/global-DMA）与父账本不搬进 Tile。
- 本文档为 RTL owner 划定边界/接口字段/合同与验收，不是文件级实施计划；凡引 Python 模型处均为**联仿对照基准**（行为参照，非原样保留对象）。

## 2. 职责、非职责和 ownership

| 状态/资源                                                                                                   | 执行 Owner（每状态唯一）                                                                                                                                                    | Python 侧（外部服务/父账本/只读镜像）                                                                  |
| ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 发射队列、InFlightOp 表、AXI/Response 匹配、分片与计费本地记账、取消状态机                                  | LSU（每 Tile 1）                                                                                                                                                            | Host checker 只读 shadow，不代发完成                                                                   |
| descriptor 解析、lane 选择、Gather/Scatter 有限队列、(i,j) 有序覆盖、首故隔离                               | MFE（`MfeDescriptor`/`GatherSegmentJob`/`ScatterSegmentJob` RTL FSM）                                                                                                       | 完成事件经事件流上报父级                                                                               |
| indexed 求址、边界/溢出/对齐判定                                                                            | SegmentResolver（RTL，纯函数）                                                                                                                                              |                                                                                                        |
| L1 物理 bank（SRAM 阵列、bank busy、SPM+cache 并存分区、逐 bank inflight 引用）                             | L1Bank ×BANKS（每 bank 独立服务；跨 bank 事务由 L1Fabric 统一汇总）                                                                                                         |                                                                                                        |
| SPM/cache 模式 shadow 切换、代际校验、成员 FSM（L1 issue gate 执行点）、范围维护阻塞表                      | Profile Member（Tile 侧；**gate 的 owner 是 Python ProfileController**，见 §3.5）                                                                                           | 12/9 步命令流调度、ACK 校验（profile_controller.py 合同）                                              |
| L1 MSHR 分配/合并/cancel 语义、L1 cache 行/脏/tag                                                           | MSHRTable(L1) + L1Cache                                                                                                                                                     | L2 MSHR/L2 cache 仍为 Python（Tile 不实例化）                                                          |
| L1 腿执行（L1_READ/L1_WRITE/L1_CACHE_LOOKUP/L1_CACHE_FILL、bank 等待）、L1 本地信用记账/归还                | L1Fabric（从现 `TransferManager` 中切除 L1 legs 后由 RTL 承接；Python 版重构为仅外部腿服务）                                                                                | Host checker 只读对照，不写也不发完成                                                                  |
| L1 Arena 逐 bank extent 账本（hold/bound 预留与回收/commit/退休）、view/pin/inflight 计数、arena generation | L1ArenaManager（RTL，Tile 本地权威；原 Python tile.plan_admission/commit_admission/abort_admission 与 arena.py L1 部分的执行职责由本模块替换，Python 只留只读静态布局镜像） | Group 持父侧 ticket 序、父账本与 `RESOURCE_COMMIT_IND` 镜像对拍（W1 §3.4）；HOLD→COMMIT 授权流（§3.7） |
| Frame Slot→Arena view 绑定/失效、frame generation                                                           | SlotFrame（RTL 执行 FRM-02 校验/FRM-04 bind，校验不在 launch 关键路径）                                                                                                     | `frame_bind_cycles` 计入 Task 启动延迟（FRM-04）                                                       |
| task 级 L1 Arena 退休                                                                                       | L1ArenaManager RTL（Group RETIRE 授权后按 ARN-05(a) 本地回收并保存 TASK_COMPLETION，§3.7）                                                                                  | R lease、父 L2 pin 释放、L2 backing final-free、root retirement（ARN-05(b)(c)，T-57）                  |
| program SRAM 物理阵列、安装写端口仲裁（RES-12 共享端口公平仲裁与如实计数）、执行中改写保护（FET-06）        | TileProgramMemory（本包物理 owner）                                                                                                                                         | 驻留身份/引用计数/PROGRAM\_\* 消息语义→W5（安装走保留通路，不占数据 Arena/MSHR，RES-17…19）            |
| L2/HBM/NoC/global-DMA 外部腿执行、外部 outstanding 池、HBM source capture/dest commit                       | （Python，不变）TransferManager 外部腿服务                                                                                                                                  | L2 view 引用 acquire/release 仍是父账本（ARN-09）                                                      |

职责不变式: InFlightOp 引用的 view/Frame 与地址在旧访问义务闭合前不复用；bank 服务占用、队列 credit、MSHR token 各在自身所属阶段闭合后返还，不能与整个 op 生命周期混成一个计数；hold/bound 容量不被并发 ticket 重复预留；每 bank `allocated + free == user_spm_bytes_per_bank` 且 `allocated = held + bound`（PRO-16 + hold 守恒，逐 bank；user 容量扣除 system reserved 与 tag/MSHR/ACK 独立面积另计）。RTL 侧任何模块不得把不可变静态 plan 当作可执行旁路；物理写只发生在 RTL bank 端口与 TileProgramMemory 安装口。

## 3. 微架构和状态机

### 3.1 LSU 结构（LsuV3 多槽改造）

```
UCE Issue ──> IngressFIFO(ContextKey, op_seq, kind) ──┐  (每 ctx 深度 ≤ outstanding 上限，参数待冻结)
UCE Event Bus <─> EventRouter                  │
VME drain/out   ──> 旁路 ingress（不占队首）    │
        +-------------+      +---------------------+
        │ IngressSel  │      │ OutStanding[ctx_id] │
        │ 可推进轮转   │<────>│  每槽 InFlightOp    │
        +-------------+      +---------------------+
              │ issue 槽位空闲 && bank 通路可用 && 数据 gate open（程序安装/维护/ACK 不查此 gate）
              v
        ┌─ L1Bank ports（BANKS 路 request/response）──┐
        └─ 外部腿 bridge（L2/HBM/NoC leg 请求，不代执行）─┘   → CompletionRouter → UCE/rob/match table
```

- IngressSel 为可推进轮转：先选非阻塞 head，再看 blocked-but-wakeable；禁止"只加深 FIFO 但保留全局队首串行化"（MEM-02）。轮转顺序/tie-break 是兼容策略（r5 §11.3），不进指纹。
- 每 ctx 记录: `ContextKey=(tile_id, ctx_slot, context_generation)、op_seq、subop_id（分片内序）、txn_id、op、addr、size、slice_id/slice_count、bank、mshr_token、credit_ids、lease_ref+scope_epoch、issued_leg、completion_state、cancel_state`。身份分域（总体设计 §3.1）: `ContextKey.context_generation` ≠ `arena_alloc_gen` ≠ `program_epoch` ≠ `profile_gen` ≠ `run_gen`，跨域字段不混用、各自单调。InFlightOp 状态机：

```
REGISTERED → ISSUING_L1 → {bank hit → COMPLETED | miss refill → REFILLING → COMPLETED}
           → (外部腿) → TRANSFERRED → CREDIT_SETTLED → COMPLETED
任一进行态 → CANCEL_REQUESTED → {COMPLETED（真实完成/全部或部分外写已发生）| ISOLATED（之后不再写）| FAULTED（错误路径）}
```

- CompletionRouter 维护 per-ctx `op_seq → completion` 匹配表，去重响应/异常回包；同 ctx 多 outstanding 合法并存（MEM-01/04/08）。
- VME drain/output 为专用旁路类进入 IngressSel 轮转，不被 load/store 长队饿死（MEM-02 对 VME 的放大效应；旁路类准入与 W3 对齐）。

### 3.2 MFE 结构（DMA-01…13, GAT-01…20）

```
UCE Issue ──> Descriptor Parser（accepted 后内容稳定，接收方拷贝，DMA-02）
         ──> load lanes / store lanes（方向分类前端，多 lane 可并行）
         ──> GatherSegmentJob FSM（index 窗口 ≤ window_entries；每段独立状态机）
         ──> ScatterSegmentJob FSM（(i,j) 有序覆盖 + per-job first-fault）
```

- **lane 选择仅为兼容策略**（validator 的"方向类内取首个空闲 lane"，r5 §11.3）：RTL 只需保证方向分类前端满时发射 stall 不丢请求并按 `engine_queue_full` 归因（MEM-11/DMA-10），硬件可自行 lane 调度；本文不再写成强制接口。
- **完成以真实握手/响应为准，不得用预测 finish_cycle 宣布完成**（联仿口径，替代现 `_MFETransferJob`/lane 链式 `start_cycle = max(cycle, tail.finish_cycle)` 的预测时序模型）：lane/descriptor 完成事件发出当拍必已收到相应 bank 端口/response 或外部腿完成返回；Gather/Scatter job 完成当拍其所有子事务已终态。预测值仅可作另列模型对照，不驱动完成或冒充实际 PMU 活动。
- Gather FSM: `INDEX_READ → resolve → LOOKUP_L1/L2 → WAIT_L1_MSHR/L2_MSHR → REFILL 类/DIRECT_RESPONSE → RESPONSE_READY → DEST_WRITE → DONE`；L2 状态名描述跨桥 leg，L2 合并由 Python L2 MSHR 完成。逐段 cache line 切分、bypass_forbidden 故障、逐级真实 line 身份维持（GAT-14）。
- Scatter FSM: `WAIT_INDEX → READY → WRITE → DONE`；同 job 内按 (i,j) 序号提交，与前序未完成段 byte-range 重叠的段必须 WAIT_PREDECESSOR；跨 job/Task 冲突由编译期拒绝与 Python dispatch 依赖承担，validator 基线无 ScatterReduce/原子/FIFO delivery（GAT-06/13）。
- **INPUT_RELEASED 语义**：编译器显式生成 SIGNAL_PHASE，UCE 按事件身份 await 该 Task 全部 L2 load 完成、且后续不再有 L2 load 后，经 W6/W1 的 EVENT 通道上报。MFE 本地 last-source-read-DONE 仅表示该 job 内部不再读源，是 internal progress，**不自动产生 Task phase**。Group 收齐该 Grid 的 `all_tasks` 后解除尚存纯读 L2 pin；随后按 buffer 的 `nest.release` 与 backing 引用闭合决定 L2 物理回收（总体设计 §4.4.1），不等计算或 Task 退休。读写 pin 不能仅凭输入阶段解除；当前没有同 Task 按 buffer／范围分批通知或单 backing 子范围回收能力。Tile 间依赖由既有具体事件承载，本包不新增 L2 barrier 要求。
- DMA-12 要区分 accepted、L2-visible、global-visible 的承诺；它们不是所有路径必经的三个阶段（直接 HBM 写可不经过 L2）。Tile 本地 store 不得直接宣称跨 Tile/HBM 可见。
- mask 语义（GAT-03）: mask-off 元素不发访问、不报错、不占 credit/segment；逐元素 mask 在现 validator 无模型（仅 tile 位图，`execution_ir.py:405-406`），列入新增 RTL/联仿真例，不称已有模型证明。

### 3.3 L1Bank（每 bank 一份；同时承载 SPM 与 cache 分区）

```
BankController:
  active_profile_mode/generation    ← Profile Member COMMIT 原子切换（由 L1ArenaManager 广播 bank 重配置完成点）
  spm_region: user SPM 窗口（容量 = user_spm_bytes_per_bank，已扣 system reserved）
  cache_region: cache 分区窗（容量由 active profile 决定，可与 SPM 并存于同一 bank）
  inflight_ref: 字节区间引用计数（>0 时该区间不可重配置/退役）
  busy_until/holder：bank 端口占用（bank access 端口级串行，多 bank 并行）
```

- `try_issue`: 目标 bank 忙→`StageWait(L1_BANK)` 待重试；空闲则原子占用（一个 leg 可含多笔多 bank 请求，各 bank 独立响应，由事务记录聚合完成与引用）。
- SPM 与 cache 在同一 bank **同时分区并存**（PRO-02 一致比例，逐 bank 同 layout）；`system_reserved_spm_per_bank`（默认 2048）不参与 arena 计账；tag/MSHR/ACK 存储独立另计（PRO-16）。
- Cache writeback/clean 决策由 L1Cache，bank 负责物理读写；bank 不理解字节内容。

### 3.4 MSHR 合并与代判（L1 MSHR 归 RTL；L2 MSHR/cache 归 Python）

- L1 MSHR entry: `merge_group((allocation_id, generation, line_offset))、token、leader、waiters、issued_refill_leg、generation`。分配：组内首个为 leader 发 refill；后续作为 waiter 合并（merge 计数去重）等 leader；**合并 key 取 provenance 三元组而非裸地址**，禁止丢代际合并（PRO-15，旧代 refill/完成拒绝，MEM-13）。
- **MSHR token 退休**：refill 到达只是第一步。保守方案保留 entry，直到已登记 waiters 各取得可靠的 response/data 所有权并结清对应 token 引用后再回收；若提前释放，必须先把每个 waiter 的数据/引用原子转移到有界响应记录，并用 token generation 防止迟到 waiter 误用新 entry。cache 行可能在 waiter 读前被逐出，不能把“写入 cache”当永久数据保证；重新 lookup/补取不重复返回或记费。
- 一切外部/内部回包到达 L1 MSHR 时：`generation` 不匹配即拒收并计数，不写新代 SPM/cache（MEM-13）。MFE `_downgrade_after_eviction`（行被维护命令逐出→释放 token、唤醒 waiter、转 bypass、计费迁移）作为 RTL 等价参考。

### 3.5 Profile Member（L1 数据访问 gate 与 12/9 步成员协议）

- **单一 writer 是 Group 侧 Python `ProfileController`**；Tile 只作成员响应，不自创命令、不把空总线当隐式 ACK。每 Tile 成员维护 `active_mode/generation、shadow_mode/generation、bus RX/TX、member 静止判定`。切换 12 步（PRO-11）：

```
ACQUIRE → CHECK_FRONTIER → CLOSE_ISSUE → DRAIN_REFERENCES → CLEAN_INVALIDATE
→ DRAIN_DOWNSTREAM → PREPARE → WAIT_READY_ACK → COMMIT → WAIT_COMMIT_ACK
→ OPEN_ISSUE → RELEASE
```

- **gate 控制语义（修订）**: issue gate 的开关由 ProfileController 经命令驱动（CLOSE_ISSUE 置 1、OPEN_ISSUE 置 0），**RTL 自身不自行打开 gate**；gate 影响的仅是**新的用户数据类 L1 访问**（LSU/MFE/RVV/VME 数据腿、`_step_task_admission` 新 Task/新 Route 登记）——旧数据访问的完成/ACK 返回流、**程序安装路径（PROGRAM_INSTALL 走 TileProgramMemory 保留通路，RES-17，T-66）**、范围维护 clean 下游事务与排空均独立不受阻。Context 可执行（取指）与程序安装是两个独立 gate；安装完成/PROGRAM_READY 合同归 W5。
- gate 打开事件沿**事件通路**（gate 由 ProfileController 拉低 → `gate_open(l1, generation)` 事件）唤醒因 gate 受阻的 Task/Route retry（§5.3）；RTL 不做逐 tick busy-poll。
- PREPARE 前成员必须静止（PRO-12）：view/pin/inflight/arena 引用计数全零、脏行 clean/invalidate 完成、MSHR 空、无 pending clean，才写 shadow 并回 READY_ACK；COMMIT 校验 shadow 与已准备事务一致后原子切 `active_mode` 并递增 L1 generation；不匹配则维持旧 active 并报错（gate 保持关）。
- 超时（模型值 2000000 cycle，PRO-17，由后续规格冻结）→ gate 锁闭进 recovery，需显式 reset_mode 重驱动、generation 递增，不自动恢复。
- 范围维护（PRO-14，9 步）仅阻塞与范围重叠的新数据请求（`range_issue_blocked(level, allocation_id, generation, offset, size)`）；脏行 clean 产生真实下游事务并等其完成才 ACK。

### 3.6 L1Cache（DeterministicLRUCache 语义）

- 行结构 tag + per-byte validity + data + dirty；provenance 绑定 `CacheLineIdentity(allocation_id, generation, line_offset)`。refill/clean/invalidate 携带 generation；`reconfigure(capacity, generation, write_policy)` 仅在 Profile Member COMMIT 后调用。
- `begin_maintenance()`/`complete_clean()` 与 bank/MSHR 配合：clean 产生真实下游腿，等待外部腿终态才 ACK，失败回滚行状态（不静默丢弃）。

### 3.7 L1ArenaManager / SlotFrame / L1Fabric（RTL 唯一执行 owner；准入与退休走父授权流）

- **准入 4 相位（总体设计 §4.3；与 W1 §3.4/§3.5 FSM 对齐）**: ① Group 发版本票据 PLAN（无副作用，ADM-01）；② PARENT_PREP（父 L2 pin/R 预留，Group 记账）后 RTL 本地 **LOCAL_HOLD**：L1ArenaManager **排他预留** extents/Frame 容量——立即从 free 账转入 **hold 守恒**（`allocated = held + bound`，逐 bank 两段各自守恒），写票据身份并 ACK(hold_ready)；hold 期间不 bind 可执行 UCE、不取指、不动 view 数据，但该容量已不可被并发 ticket 再预留（否则 PLAN→授权窗口内同一 bank 容量可被重复承诺）。③ Group 确认父+本地全部持有（P4 记账，`RESOURCE_COMMIT_IND` 镜像与 Group ledger 对拍）后发最终 **COMMIT 授权**；④ RTL 原子 **BIND_INSTALL**：held→bound 落账（写 arena generation）、SlotFrame bind、ACK(accepted)。授权前失败 → **COMPENSATE**（held→free 原子回转、release Frame，对应 `abort_admission`/`_unpin_task_l2` 语义，tile.py:1598-1631）并 ACK，票据与 hold 记录保持到释放证据可查；**最终授权后不再 ROLLBACK**，只能走 CANCEL/drain（§4.4）。同一提交不跨 Tile 联合（VB-03），无 gang commit。物理 opcode 编码待 review。
- **L1ArenaManager 数据面**: 持逐 bank extent 账本、view/pin/inflight 计数、arena generation；commit 原子、逐 bank `allocated+free==user_spm_per_bank`（PRO-16）与 ARN-08 计数一致性为 RTL 断言；plan 侧 Python 只保留不可变静态布局镜像（只读），不发物理 bank 旁路、不再执行 L1 retire（总体设计 §4.4）。
- **SlotFrame**: 每 Context 1 Frame、Slot 数为指纹值；ALLOC_L1 绑定编译期固定 Slot、FREE_L1 只失效该 Slot（FRM-01）；prepare 完成静态校验（bank 边界/系统保留下限/对齐/重叠/总额一致/generation 匹配，FRM-02，校验不在 launch 关键路径）；`frame_bind_cycles`（模型 8，FRM-04）计入 Task 启动延迟；失败=`l1_frame_fault` 不静默。bind 后 engine 只用 shadow 副本。
- **L1Fabric（RTL）**: 执行全部 L1 腿（L1_READ/L1_WRITE/L1_CACHE_LOOKUP/L1_CACHE_FILL、bank 等待）、本地 credit 记账/归还、L1 内 capture/commit 触点；原 Python `TransferManager` 中 L1 legs/data 部分被本模块替换（允许重构），TransferManager 只保留 L2/HBM/NoC/global-DMA 外部腿与父账本角色。
- **Group 驱动退休（总体设计 §4.4，CTX-15/ARN-05 RTL 本地 owner + Group 授权）**: ① 本地无旧 inflight/VME 结束后 RTL 报 **RETIRE_READY**（区别于 Task success 信号）；② Group 经 **RETIRE 授权** 关闭本地 view/Arena/Frame（含取消隔离确认）；③ RTL 完成本地回收并可靠保存 **TASK_COMPLETION** 终态记录；④ Group 收 TASK_COMPLETION 后清理剩余父 L2 pin、归还 R 并结算 Grid 记录，再走 root 自己的 L2 release/Arena 退休并发布 root 完成。输入纯读 pin 早解除与 L2 backing final-free 按 §3.2 独立推进，不等待本地 RETIRE 授权。**RTL 不等父 R 先 free 才发 TASK_COMPLETION；Python 不再执行 L1 retire**。退役控制/ACK 使用的 credit/terminal record 保留在独立池，Context 复用前旧义务必须闭合、记录与新代隔离（保守可等完成握手的旧代 Context 记录），不与新一代数据混账。
- 联仿对照基准（行为参照，非保留实现）: `tile.py:1541-1630`（plan/commit/abort）、`arena.py:789-880`（invalidate/pin/inflight）、`tile.py:893-976`（`_free_l1`）、`transfer.py:288-474`（L1 stage 行为）；per-leg 对账（leg 偏序、真实 capture/commit 观察点、credit/free_epoch、WAIT 原因；不要求 Python/RTL 周期相等）。CS1 以 RTL 为执行权威。

## 4. 接口、descriptor、寄存器和协议

### 4.1 上下游

| 边界                | 方向     | 内容                                                                                                                                                                                                                       |
| ------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UCE↔LSU/MFE         | req/resp | 发射端口（kind, **ContextKey**, **op_seq**, **txn_id**, desc_ref）；完成事件（`completion_event`, seq, kind）匹配 op_seq；容量/类型见 r5 §5.1（MEM-11 指纹）；物理 opcode 编码待 review，不自造固定位编码（总体设计 §4.3） |
| LSU/MFE→L1          | req      | bank 端口（bank_id, op, addr, bytes, op_seq, txn_id, token）                                                                                                                                                               |
| L1→MSHR             | —        | allocate/complete/cancel（merge_group, generation）                                                                                                                                                                        |
| 外部腿 bridge       | bi       | RTL `TransferOutbridge`（仅外部腿）↔ Python 外部腿服务；wire 只承载 **地址/范围/权限/ID/data**，不传 Python view 对象                                                                                                      |
| 父 L2               |          | 不得在 Tile 实例化 L2/cache/MSHR                                                                                                                                                                                           |
| Group 准入/退休     | bi       | ticket→HOLD→COMMIT 授权 / RETIRE_READY→RETIRE 授权→本地回收→TASK_COMPLETION（§3.7）；相位命令为逻辑名，编码待 review                                                                                                       |
| 页授权（GAT-15…20） | bi       | 见 §4.7                                                                                                                                                                                                                    |

### 4.2 消息契约与字段完整性

边界消息版本化、有界、可序列化（总体设计）。**最小字段集合**（任何内存请求/响应）：

| 字段                                 | 语义                                                                                                                          | 缺失后果                       |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| op_seq / txn_id / subop_id           | 操作序号/事务 id/分片 id（MEM-03 唯一匹配；分片与重排需要三者能互相定位）                                                     | 丢匹配→不变量故障              |
| **ContextKey**                       | (tile_id, ctx_slot, **context_generation**)（总体设计 §3.1）                                                                  | 缺 generation→错主风险→fault   |
| kind                                 | load/store/gather/scatter/dma/index_read 等（逻辑名）                                                                         | 无法路由→reject                |
| addr / size / slice_id / slice_count | 地址与分片（外部腿桥只按 地址/范围/权限/ID/data 表达）                                                                        | 非法→reject                    |
| generation（分域）                   | **context_gen**（ContextKey 内）/**arena_alloc_gen**/**profile_gen**/**scope(page) epoch**/**run_gen** 分域标注，不混用单字段 | 域错→fault（PRO-15/ADR-06/07） |
| credit_id / mshr_token               | 信用引用/MSHR 组引用                                                                                                          | 重复/过期归还→fault            |
| lease_ref / epoch                    | scope 页引用（仅 indexed 路径）                                                                                               | 缺失→fault                     |
| cancel_flag / final_state            | 取消请求或终态声明（§4.4 三终态）                                                                                             | 缺失则无法结算                 |

- `CreditManager` 每 ctx 信用上下限（上限=该 ctx outstanding 上限，机械参数待评审）；`issue` 占用、终态归还；`release/abort/cancel` 一致性检查防下溢/双释放/泄漏（MEM-08）；复位时 ctx 信用归初始水位并清 MSHR/match table（RST-07/CLEAR_CREDIT_CACHE）。

### 4.3 资源管理侧交互（retire/cancel）

- 取消: UCE/W6 发 `TaskCancel`；本包按 §4.4 结算并上报终态记录（保留 credit/terminal record 池，§3.7）。
- 退休（Group 驱动，总体设计 §4.4）: 本地无旧 inflight/VME 结束→RTL 发 **RETIRE_READY**（区别于 Task success）；Group 回 **RETIRE 授权**；RTL 关本地 view/Arena/Frame（含取消隔离确认）、完成本地回收、可靠保存 **TASK_COMPLETION**；Group 收后清理剩余父 L2 pin、归还 R 并结算 Grid 记录与 root 侧 release/退休。输入早释放按 §3.2，已解除的 pin 不重复回收，也不授权 L1 Arena／Frame 归还。本包不等父 R 释放；不发出未经 Group RETIRE 授权的本地关闭。
- `FreeSlot(view_id, ContextKey, arena_gen)`: SlotFrame/L1ArenaManager 检查 pin/inflight 引用后 ack；否则报引用未闭合并维持 view live。Context 复用前旧义务闭合、terminal/credit 记录与新一代隔离。

### 4.4 终态 ACK 协议（与 MEM-12/FLT-14/FLT-05 对齐）

**终态集合**: `COMPLETED | ISOLATED | FAULTED`（三态；取消路径的最终终态仍是这三者之一，不存在"只到 COMPLETED/ISOLATED"的过窄口径）。`ISOLATED` 承诺"从隔离点起不再发生外部写"，**不承诺整个 op 从未发生外部写**——此前已完成的分片/已发出的外部写义务保留；`FAULTED` 同理保留已发生的外部效果并要求错误传播与义务结算后再回收引用。`CANCEL_REQUESTED` 只是请求标记不是终态。

| 状态               | 谁发出 | 何时发出                           | 接收方动作                                                                                                                                 |
| ------------------ | ------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `COMPLETED`        | 本包   | 全部引用 leg 终态+外部写义务已结算 | 引用/credit 归还，可退休                                                                                                                   |
| `ISOLATED`         | 本包   | 隔离确认后（后续无任何外部写）     | credit/view/lease 引用归还；已发生外写义务保留记录                                                                                         |
| `FAULTED`          | 本包   | 错误传播路径（§4.5 错误码表）      | FLT-02 保留 owner，FLT-08 基线由 Group fail-stop 与 MARK_EVENTS(PENDING→RESET) 终止依赖，不能假定已有逐事件 ERROR 扇出；已发生外写义务保留 |
| `CANCEL_REQUESTED` | UCE/W6 | 取消请求发出                       | 仅是标记；最终仍按 COMPLETED/ISOLATED/FAULTED 结算                                                                                         |

- 每个已接受 op/Transfer/命令恰好一次终态；引用/credit 在**实际终态**后才回收（MEM-12，`transfer.py:1690-1696` acknowledge 合同）；重复 cancel/取消与完成竞争只产单一一致终态（FLT-05）。重复/过期响应按 `stale_response` 丢弃并计数；迟到 NoC 回包按 MEM-13 拒收并先归还 flit 信用。

### 4.5 错误码表

| 错误码                          | 语义与触发条件                      | 后续动作                                         |
| ------------------------------- | ----------------------------------- | ------------------------------------------------ |
| `address_out_of_range`          | 地址加法溢出/越界（ADR-04）         | 单事务 fault，归还 credit，传播 owner            |
| `frame_slot_fault`              | FRM-02 校验失败                     | `l1_frame_fault`，Task fault                     |
| `l1_view_double_free`           | 同 view 重复 FREE                   | 不变量故障                                       |
| `engine_queue_full`             | MFE 前端容量满                      | 阻塞所属 Context，下周期重试（MEM-11）           |
| `l1_cache_maintenance_conflict` | 切档/维护与活跃 inflight/脏行冲突   | PREPARE 失败，gate 保持关                        |
| `stale_response`                | 旧代/重复响应                       | 丢弃并计数，不算完成（MEM-13）                   |
| `scatter_segment_write_fault`   | Scatter 单段写 FAULT/CANCELLED      | 首故锁存，停止新 issue 并隔离在途                |
| `host_scope_inactive`           | 页 lease 过期/属主不符（GAT-15/19） | 目标访问发出前 fault（`memory_scope_violation`） |

### 4.6 背压

| 位置                    | 触发条件                             | 行为                                                |
| ----------------------- | ------------------------------------ | --------------------------------------------------- |
| LSU ingress             | 该 ctx outstanding 满/IngressFIFO 满 | 阻塞该 ctx，不阻塞其他 ctx，下周期重试（MEM-02/11） |
| MFE 前端                | 方向类全部 lane 满                   | `engine_queue_full` 归因，下周期重试                |
| Gather/Scatter job 容量 | 同类活跃 job 满                      | 新 dispatch 阻塞，不覆盖                            |
| MSHR 分配               | L1 MSHR 满                           | WAIT_L1_MSHR，版本跳变重试                          |
| L1 bank 忙              | 目标 bank busy                       | `StageWait(L1_BANK)`，逐周期重试                    |
| HBM/DMA/NOC             | Python 外部腿服务                    | StageWait 原因随 leg 保留并观测                     |

### 4.7 页授权接口（GAT-15…20）

- Python 侧（不变）: 页属主/页表/epoch 递增/撤销决策/`AccessLease.acquire/release` 簿记（现 `page_pool.py`，MFE indexed 尚未接入逐段 lease——GAT-15/X-23 明示）。
- RTL 侧（新增）: 解析远端索引得到 byte-range 后，先经 `lease_check(segment_range, lease_ref, epoch)` 逐页确认（跨页逐页检查），通过才发起 cache lookup/下游请求；`lease_ref/epoch` 随 InFlightOp/SubTxn 保存到终态 ACK 才由 Python 收回（GAT-19）。撤销序列（关闭新访问→等旧引用与不可撤销写结算/隔离→失效副本收 ACK→递增 epoch→换 owner，GAT-20）中"关闭新访问/失效"是本包消费的显式消息。
- 单一串行化点（GAT-19）: 对同一授权的"检查+取得引用"原子化（`AccessLease.acquire` is_valid 校验未过即 `host_scope_inactive`）；RTL 不做"先发后授权"。撤销等待仅阻塞相关请求，控制/取消/旧完成持续推进（非全局 HOL）。本条是新增端到端实现验证项，不得称已有模型证明。

### 4.8 Python 保留区（切分后）

- **外部腿服务（不变）**: L2/HBM/NoC/global-DMA 外部腿执行、外部 outstanding 池、L2 Arena 生命周期（`nest.release` 永久 forfeiture、所有 claim RELEASED/CANCELLED、`l2_extent_release`、`pool_version` 推进、root retirement，ARN-04a/04b，T-57）、R lease（CTX-13）、父 L2 pin 与 L2 view 引用 hook（ARN-09）、页授权权威（§4.7）、`admission_version`/WAIT\_\* 归因（§5.3）、Group 准入/退休授权流（§3.7）。
- **从 Python 切到 RTL 执行权威（允许重构现对象，Python 版不再保留执行路径）**: L1 腿执行与 L1 bank 等待（原 `TransferStage`/`TransferManager` 内 L1 legs 由 L1Fabric 替换），本地 credit，L1 Arena 账本（原 tile.py/arena.py 的 L1 admission/commit/回滚），SlotFrame 绑定/校验（原 l1_slot_frame.py），L1 MSHR（原 mshr.py 的 L1 使用侧），L1Cache，TileProgramMemory 物理。Python 侧只保留只读镜像/静态 plan 快照与外部腿服务——**不发物理 bank 旁路，不代写 L1，不代发本地完成，不执行 L1 retire**。
- RTL↔Python 边界 wire 只承载 地址/范围/权限/ID/data（§4.2），view 对象不跨界；对账以 per-leg 语义+cycle 事件进行（W1）。
- 小 debug FIFO 与特权外存导出的**执行**由 W6 拥有（W6 owns FIFO/导出控制，W7 只做 Host 侧格式与 checker）；本包只作为其外存事务的执行腿，该导出事务同样受取消结算约束（先按 §4.4 终态再归还 credit）。

## 5. 数据流、控制流和时序路径

### 5.1 load/store（LSU）

```
UCE issue → IngressFIFO（记录 op_seq/ctx/kind/desc）
IngressSel(逐周期): 找可推进 head
  可推进 = 信用有余 && 目标 bank 可接 && L1 数据 gate 开（与程序/维护/ACK 无关） && Frame 已 bind
  → 分片 → 逐片走 L1Bank（bank hit → CompletionRouter；miss → L1MSHR → leader 发 L2/HBM refill 外部腿）
  → 外部腿（不可 cache 或非 L1 地址）: TransferOutbridge → 外部腿服务 → 逐腿完成返回
  → 写回: 更新 L1Cache 行/脏位或 bank SPM 字节；外部 store 记录已发出的外部腿以便取消/隔离判定
每片完成 → 归还本地 credit → 更新 InFlightOp → 全部腿终态后响应 UCE（MEM-03 唯一匹配 op_seq）
```

### 5.2 Gather/Scatter（MFE；以 validator 语义对照为验收权威）

```
Descriptor 接受（内容稳定，接收方拷贝，DMA-02）
→ 方向分类前端（stall 不丢请求，engine_queue_full 归因）
→ Gather: index window ≤ window_entries；INDEX_READ → decode i32 → resolve（负索引/溢出/越界 → job 故障 GAT-12）
   → 逐段 cache-line 切分（whole-line refill / partial-line bypass，bypass_forbidden 故障） → 逐段 FSM（§3.2）
   → RESPONSE_READY 段按 ordinal 顺序 DEST_WRITE（不乱序写回，`gather_reorder_wait_cycles` 观测）
→ Scatter: 同 window；段进入 ordered-overwrite 队列；与前序未完成段重叠 → WAIT_PREDECESSOR；
   READY → SCATTER_WRITE leg（L1_READ 读源 + HBM_WRITE 提交） → DONE；
   任一腿 FAULT/CANCELLED → first-fault 锁存 → 停止新发射，隔离在途（已终态段 ack 引用）→ 等真实完成或隔离确认
→ job 完成以全部子事务实际终态为准（不用预测 finish_cycle）
```

### 5.3 数据 gate → Route/Task retry 通路（ADM-06/ADM-10）

- CLOSE_ISSUE 起，`_step_task_admission` 直接 return（tile_group.py:1231），`dispatch_role` 直接 BACKPRESSURE（tile_group.py:1191-1192）。`admission_version` token（tile_group.py:2070-2078）不含 L1 gate；ProfileController 打开 L1 数据 gate 时发出 `gate_open(l1, generation)` 事件，父侧据此推进 admission_version，`WAIT_CONTROL_RESOURCE` 等待者经事件唤醒。RTL 不自行开 gate、不做逐 tick busy-poll。
- gate 关闭期间旧完成/ACK 流、程序安装、维护 clean 持续推进（§3.5）。

### 5.4 与复位协作（RST-07/08）

- `RST-08` 超时（模型值 100 cycle，由后续规格冻结）后强制取消在途传输与 MFE job；`CANCEL_REQUESTED` 不会因超时"洗白"成成功——仍按三终态之一结算。
- 复位完成后各 stage/引擎/MSHR 计数归零（`DRAIN_SAFE` 观察点：无 HBM outstanding、无 stage 忙、NoC 静止、accepted=0、MSHR 空），credit 重建（CLEAR_CREDIT_CACHE）。
- L1 数据 gate 在 recovery 内保持关闭直至显式 reset_mode 重驱动成功（PRO-17）。

### 5.5 每路径验收矩阵（输入/输出字段 + read/write 可见性 + terminal ACK + credit）

| 内存路径                              | 请求必须字段                                | 返回/完成字段                            | read/write 可见性锚点                                     | terminal ACK（终态）                                        | credit 路径                                   | T 锚点                  |
| ------------------------------------- | ------------------------------------------- | ---------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------- | ----------------------- |
| LSU load                              | op_seq/ContextKey/kind/addr/size/arena gen  | data 或 fault_reason                     | SPM/cache 命中本地可见；miss 由 refill 合法化             | COMPLETED / ISOLATED / FAULTED                              | 申请/归还对称；下溢/双释放=不变量故障         | MEM-01…08；T-20/T-24    |
| LSU store（本地）                     | 同上+write 数据                             | terminal ACK（写回 done）                | writeback 决策由 L1Cache；bank 写入立即本地可见           | COMPLETED / FAULTED                                         | 同上                                          | MEM-02/08；T-20         |
| LSU store（外部/HBM binding，ADR-06） | 同上 + binding guard（范围/权限非重叠）     | 分级完成事件（DMA-12）                   | 按实际路径报告 L1/L2/HBM 可见 scope，不补虚构的中间层     | 外部写已发生≠回退为 ISOLATED；隔离=此后不再写；外写义务保留 | 外部 leg 计费（父侧），本地 credit 对应片归还 | MEM-09/12/10；T-24/T-48 |
| Gather indexed                        | op_seq/ctx/lease_ref/epoch/arena gen/window | per-request ordinal 完成 + payload bytes | L1 hit/L2 refill/bypass 逐级真实 line 身份                | 段级→job 级三终态；索引/越界 job 故障                       | index+段各自计费；bypass 不占 cache 位        | GAT-05…14；T-16         |
| Scatter indexed                       | 段 (i,j)+重叠检测                           | 段级完成/隔离/故障 ACK                   | 目标写经两级 cache 旁路（GAT-13）                         | 段级三终态或首故隔离                                        | 段级计费，重叠等待不重复扣                    | GAT-06/13；T-16         |
| 页授权访问                            | 同上 + lease_ref/epoch                      | 授权引用终态 ACK                         | 旧 epoch 不命中新数据；撤销先关新访问                     | 引用持到终态（GAT-19）                                      | 授权引用独立于 credit                         | T-65                    |
| MFE 直搬 lanes                        | desc_ref/op_seq/ContextKey                  | lane 完成事件（真实握手）                | per-leg L1 ref/read/write；last-read 为 internal progress | COMPLETED / ISOLATED / FAULTED                              | 本地分片 credit 独立于前端队列                | DMA-01…11；T-23         |
| Profile 切换访问                      | 命令 id/成员 id/step/generation             | PREPARE/COMMIT ACK                       | 12 步（§3.5）                                             | 失败/超时→锁 gate 进 recovery；恰一次终态                   | 不占用户 credit                               | PRO-11…17；T-46         |
| 程序安装/取指                         | slot/offset/program_id+epoch                | 安装完成事件                             | 走保留安装口，不经数据 gate                               | PROGRAM_READY 与安装 ack（W5 合同）                         | 不占数据 credit                               | RES-12/17…19；T-66      |

## 6. 配置、PPA、性能模型和 PMU

以下所有数值除标注外均为 validator `HardwareConfig` 模型基线（`hardware_config.yaml:16-151`），属 simulator_experiment 建档值，最终由 SRAM/PPA profile 冻结，RTL 以参数化生成为准（PRO-16 缺失不得默认 0）；不新增无证据参数（ctx outstanding 与窗口按受支持配置和进展证据评审）。

| 参数                       | 基线（hardware_config.yaml）                                | 备注                                                     |
| -------------------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| L1 total / banks / bank bw | 1 MiB / 16 / 512 GB/s per tile                              | `tile_l1.*`                                              |
| L1 bank access latency     | 1 cycle                                                     | `tile_l1.access_latency_cycles`                          |
| L1 cache lookup / MSHR     | 1 cycle / 16 entries                                        | `tile_l1.cache.*`（MSHR 数为 RTL 实现锚）                |
| cache line                 | 64 B                                                        | `cache.line_bytes`                                       |
| L1 profile 档位            | spm/cache per bank 65536/57344+8192/49152+16384/32768+32768 | `profile_source.l1`（PRO-16 建档）                       |
| system reserved per bank   | 2048 B                                                      | 缺失不得默认 0（PRO-16）                                 |
| program SRAM               | 65536 B（8 kB ITCM 是 Coral 参考值 [S08]）                  | `tile_program_sram.capacity_bytes`                       |
| MFE 深度/launch/通道       | pipeline 4 / launch 3 / load+store 各 1                     | `mfe.*`（指纹值）                                        |
| stream_buffer/Queue        | 0（未启用）/ depth 3                                        | `mfe.stream_buffer_bytes` / `stream_queue.default_depth` |
| Profile 命令超时           | 2000000 cycle                                               | 超时锁 gate（PRO-17）                                    |
| Frame slot / bind          | 16 slots / 8 cycles                                         | `control.slot_frame.*`                                   |

性能观测（PMU/memory trace，不作合同）: per-bank busy、per-ctx credit 峰值、MSHR 合并率、`gather_l1_hits/gather_l2_hits/gather_hbm_misses/gather_cache_bypass_requests/gather_requests/gather_reorder_wait_cycles`、`scatter_overlap_wait_cycles`、逐腿 StageWait 分布、HBM outstanding 峰值、profile 切换分步耗时、`l1_free` 与 Arena 退休计数对账。

## 7. RTL/软件实现建议

1. 以现 `TransferManager`/`TransferStage`/`tile.py`/`arena.py` 语义行为为**联仿对照基准**（不保留执行路径），RTL L1Fabric/L1ArenaManager/SlotFrame/MSHRTable 逐语义映射，比较字节与事件偏序而非周期相等；重构允许，但外部腿服务合同不动。
2. IngressSel 可推进轮转（先非阻塞 head，后 blocked-but-wakeable），初始轮转固定；策略变更正交于 ADM-07 SAME/COMPATIBLE 队首准入。
3. SegmentResolver 首轮直接移植 `address_provider.resolve_indexed_segments` 的公式与错误码；不做 Python 式负索引回绕（GAT-12）。
4. MSHR 以 token 化表实现，merge key 用 provenance 三元组；leader 数据/引用与 token 生命周期解耦（§3.4）。
5. CreditManager 每 ctx 独立计数 + 每 op 绑定；release/abort/cancel 的一致性断言（上下限/双释放）作为 RTL SVA 常开。
6. gate：RTL 暴露 gate 状态位给 ProfileController（成员协议），open 事件广播 retry；不自行开 gate。
7. Program SRAM 物理（安装写口/取指读口与共享端口仲裁 RES-12、FET-06 改写保护）由本包承载，驻留语义归 W5。

依赖顺序（包间）：

- W2（UCE/Context）提供 UCE 绑定（在本地 HOLD/COMMIT 授权之后才允许 bind）、ContextKey（含 context_generation）；本包 SlotFrame 执行绑定物理（§3.7）。
- W6（Fault/Reset/事件）是 event wait record 唯一 writer、debug FIFO/特权导出 owner（W7 只做 Host 格式/checker）；本包作为其导出事务的执行腿并按 §4.4 结算；提供复位相位触发点（DRAIN_SAFE/RESUME/CLEAR_CREDIT_CACHE 观察）与 first-fault 上报格式；Group 驱动退休的 RETIRE 授权/TASK_COMPLETION 交互经 W6/W2 消息（§3.7）。
- W3（RVV/VME）提供 VME 旁路类与 MAT-15 drain 不饿死要求。
- W5（Kernel/Image）提供驻留身份与安装请求；本包提供 TileProgramMemory 物理安装/取指端口。
- W1（Boundary Cosimulation）消费 per-leg 观测做联仿对账；L1 腿/本地 credit/Arena 账本以 Python 现行为为对照、RTL 为执行权威（§3.7/§4.8）。W7 定义 trace/PMU 可观察字段格式（本包字段名对齐 validator PMU 名，见 §6）。

## 8. 验证、bring-up 和验收标准

### 8.1 验收判据（数据/负向/满资源/取消）

- 数据正确: load/store 逐字节对比 SPM/cache/refill 后结果；Gather/Scatter 按 `resolve_indexed_segments` 公式逐元素比对（GAT-11/12）；cache 命中/miss/bypass 路径的字节等价；store 三级可见性事件分级正确（DMA-12）。
- 负向: 重复响应、异常回包、部分分片失败、同 view 重复 FREE、Frame/Arena generation 不匹配（FRM-02）、bank 越界、系统保留区下限越界、逐 bank 碎片（总空闲够但单 bank 不够）；索引负值/元素地址溢出/段越出远端 view/bypass_forbidden；Profile 缺失/重复/过期 ACK、成员未静止 PREPARE、commit shadow 不匹配、超时 recovery；页授权旧 epoch 不命中新数据。
- 满资源: 各级 outstanding/lane/job/MSHR/credit/bank/Frame 满时的 stall 与归因（MEM-02/04/11，`engine_queue_full`、WAIT\_\*），无下溢/泄漏/双释放（MEM-08）；Arena/R/Context 满→WAIT 归因后资源复用一致。
- 取消: CANCEL_REQUESTED 与 COMPLETED 同周期竞争只产单一终态（FLT-05）；已发外部腿的分片保留其义务（部分外写后取消→终态 COMPLETED 或 ISOLATED，ISOLATED 只保证"此后不写"）；FAULTED 同样保留已发生外写；引用/credit 在真实终态后才回收（MEM-12）。
- 程序通路: 冷装载走保留安装口，不占数据 Arena/MSHR/credit（RES-17…19）；数据 issue gate 关闭期间安装、维护 clean、ACK 流持续独立进展（T-66，RES-19；W5 交界）。
- L2 final-free 时序: Python 权威，RTL 只观察；私有 backing 引用闭合早于 root 退休唤醒容量等待、共享 readonly backing 晚于 producer root 退休（T-57，ARN-04a/04b）。

### 8.2 与 r5 T 场景映射

| T 场景    | 判据锚点（本包）                                                                                               |
| --------- | -------------------------------------------------------------------------------------------------------------- |
| T-16      | §5.2 Gather/Scatter 逐段有序、重复索引、mask 语义、无额外非法访问                                              |
| T-17      | §4.2 addr/size 溢出、窗口与 L1/L2 基址隔离、`address_out_of_range`                                             |
| T-18      | §3.5 SAME/COMPATIBLE 兼容驻留不重配不递增 gen；显式 ProfileReconfigDesc 才切档                                 |
| T-19      | §3.5 PREPARE/CHECK_FRONTIER 静止+活数据写出、等待、重配、读回顺序                                              |
| T-20      | §3.7 授权前失败仅补偿释放+ACK 失败；授权后不回滚走 CANCEL/drain                                                |
| T-21      | §5.3 gate → Task/Route 事件驱动重试；不忙等；类内不越队首                                                      |
| T-22      | §3.7 提交依赖由父级 admission 保证；本包提供静止/活跃可查询点                                                  |
| T-23      | §4.6 各级容量有限、事件/终态不丢不变形                                                                         |
| T-24      | §4.4/§5.5 取消/隔离/故障三态；外部写义务先结算再回收                                                           |
| T-41      | §3.7 FREE_L1/ARN-04 只失效 view 不还 extent；Arena/R 安全退休才归还                                            |
| T-42      | §4.6 commit 失败归因 WAIT_CAPACITY/WAIT_FRAGMENTATION/WAIT_SLOT；hold 容量占用计入 WAIT 归因；Grid 等全 retire |
| T-43      | §4.6 有限队列/容量归因分类                                                                                     |
| T-44      | §4.3 UCE 钉住：唯一 ctx 占用→该 Tile WAIT_SLOT，其他 Tile 继续；越界钉住编译/加载期拒绝                        |
| T-45      | §4.4 阶段信号与终态恰好一次；重复/过期响应去重                                                                 |
| T-46/T-47 | §3.5/§3.6 12/9 步、ACK 异常、范围维护仅阻塞重叠、超时 recovery                                                 |
| T-48      | §4.4 真实完成 vs 隔离确认 vs 故障终态三态可区分；旧代响应/成员 ACK/NoC 回包拒收（MEM-13）                      |
| T-55      | §3.3/§3.7 逐 bank 守恒（含 hold/bound）、padding 不可见、碎片归因；hold 容量可重复预留检测                     |
| T-56/T-57 | §3.7/§4.8 生命周期控制；L2 final-free Python 权威；Arena/Frame 计数对账                                        |
| T-65      | §4.7 页授权逐段检查、epoch 引用持终态、撤销先关新访问、等待不阻塞控制/取消                                     |
| T-66      | §3.5/§4.8 程序通路不占数据 Arena/MSHR/FIFO；冷装载与数据 gate/用户数据饱和共存                                 |

### 8.3 下沉实施顺序建议（read→implement invariants→prove）

1. 在 Python 对照集上锁定不变量：逐 bank 守恒（含 `allocated=held+bound` 两段各自守恒、hold 不被重复预留）、信用/引用双向守恒、MSHR merge 唯一 leader、代际单调；RTL 实现时逐条映射为断言。
2. SegmentResolver + Frame/Arena plan/commit 的纯函数部分先行对照通过，再动 LSU/MFE FSM。
3. 联仿 per-leg 对账（leg 偏序、真实 capture/commit 观察点、credit/free_epoch、WAIT 原因；不要求 Python/RTL 周期相等），Python 现行为作基准。
4. 停止条件：§8.1 全部用例在 Python 基准覆盖的路径与 RTL 间得到一致终态、外部可见字节和合法事件偏序；mask/设备页授权等新增合同用独立参考验证；否则不进下一阶段。

### 8.4 不在本包范围

- L2/HBM/NoC/global-DMA 与父账本合同的重新设计；消息适配与唯一 owner 切分所需内部重构仍在范围内（§2/§4.8 Python 保留区）；SharedAmManager/Gemmini/共享 AM（总体设计 排除）；FPGA/PPA 宣称；多 VME 实例（W3 的 MAT-18…23）。
- W5 拥有的 program 驻留身份/引用计数/PROGRAM\_\* 消息语义；本包只负责物理安装/取指端口与共享端口仲裁（§7 建议 7）。

## 9. 风险、取舍和后续细化方向

- 参数未冻结: MSHR 深度、窗口、lane 数、per-bank 计费粒度由 SRAM/PPA profile 冻结（§6）；ctx outstanding 上限无模型值，**不预设**，由联仿数据评审后冻结。
- Scatter 与 cache 写回/refill 的仲裁顺序: `_downgrade_after_eviction` 是一种合法收敛路径，RTL 取不同等效顺序须证明外部可见字节/事件序列一致（r5 §11.3 策略非体系要求）。
- mask 逐元素语义（GAT-03）现 validator 无模型；本包为新增 RTL/联仿真例，不得称已有模型证明。
- 页授权端到端（GAT-15…20，X-23）当前仓库无实现，本包设计为新增通路；联仿证据依赖 W1/W7。
- L1↔L2 MSHR 合并 key（provenance 三元组）一致性是跨桥合同点，漂移会造成假 miss/重复 refill；列入 W1 对账项。

### 9.1 证据与引用

| 证据                            | 位置                                                                                                                                                                                                 | 用途                          |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| LsuV3 单 SuperSlot/队首         | [H05] `scalar/Lsu.scala` @ f79522157fc0f1d7311f44e209fc36c9c8b7a039: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/scalar/Lsu.scala | r5 §3.2/§10.2 改造入口        |
| CoreAxi 全核门控                | [H06] `CoreAxi.scala` 同上 tag: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/CoreAxi.scala                                         | 门控与桥接入口                |
| ITCM/DTCM 8/32 kB、S_AXI 启动   | r5 §3.2/§11.6.1 已引用 [S08][C04]（未重新拉源码，不宣称构建成功）                                                                                                                                    | 程序/数据分离背景             |
| TransferStage/Manager           | `../../../pipeline_validator/memory/transfer.py:288-474`（Stage 语义）、`:1415-1560`（腿推进/取消）                                                                                                  | 外部腿对照/取消合同源         |
| MSHR                            | `../../../pipeline_validator/memory/mshr.py:46-179`                                                                                                                                                  | 合并/代际/取消语义对照        |
| MFE FSM（行为对照）             | `../../../pipeline_validator/engines.py:239-396`（lane/job）、`:661-1014`（发射/取消/隔离）、`:1119-1750`（Gather/Scatter tick；其预测 finish_cycle 联仿口径按 §3.2 修订）                           | 预测时序→真实握手修订依据     |
| Segment 求址                    | `../../../pipeline_validator/memory/address_provider.py:20-73`                                                                                                                                       | GAT-11/12 公式与错误码        |
| Profile 控制器                  | `../../../pipeline_validator/memory/profile_controller.py:377-563`（submit/gate/step）、`:601-1013`（成员校验/总线）                                                                                 | gate owner 的 Python 合同源   |
| PagePool/Lease                  | `../../../pipeline_validator/memory/page_pool.py:38-153`（lease/epoch）                                                                                                                              | GAT-15…20 Python 现状（X-23） |
| Arena/view/pin                  | `../../../pipeline_validator/memory/arena.py:789-880`（invalidate/pin/inflight）、`arena.py:1312`（l2_extent_release）                                                                               | ARN-04/04a/04b/05 对照        |
| L1 Slot/Frame                   | `../../../pipeline_validator/memory/l1_slot_frame.py:102-286`                                                                                                                                        | FRM-01…04 对照                |
| Tile plan/commit/abort          | `../../../pipeline_validator/tile.py:1541-1630`、`tile.py:893-976`（`_free_l1`）                                                                                                                     | L1 admission 对照             |
| Gate/Task 准入                  | `../../../pipeline_validator/tile_group.py:1159-1263`、`:2070-2078`（admission_version 不含 L1 gate）                                                                                                | ADM-06/ADM-10                 |
| tile 位图 mask（无逐元素 mask） | `../../../pipeline_validator/execution_ir.py:405-406`                                                                                                                                                | GAT-03 无模型证据             |
| 硬件基线                        | `../../../pipeline_validator/hardware_config.yaml:16-151`                                                                                                                                            | §6 数值来源                   |
| 需求条款                        | r5 §10（行 657-735）、§11（行 744-862）、§15.2（行 1107-1126）、T 表（行 1453-1503）                                                                                                                 | 行号对齐                      |

（行号基于当前工作树 2026-11-05 快照；未运行构建/测试/格式化，与 总体设计 约定一致。）

[C04]: https://github.com/google-coral/coralnpu-compiler/blob/d715556c3853719e414182c0bf5f4a10f248f1d9/runtime/sim/fpga/simulator.cpp
[H05]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/scalar/Lsu.scala
[H06]: https://github.com/google-coral/coralnpu/blob/f79522157fc0f1d7311f44e209fc36c9c8b7a039/hdl/chisel/src/coralnpu/CoreAxi.scala
[S08]: https://developers.google.com/coral/guides/hardware/integration
