# W4 内存子系统实施计划（r1）

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

状态：**实施计划**：本轮只写本计划文件，不执行 build/test/formatter，不创建 `hardware/` Python 目标代码。
依据：用户已 review 批准 `../Coral_Nexus_Tile_Design.md`＋W0–W7；本计划按 `../plan.md`（下称"合同"，含其 §2/§3 与未来命令合同）与 `../components/W4_Memory_Subsystem.md`（下称"W4 设计"）展开 W4-01…W4-06。只写本文件；不改源码、设计文档、其他包计划。

W1 权威接口对齐（按 PlanW1 计划）：协议冻结产物由 W1-01 单点生成（`protocol/cn_protocol_v1.yaml` 等）；本包只写 fragment `hardware/coral_nexus/protocol/field_requests/w4.yaml`＋`protocol/w4_memory.contract.yaml`，由 W1-01 `gen_protocol.py` 读取合并、`freeze_check.py` 校验 path+hash 进 `PROTOCOL_FREEZE.json`。位宽/编码由 W1-01 按本包提供的**上界推导表**产出；本计划不自造硅片位编码。`message_seq`（边界层）／`op_seq+subop_id`（归因层，subop_id 不上边界线）分域；`root_launch_generation==grid.launch_generation` 只存一份。libcn_cosim_stub.so 仅单元，禁用于 CS1 判定。

## 0. 阅读顺序约定

1. 合同（权限、ID、DAG、已选方案 7/8、未来命令）；
2. W4 设计 §2/§3（owner 表、微架构）、§4（接口）、§5（数据流/时序）、§6（参数表）、§8.2（T 映射）；
3. 本计划 §2（文件证据/待创建清单）、§3（依赖 DAG）、§4（基线裁决）、§5（任务）、§6（T 映射）、§7（冲突）、§8（命令表）。
4. 实现轮只读本计划＋对应设计段＋引用 validator 行号；先读后改；缺接口回 owner 报证据，不自行改合同。

## 1. 已选方案摘要、非目标与本包 owner

**已选方案（合同"已选方案"第 7/8 条＋W4 设计 §1–§4，均为推荐默认）：**

- Tile 内一切可变内存状态唯一执行 owner 是本包 RTL：LSU/MFE/L1 bank/SPM+cache/MSHR/Arena 账本/Frame/TileProgramMemory 物理。Python 只保留 Tile 外部服务（Group 调度、L2 SRAM/cache/MSHR、NoC、HBM、global-DMA）与父账本（R lease、L2 pin、页授权权威、root retirement）＋只读静态镜像；**不代执行、不代写 L1、不代发本地完成**。
- **地址基线（Main 裁定，`../plan.md` §2.1 决策表**"地址／memory map"行**）**：RV32 scalar/RVV 仅访问 Context 已绑定 Frame／local／program 窗口；外侧 L2 view／HBM binding 经 **typed binding/view ID＋offset** 由 MFE 生成完整系统地址，宽度按配置最大容量从 config 推导冻结；高位范围与加法溢出必须检测；普通 kernel 不持自由 global pointer。本计划不保留"全宽 pointer vs 窗口"任选。
- 推论（需 Main 确认为 W4 裁决）：按 ADR-06"普通 kernel 对全局内存的访问只经 MFE 与 indexed descriptor"，**LSU 数据腿不存在外部 HBM/store 路径**；W4 设计 §5.1"外部腿（不可 cache 或非 L1 地址）"与 §5.5"LSU store（外部/HBM binding）"两行按此裁为"地址窗口外→fault"，外写义务（T-24/T-48）全部落在 MFE load/store/scatter 的外部腿上。见 §7 冲突 C-4。
- MSHR：合并 key 用 provenance 三元组（`(binding_id, binding_generation, line_offset)`，对齐 `cache.py:CacheLineIdentity`/`allocator.py` 纹理），profile generation 分域；**token 保留至已登记 waiters 获得可靠 data/response 并结清 token 引用**，不以 cache 写入等价永久数据存活（合同"已选方案 7"；`../plan.md` §2.1 MSHR 行）。每条目持有**行填充缓冲**（64＋64 validity bits）至最后一个 waiter 消费，cache 行被逐出不影响 waiter 数据（§4.4）。
- Ingress 可推进 RR（MEM-02）：先非阻塞 head、后 blocked-but-wakeable，禁止"只加深 FIFO 保留全局队首串行化"；轮转顺序仅兼容策略不进指纹。
- Group 页授权唯一权威：**真实外侧路径**——`PagePoolRegistry.lease`（`memory/page_pool.py:344-357`）在当前 validator **MFE indexed 尚未接入**（engines/tile 无 `.lease(` production 调用；`ExecTileGatherDesc.scope` 仅编译期静态校验 `simulator.py:443-477`，运行期不消费）。CS1 补真实外侧服务：Tile 每段先 `LEASE_ACQUIRE`（原子 check+take，GAT-19 单串行化点）→携带 lease_ref 到终态 ACK→Python `release`；revoke 走"关新→等旧引用→副本失效收 ACK→epoch 递增→换 owner"（GAT-20），**不得假 claim**。
- gate：唯一 writer 是 Python `ProfileController`（`_profile_gates`，`profile_controller.py:174,229-230,658-661,714-716`）；`GATE_NOTIFY(level, closed/open, profile_generation)`（W1 计划 §4.2）驱动 Tile 成员；RTL 不自开 gate、不逐 tick 轮询；gate_open 事件唤醒 Route 登记与 Task 提交（合同"已选方案 7"，ADM-06/ADM-10）。
- L1 gate 双唤醒（W1-04 执行，见 §7 R 共享接线）：`OPEN_ISSUE` 时 Group 唤醒受阻 Route 登记与 Task 提交重试（事件驱动，非轮询）。
- hold/bound：本地 hold 在 `HOLD_REQ` 后排他预留 extents/Frame 容量（`allocated=held+bound` 逐 bank 守恒），COMMIT_FINAL 才 bind；授权前失败 COMPENSATE，授权后不 ROLLBACK。**当前 validator 无 hold 语义**（`arena.py`/`allocator.py`/`tile.py`/`tile_group.py` 全库 grep `HOLD*\b|\_hold\b`仅`held` 出现于 L2 slack 变量名），Plan/commit 原子（`tile.py:1572-1596`）；hold 语义是 RTL 新增＋Python 镜像新增（§5.1）。
- L2 早释放（合同"已选方案 8"，用户已勘误，**不新增范围协议**）：Task 全部 L2 load await 且后续无 L2 load 才 `SIGNAL_PHASE(input_released)`；Group all_tasks 聚合后解除尚存纯读 pin（`tile_group.py:1010-1025`）；安全退休清理自己剩余 pin（不要求等别人）；`nest.release` 按 buffer，finalizer 全闭合后归还整段 padded span（`arena.py:1275-1313`）；INPUT_RELEASED 不归还 L1/Frame/UCE/R、不递减程序驻留引用；已早解 pin 退休不重复解。同 Task 按 buffer/范围分批通知与单 backing 子范围回收**不在范围**。
- 12/9 步 Profile/维护（`profiles.py:22-46` `PROFILE_STEPS`/`MAINTENANCE_STEPS`）＋范围维护仅阻塞重叠（PRO-14）；命令超时锁 gate 进 recovery，需显式 `reset_mode` 重驱动（PRO-17，模型 2,000,000 cycle）。
- 已接受外写不可凭 generation/timeout 撤销（CTX-16、FLT-04）；真实外写取消隔离按 W4 设计 §4.4 三终态（COMPLETED/ISOLATED/FAULTED；`CANCEL_REQUESTED` 仅请求标记）。

### 1.1 非目标

- L2/HBM/NoC/global-DMA 物理执行与父账本合同重设计（Python 保留；接口改造见 §7 请求清单 R1–R7，`memory_service.py` 归 W1-02）。
- 多 VME 实例（W3）；程序驻留语义/引用计数/PROGRAM\_\*（W5；本包只做 TileProgramMemory 物理＋安装/取指端口仲裁，W4 设计 §2 行 30/§7.7）。
- 本地事件表/WAIT 记录/first-fault 锁存本体（W6；本包只作其完成通知源与取消/隔离执行腿）。
- CS2/CS3、FPGA、PPA、利用率估算（VB-14/D-29）；共享 AM（排除）。
- Per-Context 越界钉住、局部故障恢复（FLT-02a 不在 CS1）。
- L1 issue gate 的 OPEN/CLOSE 决策本身（Group `ProfileController` 唯一写者）。

### 1.2 本包 owner 表（每状态唯一）

| 状态/资源                                                                                                  | 执行 owner                             | Python 侧角色                                                        |
| ---------------------------------------------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------- |
| L1 SRAM bytes／bank busy／SPM+cache 并存分区／逐 bank inflight 引用                                        | RTL W4（`rtl/memory/`）                | 无（只读镜像）                                                       |
| SPM/cache 模式 shadow 切换、代际校验、成员静止判定（PREPARE 静止检查）                                     | RTL W4 Profile Member                  | `ProfileController` 是唯一 writer（gate/命令流）                     |
| L1 arena extent 账本（hold/bound）、SlotFrame、view/inflight 计数、arena generation                        | RTL W4 L1ArenaManager                  | 只读布局/账本镜像（W1-04 经 RESOURCE_COMMIT_IND 对拍）               |
| L1 cache tag/dirty/validity、MSHR token/merge/代判、行填充缓冲                                             | RTL W4                                 | 无                                                                   |
| TileProgramMemory 物理阵列、安装写口/取指读口仲裁、FET-06 改写保护                                         | RTL W4                                 | 驻留身份/引用计数归 W5                                               |
| L1 腿执行（L1_READ/L1_WRITE/L1_CACHE_LOOKUP/FILL）、本地 credit                                            | RTL W4（L1 执行）                      | TransferManager 外部腿服务（L1 腿切除后）                            |
| Frame/Arena hold-retire 链路（RETIRE_READY→RETIRE→本地回收→TASK_COMPLETION 保存）                          | RTL W4（执行）＋W6（保存/发送）        | Group 授权与父账本                                                   |
| MFE descriptor 解析/lane/Gather/Scatter FSM、SegmentResolver、mask、页授权 Tile 侧检查点                   | RTL W4                                 | 页授权权威（Python）；完成事件经事件流上报                           |
| 外部腿（L2/HBM/NoC/global-DMA 事务、L2 cache/MSHR）、页授权权威、R lease、父 L2 pin、L2 release/final-free | **Python（不变）**                     | TransferManager 外部段＋ArenaPool(L2)＋`MemoryService`（W1-02 新建） |
| 取消隔离/外写义务结算/终态 ACK（COMPLETED/ISOLATED/FAULTED）                                               | RTL W4                                 | Group drain/poison 权威                                              |
| `confirm_isolation` 隔离确认语义（现 `transfer.py:1796-1804`）                                             | Python `MemoryService`＋RTL cancel FSM | 仅测试注入确认；生产取消走"真实完成或隔离确认"                       |

## 2. 文件证据与创建清单

### 2.1 已存在证据（HEAD `814bdf9` "reviewed design"）

| 事实                                                                                                                   | 位置                                                                                             | 用途                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| TransferStatus/TransferLeg/TransferOp/TransferLegKind/StageWaitReason 枚举                                             | `pipeline_validator/memory/transfer.py:94-163`                                                   | 路由/腿/状态枚举基线；**L1 腿切除时的 owner 切分依据**（§4.2 表）                                    |
| TransferStage：bank 占用/`try_issue`/`_release_resources`/`step`                                                       | `.../transfer.py:288-474`                                                                        | bank 端口模型（逐 bank busy/holder）；`_issue_parsed` 多 bank 同步 issue（:359-419）                 |
| L1 stage：16 banks、burst 32B/cycle、1 cycle latency                                                                   | `.../transfer.py:646-673`                                                                        | `_l1_stage`（512 GB/s÷16 banks÷1 GHz＝32B）＝256-bit bus 行宽（上游 `Parameters.scala:165`）一致性锚 |
| MSHR：`MshrTable` allocate/merge/wait/complete、token 单调、`complete` 立刻弹条目再回调 waiters                        | `pipeline_validator/memory/mshr.py:53-137`                                                       | **token 不能早复用**合同基线：RTL 端 token 复用规则须强于此模型（§4.4 M1）                           |
| ArenaPool：`plan_arena`/`commit_arena`/`retire_arena`/`invalidate_view`/`pin`/`unpin`/`begin_inflight`/`end_inflight`  | `pipeline_validator/memory/arena.py:312-523/543-599/789-880/961-1022`                            | L1 admission 对照（tile.py 计划消费）；retire 语义                                                   |
| per-bank 守恒检查：`allocated+free==user_spm_per_bank` 逐 bank 断言                                                    | `.../arena.py:1424-1451`（snapshot 内 `if sum(reserved)...!=user_spm_per_bank: raise`）          | T-55/PRO-16 守恒 oracle                                                                              |
| SlotFrame：`prepare`/`bind_view`/`bind`/`release`/`assert_slot_binding`/`release_slot`                                 | `pipeline_validator/memory/l1_slot_frame.py:88-295`                                              | FRM-01..04 对照                                                                                      |
| tile plan/commit/abort admission                                                                                       | `pipeline_validator/tile.py:1541-1632`                                                           | W4 与 W1/W2 切分点（§7 R）                                                                           |
| `_free_l1`：preflight→final-free→invalidate                                                                            | `pipeline_validator/tile.py:923-977`                                                             | FreeSlot 语义基线（inflight/pin 检查）                                                               |
| MFE 发射/取消/隔离、gather/scatter tick、`_downgrade_after_eviction`                                                   | `pipeline_validator/engines.py:239-396/661-1014/1119-1204/1268-1578/1583-1762`                   | FSM 对照基线                                                                                         |
| SegmentResolver（indexed 公式/错误码）                                                                                 | `pipeline_validator/memory/address_provider.py:20-73`                                            | GAT-11/12 参照                                                                                       |
| PagePoolRegistry：`lease`/`Acquirelease`/`is_valid`/`release`、`allocate`/`free` 拒绝 `host_pool_busy`                 | `pipeline_validator/memory/page_pool.py:98-155/344-408/415-446`                                  | GAT-15...20 现状（X-23： indexed 未接）                                                              |
| `AccessLease.is_valid` 仅按 page epoch/owner，不持事务引用                                                             | `.../page_pool.py:98-144`                                                                        | 假 claim 风险点（§4.5 M2）                                                                           |
| L1 issue gate 检查（每 tick 轮询，不在 admission_version token）                                                       | `pipeline_validator/tile_group.py:1191-1192,1231,2070-2078`                                      | ADM-06/10 差异项（合同"已选方案 7"）；gate_open 事件唤醒为新增                                       |
| `range_issue_blocked` 定义                                                                                             | `.../profile_controller.py:479-491`                                                              | **production 无 caller**（全库 grep；见 §7 冲突 C-2）                                                |
| Profile member `PREPARE` 静止判定（读 `pool.snapshot()`/`mshr.stats.active`/`cache_stats`）                            | `.../profile_controller.py:1013-1043`                                                            | CS1 须换 RTL 状态（§5.5）                                                                            |
| L2 引用 acquire/release hook（begin/end_inflight 逐 view 去重）                                                        | `pipeline_validator/tile_group.py:370-401`                                                       | 外侧权威接口                                                                                         |
| Tile L1 引用计数表（`allocation_id→generation`）                                                                       | `.../arena.py:789-880`                                                                           | 退役链对拍                                                                                           |
| `_has_l2_endpoint`/`submit` route 选择与 `_build_route` 逐 op route                                                    | `pipeline_validator/memory/transfer.py:832-970`                                                  | MEMORY 通道 route 切分（§4.2）                                                                       |
| HardwareConfig `tile_l1_*`/`mfe_*`/`control.slot_frame`/`tile_program_sram`                                            | `pipeline_validator/hardware_config.yaml:69-130`                                                 | §5 各参数基线（模型值，非最终冻结）                                                                  |
| StallReason 12 成员（含 `WAIT_L1_BANK/WAIT_MSHR`）                                                                     | `pipeline_validator/pmu.py`（W2 计划 §2.1 L15-30 引用同位）                                      | MEM-11 归因枚举                                                                                      |
| Upstream Lsu.scala @ f795221...：LsuV3（`L2029`）单 `LsuSuperSlot`（`L659`）、`CircularBufferMulti`（`L2032` depth≥4） | [H05]（W4 设计 §9.1 URL）＋已核对 `RetirementBuffer.scala:28` PC 匹配、`L28 Valid(PC)` 与 `L588` | LSU 改造入口                                                                                         |
| Upstream `Parameters.scala`：`lsuSlotTxidEntries=8`、`lsuDataBits=256`                                                 | pin `Parameters.scala:172,165`                                                                   | txid/行宽基线                                                                                        |
| Upstream `CoreAxi.scala`：`tcmPortCount`（`L65-66`）、TCM 读写路径、`cg.io.enable`                                     | [H06]（W4 设计 §9.1 URL）                                                                        | W6/W2 CoreAxi 改造请求输入，**本包只发请求不编辑**                                                   |
| `AccessLease.is_valid`/release 语义                                                                                    | `pipeline_validator/memory/page_pool.py:98-155`                                                  | §4.5 L2 服务检查点输入                                                                               |

### 2.2 待创建（本包目标；根 `hardware/coral_nexus/`，**当前不存在**，W0-01/02 建目录后落位）

| 路径                                                                                                                                                                                                                                                                                                                          | 产物                                                                                                                                                                                | owner 任务                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `hardware/coral_nexus/protocol/w4_memory.contract.yaml`                                                                                                                                                                                                                                                                       | 内存/Frame/Arena/页授权/Profile 成员/hold/retire **字段与语义 fragment**（逻辑名，不含位编码）                                                                                      | W4-01                                   |
| `hardware/coral_nexus/protocol/field_requests/w4.yaml`                                                                                                                                                                                                                                                                        | W4 侧对 `cn_protocol_v1` 的字段/消息请求（W1-01 单点合入）                                                                                                                          | W4-01                                   |
| `hardware/coral_nexus/configs/w4_memory_constraints.yaml`                                                                                                                                                                                                                                                                     | 每参数推导式＋来源＋冻结状态（不写未实测值）；与 `derive_w4_bounds.py` 同源                                                                                                         | W4-01                                   |
| `hardware/coral_nexus/scripts/derive_w4_bounds.py`                                                                                                                                                                                                                                                                            | 从 current `hardware_config.yaml`/`profiles.py`/语料推导 MSHR token 宽、credit 上界、bank busy 序列上界；输出 `configs/w4_memory_constraints.yaml` 建议 + 硬校验（越界即非 0 退出） | W4-01                                   |
| `hardware/coral_nexus/tests/verify/corpus_scan.py`                                                                                                                                                                                                                                                                            | 逐 T 证据语料扫描（不执行 build/test/formatter）                                                                                                                                    | W4-01                                   |
| `hardware/coral_nexus/rtl/memory/`（Verilog-2005/SV，**手写，非 Chisel**）：`l1_bank.sv`、`mshr_table.sv`、`cache.sv`、`slot_frame.sv`、`arena_ledger.sv`、`lsu_mslot.sv`、`mfe_top.sv`、`segment_resolver.sv`、`lease_checkpoint.sv`、`profile_member.sv`、`transfer_outbridge.sv`、`page_authority.sv`、`cancel_tracker.sv` | 本包 RTL 唯一执行 owner；每模块配 Verilator tb（`rtl/memory/tb/`）                                                                                                                  | W4-02..06                               |
| `pipeline_validator/memory/transfer.py` 外侧腿服务改造（**仅 W4 编辑**）：外侧腿抽为 `ExternalLegService`（`memory/transfer.py:832-970` 拆分），TransferManager 保留外部腿，L1 腿改走 TilePort（W1 侧接口）                                                                                                                   | Python 外侧服务合同源                                                                                                                                                               | W4-01(设计)+W4-02(实编)+W1-02(共享接线) |
| `pipeline_validator/memory/arena.py` hold 语义扩展：`hold_arena`/`bind_arena`/`compensate_held`/`retire_arena`(既有)                                                                                                                                                                                                          | Python 镜像（RTL/Python 同账本，供 W7-03 对拍）                                                                                                                                     | W4-05                                   |
| `pipeline_validator/memory/page_pool.py`：`PAGE_TABLE_INSTALL`/`acquire_range`（原子 check+take）＋`closing` 状态机＋`revoke`（epoch 递增/新 owner 发布）                                                                                                                                                                     | GAT-15..20 外侧权威                                                                                                                                                                 | W4-04                                   |
| `pipeline_validator/memory/profile_controller.py`：gate_open 事件（非 tick 轮询）＋member ack 收 `RTL_TILE_READY`                                                                                                                                                                                                             | gate 双唤醒 Python 侧                                                                                                                                                               | W4-05                                   |
| `pipeline_validator/tile.py`（本包新增 L1ArenaManager Python 镜像类，供 W7 对拍）                                                                                                                                                                                                                                             | RTL/Python 双账本                                                                                                                                                                   | W4-05                                   |
| `pipeline_validator/tests/test_w4_mshr_cache.py`、`test_w4_hb_lease.py`、`test_w4_hold_retire.py`（新文件）                                                                                                                                                                                                                   | 单元正/负例（**当前可执行** pytest 入口）                                                                                                                                           | W4-02..05                               |

## 3. 任务依赖与冻结门

| 任务  | 主题                                         | 硬前置                                                          | 并行窗口                                                                                                                          | 供体/冻结门                                                     |
| ----- | -------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| W4-01 | 内存/Frame/Arena/地址接口与上界              | —（读 W4 设计＋validator 行号＋P0 已锁上游即可）                | 与 W1-01、W2-01、W3-01、W5-01、W6-01 并行（互不 DONE-gate）                                                                       | 向 `PROTOCOL_FREEZE.json` 输 fragment（W1-01 收），不反等 W0-04 |
| W4-02 | L1 bank/cache/MSHR/program 物理端口          | W4-01 fragment 冻结 hash；W0-03（P0 原核）                      | 与 W1-02、W2-02 并行                                                                                                              | 供 W4-03/04/05 消费                                             |
| W4-03 | LSU 多槽多在途                               | W4-02 L1 bank/cache/MSHR;W2-01 owner-tag 合同(fragment hash)    | 与 W4-04 并行                                                                                                                     | 供 W2-05（W2 计划硬前置 W4-02 L1）、W4-05                       |
| W4-04 | MFE/直搬/indexed/mask/页授权                 | W4-01;W4-02;W1-02 `MemoryService`（外侧服务）                   | 与 W4-03 并行                                                                                                                     | 供 W3-04（VME drain）、W5-03                                    |
| W4-05 | Profile 12/9 步/维护/hold-retire/gate 双唤醒 | W4-02（bank/cache/MSHR 计数口）;W4-03（真在途）;W4-04（真外写） | 与 W3-05、W5-05 并行                                                                                                              | 供 W6-03（drain 判据字段）                                      |
| W4-06 | 取消隔离/资源压力/真实外写结算闭环           | W4-04;W4-05;W6-01 fragment（终态/TASK_COMPLETION 接口）         | 最后（CN-G4/G5）；不以 W1-05 四 Tile 汇合为 DONE 硬前置——**oneTile 真实 loop（W1-03/04）为验收载体**，4-Tile 汇合仍归 W1-05/W7-04 | 供 W7-03/04                                                     |

**合同冻结门（CN-G1）**：W1-01 `gen_protocol.py` 读取本包两份 fragment 与全部他包 fragment 合并生成 `PROTOCOL_FREEZE.json`（"齐了才 seal"，不与任何 -01 DONE 互等）；W4-02 起实施任务消费冻结 hash，不依赖其他 -01 DONE。

## 4. 基线裁决（本包独有、跨包引用的语义裁定）

### 4.1 地址/Frame window 基线（W4-01 冻结对象）

- RV32 侧每 Context 一张 **window table**，由 HOLD_ACK/BIND 时的 slot table（W4-01 fragment §3）填充；普通 kernel 只持 Frame window/slot 相对指针。
- 系统侧唯一地址格式：**typed binding/view ID（binding 类：`L2_VIEW`/`HBM_BINDING`）＋ 64-bit offset**。Tile 侧不持自由全局指针；MFE 在发起 `MEM_REQ` 前把 (binding_id, offset) 拼为完整系统地址并逐段校验（宽度按目标 HBM/B 容量从 config 推导，`derive_w4_bounds.py` 产 addr_width 建议值＋校验）。
- 溢出/越界/权限缺失统一 `address_out_of_range`/`frame_slot_fault`（沿用 W4 设计 §4.5 表，不新增错误码）。

### 4.2 L1 腿归属切分（transfer.py 外侧保留 vs RTL 承接）

| TransferLegKind                                                  | 保留在 Python（W1 外部腿服务）                           | 切到 RTL W4                                                              |
| ---------------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| `HBM_READ`/`HBM_WRITE`/`GLOBAL_DMA`/`NOC_REQUEST`/`NOC_RESPONSE` | 是（`transfer.py` 外部段）                               | —                                                                        |
| `L2_READ`/`L2_WRITE`/`L2_CACHE_LOOKUP`/`L2_CACHE_FILL`           | 是（L2 SRAM/cache/MSHR Python owner）                    | —                                                                        |
| `L1_READ`/`L1_WRITE`/`L1_CACHE_LOOKUP`/`L1_CACHE_FILL`           | 否（`_l1_stage`/`_l1_cache_stage` 为 tile-local 模型腿） | 是（`l1_bank`/`mfe_top` 承接；TransferManager 对 RTL Tile 拒绝发 L1 腿） |
| `LOCAL_DMA`（tile 侧 MFE 前端，`Engines.py:488-504`）            | 否（描述/queue 由 W2 提供，腿由 RTL MFE 执行）           | 是                                                                       |

**transfer.py 改造动作**（W1 共享接线 R1＋本包 API）：`TransferManager.submit` 按腿空间（`TransferLeg.src_space/dst_space`）路由：全外部腿照旧；含 `L1_*` 腿的事务在 RTL Tile 上由 `l1_bank/mfe_top` 承接（Tile 端 `TilePort.l1_*` 由 W1-02 adapter 生成）。本包不改 `pipeline_validator/memory/transfer.py` 之外任何 Python 文件；对 Tile-local L1 腿**保留 TilePort 分流（W1-02）**，不添加兼容 shim。

### 4.3 真实外侧路径（页授权，GAT-19 原子性）

- `PagePoolRegistry.lease` 当前**不持事务引用**、`release` 不幂等（`page_pool.py:132-144` 重复 release → 下溢/双释风险）：**`AccessLease.release` 改为 per-txn 幂等**（持有 `set[txn_id]`，重复 release 返回 False 计数）。
- 原子 check+take：`acquire_range(scope, binding, offset, size, txn_id)`（新增；内部 = `scope_owns_range`（`page_pool.py:331-343`）→ grant lease_ref+epoch → `AccessLease` per-txn 注册），**在单次 `registry.lease` 调用内完成 check 与 take**；`lease()` 返回的 handle 供 Tile 端 `lease_checkpoint` 持 `lease_ref+scope_epoch`。
- `AccessLease.release`/`is_valid` 的**真实外侧路径**：**同一物理事务的终态 ACK**（`MEM_TERMINAL_ACK`）触发 Python `AccessLease.release(txn_id)`，**不可替代为 tile-only 本地记账**（GAT-19"引用持终态"）。RTL 侧 `lease_checkpoint` 持引用直到终态 ACK 才释放。
- **禁止假 claim**：当前 validator `PagePoolRegistry.lease` 仅 Host 端检查（`page_pool.py:344-357`），Tile indexed 尚未接入（X-23）；CS1 必须补真通道（W1-02 `MemoryService`＋本包 `lease_checkpoint.sv`），不得用"本地缓存已授权"替代逐段 check+take。

### 4.4 MSHR token 生命周期（不得早复用破坏 waiter 数据，合同 §2 决策表 MSHR 行）

- token 生命周期：`ALLOCATED → ISSUED_REFILL → WAITERS_PENDING → FREE`；仅当 (a) 已登记 waiters 全部获得可靠 data/response，且 (b) 各 waiter 的 token 引用（含 per-op 引用与 `cancel_tracker` 引用）已结清，才 `FREE`。
- fill 缓冲（行数据＋per-entry validity bitmap）驻留在 MSHR entry 内直至 (a)，**不依赖 L1 cache 残留**（`_downgrade_after_eviction` 的"行被逐出"场景由 fill 缓冲消解，缓存行被驱逐不破坏 waiter 数据）。
- 同一行多 waiter 合并：`MshrTable.merge_group`（`mshr.py:76-96`）在 RTL 侧按 `(binding_id, binding_generation, line_offset)`（provenance）分域；**profile generation 分域**（`MshrTable.generation`，`mshr.py:139-145`），合并只允许同 generation。
- 负例：token 在 waiter 数据未消费前被复用 → 下次 allocate 返回同 slot token 而旧 waiter 仍读旧 token 数据 → 字节错误必须被 W4-02 单元负例检出（`tb/mshr_table_tb`：先 allocate/merge 两 waiter，手动 `complete` 前抢 token 复用注入），**此为不可让步不变量**（合同"阻止 MSHR token 早复用破坏 waiter 数据"）。

### 4.5 Python 侧保留/替换明细（切后）

| 模块                                     | 保留（Tile 外视角）                                                                            | 替换/改造（W4 owner）                                                                                                                                                                                         |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `memory/transfer.py`                     | 外部腿（HBM/DMA/NoC/L2/cache）全部 stage/route；`TransferManager.submit/step/acknowledge` 外段 | L1 腿改走 TilePort adapter（W1-02 共享接线）；`confirm_isolation`（`:1796-1804`）保留，**其触发方为 W4-06 cancel 隔离链**，不得只留在测试                                                                     |
| `memory/arena.py`                        | L2 ArenaPool 全部（root owner/l2 extent/pin/release）；`retire_arena`/`reconfigure` 语义       | **新增** hold/bound L1 镜像 API（`hold_arena`/`bind_arena`/`compensate_held`，本包 owner，供 W7 对拍）                                                                                                        |
| `memory/l1_slot_frame.py`                | 类与校验逻辑（`prepare`/`bind_view`/`assert_slot_binding`）供 W7 对拍                          | **不再作为 Tile 侧执行路径**；Tile 端 Frame 由 RTL `slot_frame.sv` 唯一执行 owner                                                                                                                             |
| `memory/mshr.py`                         | 类逻辑供 W7 对拍                                                                               | 同上；**token 复用规则升级**（§4.4 M1），validator 端 `MshrToken` 单调整型（`mshr.py:91-92`）不构成复用约束，RTL 端新增 per-entry `generation` 字段（token 复用仅在全引用结清后允许，且 waiter 残留检查必跑） |
| `memory/cache.py`                        | L2 cache（Python）                                                                             | L1 cache 移交 RTL；`DeterministicLRUCache` 保留为 Tile 侧行为 oracle（L1 侧 Python 类不再驱动）                                                                                                               |
| `memory/page_pool.py`                    | 页表/租约权威（Host 侧，不变）                                                                 | **新增 `acquire_range`/`revoke` 状态机＋per-txn release 幂等**（本包 owner edits）                                                                                                                            |
| `engines.py` MFEEngine                   | 类保留（行为 oracle）                                                                          | Tile 侧 MFE 执行由 `mfe_top.sv` 唯一 owner；`engines.py` 不再驱动 RTL Tile                                                                                                                                    |
| `tile.py` `_free_l1`/`plan_admission` 等 | Tile 侧 L1 准入/退役链路（`tile.py:893-977`/`1541-1632`）                                      | 退役链路改 `RETIRE_READY→RETIRE→TASK_COMPLETION`（W1-04 硬前置；本包提供 tile 级证据字段）                                                                                                                    |
| `tile_group.py` / `simulator.py`         | Group 侧全部（事件表/复位/FaultRing/R lease/页权威）                                           | **零直接编辑**；共享接线（L1 gate 事件唤醒/`_close_l2_access` RTL Tile 端钩子）走 W1-04/W1-05                                                                                                                 |

## 5. 逐任务

> 每条正/负例：具体输入/期望 bytes 或状态/断言；命令标 **【当前可执行】**（现有 CLI/pytest）或 **【未来，待 W0-02/W1-02 创建】**（`bash hardware/coral_nexus/scripts/cn.sh ...`）。每任务须先跑正例、后跑负例、再跑取消/背压/旧代；负例必须**先失败后通过**（先在缺陷状态跑出失败，修复后重跑通过）。

### 5.1 W4-01 内存/Frame/Arena/地址接口与上界（合同）

- **优先级**：最高（CN-G1 合同冻结）。**硬前置**：无 DONE 门（读 W4 设计＋validator 行号＋P0 锁定上游即可）。**并行窗口**：与 W1-01、W2-01、W3-01、W5-01、W6-01 并行。
- **输入**：W4 设计 §2/§4/§5/§6；`memory/transfer.py`/`arena.py`/`l1_slot_frame.py`/`mshr.py`/`page_pool.py` 对应行；上游 `Parameters.scala:165-193`、`Lsu.scala:2029-2036`。
- **产物**：§2.2 表前四个文件（`w4_memory.contract.yaml`、`field_requests/w4.yaml`、`configs/w4_memory_constraints.yaml`、`scripts/derive_w4_bounds.py`、`tests/verify/corpus_scan.py`；fragment 不设位编码，仅登记语义与**上界**）。
- **机械步骤**：
  1. 逐条列 W4 设计 §4.2 最小字段集到 fragment：`op_seq/subop_id/txn_id/ContextKey/kind/addr/size/slice_id/slice_count/generation 分域/credit_id/mshr_token/lease_ref+epoch/cancel_flag/final_state`。
  2. 上界推导（`derive_w4_bounds.py` 输入＝`hardware_config.yaml`+W2 计划的 Context/分片/向量展开上界；输出逐参数推导式）：bank busy 序列上界（`_issue_parsed:359-383`）、MSHR token 位宽（`mshr.py:53-96` 按 `MshrStats.active` 上限 16 推导）、lease_ref 宽（`page_pool.py:344-357` scope×page 上界）、frame slot 上界（`l1_slot_frame.py:102-108` 16 slot）、arena extent 上界（`arena.py:312-523`）。
  3. hold/bound 语义 fragment：HOLD_REQ→ACK(hold_ready)→BIND_INSTALL→ACK(accepted)，授权前失败 COMPENSATE（W4 设计 §3.7；W1 计划 §3.4/§3.5 为路由 owner）。写清"hold 不超卖"判定式（`arena.py` 逐 bank `allocated=held+bound`）。
  4. 页授权 fragment：`acquire_range/release/revoke` 三消息＋`PagePoolRegistry` API 变更清单（`.lease` per-txn release 幂等；GAT-19 单串行化点＝`acquire_range` 原子段）。
  5. `address_provider` 公式逐字核对（GAT-11），把 `indexed_address_out_of_bounds/overflow` 错误码入 fragment 错误表；`window_entries>0`（`execution_verifier.py:269-273`）。
  6. `field_requests/w4.yaml` 列 W4 对 memory/ctrl 通道的字段请求（不写位编码）；W1-01 合入。
- **消费者**：W1-01（合入 `cn_protocol_v1`）、W2-01（ContextKey/op_seq/LSU owner tag）、W3-01（drain bypass/VME drain 类）、W4-02..06（实施）、W5-01（binding manifest 引用）、W6-01（drain 判据字段）。
- **不变量**：MSHR token 生命周期（§4.4）；`allocated=held+bound` 逐 bank（PRO-16+hold 守恒）；`lease_ref` 到终态 ACK 才释放（GAT-19）；`AdmissionWaitReason` WAIT 枚举不动（`allocator.py:152-158`）。
- **验证**（正/负例）：
  | 用例 | 类型 | 输入/断言 | 命令 |
  |---|---|---|---|
  | U1-a | 正 | 跑 `l2-admission-wait`（【当前可执行】`bash examples/run.sh l2-admission-wait`）＋`derive_w4_bounds.py`（未创建：本轮为**未来命令**占位，不执行） | `derive_w4_bounds.py --from pipeline_validator/hardware_config.yaml` 后比对 `w4_memory_constraints.yaml` 逐参数与手算一致 |
  | U1-b | 正+负（合同级） | hold/bound 对拍断言（`arena.py:1424-1451` 同式） | 【未来，本任务交付后可跑】`pytest -n0 pipeline_validator/tests/test_w4_hold_retire.py`（新文件，非当前存在） |
  | U1-c | 负 | MSHR token 复用：手动 `mshr.complete` 令 token 可复用前 waiter 未消费 | 【未来，本任务交付后可跑】`pytest -n0 pipeline_validator/tests/test_w4_mshr_cache.py::test_token_reuse_faults_waiter`（新文件；负例先行：未修复版 fixture 先 RED，修复后 GREEN） |
  | U1-d | 负 | `AccessLease.release` 双释放 | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_page_pool.py -k "lease"`（现有测试改造点，改造后先 RED 后 GREEN） |
- **退出条件**：全部 fragment 语义无"待定"；`field_requests/w4.yaml` 被 W1-01 `PROTOCOL_FREEZE.json` 收并 hash 命中；U1-a 比对 0 差异；U1-b/c/d 全 PASS。**不写"再 review"**。
- **T 追踪**：GAT-11/12/15/19、MEM-03/05/08/12/13、PRO-16、FRM-01..04、ADR-06/07、T-55/L2 早释放（合同方案 8）。

### 5.2 W4-02 L1 bank/slot/cache/MSHR/program 物理端口（W1 bank 端口）

- **优先级**：P0。**硬前置**：W4-01 fragment 冻结 hash；W0-03 P0 原核（提供真实 RTL 验证工作台）。**并行窗口**：与 W1-02（`TilePort`）、W2-02（RF/ROB）并行。
- **输入**：W4 设计 §3.3/§3.4/§3.6、§4.6、§6 参数表；`mshr.py`/`cache.py`/`l1_slot_frame.py` 基线。
- **产物**：§2.2 表 `l1_bank.sv`、`slot_frame.sv`、`arena_ledger.sv`、`mshr_table.sv`、`cache.sv`、`transfer_outbridge.sv`、`l1_program_port.sv`（含 `tile_program.sv` 安装/取指端口仲裁）＋各 tb；`pipeline_validator/tests/test_w4_mshr_cache.py`。
- **机械步骤**：
  1. `l1_bank.sv`：逐 bank 独立 busy/holder，`try_issue` 语义（`transfer.py:346-474` 参照：请求多 bank 同步占用/释放）；SPM+cache 分区由 profile 驱动（PRO-02 一致比例逐 bank 同 layout），`system_reserved_spm_per_bank`（默认 2048）不参与 arena 计账。
  2. `slot_frame.sv`：`prepare`/`bind_view`/`bind`/`release`/`assert_slot_binding` 语义（`l1_slot_frame.py:127-295`），Frame FSM 8 states×1 cycle（`...:52-62,209-229`），校验不在 launch 关键路径（FRM-02）。
  3. `arena_ledger.sv`：hold/bound 账本（§5.1 fragment）；HOLD/COMMIT/COMPENSATE 状态迁移；W4 设计 §3.7 顺序合同。
  4. `mshr_table.sv`：token 分配/合并/cancel（`mshr.py:46-179` 语义），**token 生命周期按 §4.4**；每 entry 附 64B＋validity fill 缓冲；`profile generation` 分域；`_downgrade_after_eviction` 语义（`engines.py:1169-1204` 参照）。
  5. `cache.sv`：行结构 tag+validity+data+dirty；refill/clean/invalidate携 generation；`reconfigure`（`cache.py:546-563`）仅在 commit 后生效；`begin_maintenance`/`complete_clean`（`cache.py:441-519`）。
  6. `transfer_outbridge.sv`/`l1_program_port.sv`：外段腿（L2/HBM/NoC）桥接到 W1 `MemoryService`；program 安装/取指端口仲裁（W4 设计 §7.7；RES-12 共享端口公平仲裁与如实计数；FET-06 改写保护＝text 写入仅在无执行引用时）。
  7. L1 腿切除落地：`transfer.py` 外部腿改造按 §4.2 表（TilePort 桥由 W1-02 提供端口；本任务只产 L1 端实现）。
- **消费者**：W4-03（LSU）、W4-04（MFE）、W4-05（profile/hold-retire）、W1-02（TilePort 桥）。
- **不变量**：逐 bank `allocated+free==user_spm_per_bank`、`allocated=held+bound`（`arena.py:1424-1435`）；MSHR token 不早复用（§4.4）；Frame gen==Arena `allocation_generation`（FRM-04，`l1_slot_frame.py:123-125`）；`system_reserved` 不参与 Arena 计账（W4 设计 §3.3）；`profile` 切换仅由 12 步 COMMIT 驱动（不自开 gate）。
- **验证**：
  | 用例 | 类型 | 输入/断言 | 命令 |
  |---|---|---|---|
  | U2-a | 正 | 单 bank 连续读写（full_memory ByteStore 侧基线） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_memory_invariants.py -k "BankedFreeExtent"` |
  | U2-b | 正 | 12/16 bank 多 context gather/scatter 命中/合并逐 bank byte 守恒（`gather_bytes=48B` 案例，`test_tile_indexed.py:86-155`） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_tile_indexed.py -k "conservation or ordered"` |
  | U2-c | 负 | MSHR token 复用/早释放（U1-c 同源扩展：RTL tb 注入） | 【未来】`bash hardware/coral_nexus/scripts/cn.sh unit --suite w4 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/w4`（W0-02 创建后）；单元未创建时以 `tb/mshr_table_tb` 设计说明占位，不冒充已跑 |
  | U2-d | 负 | per-bank 越界/越 reserved 下限/双 free（`l1_slot_frame.py` `pmu_permission_fault_count`） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_memory_invariants.py -k "double_release or stale_generation"` |
- **退出条件**：全部 tb 正/负例 PASS 且 RTL 侧 MSHR/hold 守恒断言在 tb 内开启；`test_w4_mshr_cache.py` 每用例**先 RED 后 GREEN**；无"wiring 通过即完成"。
- **T 追踪**：MEM-02/11/13（bank busy/MSHR fill 归因）、PRO-16（逐 bank 守恒）、T-55、T-17（地址越界）、T-45（终态不丢）、T-57（L2 final-free Python 权威，本包只观察）。

### 5.3 W4-03 LSU 多槽多在途（每 ctx 独立 SuperSlot＋bank 端口）

- **优先级**：P0。**硬前置**：W4-02（bank/cache/MSHR 端口）；W2-01 `owner_tag`/`ContextKey` fragment（合同引用，不互等 DONE）。**并行窗口**：与 W4-04 并行；**W2-05 依赖本任务的 L1 端口＋W5-02 kernel**，但不依赖 W4-04。
- **输入**：W4 设计 §3.1/§5.1；上游 `Lsu.scala:2029-2036`（LsuV3 单 SuperSlot）与 `Parameters.scala:172`（txid=8）；W2 计划 `w2-to-w4-lsu-owner-tag.md` 的 `(ContextKey, owner_op_seq)` 键。
- **产物**：`lsu_mslot.sv`（per-Context ingress/in-flight/scoreboard/bank 端口）＋tb；`pipeline_validator/tests/test_w4_lsu.py`（新）。
- **机械步骤**：
  1. per-Context LSU：每 Context 独立 ingress FIFO（上限＝该 ctx outstanding 上限，上界由 `derive_w4_bounds.py` 输出），bank 端口仲裁可推进 RR（先非阻塞 head，后 blocked-but-wakeable，禁止全局队首串行化；MEM-02）。
  2. InFlightOp 状态机：`REGISTERED→ISSUING_L1→{COMPLETED|REFILLING→COMPLETED}→TRANSFERRED→CREDIT_SETTLED→COMPLETED`（W4 设计 §3.1），cancel 路径接 W4-06。
  3. bank 端口复用：load/stores 走 `l1_bank`/`cache`/`mshr`；L1 miss 由 `mshr_table` leader 发外部腿请求（W4 设计 §5.1），外部腿桥到 W1 `MemoryService`（W1 计划 `memory_service.py`）。
  4. `(ContextKey, owner_op_seq)` 键落 store 完成/写回匹配（W2 计划 §3.4 对拍断言）。
  5. per-ctx credit：独立计数/终态归还（`CreditManager` 语义，下溢/双释放断言常开，W4 设计 §4.2）。
- **消费者**：W2-05（native scalar 闭环）、W3-02（RVV/VRF）、W4-05（quiescence 计数源）。
- **不变量**：bank busy 序列逐 bank 上界可计算（§5.1）；credit `issued-in-flight-terminal` 守恒；每 outstanding 上限不阻塞其他 ctx；`engine_queue_full` 归因（MEM-11）。
- **验证**：
  | 用例 | 类型 | 输入/断言 | 命令 |
  |---|---|---|---|
  | U3-a | 正 | 双 ctx 独立 load/store，`bank busy` 逐 bank 忙位检查＋单 ctx 长延迟不阻塞其余 | 【未来】`cn.sh unit --suite w4`（单元 tb 未创建，以 `tb/lsu_mslot_tb` 设计占位，不冒充已跑） |
  | U3-b | 正 | gather/scatter 走 MFE path（W4-04 交叉集成一例） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_tile_indexed.py`（先 RED 后 GREEN 于 MFE 侧） |
  | U3-c | 负 | per-ctx credit 下溢/双释放/泄漏（MEM-08） | 【未来】单元 tb 注入；`cn.sh unit --suite w4` |
  | U3-d | 负 | `WAIT_L1_BANK`/`WAIT_MSHR` 归因（不丢请求不丢响应） | 【未来】单元 tb |
- **退出条件**：U3-a..d 全 PASS；`storeComplete` 键与 W2 计划断言逐条落地；无队首串行化回归。
- **T 追踪**：MEM-01..08、T-10/12/20、T-42/WAIT 归因（FRAGMENTATION/SLOT/CONTROL_RESOURCE）。

### 5.4 W4-04 MFE 直搬/indexed/mask/页授权（真实外侧路径）

- **优先级**：P0。**硬前置**：W4-02；W1-02 `MemoryService`；W2-01（descriptor 输入，不互等 DONE）。**并行窗口**：与 W4-03 并行。
- **输入**：W4 设计 §3.2/§4.7/§5.2；`address_provider.py:20-73`（GAT-11 公式）；`page_pool.py:98-155/344-357`；`engines.py:751-960`（segment 切分/bypass_forbidden 合同）。
- **产物**：`mfe_top.sv`、`segment_resolver.sv`、`lease_checkpoint.sv`、`page_authority_bridge.sv`＋tb；`pipeline_validator/page_pool.py` `acquire_range/revoke` 改造＋`test_w4_hb_lease.py`（新）；`TransferManager` 外部腿服务化（§4.2 表；W1 共享接线 R1 列入）。
- **机械步骤**：
  1. `segment_resolver.sv`：逐字移植 `address_provider.resolve_indexed_segments`（GAT-11 公式：`remote_element = index[i] × index_scale + offset + t × task_stride + j × stride`，`local_element=(i×repeat+j)×segment`；错误码/顺序与该函数完全一致，不做负回绕）。
  2. mask：沿用 RVV/LSU 侧既有 per-element mask（上游 `Lsu.scala` `vectorMask`/`canMasked` 路径），**不动**现有 `LsuCmd.vm` 语义；mask-off 元素不发访问、不报错、不占 credit（GAT-03），落单元例（`tb/segment_resolver_tb` 注入 mask-off 输入逐 byte 断言）。
  3. `lease_checkpoint.sv`：每段逐页 `lease_check`（GAT-18），真实外侧路径走 W1 `MemoryService.acquire_range`（§4.3）；**`AccessLease.release` per-txn 幂等**（`page_pool.py:132-144` 改造点）。
  4. Gather/Scatter FSM：`_tick_gather`/`_tick_scatter`（`engines.py:1483-1591/1677-1742`）语义逐态移植；ordered scatter `(i,j)`、`first-fault` 锁存、`bypass_forbidden`（`engines.py:781-811`）；`_downgrade_after_eviction` 语义落 MSHR（§5.2 U2-c 载体）。
  5. Indexed cache 路径：逐级真实 line 身份（`cache.py:36-45` `CacheLineIdentity`）、whole-line refill 不越授权段（GAT-18）；`gather_cache_bypass_requests` 逐案计数。
  6. 页授权/撤销序列（`MemoryService.revoke`，W1 侧共享接线 R3）：`关新→等旧引用→副本失效收 ACK→epoch 递增→换 owner`（GAT-20），Python 侧权威在 `PagePoolRegistry`，Tile 侧仅 lease 引用计数。
- **消费者**：W3-04（VME drain 外侧腿）、W4-05（MSHR/clean 联动）、W7-03（负向检出）、W5-03（BF16 gather 场景）。
- **不变量**：GAT-11 逐元素等价；GAT-18 整行 refill 逐页检查；GAT-19 lease_ref 持终态 ACK；GAT-20 页撤销顺序；DMA-11 `input_released` 晚于真实最后一次读（Task 级整体，非 MFE 内部信号）；ordered scatter 逐段 `(i,j)`；每段真实终态恰一次。
- **验证**：
  | 用例 | 类型 | 输入/断言 | 命令 |
  |---|---|---|---|
  | U4-a | 正 | indexed gather/scatter 公式逐 byte 对拍（`engines.py` 基线 vs SegmentResolver） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_tile_indexed.py -k "resolved_fidelity or ordered"` |
  | U4-b | 正 | `AccessLease.release` per-txn 幂等（每 lease 一 ref、double release 拒） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_w4_hb_lease.py`（新文件交付后可跑） |
  | U4-c | 负（假 claim 检出） | Tile 端 `lease_checkpoint` 不经 `MemoryService.acquire_range` 直接发起 indexed 请求（缺陷注入） | 【未来】`cn.sh unit --suite w4` 或联仿注入负例（先 RED 后 GREEN） |
  | U4-d | 负 | mask-off 段被发出请求/占 credit | 【未来】单元 tb 注入 |
  | U4-e | 负 | 页撤销序列顺序错误（`epoch` 未递增先发 invalidate 等） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_page_pool.py -k "epoch or residual"`（现有）＋【未来】单元 tb（撤销序列注入） |
  | U4-f | 正 | 真实外侧路径端到端：`l2-admission-wait` 场景＋seed 真实 A_IN bytes（`--input-data`），A_IN 131072B 在 final-free 后被 B 同 segments 复用 | 【当前可执行】`bash examples/run.sh l2-admission-wait --trace-json /tmp/w4-a.trace.json --report /tmp/w4-a.report.json`（Python 基线，父已跑）；RTL 侧【未来】`cn.sh cosim --scenario ... --backend rtl` |
- **退出条件**：U4-a/b 全 PASS；U4-c/d/e 均"先 RED 后 GREEN"；U4-f Python 基线 3 项检查通过（父已核）＋RTL 侧断言（外部腿 byte 等价＋lease_ref 终态释放）在单元/联仿真例落地。**L2 早释放语义不新增范围协议**（`nest.release` 按 buffer、finalizer 全闭合归还，合同方案 8）。
- **T 追踪**：GAT-01..20、T-16/17/23/24/50/65、MEM-09/12/13、T-57。

### 5.5 W4-05 Profile 12/9 步/维护/hold-retire/gate 双唤醒

- **优先级**：P0（CN-G3/G4 关口）。**硬前置**：W4-02（bank/cache/MSHR 计数端口）；W4-03（真在途）；W4-04（真外写）。**并行窗口**：与 W3-05、W5-05 并行。
- **输入**：W4 设计 §3.5/§3.6/§3.7、§4.7；`profile_controller.py:473-491/551-594/607-752/1013-1043/1306-1358/1385-1419`；`cache.py:441-519`；`arena.py:961-1022`。
- **产物**：`profile_member.sv`、`arena_ledger.sv` 扩展（hold-retire）＋tb；`profile_controller.py`（gate_open 事件发出/事件驱动 retry，非 tick 轮询）＋`tile.py` L1ArenaManager Python 镜像＋`test_w4_hold_retire.py` 既有/新例。
- **机械步骤**：
  1. `profile_member.sv`：12 步（ACQUIRE→…→RELEASE，`profiles.py:22-35`）/9 步（`profiles.py:36-46`）逐 step FSM；PREPARE 静止判定（`profile_controller.py:1013-1043` 四条：cache pending_clean==0、dirty==0、MSHR active==0、Arena/View pin/inflight==0，**换成 RTL 侧计数**）；COMMIT 校验 shadow 与已准备事务一致后原子切 active 并递增 generation（PRO-12/15）；失败/超时→gate 锁闭进 recovery，需 `reset_mode` 显式重驱动（PRO-17，模型 2,000,000 cycle）。
  2. gate 双唤醒（合同方案 7/ADM-06/ADM-10；W1 计划 §5.4 为 Python 侧共享接线）：Tile 侧不再 tick 轮询 gate——`profile_controller.py` 由每 tick 主动检查改为**事件驱动** retry（GateNotify 消息触发 `_step_task_admission`/`dispatch_role` 重试）；Python 侧 `admission_version` token 补 L1 gate 项（`tile_group.py:2070-2078` 差异项，由 W1-04 落）。**Tile 端口 passive**（W6 计划 §5.1：`Tile 端点 passive 执行组命令、不自动推进`）。
  3. hold-retire 链路：hold（W4 设计 §3.7）→COMMIT_FINAL→RETIRE_READY→Group `RETIRE` 授权→RTL 本地回收（view/Frame/Arena）→W6 保存 `TASK_COMPLETION`（不归还 L1/Frame/UCE/R、不递减程序驻留引用）；已早解 pin 退休不重复解（合同方案 8）。
  4. 范围维护（PRO-14，9 步）：`range_issue_blocked` 真接消费侧（现 `profile_controller.py:479-491` 无 production caller，§7 冲突 C-2）；脏行 clean 产生真实下游事务并等完成（`_maintain_caches:1306-1358` → `_clean_transactions_done:1342-1356`）才 ACK。
  5. gate 开关期间行为：旧数据访问完成/ACK 返回、程序安装路径（PROGRAM_INSTALL 保留通路，RES-17）、范围维护 clean 下游事务与排空均独立不受阻（W4 设计 §3.5）。
- **消费者**：W6-01/03（复位 FSM/drain 判据）、W5-04（PROGRAM READY 与 gate 交互）、W7-03（负例）。
- **不变量**：PREPARE 前静止四条（RTL 侧）；COMMIT 校验失败维持旧 active 报错、gate 不开；重复/过期/缺失 ACK 忽略并计数；12/9 步序列与 `profiles.py` 表完全同序；MSHR/clean/范围 gate 不互锁。
- **验证**：
  | 用例 | 类型 | 输入/断言 | 命令 |
  |---|---|---|---|
  | U5-a | 正 | 12 步完整 happy path+9 步范围维护 happy path（L1-only/L2-only/L1+L2） | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_profile_runtime.py -k "MemberProtocol or ByteOracle or LeaseRetirement"` |
  | U5-b | 负 | 缺失/重复/过期成员 ACK；成员未静止 PREPARE；commit shadow 不匹配；超时 recovery；range gate 期间非重叠事务继续 | 【当前可执行】同上 `-k "range_maintenance or timeout or stale or recovery"`；【未来】单元 tb（gate 期间新数据访问 stall 而不 fault） |
  | U5-c | 正 | hold-retire 真实循环：HOLD→COMMIT_FINAL→执行→RETIRE_READY→RETIRE→TASK_COMPLETION，逐 bank `allocated+free==user_spm` | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_w4_hold_retire.py`（新，先 RED 后 GREEN） |
  | U5-d | 负 | hold 占用期间并发 ticket 尝试同 bank 预留→拒绝（不超卖） | 同上 tb/pytest 负例 |
- **退出条件**：U5-a..d 全 PASS；`profile_member.sv` 与 Python `ProfileController` 联仿对拍（W1 侧）完成；`task_retirement_wait` 归因不丢（`pmu.py:15-30`）；无 gate 轮询回归（单元 tb 逐 cycle 断言无 tick 轮询）。
- **T 追踪**：T-18/19/20/46/47/56、PRO-01..17、FRM-01..04、ADM-06/07/08/10、T-57（L2 final-free Python 权威）、T-58/T-66（程序通路与 gate 共存）。

### 5.6 W4-06 取消隔离/资源压力/真实外写结算闭环

- **优先级**：P0（CN-G4 关口）。**硬前置**：W4-04（外写路径）；W4-05（quiescence/drain 证据）；W6-01 fragment（终态/TaskCancel/RESUME 接口，不互等 DONE）。**并行窗口**：最后（CN-G4/G5）；**oneTile 真实 loop 验收**（W1-03/04）为载体，4-Tile 汇合不回依赖 W1-05。
- **输入**：W4 设计 §4.4（终态三集合）、§4.5（错误码表）、§5.3/§5.4、§8.1/§8.2；`transfer.py:1415-1560/1690-1810`（cancel/isolation/acknowledge）、`tile_group.py:1049-1125`（release_context_memory/drain）、`reset_domain.py:91-105/167-219`。
- **产物**：`cancel_tracker.sv`＋tb；`TransferManager.confirm_isolation` 调用点接入（生产路径；`engine 侧 cancel/隔离确认`，`engines.py:962-1098` 对齐）；`test_w4_cancel_isolation.py`（新）。
- **机械步骤**：
  1. `cancel_tracker.sv`：每 accepted op/Transfer/命令终态集合 `COMPLETED|ISOLATED|FAULTED`（W4 设计 §4.4；取消最终终态是三者之一）；`CANCEL_REQUESTED` 仅请求标记；外写义务在真实完成/隔离确认后才回收（MEM-12，`transfer.py:1690-1711` acknowledge 合同）。
  2. 真实外写取消/隔离：accepted 外写不可凭 generation/timeout 撤销（CTX-16、FLT-04，W6 计划 §5.2 原文）；隔离确认（`confirm_isolation`，`transfer.py:1796-1804`）作为**生产路径触发点**接入（现仅测试调，§7 冲突 C-1），隔离失败 poison 并拒绝新启动直至显式静止复位（RST-08，`tile_group.py:1992-1993/2108-2110/2235-2267`）。
  3. 资源压力闭环：T-42/T-43 各级容量（outstanding/MSHR/credit/job/Frame）满时 stall 与归因（MEM-02/04/11）；零泄漏（每 accepted 事务终态＋引用/credit 全归还，MEM-08；`transfer.py:815-831` tombstone 预留合同）。
  4. `DRAIN_SAFE` 判据（W6-01 fragment 引用）：真实接受计数 0（`reset_domain.py:167-219`）＋MSHR/传输静止（`l1_mshr.stats.active==0`）。
  5. 竞争闭环：同周期 cancel/complete 只产单一终态（FLT-05）；重复终态/重复 credit 归还/双释放全拒（MEM-08/IF-10）。
- **消费者**：W6-03（reset FSM）、W6-05（竞争集成）、W7-03/04（负例与最终验收）。
- **不变量**：三终态语义（W4 设计 §4.4 表）；重复 cancel/完成竞争恰一次终态（FLT-05）；复位零泄漏（`reset_domain.py:167-219` 全计数）；隔离失败 poison（RST-08）；已接受外写不可凭 generation/timeout 撤销。
- **验证**：
  | 用例 | 类型 | 输入/断言 | 命令 |
  |---|---|---|---|
  | U6-a | 正+负 | 取消发生在 store 发出后、响应前（T-24）：真实完成 vs 隔离确认两出口外部结果可区分 | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_memory_invariants.py -k "cancel"`（现有基线）；【未来】单元/联仿 tb 外部结果双检 |
  | U6-b | 负 | 外写撤销尝试（generation/timeout 驱动）→ 拒；隔离失败 poison | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_runtime.py -k "CycleCap or FaultReset"`（现有）；【未来】注入负例 |
  | U6-c | 正 | 资源压力闭环：多 ctx/多 MSHR/job/Frame 满 + 排空 + 归还守恒，终态零泄漏 | 【当前可执行】`pytest -n0 pipeline_validator/tests/test_memory_invariants.py -k "cancel_all or double_release"`（现有）；【未来】单元 tb |
  | U6-d | 负 | 复位在途传输/MFE job/Profile 命令（RST-08 超时强制取消）；隔离失败 poison | 【当前可执行】同上 `-k "reset"`（现有）；【未来】注入负例 |
- **退出条件**：U6-a..d 全 PASS（先 RED 后 GREEN）；`DRAIN_SAFE` 证据字段真读（非文件存在检查）；零泄漏在每次 run 终局断言；oneTile 真实 loop 取 `W1-03/04` 载体（`--rtl-tile-mask 0x1`）。
- **T 追踪**：T-24/45/48/52、MEM-07/08/12/13、FLT-01/02/04/05/14、RST-07/08/10。

## 6. T 场景映射（W4 负责/共同/排除）

| T                   | W4 职责                                                           | 共同/接口                   |
| ------------------- | ----------------------------------------------------------------- | --------------------------- |
| T-10/12             | per-ctx 多在途/并发 bank 端口（W4-03）                            | W2-05 载体                  |
| T-14/15             | gather/scatter 多消费者/staging（W4-04 直搬/indexed）             | W1-04 早释放                |
| T-16/17             | Gather/Scatter/indexed（W4-04）                                   | W3-02/W5-03 kernel          |
| T-18/19             | SAME/COMPATIBLE、跨不兼容 profile（W4-05 12 步）                  | W1-04/W3-01                 |
| T-20/21/22/23       | 准入/retirement/早释放/WAIT 归因（W4-05 hold-retire/gate 双唤醒） | W1-04/W6/W2                 |
| T-24/41/42/43/44/45 | 终态/取消/容量/碎片/钉住/阶段信号（W4-06 + W4-02 ledger）         | W1-04/W6                    |
| T-46/47             | 12/9 步/范围维护（W4-05）                                         | W1（gate 事件）             |
| T-48                | 迟到回包/代际（W4-02 MSHR + W4-06）                               | W6                          |
| T-55/56/57          | Arena 守恒/生命周期/L2 final-free（W4-05；**L2 侧 Python 权威**） | W1-04（早释放链）           |
| T-65/66             | 页授权/程序通路（W4-04 lease_checkpoint/W4-05 gate 独立进展）     | W1-02 `MemoryService`/W5-04 |
| T-68                | VME 区域（**非 W4**）                                             | W3                          |
| T-59/60             | CS2/CS3（**不在范围**）                                           | —                           |

**不含 L2 行不整条跳过**：T-59/60 及明确 N≥2 子项延后；带 L2 的用例（T-14/15/57 等）仍验其 CS1 的 Tile 侧/外侧协作部分。

## 7. 冲突、共享接线请求与给实现 agent 的手册

### 7.1 冲突清单（Main 已裁决 2026-11-05；逐项落定）

- **C-1（已裁决）**：`transfer.py:1796-1804` 的 `confirm_isolation` 当前仅测试调用（`test_memory_invariants.py:533,565,620,779,921,1433`），无 production 常规触发点；**裁决＝W4-06 接入生产**（cancel/隔离执行腿为 RTL 侧；Group 在 drain 中发 TaskCancel 并等待三终态，不再依赖测试注入）。W4 执行腿、Group 触发点经 W1-04。
- **C-2（已裁决）**：`profile_controller.py:479-491` 定义、W4 设计 §3.5 引用，但在 `dispatch/engines/tile` 无调用侧；**裁决＝W4-05 补真实消费点**（范围 gate 期间重叠请求阻塞、非重叠继续；新增功能行为，先 RED 后 GREEN）。
- **C-3（已裁决）**：L1 腿内留 RTL，L2/HBM/NoC/global-DMA 外段走 `MemoryService`（W1-02），`TransferManager` 只留外部腿服务、不再持有任何 L1 执行状态——与本计划 §4.2 表一致；Main 已与 W1 计划对齐。
- **C-4（已裁决）**：按 ADR-06，RV32 LSU 数据腿不存在外部 HBM/store 路径；窗口外访问＝fault。W4 设计 §5.1/§5.5 的 LSU 外部 store/HBM binding 行属设计文档待勘误项（下轮文档修订时改，本计划已按裁决口径执行）；T-24/T-48 载体改 MFE 外部腿，验收要求不变。
- **C-5（已裁决）**：RTL 端必须实现 §4.4 生命周期（waiter 全消费才 FREE）＋per-entry `generation` 作为新增不变量；validator 的 `MshrToken` 单调性不构成复用约束的放松理由。W4-02 单元负例 U1-c 为此设。
- **C-6（事实记录）**：所有 `hardware/coral_nexus/**` 路径在未来 W0-01/02 落盘之前不存在；本计划中的"当前可执行"仅指 `pipeline_validator` CLI/pytest，未来命令**未创建**，不得冒充已执行。

### 7.2 需 W1 协调处理（W4→W1 请求清单，W1-01/02/04/05 各自消化）

- **R1（TransferManager 外部腿服务化 + W1 TilePort 桥）**：`memory_service.py`（W1 计划 §2.1 L104/§2.2 L209 已列）承担 MEM*REQ→外部腿服务；W4 供 `TransferManager.externals_only`/`TilePort.l1*\*` 端口（§4.2 表）。共享接线归 W1-02，本包不编辑 W1 文件。
- **R2（tile_group 退役链改 RETIRE_READY/RETIRE/TASK_COMPLETION）**：W4 供 `RETIRE_READY(ContextKey, TaskKey, drain_evidence)` 与 `TASK_COMPLETION` 保存端口；Python `_step_tiles`（`tile_group.py:1669-1722`）退役链拆分由 W1-04 落。
- **R3（页授权 revoke 真通道）**：`MemoryService.revoke`（GAT-20 序列）由 W1-02 落 wire；W4 供 `PagePoolRegistry`/`AccessLease` API 改造（`acquire_range`/per-txn release 幂等）＋`lease_checkpoint.sv`。
- **R4（L1 gate 事件驱动双唤醒）**：`profile_controller.py` gate_open 事件发出＋`_step_task_admission`/`dispatch_role` 事件驱动 retry（替换每 tick 轮询，`tile_group.py:1191-1192/1231`）由 W1-04 落 Python 侧；`admission_version` token 补 L1 gate 项（`tile_group.py:2070-2078` 差异）。Tile 侧只收 GATE_NOTIFY（passive，W6 计划 §5.1）。
- **R5（`l1` domain 在 `admission_version`/`GATE_NOTIFY`/`RESOURCE_COMMIT_IND` 的镜像对拍）**：W4 供账本镜像（hold/bound/extent/quiescence 计数），对拍由 W7-03/W1-04。

### 7.3 给实现 agent 的手册（低能力模型）

1. **只改本计划 §2.2 列出的文件/新文件**；任何 `hardware/coral_nexus/` 路径在未来 W0-01/02 之前不存在，先落目录再写文件，勿把未来命令当已执行。
2. **先读后写**：每个改造点先读 W4 设计对应段＋validator 引用行号（§2.1 表），再实施；缺接口时回 owner 报证据（`confirm_isolation` 触发点、`range_issue_blocked` 消费点、`admission_version` 补门）。
3. **负例先 RED 后 GREEN**：每个正例先在缺陷状态跑出失败（注入/回退实现），再修复重跑通过；不许"wiring 通过即完成"。
4. **字节与状态为准**：终态/归因/守恒断言逐条读具体字段（bytes、计数、状态），不读"文件存在"或"非空返回"；不比较 Python/RTL 周期数（W4 设计 §7.1）。
5. **MSHR/lease/hold 三处为不可让步不变量**：§4.4 MSHR token 生命周期、§4.3 lease_ref 到终态 ACK、§5.1 hold 不超卖（`allocated=held+bound` 逐 bank）；任何简化须走 Main 报告。
6. **跳过 build/lint/test/formatter**（本轮）；实现轮只在任务验收阶段跑【当前可执行】

命令，未来命令（`cn.sh`）等 W0-02 落盘后可用。7. **失败上报**：任何跨包字段不匹配（W2 `owner_tag`/W5 `binding manifest`/W1 `cn_protocol_v1`）→ 回 `agent://PlanW4` 之外直接报 `agent://Main`，不自行改合同。

## 8. 命令与环境差异说明（当前可执行 vs 未来）

| 命令                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 状态                                                                                                      | 说明                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `python -m pipeline_validator --ir-file examples/scenarios/l2_admission_wait.mlir --hw-config examples/configs/profile_l2_256k.yaml --hw-override num_dma_channels=2 --hw-override hbm_fixed_latency_cycles=10 --context-mode 2 --device-context-mode 2 --input-binding A_IN=0x100000:131072:rw --input-binding A_OUT=0x200000:131072:rw --input-binding B_IN=0x300000:131072:rw --sim-override fidelity=full_memory --memory-trace --trace-json <p> --json --report <p>` | 当前可执行（run.sh 场景封装不透传 trace/report flag，须用完整 CLI；见 Design_Validation §2.4 已验证命令） | Python full_memory 输入早释/准入门（父已跑 3 项检查通过；A 输入 cycle 3311 回收并被 B 复用，A 四个 Task 19833+ 才退休）        |
| `conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -v`                                                                                                                                                                                                                                                                                                                                                                                             | 当前可执行                                                                                                | 现有单元（`test_page_pool.py`/`test_memory_invariants.py`/`test_tile_indexed.py`/`test_profile_runtime.py`/`test_runtime.py`） |
| `bash hardware/coral_nexus/scripts/cn.sh unit --suite w4 ...`                                                                                                                                                                                                                                                                                                                                                                                                             | 未来，待 W0-02 创建                                                                                       | W4 单元 suite；未创建不执行、不冒充已跑                                                                                        |
| `bash hardware/coral_nexus/scripts/cn.sh cosim --scenario hardware/coral_nexus/verification/scenarios/<具体场景>.yaml --backend rtl --rtl-tile-mask 0x1 --config hardware/coral_nexus/configs/cs1_c2.yaml --out out/coral_nexus/<run_id>`                                                                                                                                                                                                                                 | 未来，待 W1-02/03 创建                                                                                    | oneTile 真实 loop（W4-06 验收载体）；场景路径在 W7-01 schema 冻结后由各任务填入                                                |

## 9. 本轮计划检查记录

- 本轮**未执行**任何 pytest/build/formatter/cosim；上述"当前可执行"仅为现有 CLI/pytest 语义引用。
- 全库 grep 断言（本轮内完成，不依赖后续运行）：`hold/HOLD_\b` 无 production 语义（`arena.py:976/1003` 仅 `held_slack` 变量名）；`MshrToken` 单调整型（`mshr.py:91-92`）；`confirm_isolation` production 无 caller；`range_issue_blocked` production 无 caller；`hardware/`/`design/coral_nexus_tile/implementation/` 在本轮写文件前不存在（`ls` 断言）。
- 任务数：6（W4-01…W4-06）；依赖无环（W4-01 与他包 -01 并行，无 DOWN 级互等 DONE）；每任务带输入/产物/步骤/消费者/正负例/退出条件/T 追踪。
