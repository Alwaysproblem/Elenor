# CN-G4 执行页：四 RTL Tile 完整 CS1（C055–C068）

> 本页把 W4-04/05/06、W6-03/04/05、W5-04/05、W7-03/04/06、W1-05 落为 14 个可直接执行的 commit 卡。共享语义、命令约定（`H`/`V`/`C1`/`C2`/`P0`/`cn`/`EVIDENCE`）见总入口 `execution_guide.md`；卡片格式与依赖以共享合同为准。所有 `cn` 命令为未来命令，可用性以前置卡交付为准；本页不创建任何代码。**证据与交接约定**：每卡示例用 `: "${EVIDENCE:?按总入口设置本卡新attempt}"` 取由总入口分配的新 attempt 目录，重试用更大的 N，不覆写旧证据；每个 `cn` run 的 `--out` 目录产 **run 级 `manifest.json`**，多 run 汇总为 **`cs1_manifest.json`**（aggregate）；`cn gate` 只从自身 out 目录的 `inputs.json` 显式读前置 run manifest 路径＋hash＋检查类型。**卡片 DONE ≠ 原 Wn-xx DONE**：原任务 DONE 遵循 master 关闭表（全部对应 commit 合入＋端到端 acceptance）。

<a id="c055"></a>

## C055 — indexed 逐段求址与 per-element mask

- **任务**：W4-04
- **前置**：C020, C032
- **建议 commit**：`feat(coral-w4): indexed segment resolver RTL with per-element mask semantics`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §5.4（W4-04 机械步骤 1/2、不变量）；W4 设计 §3.2/§5.2；Python 参考语义 `pipeline_validator/memory/address_provider.py:20-73`（GAT-11 公式）、`pipeline_validator/engines.py:751-960`（segment 切分/错误码/bypass_forbidden）；`PROTOCOL_FREEZE.json`（C020 封印 hash）中 `w4_memory.contract.yaml` 的 descriptor 字段。
- **修改边界**：H。新增 `hardware/coral_nexus/rtl/memory/segment_resolver.sv`＋`rtl/memory/tb/segment_resolver_tb.sv`（Verilator tb，含 mask 注入例）；在本卡的 w4 unit suite 注册表（C019 runner 入口）注册 tb case。不改 `lsu_mslot.sv`/`mfe_top.sv`（C031/C032/C056 owner）；Python 侧只读参考，不改 `address_provider.py`。
- **实现步骤**：
  1. 逐字移植 `resolve_indexed_segments`：`remote_element = index[i]×index_scale + offset + t×task_stride + j×stride`、`local_element=(i×repeat+j)×segment`，错误码与返回顺序和该函数完全一致（含加法溢出/越窗检测，不截断高位，不做负回绕）。
  2. mask 侧面独立：mask-off 元素不发访问、不报错、不占 credit（GAT-03）；tb 注入 mask-off 输入逐 byte 断言输出请求集为空且 credit 计数不动。
  3. 对拍闭环：tb 驱动与 Python `address_provider`（i32 索引、repeat/segment/stride 参数化）逐 case 输出相等后才算移植完成；溢出/越界 case 双侧同错误码。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  # 先读基线（Python 参考语义，现仓可跑）：
  conda run -n elenor-validator python -m pytest -n0 pipeline_validator/tests/test_tile_indexed.py \
    -k "resolved_fidelity or ordered" -q
  # 本卡 tb（provider：C019 unit runner＋本卡注册 case）：
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  ```
- **通过/交接**：tb 报告含逐 case 双侧求址相等（元素数、每元素 remote/local 地址、错误码），mask-off 例请求数=0、credit 无消耗；负例＝注入"mask-off 段仍发请求/占 credit"缺陷必须先 RED 后 GREEN。消费者：C056（gather/scatter FSM 消费求址结果）、C067（T-16/T-50 对拍）。W4-04 须待 C055–C057 全部合入并通过端到端验收才 DONE。

<a id="c056"></a>

## C056 — gather/scatter 实字节与有界窗口

- **任务**：W4-04
- **前置**：C055, C032, C040
- **建议 commit**：`feat(coral-w4): MFE gather/scatter FSM with real bytes, bounded windows and ordered scatter`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §5.4（步骤 4/5、不变量、U4-a）；`engines.py:1483-1591/1677-1742`（`_tick_gather`/`_tick_scatter` 逐态）、`engines.py:781-811`（first-fault 锁存/bypass_forbidden）、`cache.py:36-45`（`CacheLineIdentity`）；C032 交付的 `mfe_top.sv` 直搬腿与逐事务结算；语料 `examples/workloads/indexed_gather_scatter.mlir`（`test_tile_indexed.py:28` FIXTURE）。
- **修改边界**：H。扩展 `hardware/coral_nexus/rtl/memory/mfe_top.sv`（gather/scatter FSM）＋`rtl/memory/tb/mfe_gather_scatter_tb.sv`；w4 suite 注册 case。真实外侧腿经 C023/C024 `MemoryService`，不新写外部服务。V 不改。
- **实现步骤**：
  1. Gather FSM：逐 segment 消费 C055 求址结果，真实字节搬运（fixture 真值 DATA rows `[10..13]/[20..23]/[30..33]/[40..43]`、gather idx `(3,0,3)`、scatter idx `(1,1,2)` 同 `test_tile_indexed.py:37-39` 语义）；窗口有界（每 segment 落入已授权窗口才发）。
  2. Scatter FSM：ordered `(i,j)` 逐段顺序、同址覆盖语义与 Python 对拍一致；first-fault 锁存后停止发新段但已发段照常结算；`bypass_forbidden` 逐案拒绝。
  3. Indexed cache 路径：逐级真实 line 身份（`CacheLineIdentity` 对齐），whole-line refill 不越授权段（GAT-18），bypass 请求逐案计数并与 Python `gather_cache_bypass_requests` 对拍。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  # oneTile 真字节端到端（provider：C040 cosim 入口；场景本卡生成并注册 schema）：
  cn cosim --scenario "$H/verification/scenarios/w4_indexed_bytes.yaml" \
    --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"
  ```
- **通过/交接**：cosim 输出 OUT 字节与 fixture 期望逐 byte 相等（OUT row1=`[10,11,12,13]`、row2=`[40,41,42,43]`、row0/3 未写），gather/scatter 各 48 B、credit 与 L2 闭合检查通过；负例＝注入"scatter 乱序 (i,j)"或"first-fault 后继续发段"缺陷先 RED 后 GREEN。消费者：C057（页授权闭环以本卡外写为载体）、C067（T-16）。

<a id="c057"></a>

## C057 — 页授权 acquire/revoke 与终态引用闭环

- **任务**：W4-04
- **前置**：C056, C024
- **建议 commit**：`feat(coral-w4): page authority acquire/revoke closure with per-txn idempotent leases`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §4.3（真实外侧路径/GAT-19 原子性）、§5.4（步骤 3/6）；**先读现码** `pipeline_validator/memory/page_pool.py:98-155/291-357`——`AccessLease.acquire(txn_id)/release(txn_id)` 与 `_acquired` 事务集合**已存在**，本卡只做语义收口，不另起第二套事务记账；C024 `MemoryService` 端点；W4 设计 §4.7。
- **修改边界**：H＋V。H：新增 `rtl/memory/lease_checkpoint.sv`、`rtl/memory/page_authority_bridge.sv`（Tile 侧桥；**页授权权威在 Python，不新增 `page_authority.sv` 权威实体**）＋tb，w4 suite 注册。V：`pipeline_validator/memory/page_pool.py`——先核对两本账（对象引用＝per-scope `leases` 计数 `_release_lease`；事务引用＝per-lease `_acquired` 集合），新增 `acquire_range(scope,binding,offset,size,txn_id)` 原子 bridge（单次调用内 check+take＝`scope_owns_range`→grant lease_ref+epoch→txn 注册），**保留旧 `lease()/acquire()/release()` 的合法语义**；transport 层迟到重复可按幂等键去重，但已结算 lease 的非法 double-release 是响亮错误（错误码＋计数），**不得全局吞成 False 假装成功**；`closing` 状态机＋`revoke`（关新→等旧引用→invalidate 收 ACK→epoch 递增→换 owner）。新 `pipeline_validator/tests/test_w4_hb_lease.py`。不改 `arena.py`（L2 finalizer 既有语义）。
- **实现步骤**：
  1. `lease_checkpoint.sv`：每段逐页 `lease_check`（GAT-18），持 `lease_ref+scope_epoch` 至事务终态 ACK 才释放；grant 后下游 stall 保存 owner/pending，cancel 需结算 grant 后释放（不得把 unaccepted 当无资源）。
  2. Python 权威收口：`acquire_range` 原子 bridge＋旧 API 合法语义保留；撤销序列按 GAT-20 固定次序"关新→等旧引用→副本失效收 ACK→epoch 递增→换 owner"（invalidate 先于 epoch++ 是合法次序）；RTL 侧仅 lease 引用计数，Python `PagePoolRegistry` 唯一权威（不新增双执行记账）。
  3. 终态闭环：每个 accepted indexed/MFE 事务恰一终态，`lease_ref` 引用在终态与隔离结算后归零；迟到回包按完整键忽略/计数不反扰。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  conda run -n elenor-validator python -m pytest -n0 \
    pipeline_validator/tests/test_w4_hb_lease.py pipeline_validator/tests/test_page_pool.py -q
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  ```
- **通过/交接**：`test_w4_hb_lease.py` 断言合法 release 成功、已结算 lease 的 double-release 响亮报错（错误码＋计数，非静默 False）；tb 断言"不经 `acquire_range` 直接发 indexed 请求"的缺陷注入被检出；撤销负例＝反向注入"**invalidate ACK 未收齐就递增 epoch/换 owner**"先 RED 后 GREEN。消费者：C058（MSHR/clean 联动）、C062（旧代/永阻注入）、C067（T-50/T-65）。

<a id="c058"></a>

## C058 — Profile 12 步与范围维护 9 步

- **任务**：W4-05
- **前置**：C030, C032, C036, C057
- **建议 commit**：`feat(coral-w4): profile member 12/9-step FSM, range maintenance and hold-retire ledger`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §5.5（步骤 1–5、U5-a..d）；`profiles.py:22-46`（`PROFILE_STEPS`/`MAINTENANCE_STEPS` 次序）；`profile_controller.py:473-491/1013-1043/1306-1358`（静止判定/range gate/维护 clean）；`arena.py:961-1022`；C030 bank/cache/MSHR 计数端口、C036 gate 事件唤醒。
- **修改边界**：H＋V。H：新增 `rtl/memory/profile_member.sv`、扩展 `rtl/memory/arena_ledger.sv`（hold-retire）＋tb。V：**只改** `pipeline_validator/memory/profile_controller.py`（gate_open 事件驱动 retry 替代每 tick 轮询；member ack 收 `RTL_TILE_READY`；补 `range_issue_blocked` 真实消费点：范围 gate 期间重叠请求阻塞、非重叠继续）。**禁止在 `arena.py`/`tile.py` 建第二可执行 hold 账本**：hold/retire 的 Python 侧只读镜像归 W1 cosim ledger（仅由真实边界消息更新、不回驱 RTL）；`arena.py` 的 L2 Arena 语义不改。新 `tests/test_w4_hold_retire.py`（断言走 W1 只读 ledger 快照与 ProfileController 状态）。
- **实现步骤**：
  1. `profile_member.sv`：12 步/9 步 FSM 与 `profiles.py` 表完全同序；PREPARE 静止四条换成 RTL 侧计数（cache pending_clean==0、dirty==0、MSHR active==0、Arena/View pin/inflight==0）；COMMIT 校验 shadow 与已准备事务一致后原子切 active 并递增 generation；失败/超时锁 gate 进 recovery，仅显式 `reset_mode` 重驱动。
  2. gate 双唤醒接 RTL：Tile 端口 passive，GateNotify 消息触发重试（C036 token 补 L1 gate 项）；单元 tb 逐 cycle 断言无 tick 轮询回归。
  3. hold-retire 链与范围维护：hold 排他预留（`allocated=held+bound` 逐 bank 守恒）→COMMIT_FINAL→RETIRE_READY→Group RETIRE 授权→**RTL 实际本地回收 L1/Frame（view/Arena/本地槽位）**→可靠 TASK_COMPLETION→Group 结清剩余父 pin/R/Route 与程序驻留引用（对齐总入口 §5 退休链）。"不归还 L1/Frame/UCE/R、不递减程序驻留引用"是 **INPUT_RELEASED 早释放信号**的限制，不得套到退休链上；已早解 pin 退休不重复解。9 步范围维护的脏行 clean 产生真实下游事务并等完成才 ACK。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  conda run -n elenor-validator python -m pytest -n0 \
    pipeline_validator/tests/test_profile_runtime.py -k "MemberProtocol or ByteOracle or LeaseRetirement or range_maintenance" -q
  conda run -n elenor-validator python -m pytest -n0 pipeline_validator/tests/test_w4_hold_retire.py -q
  cn unit --suite w4 --config "$C2" --out "$EVIDENCE/unit"
  ```
- **通过/交接**：12/9 步 happy path（L1-only/L2-only/L1+L2）与 hold-retire 循环逐 bank `allocated+free==user_spm` 成立；负例＝hold 占用期间并发 ticket 同 bank 预留被拒（不超卖）、commit shadow 不匹配维持旧 active 报错，均先 RED 后 GREEN。消费者：C059（DRAIN/复位联动）、C066（T-42/43/46/47）、C067（T-18/19/20/46/56/58/66）。

<a id="c059"></a>

## C059 — 外写隔离、poison 与完整复位（W4-06＋W6-03）

- **任务**：W4-06, W6-03
- **前置**：C058, C034
- **建议 commit**：`feat(coral-w4/w6): cancel/isolation terminals, first-fault arbitration and tile reset endpoint`
- **输入/先读**：`../implementation/W4_Memory_Subsystem_Plan.md` §5.6；`../implementation/W6_Events_Fault_Reset_Plan.md` §5.3（数据结构 1–7、U3-a..i）；`transfer.py:1415-1560/1690-1810`（cancel/acknowledge）、`transfer.py:1796-1804`（`confirm_isolation` 现仅测试调用）、`tile_group.py:1049-1125/1992-1993/2108-2110/2235-2267`、`reset_domain.py:11-233`；C032 已交付的 cancel_tracker 逐事务结算基础与 C034 已交付的 first-fault/终态端点基础。
- **修改边界**：H＋V。H：**扩展已有端点，不重复创建**——`rtl/memory/cancel_tracker.sv`（C032 基础上补三终态集合/取消结算）、`rtl/w6/first_fault_latch.sv`（C034 基础上补完整仲裁与 `suppressed_fault_count`）、`rtl/w6/reset_endpoint.sv`（C034 基础上补完整 8 阶段 FSM）＋tb 扩例。V：`memory/transfer.py`（`confirm_isolation` 接入生产触发点：Group drain 中发 TaskCancel 并等三终态，不再依赖测试注入）、`runtime/reset_domain.py` 行为适配（仅加 Tile 段执行/证据上报，不重命名/不改字段；共享接线归 W1 已在 C035/C036 落）、新 `tests/test_w4_cancel_isolation.py`。
- **实现步骤**：
  1. `cancel_tracker.sv`：每 accepted op/Transfer/命令终态集合 `COMPLETED|ISOLATED|FAULTED`；`CANCEL_REQUESTED` 仅请求标记；已接受外写不可凭 generation/timeout 撤销，只在真实完成或隔离确认后结算；同周期 cancel/complete 恰一终态、重复终态/双释放全拒。
  2. first-fault 与复位端点：latch first-fault-wins，同拍仲裁按 `source_id→engine_id→ctx_slot→op_seq` 字典序（不用哈希）；`FAULT/RESET_CMD→FREEZE_ISSUE→CANCEL_UNISSUED→DRAIN→MARK_EVENTS→INVALIDATE_RESIDENCY→REBUILD_CREDIT→RESUME/DONE` 固定次序，Group 唯一顺序权威、Tile 被动 ACK；**DRAIN 判据只含真实义务闭合（引擎 accepted 归零、MSHR/传输静止、无未决 debug 导出），绝不把 event PENDING=0 写进判据**（否则与 MARK_EVENTS 死锁）；**MARK_EVENTS 由 Tile 实扫并写自己的本地事件表（PENDING→RESET，已终态保持）再 ACK 计数**，Group 只写 Group 表；超时只触发强制取消升级，永不判排空成功；watchdog 归 Group，Tile 不含改变功能状态的计时器。
  3. poison 与故障域：隔离失败不返回 ISOLATED，占位至真零证据并 poison（Group 独占），拒绝新启动直至显式静止＋证据清零＋显式恢复；生产故障域=GROUP（RTL FAULT_REPORT 的 tile_id 仅归因记录）；`CANCEL_TASK` 旧代/stale 幂等 ACK 计数，live 但未知＝协议错误进故障链。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  conda run -n elenor-validator python -m pytest -n0 \
    pipeline_validator/tests/test_w4_cancel_isolation.py \
    pipeline_validator/tests/test_memory_invariants.py -k "cancel or reset" -q
  conda run -n elenor-validator python -m pytest -n0 \
    pipeline_validator/tests/test_runtime.py -k "TestFaultReset or TestCycleCap or TestCycleCapPoison" -q
  # oneTile 全链复位（场景本卡生成）：
  cn cosim --scenario "$H/verification/scenarios/w6/u3b_full_reset_drain.yaml" \
    --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"
  ```
- **通过/交接**：复位各阶段信号与 `ResetState` 一一对应且次序不乱，MARK_EVENTS ACK 带 PENDING→RESET 计数；外写取消两出口（真实完成 vs 隔离确认）外部结果可区分；隔离失败 poison（`poisoned_reason` 非空、新启动全拒）。负例＝注入"DRAIN 等 PENDING=0"缺陷必须死锁/超时被检出。**安全故障注入场景本身 PASS（受控行为），但 DUT run 终态非 SUCCESS（Task 终态唯一非成功、零泄漏）——case PASS 不等于 run SUCCESS**。消费者：C060–C062、C067（T-24/25/26/45/48/52）。

<a id="c060"></a>

## C060 — Context park 与时钟门控存活路径

- **任务**：W6-04
- **前置**：C059
- **建议 commit**：`feat(coral-w6): hierarchical clock gating with context park liveness`
- **输入/先读**：`../implementation/W6_Events_Fault_Reset_Plan.md` §4.4（CR-W6-W2-01/02）、§5.4（数据结构 1–3/5、U4-a..c）；上游 `CoreAxi.scala`/`ClockGate.sv`（C001 锁定 pin）；`tile.py:300-313`（halt 口径）；C027 已暴露 Tile 级 `gate_ctl` 输入接口。
- **修改边界**：H。新增 `rtl/w6/gate_ctl.sv`＋tb（含分层 enable 与 `force_wake`）；**同卡完成真实门控接入**：本卡为集成 owner，把分层 enable 真正接进 CoreAxi/各单元 caller（W2 实体，落增量 patch 并迁移全部调用点），不能只"消费 C027 暴露的接口"而不接算法。V 不改。
- **实现步骤**：
  1. 三层门控：`Context park`（只关该 Context 取指/译码时钟，排除 RF/VRF 回写端口，不影响在途/他 Context/唤醒）；`unit idle`（各单元 accepted/occupied 计数驱动，MFE/RVV idle 计数不参与 Context park 门）；`Tile quiescence`（全部 park ∧ 在途表全空 ∧ 完成通道无待处理项）。`io.wfi` 仅 park 提示，不进 Tile 级关钟条件。
  2. 唤醒存活性：门控判定到时钟关断之间到达的完成按未门控处理（不丢唤醒/不漏写回/不重复完成）；freeze/cancel/reset 命令经 `force_wake` 强制解除目标单元门控，不依赖被门控单元已醒。
  3. debug halt：停在指令/launch 边界，不破坏信用与 descriptor 一致性；halted 期间**返回/唤醒/复位路径照常服务**（停新发、完成照常处理），不是"一切状态不变"。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w6 --config "$C2" --out "$EVIDENCE/unit"   # provider：C019 runner＋本卡注册 case
  ```
- **通过/交接**：单 Context WAIT park 时 `tile_clk_en==1`、仅该 Context `front_en==0`、在途写回正常；全 park＋在途＋门控边界事件下唤醒零丢失/重复（T-27/T-36 判定）；负例＝注入"WFI 单 Context 直连全核关钟"缺陷（去分层）被 tb 抓出（先 RED 后 GREEN）。消费者：C061、C067（T-26/27/51/54/67）。

<a id="c061"></a>

## C061 — 小 debug FIFO 与受限 HBM 导出

- **任务**：W6-04
- **前置**：C060
- **建议 commit**：`feat(coral-w6): debug FIFO with drop-new policy and privileged bounded HBM export`
- **输入/先读**：`../implementation/W6_Events_Fault_Reset_Plan.md` §5.4（数据结构 4、U4-d1/d2/e/f）；C015 冻结的 `w6_terminal_reset.contract.yaml` fragment 与 C020 `PROTOCOL_FREEZE.json`——debug 导出/门控字段与深度推导（`derive_w6_params.sh` 产 `cs1_*.yaml` 与 target manifest 值）**已在封印内**。旧 W6 计划 §5.4"实现期追加字段更新 hash"与一次封印规则冲突，按共享合同修正：字段缺失时走**新 hash 新 seal** 报 parent，不得向已冻结文件追加。
- **修改边界**：H。新增 `rtl/w6/debug_fifo.sv`＋tb；外存导出事务腿复用 C056/C057 MFE/页授权路径（目标 ring/范围/权限检查在 W4 执行侧）。V 不改。
- **实现步骤**：
  1. 正常模式默认关：无片上历史 memory、无 HBM 日志 ring，仅 PMU 计数/first-fault 摘要/溢出 sticky。
  2. 满则丢新：丢弃不阻塞 issue/state_free/唤醒/终态/故障处理；`dropped_count` 饱和计数＋"窗口不完整" sticky；深度用冻结值（深度=1 为正确性边界参数，不是工程默认）。
  3. 特权导出与关闭排空：`DEBUG_ENABLE/限额/捕获配置` 走 CTRL 特权清单（默认关）；已接受 HBM 日志写保留 owner/addr/credit，须真实完成或隔离确认（C059 三终态），未排空状态可查 `pending`/`poison`，不能清 FIFO 冒充完成。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn unit --suite w6 --config "$C2" --out "$EVIDENCE/unit"
  ```
- **通过/交接**：深度=1 连续两记录第 2 条被丢（`dropped_count`=1、sticky 置位、无功能阻塞）；满注入下 issue/唤醒/终态不受影响；特权未开时导出尝试无任何 HBM DMA/ring 写（仅计数标记）；负例＝注入"FIFO 满阻塞功能完成"缺陷先 RED 后 GREEN。消费者：C062（满日志行）、C067（T-51/67）。

<a id="c062"></a>

## C062 — 竞争、旧代、回绕与永阻注入矩阵

- **任务**：W6-05
- **前置**：C061
- **建议 commit**：`test(coral-w6): race/stale/wrap/permanent-block injection matrix on oneTile loop`
- **输入/先读**：`../implementation/W6_Events_Fault_Reset_Plan.md` §5.0（单周期线性化点）、§5.5（注入矩阵全表、U5-a..d）；C059/C060/C061 全部端点；`GEN_WRAP_QUIESCE` 消息语义（C015 fragment：Tile 只查不超时，Group 响应授权复用）。
- **修改边界**：H。新增场景 `hardware/coral_nexus/verification/scenarios/w6/u5b_<row>.yaml`（同拍竞争/满表/满日志/旧代/回绕/外部永阻/隔离失败各行）、`u5c_external_permanent_block.yaml`（C016 冻结的 scenario schema：explicit package/input bindings/seed data）；判读字段读边界日志与 C059 计数。V 不新增执行路径（判读所需计数器均已在前置卡冻结字段内）。
- **实现步骤**：
  1. 每行先在无注入 baseline 上通过正常终态一致性，再注入冲突；同拍竞争（completion/fault/cancel/reset 组合、同 Context vs 异 Context）表终态与 Group 观察一致、单一线性化点可对拍。
  2. 旧代/回绕：旧 generation 完成/ACK/signal 按完整键拒绝/忽略/计数、不写新代状态、迟到回包归还信用；回绕前发 `GEN_WRAP_QUIESCE(ContextKey)`，Group 确认"无旧代在途/队列/ACK 缓存"并授权后 Tile 才回绕位型（仅 Tile 本地静止不足；同位型古老副本在途必须阻止回绕）。
  3. 永阻与隔离失败：外写/外存事务永不响应时 DRAIN 阻塞至 `max_drain_cycles`→强制取消→仍无真实终态→poison；`is_quiescent` 不置 1、`completed=False`、`poisoned_reason` 非空、`reset_domain.is_done=False`——**该行退出口径是"永不判成功排空"，不允许"等更久就成功"**。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  conda run -n elenor-validator python -m pytest -n0 pipeline_validator/tests/ -q   # baseline 不退化
  for s in u5b_same_cycle u5b_full_table u5b_stale_gen u5b_gen_wrap u5b_iso_fail u5c_external_permanent_block; do
    cn cosim --scenario "$H/verification/scenarios/w6/${s}.yaml" \
      --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim/${s}"
  done
  ```
- **通过/交接**：注入矩阵逐行判据成立（状态/字节/计数），合法故障注入场景 PASS 而缺陷 mutation（去同拍 bypass、去 event_generation、提前释放、超时判成功）被检出 FAIL；永阻行最终态停在 poison。注意：本卡 checker verdict 由 C064 消费出正式报告，本卡以边界日志自判。消费者：C064（mutation 判读）、C067（T-13..15/23..27/45/51/52/54/67）。

<a id="c063"></a>

## C063 — 全 kernel 包链与多映像冷热/失效（W5-04＋W5-05）

- **任务**：W5-04, W5-05
- **前置**：C062, C052
- **建议 commit**：`feat(coral-w5): install service, multi-slot residency and sealed package e2e chain`
- **输入/先读**：`../implementation/W5_Kernel_Image_Runtime_Plan.md` §4.4/§4.5（产物/步骤/验证）；C052 的计算库映像映射与资源再核验；C038/C039 的 sealed package、`slot_freeze.json` 实测容量与 **C039 已交付的 `install_service.sv`/`program_table.py` 安装与多 slot 驻留基线**；C030 程序物理端口。
- **修改边界**：H＋V。H：**扩展 C039 已交付的 `rtl/w5/install_service.sv`**（不重建）＋tb 扩例（live kernel 并发安装/失效）；场景 `$H/verification/scenarios/t58_cold_warm.yaml`、每族 kernel oneTile 场景与多程序并存场景。V：`runtime/program_table.py` 演进（pending install 记录、引用计数全局一份、累计容量账本、多 slot 位图、epoch）、新 `tests/test_w5_residency.py`、`tests/test_w5_e2e.py`。
- **实现步骤**：
  1. 安装 RTL 四步 pull：INSTALL 头→READ_REQ→READ_RESP→INSTALL_ACK（实测 hash/长度＋本地句柄），有界 FIFO/分段背压，校验失败即 fault 无半驻留；不占数据 Arena/MSHR/credit（RES-17/18）。
  2. 驻留状态机：`ABSENT→PENDING_INSTALL→RESIDENT`，invalidating 串行点后无新热命中；冷装 A 运行中冷装 B（不同 slot、不重叠链接地址）无全 Tile stall；安装中提交同映像只加引用共享完成点；引用归零＋显式失效收齐 ACK 才释放 slot；累计超容量→有界等待或拒绝。
  3. 全链端到端：`package`→`cosim` 串联（真实上传→安装→accepted→执行→retire→TASK_COMPLETION）；warm＝零 Tile 外部 text 字节（边界计数器；本地取指与 descriptor patch 合法）；同映像双 Context 重入可写区不相交、引用守恒；全部构建期拒绝负例在链路中复现。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  conda run -n elenor-validator python -m pytest -n0 \
    pipeline_validator/tests/test_w5_residency.py pipeline_validator/tests/test_w5_e2e.py -q
  cn cosim --scenario "$H/verification/scenarios/t58_cold_warm.yaml" \
    --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim"
  ```
- **通过/交接**：热命中 READ_REQ=0、冷装并发无 stall、`assert_l2_closed` 与资源零泄漏成立；全族 oneTile 场景执行完成且 manifest 登记不以"封印成功"替代执行；负例＝warm 改写 resident text 被拒、patch 越出 patch_mask→invalid descriptor fault、INVALIDATING 中提交不获热命中，均先 RED 后 GREEN。消费者：C067（T-30/49/53/56/58/61/63/66）。

<a id="c064"></a>

## C064 — 只读资源镜像报告与完整 checker 检出

- **任务**：W7-03
- **前置**：C063, C054
- **建议 commit**：`feat(coral-w7): C1..C7 checkers, readonly resource-mirror reports and mutation suite`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §1.1（只读 ledger 口径）、§8（产物/步骤/负例）；C016 冻结的 trace/verdict/t_matrix schema 与 `cn_trace/v1`；W1 只读 ledger/snapshot API（C025 `bp_status` 计数、C035 `retirement.py`/`incomplete.py` 的 `l1_ledger_snapshot()` 等——只读、derived only from `RESOURCE_COMMIT_IND/RETIRE_ACK/TASK_COMPLETION` 等真实消息）；C020 各合同 hash。
- **修改边界**：H＋V。H：**扩展 C041/C054 已合入的 checker 骨架（C1/C2 与 C4 数值判定）至完整 C1–C7**，补 `mutator.py`/`not_empty.py`/资源镜像对账报告，不另起重复实现。V：`report.py` L1 快照行**本卡实施**迁移为消费 W1 只读 snapshot（report.py 修改权在 W7，不再推迟到 C068）；新 `pipeline_validator/tests/test_w7_checkers.py`；扩展 `cn verify` 判定路径（命令宿主 C007 交付，本卡接入 C1..C7 真实执行）。不写 RTL 执行路径。**C6 范围限定**：本卡 C6 只验窗口/bundle 封窗完整性与 `NOT_FULLY_REPLAYABLE` 判定；完整行为重放（逐条驱动 `bp_*` 比对请求）C069 才交付，本卡与前置卡不得强制未来重放能力。
- **实现步骤**：
  1. C1/C6/C7 骨架先落（schema/seq/唯一性/封窗/manifest），C2 偏序（退休链、早释放链不强加在 Task completion 后）、C3 守恒（credit、R/Arena 逐 bank `allocated+free==user_spm`、每已接受命令恰一终态、L2 收支由 `l2_extent_release` 与原始 endpoint 消息交叉验证）消费 W1 只读快照，**禁止读未实例化 Python Tile L1 或拿 mock bool/单侧 echo 当 PASS**。
  2. C4 数值接 C047 oracle（KD-01 由 bit oracle 定 normative、mask 类仅 bit oracle、不允许 ULP 放宽）；C5 终态（超时=FAIL 附重放包、`pending≠完成`、外部永阻不冻结独立工作）。
  3. mutation 套件：每个 checker 至少一负例（seq 断档、消费者早读、旧代唤醒新代、释放早于最后读取、终态早于 store_done、双重回收、`pending冒充完成`、尾部缺失仍宣称可重放、dangling 引用），先跑负例确认 FAIL 命中字段、再跑正例 PASS，报告成对留证；资源镜像对账报告（账本字段＋差异行）随 verdict 落盘供 C067/C068 gate 消费。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  conda run -n elenor-validator python -m pytest -n0 pipeline_validator/tests/test_w7_checkers.py -q
  # 先把一条已存在的 C062 场景真实跑到本卡目录，再 verify 实际 manifest json（路径按 C040 cosim 输出约定）：
  cn cosim --scenario "$H/verification/scenarios/w6/u5b_same_cycle.yaml" \
    --backend rtl --rtl-tile-mask 0x1 --config "$C2" --out "$EVIDENCE/cosim/u5b_same_cycle"
  cn verify --manifest "$EVIDENCE/cosim/u5b_same_cycle/manifest.json"
  ```
- **通过/交接**：全部负例期望 verdict 与命中字段断言成立（先 RED 后 GREEN）；`not_empty` 公共实现拒绝非空 PASS；退出码 0 不充分——报告须含逐 checker 判定与资源对账差异（成功例差异为空）。消费者：C066/C067（verdict 生成）、C068（gate 重放 C1..C7）。

<a id="c065"></a>

## C065 — 四 RTL Tile 集成与 mixed 调试隔离

- **任务**：W1-05
- **前置**：C064
- **建议 commit**：`feat(coral-w1): four RTL tile integration harness with mixed-debug isolation`
- **输入/先读**：`../implementation/W1_Boundary_Cosimulation_Plan.md` §11（产物/步骤/验证）；C040 的 `cn cosim` 入口与 oneTile 链；C019/C020 的 `cs1_c1.yaml`/`cs1_c2.yaml`；C055–C063 各 RTL 端点。
- **修改边界**：H＋V。H：**新 4-Tile Verilator top（`VnCosim` 4 Tile 实例）＋`sim/cabi` BUILD 多 Tile 实例化与 host router 真实改造**（每 Tile 独立 IO/返回队列路由、预分配不变、steady-state 无新增堆分配），不是只加 CLI mask flag；`scripts/cn.sh` cosim 分派扩展（4 Tile mask `0xf`、mixed 子集 mask、参数校验/证据落盘管道）。V：`pipeline_validator/cosim/run.py`（`run_cosim` 退出码 0/1/2/3：0 全 PASS/1 checker FAIL/2 INCOMPLETE/3 异常）、新 `tests/test_w1_integration_gate.py`（pytest 桥，实际执行经 `cn.sh`）。
- **实现步骤**：
  1. 4 RTL 最终门：`cs1_c1`（退化回归）与 `cs1_c2` 双配置 × mask `0xf`，P0–P4 全链；运行 `l2_admission_wait` 与 transformer prefill 语料（mandatory 能力集不缩水为 matmul-only）；`assert_l2_closed`/`unclosed_l2_objects`/`credit_invariants_hold` 三项终态检查；**被替换 Tile 无任何 Python L1/引擎执行影子（T-64；`PythonTileBackend` 仅是未替换 Tile 的合法 owner）**。
  2. mixed 仅调试：部分 mask（0x1/0x3/0x5）bring-up，每次 run 输出 `evidence: mixed_debug_only`；C064 checker 拒绝 mixed 报告冒充 final PASS。
  3. verdict 两态区分：协议/运行时 fault 照常走 Group fail-stop 排空（排空完成且账本清零→`FAULT`，附完整链证据）；真实 crash/不可恢复 fatal→`INCOMPLETE`（无 `run_end.json` 封窗、退出码<0、`credit_invariant=False`、`unclosed` 非空、seq gap、accepted 未归还 credit 任一成立）；每次 run 落 `report.json`（evidence_tier/cycles/PMU/BP 快照/守恒布尔/`drained`/`unclosed`）。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  conda run -n elenor-validator python -m pytest -n0 pipeline_validator/tests/test_w1_integration_gate.py -q
  cn cosim --scenario "$H/verification/scenarios/w1_four_tile_smoke.yaml" \
    --backend rtl --rtl-tile-mask 0xf --config "$C2" --out "$EVIDENCE/cosim"
  cn cosim --scenario "$H/verification/scenarios/w1_mixed_debug.yaml" \
    --backend rtl --rtl-tile-mask 0x5 --config "$C2" --out "$EVIDENCE/mixed"
  ```
- **通过/交接**：4-RTL tier 报告逐字段断言（`bp_status.fatal_code==0 && credit_ok && message_seq_complete && unclosed==[] && l2_closed && evidence_tier=="four_tile_rtl"`）；Python 侧对拍限**合同级偏序＋声明输出 bytes**一致（不要求 cycles/全 trace hash 相等）；负例＝mixed report 被当 final PASS 时 C064 checker FAIL、crash 无封窗→退出码 2/INCOMPLETE。**mixed 不替代四 RTL 最终门**。消费者：C066/C067（场景宿主）、C068（G4 门）。

<a id="c066"></a>

## C066 — 真实 L2 早释放、准入与共享输入场景

- **任务**：W7-04
- **前置**：C065
- **建议 commit**：`test(coral-w7): real-bytes L2 early release, admission and shared-input scenarios on four RTL tiles`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §10.1（T-14 命令与"Python 侧只是必要条件"口径）；早释放勘误合同（共享合同"早释放"条）：Task 级整体信号、Group `all_tasks` 聚合、纯读 pin 解除、按 buffer owner release、全闭合物理 final-free，**不新增同 Task 分批通知/子范围回收**；C035 准入链、C058 Profile、C065 四 RTL 宿主。
- **修改边界**：H。新增场景 `$H/verification/scenarios/{t14_early_release_realbytes,t57_l2_closure_dual_reader,t41_r1_lease,t42_l1_capacity,t43_attribution,t47_range_capability,t15_output_ready}.yaml`（C016 schema：explicit package/input bindings/seed data/expected outcome，真实 seed bytes）＋ `t_matrix` 场景列填充。V 不改（判读全走 C064 checker）。
- **实现步骤**：
  1. T-14 真字节版：以 `l2_admission_wait.mlir` 同图形在 4-RTL 上跑，A_IN 131072 B 真实 seed bytes；断言 final-free 后源空间被 B 复用（同 bank segments）时 **L1 staging 保持、计算结果仍逐 byte 正确**，且 `task_lease_release` 均晚于 `l2_extent_release`；**INPUT_RELEASED 不归还 L1/Frame/UCE/R、不递减程序驻留引用**（资源不提前还）。
  2. T-57 双 reader＋producer 先退休：私有 buffer 闭合、final-free 恰一次；T-41 R=1 第二 Task 归因（`tile.free/input_released/output_ready/取消` 后 lease 仍占用、`WAIT_CONTEXT_LIMIT`）；T-15＝**`output_ready` 发布不等外部 store/HBM 完成、失败路径不撤销已发布输出**（共享只读输入多消费者语义归 T-14 场景，不重复建）。
  3. 准入容量与范围维护能力：T-42 单 Tile L1 占满/尾部 Task 部分放置/零 dispatch；T-43 五类独立归因、空池→PERMANENT fault 不入等待队列；T-47＝**范围维护的两种能力目标**（前提：HBM 写入后存在缓存读者，非容量压力、非取消压力——取消链已在 C059/C062 覆盖）：正例＝**仅范围能力**——只阻塞重叠范围、非重叠流量继续，脏行 clean 产生真实下游事务并等完成才 ACK；负例＝**仅全域能力且无静止边界→编译失败**（构建/装载期拒绝并落 reject receipt，不经 cosim 执行，对齐需求 T-47 判据）。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  for s in t14_early_release_realbytes t57_l2_closure_dual_reader t41_r1_lease t42_l1_capacity t43_attribution t47_range_capability t15_output_ready; do
    cn cosim --scenario "$H/verification/scenarios/${s}.yaml" \
      --backend rtl --rtl-tile-mask 0xf --config "$C2" --out "$EVIDENCE/cosim/${s}"
  done
  cn verify --manifest "$EVIDENCE/cosim/t14_early_release_realbytes/manifest.json"
  ```
- **通过/交接**：各场景 `t_id` verdict=PASS 且 `not_empty_pass=true`（T-14 含逐 byte 输出对拍与 staging 保持断言；T-47 正例＝重叠被挡、非重叠继续、clean 下游完成才 ACK）；T-47 负例＝仅全域能力且无静止边界的编译失败 receipt 在场；负例＝"未聚合就复用"构造（`SIGNAL_PHASE` 缺全 `all_tasks`）→ C2 报 `PREMATURE_RELEASE` 先 RED 后 GREEN。消费者：C067（t_matrix 行）、C068。

<a id="c067"></a>

## C067 — T-01..T-68 适用矩阵完整执行

- **任务**：W7-04
- **前置**：C066
- **建议 commit**：`test(coral-w7): full T-01..T-68 applicability matrix execution and evidence registration`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §9（68 行矩阵列义/例外）、§10.2（批次 .1–.7）、§10.3；C016 `t_matrix.yaml`/`scenario_contract.yaml`；C041/C054 已完成的 G2/G3 批次产物（作为进度证据被本卡 manifest 引用并重验 hash）。
- **修改边界**：H。逐 T 场景 `hardware/coral_nexus/verification/scenarios/<t_id>_<slug>.yaml` 补齐（G4 批次 .5 内存/授权：T-38/47/48/50/52/55/57/65；.6 Group/包装/build：T-37/42/43/46/56；.7 程序/映像：T-33 执行子集/49/53/58/61/63/66）；**同卡实现批量聚合器** `hardware/coral_nexus/verification/scripts/run_t_matrix.py`（读 `t_matrix.yaml` 场景列→逐条调用 `cn cosim`→收集各 run 级 `manifest.json` 汇总为 `cs1_manifest.json`，含逐行 verdict/证据 hash）。V：仅 `pipeline_validator/tests/test_w7_checkers.py` 增补批次断言（如需）。
- **实现步骤**：
  1. 执行全部 CS1 适用功能行：G2/G3 批次行在 4-RTL 环境复跑（同一 run_id 规则，不漂移）；G4 批次行按 C066 与本卡新增场景执行；每族 kernel 至少 2 条真实场景（`T-xx_a1/_a2`），族缺席＝该族 BLOCKED 报 W7-06，不得静默删族。
  2. **BUILD/LOADER 子项证据按 T 行证据类型落**：T-37 是 build-time 检查（不进 cosim 命令，产物为构建 manifest 断言）；T-33 拒绝边界/T-53 loader 级以 `BUILD+CS1_LOG` 双证据登记（构建期拒绝＋执行子集日志）；T-49 Python 基线行以 `CS1_PY` 登记并注明 RTL 侧冷热在 T-58。
  3. 例外登记与 L2 协作：**延期（`DEFERRED_NOT_APPLICABLE`）仅限 T-59/T-60 与 N≥2 子项（T-28/T-68 的多实例行），各注明启动条件（D-24/D-25、N2），不跑**；其余一律不写延期。按子项区分 G4 功能证据与 G5 待执行：批次 .8 的 T-35 P0/P1/PN、T-36、T-67 的完整窗口单侧重放/PERF 缺口拒绝子项登记为 **G5 待执行（mandatory，提供者 C069–C072）**，本卡只登记不执行；T-35 的 PS 子行在 `perf_gates` 以 `not_run` 占位（W7-05 口径），**不属延期清单**；T-67 的 G4 debug 子项（正常模式零留存、溢出只丢诊断）以 C061/C062 证据在本卡登记。**包含 L2 的行仍须验其 CS1 的 Python 外侧协作部分（Group/L2/页授权账本），不得整行跳过**。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  # 本卡聚合器批量执行全表（内部逐条 cn cosim，不拿单场景冒充全表）；明确输出＝逐 T verdict＋证据 hash 的 cs1_manifest.json：
  python3 "$H/verification/scripts/run_t_matrix.py" --t-matrix "$H/verification/matrix/t_matrix.yaml" \
    --config "$C2" --rtl-tile-mask 0xf --out "$EVIDENCE"
  cn verify --manifest "$EVIDENCE/cs1_manifest.json"
  ```
- **通过/交接**：68 行全部有 status＋场景/证据引用闭合（无 dangling）；所有适用 mandatory 行中已到执行时点者 PASS 且 `not_empty_pass=true`（BUILD/LOADER/CS1_PY 行按其证据类型判定，不强制 CS1_LOG）；仅 T-59/T-60 与 N≥2 子项带延期原因与启动条件；G5 待执行子项（T-35 P0/P1/PN、T-36、T-67 重放/PERF）登记批次与提供者，不冒充已执行；负例＝dangling evidence 引用或 deadline 到点当通过被 C7 拒。消费者：C068（G4 门唯一输入）。

<a id="c068"></a>

## C068 — CN-G4 完整 CS1 功能门

- **任务**：W7-06
- **前置**：C067
- **建议 commit**：`feat(coral-w7): CN-G4 full CS1 functional gate`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §10.4/§12；总计划 §4 CN-G4 出口定义（四 RTL Tile、无 Python 执行影子、完整 MFE/授权/Profile/故障/debug/冷暖/输入早释放闭环）；C007 `cn gate` 宿主（G0 证据基础）与 C041/C054 的 G2/G3 stage gate 先例。
- **修改边界**：H。`cn gate --stage G4` 判定集（复用 C007 宿主框架，本卡交付 G4 证据集读取与重验）；**gate 输入按总入口 §4.3**：执行前先从 C067 `cs1_manifest.json` 与各 run receipt 读取实际产物路径与 hash，连同检查类型写入 gate 输出目录的 `inputs.json`；gate 只读该显式清单重验，不按 mtime 扫 `out/` 猜最新。V 不改执行路径（`report.py` 迁移已在 C064 实施）。
- **实现步骤**：
  1. 证据集重验：读入 C067 `cs1_manifest.json` 全部产物 hash，重放 C1..C7 判据（不只查文件存在）；G4 必备闭环逐项在场：四 RTL Tier、T-14 真字节早释放、T-58 冷热、T-50/65 页授权、T-24/25/26/52 取消复位、T-51/T-67 的 G4 debug 子项（正常零留存、溢出只丢诊断；重放与 PERF 缺口拒绝子项归 C069/C070）、T-64 无影子。
  2. 影子执行断言（按子项证据类型区分）：凡以 cosim 场景为证据的 **RTL 执行子项**必须是 `evidence_tier=="four_tile_rtl"`（mixed_debug_only/stub/unit 不计入 final）；BUILD（T-37、T-33/T-53 构建期拒绝）、LOADER 级与 T-49 的 `CS1_PY` 基线按其登记的证据类型接受，不要求四 Tile tier；`--rtl-tile-mask 0x0` 纯 Python 基线仅作对拍参照，不充当 RTL 证据。
  3. 出口裁决：缺文件/FAIL/BLOCKED→非零退出并逐条列缺失 owner；**G5 待执行子项（T-35 P0/P1/PN、T-36、T-67 重放/PERF）在 G4 门中既不判 PASS 也不标 SKIP——保持 mandatory，以 `pending_g5` 清单移交 C069–C072**；全部通过→`cs1_g4.json`（verdict＋证据 hash 快照＋`pending_g5`）落盘，供 G5 重放/性能链消费。
- **验证**：
  ```bash
  : "${EVIDENCE:?按总入口设置本卡新attempt}"
  cn gate --stage G4 --out "$EVIDENCE"
  ```
- **通过/交接**：`cs1_g4.json` 中 G4 闭环逐项 PASS、资源对账差异为空、退出码 0；负例＝从 manifest 摘除任一必备证据（如 T-14 场景）后重跑 gate 必须非零并指名缺失项。消费者：C069（重放）、C071/C072（性能）、C074（G5 出口）。W7-06 的 G5 最终出口部分归 C074，本卡只关 CN-G4。

---

## 本页口径修正记录（旧计划残留，已按共享合同改正）

1. **W6 计划 §5.4"debug 字段实现期追加、更新 fragment hash"** 与 `PROTOCOL_FREEZE.json` 一次封印规则冲突：C061 改为"字段须已在 C015 fragment 封印内；发现缺失走新 hash 新 seal 报 parent，不追加已冻结文件"。
2. **W6 计划场景路径 `hardware/coral_nexus/scenarios/w6/`** 统一改为合同命令样例的 `$H/verification/scenarios/`（保留 `w6/` 子目录），场景须符合 C016 冻结 schema（explicit package/input bindings/seed data）。
3. **W7-04 批次 .8（T-35/T-36/T-67）** 依赖 W7-05 重放/性能产物，属 G5 提交（C069–C072）；C067 只登记不执行，避免把 G5 证据混入 G4 门。
4. **W4 计划 §4.3 表述"AccessLease 无事务引用"** 与现码不符：`page_pool.py:132-143` 已有 `acquire/release(txn_id)` 与 `_acquired` 集合；C057 只做 per-txn 幂等收口与 `acquire_range` 原子化，不另建记账。
