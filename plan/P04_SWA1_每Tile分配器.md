# P04 · SW-A1 每 Tile AM 拓扑与隔离（含 target identity 升级与 Q09 永久回归）

状态：未开始
前置：P01 通过（P02 硬件环境不是本任务前置；软件层独立验收）。
对应原方案：SW-A1（[设计 v0.5](../design/proposal/BOA_Gemmini_Integration_Design.md) §1.3、§7.2 SW-A 表）；Q08/Q09（§8.1b）；总则 §3.1。

## 1. 目标与不做

目标：把 AM 从 "Tile Group 一个共享 manager（8 槽由全部 Tile 竞争）" 改为 "每个 ComputeTile 独立 manager（每 Tile 4 groups×2 slots=8 槽，Tile 间互不可见）"；同步 report/snapshot 按 Tile 汇总；补齐 Q08 拓扑回归与 Q09 永久 runtime 回归。**本任务是可观察语义变化，必须在任务内完成 target identity 升级，保证旧 Group 共享拓扑的 artifact 不会被新拓扑静默解释**（总则 §4.1 修订版要求）。

不做：不改 engine/affinity（P05）；不改数值（P06）；不改终态记录顺序（P07，但 Q09 断言只依赖偏序不依赖本任务的记录位置）；不加新配置键（每 Tile 槽位仍是 4×2、quota 2 不变）；不实现硬件 allocator。

## 2. 前置与输入

- [00 执行总则与共享合同](00_执行总则与共享合同.md) §1（仓库证据）、§3.1、§4.1（fingerprint/版本规则）、§5（计数解释）。
- 设计 §7.2 SW-A1 行、§8.1b Q08/Q09 行、§8.2 不变量（per-owner 0..2、全局 0..8 改为按 Tile 计）。
- P01 evidence 的版本事实段（当前 schema v4 / `compiler_abi "v4"`、`target_fingerprint` 结构）。

## 3. 当前证据与修改文件

现状（符号定位，行号以当日为准）：

- `pipeline_validator/tile_group.py`：`TileGroup.__init__` 创建 `SharedAmManager(..., tile_id=0)` 并作为 `shared_am=` 注入每个 `ComputeTile`。
- `pipeline_validator/tile.py`：`ComputeTile.__init__` 已有本地构造路径 `shared_am or SharedAmManager(..., tile_id=tile_id)`；`TileUCE.shared_am` 由 host tile 注入。
- `pipeline_validator/runtime/shared_am.py`：`request/_grantable/_grant_pending_locked/_allocate_lease/begin_step/commit_step/release_final/abort_lease/abort_owner/reset` 全部按单 manager 实例工作，天然适配 per-Tile 实例化，无需协议改动。
- `pipeline_validator/report.py`：`group_snapshot["shared_am"]`（单 dict）与 `am_slot_zero_leak`/`am_fault_expected` 检查。
- `pipeline_validator/execution_verifier.py`：`target_fingerprint` payload（`_HW_TARGET_FIELDS` 等分组）。
- 测试：`pipeline_validator/tests/test_boa_am.py` 中 `simulator.group.shared_am`、`test_am_owner_uses_physical_tile_identity`、`test_am_fault_isolation_by_physical_tile`、Q01–Q07 系列基于 Group 级 manager 构造。

修改 ownership：`tile_group.py`、`tile.py`（仅 manager 装配与 snapshot 聚合）、`runtime/shared_am.py`（如需 per-tile 汇总辅助）、`report.py`、`execution_verifier.py`、`tests/test_boa_am.py` 及受影响 runtime/trace 测试、`plan/README.md` 状态行。禁止动 P05/P06/P07 的领域（engine 选择、数值、终态顺序）。

## 4. 输入输出、接口与不变量

- 接口变化 1（拓扑）：设备 AM 容量从 “全体 Tile 共享 8 槽” 变为 “num_tiles × 8 槽”；owner 已含 `tile_id`，`(ctx_id, epoch, tile_id)` 在 per-Tile manager 下唯一。
- target_fingerprint增加固定实现语义标识（如payload中的boa_am_topology=per_tile_v1），不是允许切回Group共享的用户配置旋钮。旧target artifact立即拒绝；若序列化字段也变按严格规则升版。P06另负责数值ABI cutover，不能借后续升级推迟本任务的错target保护。
- 不变量：每 Tile 内协议不变（一次申请/挂起/唤醒、bundle、FIFO、quota 2、`(group,index)` 取槽）；跨 Tile 无任何共享或借用；fault/reset/rebind 只影响本 Tile；全局计数改为 Σ per-Tile。

## 5. 实施步骤

1. `TileGroup` 停止创建共享 manager：不再传 `shared_am=`，让每个 `ComputeTile` 用本地路径构造 `tile_id=i` 的实例（或显式循环构造，二选一，避免 None 兼容分支）。迁移所有 `group.shared_am` 读取方。
2. snapshot 聚合：`TileGroup` 提供 per-Tile 列表（建议键 `shared_am_per_tile`）并把 `shared_am` 键改为聚合视图（计数求和；`free_slots`/`live_leases` 求和，`total_slots=num_tiles×8`）；`report.py` 的 `am_slot_zero_leak` 改为逐 Tile 校验（每 Tile live=0 且 free=total），`am_fault_expected` 按 Tile 归属。列明 producer/consumer 全部调用方（report、trace、测试）。
3. target identity：按 §4 增加 fingerprint 字段；保存一个 P04 前编译 artifact 的 bytes+hash（放 `/tmp` 或 evidence 引用路径），实施后证明其加载被拒。
4. Q08 回归：改写 `test_am_owner_uses_physical_tile_identity`/`test_am_fault_isolation_by_physical_tile` 为 per-Tile manager 语义：Tile0 8 槽全满时 Tile1 同 context 仍能授予；Tile0 fault 后 Tile1 lease/请求不受影响；per-Tile snapshot 断言。
5. Q09 永久回归（补齐实施报告 §4.1 的一次性 smoke）：`boa_am_slots_per_context=1` 的 runtime 用例——同 owner 两条顺序链都用 logical slot0；第一条 FINAL 受理后（输出背压压至 3000 cycles）发起第二条 ALLOC(slot0)，随后 `await` 第一条旧事件；断言：完成、`requests=2 parks=1 wakes=1 faults=0`、旧事件在授予后仍可 await、结束时 live/pending/reserved=0、C bytes 与独立 oracle 一致。注册为常驻 pytest（不是一次性脚本）。
6. 重跑split-K/multicontext/gather多Tile真实场景，记录新周期与per-Tile统计；替换只固定旧实现周期的脆弱测试为容量/隔离/进展不变量，不能单纯把旧周期常量改成新测量值。历史报告留标签。
7. 全量回归：`pre-commit run -a` + `pytest -q -n 4`。
8. 交付可独立pytest/CLI执行的case和smoke入口；shared-am-functional统一容器target由P09注册并消费这些入口。P04不隐式依赖P02，也不留下当前范围未实现的runner。

## 6. 验收表

| Case ID                   | 构造                               | 必须观测                                                                | 失败特征                                |
| ------------------------- | ---------------------------------- | ----------------------------------------------------------------------- | --------------------------------------- |
| Q08a Tile 隔离容量        | 4-Tile 样例，Tile0 8 槽满          | Tile1 新申请同窗口授予；`shared_am_per_tile[1]` 计数独立                | Tile1 被 Tile0 满槽阻塞                 |
| Q08b Tile 隔离故障        | Tile0 fault/reset                  | 仅 Tile0 lease/pending 清零；Tile1 snapshot 不变；报告 fault 归属 Tile0 | abort 波及 Tile1                        |
| Q09 旧事件跨 slot 复用    | 步骤 5 场景（quota=1+背压）        | 全部断言通过且为常驻测试                                                | 第二条 ALLOC 挂死或旧事件不可 await     |
| ID-P04-1 旧 artifact 拒绝 | P04 前 artifact 在 P04 后加载      | 报 `target fingerprint mismatch`，非零退出                              | 静默按新拓扑跑完                        |
| ID-P04-2 新fingerprint    | 保存迁移前artifact、迁移后重新编译 | 前者拒绝、后者replay正确；固定拓扑身份不可由用户切回旧实现              | 旧artifact静默reinterpret或新增兼容后门 |
| RPT-P04-3 per-Tile 报告   | 任一 typed 样例 `--json --report`  | JSON 含 per-Tile AM 统计；zero-leak 逐 Tile；fault 归属正确             | 只有聚合数无法定位 Tile                 |
| REG-P04-4 全量 gate       | 步骤 7                             | exit 0，计数入 evidence                                                 | 漏改的 Group 级测试失败                 |

## 7. 可执行命令

- 回归：`conda run -n elenor-validator pre-commit run -a`；`conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -q -n 4`。
- 样例：`bash examples/run.sh matmul-splitk-multicontext-pipeline --json --report /tmp/p04/sk.report.json`（fidelity 由 run.sh/run_all 注入，不手写重复 override）。
- 本任务使用上述真实pytest/CLI；P09负责注册 `hardware/scripts/check shared-am-functional --out /tmp/p09/am --case q09-old-event-cross-slot`，复用本任务回归，不把尚无P02脚本列为P04通过条件。

## 8. 产物、交接与停止条件

交付代码、回归、独立smoke及P04 evidence（新周期/per-Tile统计/旧artifact拒绝/拓扑标识），供P05/P07/P09消费。若发现artifact字段含旧拓扑语义，保存证据并按严格规则同步升级全部消费者，不能只知会一个不存在的Main而停在半迁移。
