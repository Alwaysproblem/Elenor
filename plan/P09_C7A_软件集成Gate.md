# P09 · C7-A / SW-A 软件全链路 gate 与新基线

状态：未开始
前置：P04、P05、P06、P07、P08 全部通过（P02/P03 证据可引用；P01 基线作对照）。
对应原方案：C7-A / SW-A gate（[设计 v0.5](../design/proposal/BOA_Gemmini_Integration_Design.md) §7.2 C7 行、SW-A guide）；总则 §2、§5、§6。

## 1. 目标与不做

目标：在新合同（per-Tile AM、多 engine affinity、逐 MAC 数值、终态偏序、INT8/epilogue/软件 Conv）下运行并记录完整软件 gate：统一 target 全绿、全量 pytest/pre-commit、typed fixtures 与 P03 golden 的 bytes 链、负/边界 case（含零 K/零维、旧 artifact 拒绝）、新软件基线数字与文档同步。产出 P10+（协议/RTL）赖以起步的软件基线。

不做：不实现新功能；不把 IR13/旧 S0 的延期项自动变成新源级 CFG；不新增 `tile.await_any`/TRY/轮询 AM；不重跑或解释硬件任务；不修改其他任务的实现文件（发现缺陷回对应任务修复或记录阻塞）。

## 2. 前置与输入

- P04–P08 的 evidence 与交接（版本决策、配置拼写表、新拓扑/数值/终态/Conv 语义）。
- P03 vectors 与 `numeric_freeze.md`；P01 的旧基线（历史标签对照）。
- 总则 §5（验证纪律、计数解释、F03 恒等式适用范围）、§6（非目标）；设计 §8.1b Q 项软件证据列。

## 3. 当前证据与修改文件

输入为P03/P06/P08已注册numeric、numeric-profile、soft-conv，以及P04/P05/P07独立pytest/CLI case。P09负责在P02统一分派器注册shared-am-functional、two-slot-progress、shared-am-cycle，复用真实入口，不复制算法/虚构PASS。

本任务是软件集成owner：补上述runner并核对文档；发现前序集成缺陷可修对应唯一实现、重跑受影响gate并更新evidence，不把可修代码缺陷只“回派”后停在半完成。不顺带增加范围外功能。

## 4. 输入输出、接口与不变量

- 输出：新软件基线（每样例 makespan/cycles/AM 统计/bytes hash，统一标注 “per-Tile + 多 engine + bf16_fp32_mac_rne_v1 + 终态偏序 合同基线，2026-xx-xx”）、全 target manifest/summary、缺陷回派清单。
- 不变量：`PASS` 只能由真实断言产生；旧周期数字不得出现在任何新基线表中（对照列必须标 “历史/旧合同”）；所有数值 runs 为 full_memory/ByteStore + 独立 golden；F03 的 `request=grant=ALLOC、final=slot_free、park=正常wake` 恒等式只用于全成功运行，混合故障用独立分母。

## 5. 实施步骤

1. 依赖核验：逐个读取 P04–P08 evidence，确认状态通过、版本链一致（当前 schema/ABI = P08 落地版）、`hardware/configs/` 实例存在；任一未通过即停止并记录阻塞。
2. 注册三个软件协议/周期runner后，真实执行六个target，各自out目录/固定seed、明确F或C；汇总case结果，失败保留最小复现并修唯一实现，不以修改expected bytes消错。
3. 宿主全量 gate：`conda run -n elenor-validator pre-commit run -a`；`conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -q -n 4`；记录当日计数。
4. 跑 `bash examples/run_all_workload.sh --full-memory`，逐case检查completed/report/资源；BOA-only、Conv与已实现数值路径比较完整C。split-K等含非数值EVU combine只比较BOA partial和调度/资源，不把整个组合C当已实现数值输出；记录此明确不变范围。
5. 负/边界 case 集（P09 明确覆盖，均走现有公开入口，不新增源级控制流）：
   - 零 K（±bias）与 m=0/n=0 空工作：合法完成、bytes==参考；
   - P06数值ABI切换前、P08字段升级前artifact分别版本拒绝；新版artifacts以不同有效geometry/arrays/numeric_tile_k测试fingerprint错配；
   - 未知numeric_profile由配置加载明确拒绝，不能要求它先生成可执行target_hash；唯一合法profile的身份仍入fingerprint；
   - quota/bundle 边界：两条未受理 FINAL 后第三链申请被拒且无副作用（T01 运行时面）；FULL→WAIT_ENGINE_QUEUE 不丢 seq（T14）；
   - INT8 非法参数与背压恢复（Q07）；
   - 混合故障计数：注入 fault 的运行用独立分母记账，检查 `park=正常wake+挂起cancel` 闭合，成功恒等式不套用。
6. 合同审计（grep/结构证据，写入 evidence）：
   - 非目标未混入：无源级 CFG/TRY/`await_any`/AM 轮询；无 EVU 数值 combine；无硬件 Conv；无跨 Tile AM；无无限缓冲/默认 FP32 输入路径；
   - 全链完整性：epilogue 字段在 dialect/DTO/lowering/codec/verifier/runtime 六处齐全；`run.sh` 无硬编码 `--sim-override fidelity`（fidelity 由 `fidelity-of`/run_all 注入）；
   - oracle 隔离：生产代码无 `hardware/boa/reference` import。
7. 新基线表：按样例记录 cycles/makespan/AM 计数（requests/parks/wakes/live/reserved/terminal 字段）与 bytes hash；与 P01 旧基线并列时旧列标 “历史（Group 共享 + binary64 + 旧终态）”。
8. 文档同步：核对 `pipeline_validator/README.md`、`IR_SPEC.md`、`Limitation.md`、`design/proposal/BOA_Software_Managed_AM_IR_Implementation_Report.md` 的现状描述与新合同一致；过期陈述更新并保留历史数字的历史标签。
9. 交接包：`plan/evidence/P09.md` 汇总命令/exit status/manifest 路径/基线表/回派清单；更新 `plan/README.md` P09 状态。

## 6. 验收表

| Case ID | 构造 | 必须观测 | 失败特征 |
| --- | --- | --- | --- |
| G-P09-1 统一 target | 步骤 2 全部命令 | 六个 target 全 PASS（真实断言），manifest 齐全 | 空 target 或仅 echo |
| G-P09-2 宿主 gate | 步骤 3 | pre-commit/pytest exit 0，当日计数入 evidence | 用历史计数冒充 |
| G-P09-3 bytes链 | BOA-only/Conv及真实numeric部分 | 全输出或BOA partial逐byte等于golden，EVU timing部分明确不宣称数值 | 整体exit0或hash存在冒充正确性 |
| G-P09-4 零 K/零维 | 步骤 5 集合 | 合法完成且 bytes 正确；无挂死 | 异常拒绝 |
| G-P09-5 旧artifact升级链拒 | P06前、P08前artifact | 旧ABI版本明确拒绝；另测新版错target拒绝 | 静默升级/执行旧语义 |
| G-P09-6 target拒绝 | 有效arrays/geometry/块长错配与未知profile | 前者fingerprint mismatch，后者配置错误，各自路径正确 | 为追求同错误信息放过非法配置 |
| G-P09-7 混合故障计数 | fault 注入运行 | 独立分母与闭合等式成立记录 | 对故障运行硬套成功恒等式 |
| G-P09-8 合同审计 | 步骤 6 | 三组审计均有 grep/清单证据 | 只有口头结论 |
| G-P09-9 基线与文档 | 步骤 7/8 | 新基线表完整；历史列有标签；文档无过期断言 | 新旧数字混写 |

## 7. 可执行命令

除步骤2新增三个真实target外均使用前序入口；所有runner按总则接受--out/--config/--seed/--case并创建目录。前序应交付的入口缺失是缺陷，当前集成owner须修复或给出真实外部阻塞，不以空target替代。

## 8. 产物、交接与停止条件

交付六target全记录、新基线、完整bytes与分层限制、三个协议/周期runner、文档同步和P09 evidence；给P10/P20/P29作为软件权威输入。任何必达case失败都不能通过，修复可达缺陷后重跑，外部缺口按总则明确阻塞。
