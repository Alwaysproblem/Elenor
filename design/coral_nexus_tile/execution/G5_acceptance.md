# CN-G5 验收／性能／复现执行阶段执行指南（C069–C076）

本页把总计划 CN-G5 阶段（W7-05 重放／性能、W0-05 双复现、W7-06 最终出口）拆成 8 个可直接执行的 commit 卡。shell 变量 `H`/`V`/`C1`/`C2`/`P0`/`cn()`/`EVIDENCE` 由总入口 `execution_guide.md` 定义，本页直接复用；`EVIDENCE` 由总入口为每卡设置全新 attempt 目录（卡内以 `: "${EVIDENCE:?…}"` 断言已设置），不覆写旧证据。所有 `cn` 子命令为未来命令，提供 commit 在各卡"验证"节显式列出；本页不声称任何新命令当前已存在。旧分包计划与本卡冲突处，以本卡口径为准（冲突在卡内标注并汇总于文末"执行顺序澄清"节）。

阶段总原则：

- 本阶段依赖 C068 的完整功能证据，实施重放、度量、复现与出口裁决。发现功能缺陷必须回归所属模块做独立修复并重跑受影响闭包；不能以“本阶段不修功能”为由带病验收。
- 性能比较口径：P1/PN 用同一 `cs1_c2.yaml` target、同一 RTL hash，仅活跃 Context 数不同（P1=1，PN=多）；P0 用 `p0_coral.yaml` 原核，是**另一 RTL hash**，仅 full RetirementBuffer 配置对齐（enableVerification 全量），P0 结果单列比较，禁止表述为"P0 同 RTL"。PS 只登记 `not_run` 占位。
- makespan ＝ 登记的测量起点事件到**最终承诺完成**的 cycle 跨度——终点含 Group 输出写回（out-store）与 root done 完成语义，不只最后一个 Task 的 end；起点/终点事件名在 C071 登记并供 C072 提取器消费。取 CS1_LOG 原始整数 cycle，**不是**各任务 duration 之和；两类定义级反例：重叠任务（duration 之和＞span）与带空隙的串行（之和＜span）。W7 旧计划 §11.2 的 "sum(start..end)" 表述已修正（见文末澄清）。
- 重放等价口径：跨模型替换对拍（VER-21，`--rtl-tile-mask 0x0` 纯 Python vs `0xf` RTL）允许"仅偏序、周期可不等"；**同 RTL hash 重放不允许沿用该例外**——同 bundle seed/延迟/背压下逐条 cycle 必须相等（确定性 RTL），否则 FAIL（W7 旧计划 §11.1 的"周期不要求相等"仅指跨模型场景，见文末澄清）。
- 性能阈值先登记后测量：`perf_gates.json` 在测量 commit 之前先 commit（预登记即该 commit 本身）；每行阈值必须带 `threshold_source`（明确来源），禁止无来源数值（如凭空的百分比回归容忍、加速比）与占位 0。
- 证据可取回：`out/` 是 gitignored 运行目录，**本身不构成可取回归档**；可取回性＝仓库内 `evidence_index/`（逐条 sha256＋run_id＋真实可访问归档 URI）＋MAINTENANCE.md 登记的归档位置＋按 URI+sha256 重取并复核 hash 的取回验证。`run_id` 目录互斥创建，覆盖既有 run 一律拒绝非零退出。测量 manifest 只落代码树 `tree_digest`（C007 digest 函数）；git commit receipt 由 `evidence_index` 外部后置登记（不为登记 receipt 强制新增代码提交），**禁止在 commit 内容里自引用本 commit 的 git hash**。
- C076 审计支持矩阵与维护交接；组件文档应随对应实现提交。发现遗漏由所属 owner 在明确范围内修复并复验，不为赶最终门留下文档与实现不一致，也不改写已发布历史。

<a id="c069"></a>

## C069 — 完整窗口单侧已知故障重放与拒绝路径

- **任务**：W7-05（重放半部：单侧重放驱动 + 拒绝负例 + 已知故障复现）
- **前置**：C068
- **建议 commit**：`feat(coral-verify): one-sided window replay driver with known-fault reproduction and integrity rejection`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §11.1、§5（verdict schema／产物路径）、§4（replay 退出码 0/1/2/3）；C016 冻结的 `replay_bundle_contract_v1.yaml`、`checker_verdict_contract_v1.yaml`、`trace_window_contract_v1.yaml`（C020 封印 hash）；C025 的 `libcn_cosim.so` 八函数 C ABI；C059/C062 的故障注入场景窗口（C0 非法指令、双 SET、同拍多故障 first-fault）；C068 的最终 `cs1_manifest.json` 与封窗 bundle。
- **修改边界**：H=`hardware/coral_nexus`。新增 `H/verification/replay/{driver.py,compare.py,reject_cases.py}`；`H/scripts/cn.sh` 注册 `replay` 子命令（本卡提供；C003 入口只 dispatch 已实现 handler）；`cn unit --suite w7` 注册 `test_replay_equivalence.py`、`test_replay_reject.py`（runner 由 C019 提供）。V 不改。
- **实现步骤**：
  1. bundle 装载与 C6 门：读窗口 manifest（`first_seq/last_seq/record_count/gap_status/blob_manifest/seal`），逐 blob sha256 复核；`gap_status≠none`、缺 blob／缺初始快照／缺映像 hash、尾部缺断但前缀连续 → 直接 `NOT_FULLY_REPLAYABLE`，exit 3，报告列出第一个缺口（seq／blob id），不静默降级。
  2. 单侧驱动：按日志 Host 侧记录逐 seq 驱动 bp API（`bp_abi_info→bp_begin→bp_layout→bp_drive→bp_tick(P1–P3)→bp_collect→bp_status→bp_destroy`，不用任何 stub so；Host 被动性的唯一例外就是重放，DUT 侧不驱动）。**响应注入前置**：一条 resp 类日志记录只有在与之配对的 req 已被实际匹配后才允许驱动/比对，禁止先 drive 未匹配请求的响应。逐条比对 protocol/port/dir/身份/`payload_hash`，跨记录偏序一致；**bundle 产生 run 与重放 run 的 RTL hash 相同（同 seed/延迟/背压）时逐条 cycle 相等**，不同则 FAIL（见阶段总原则）。首个 mismatch 报 `seq/cycle/字段` 三元组，不批量模糊。
  3. 已知故障单侧重放：取 C059/C062 注入矩阵的完整窗口（C0 非法指令、双 SET 竞争、同拍多故障）。期望安全故障场景的**场景级 verdict＝PASS 且 DUT run 终态＝ERROR/FAIL**（两者并存分别记录，不混写）；重放判据＝DUT 侧行为记录复现（同 first-fault id、终态种类、复位相位 ACK 序），不要求 checker FAIL、也不要求 DUT 变 SUCCESS。
  4. 拒绝负例套件（`reject_cases.py`，每例独立 bundle，先失败后通过成对留证）：quota 截断、R1 drop（删 task/ctx/终态条目→C1 FAIL）、truncation 前缀连续、缺 blob、缺快照、缺映像 hash——各断言 exit 3 与命中的 checker/字段。**两类篡改分开断言**：(a) 日志内 `payload_hash` 与 payload bytes 不符 → 完整性/hash 校验类拒绝（C1/C6，exit 3）；(b) 重放中 DUT 实际响应与日志记录不等价 → 等价性 FAIL（exit 1，首 mismatch 定位 seq）——不得互相冒充。
- **验证**（`cn replay` 本卡提供；suite runner C019 提供）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"   # 本卡 run 级 manifest.json 落此目录
  # 前置 manifest 路径/摘要/检验类型先写入 "$EVIDENCE/inputs.json"（总入口 §4.3），
  # bundle 路径从 C068 汇总 cs1_manifest.json（aggregate）的 replay_bundles 字段解析，
  # 不按 mtime 猜"最新产物"
  CS1_MANIFEST=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["inputs"]["cs1_manifest"])' "$EVIDENCE/inputs.json")
  BUNDLE=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["replay_bundles"]["known_fault_c0"])' "$CS1_MANIFEST")
  cn replay --bundle "$BUNDLE" --out "$EVIDENCE/replay"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"
  ```

- **通过/交接**：`replay/report.json` 含逐记录等价表（mismatch=0、同 RTL `cycle_diff=0`、无"未匹配先注入响应"违例）、已知故障 DUT 行为复现表（≥3 故障类各一窗口，场景 verdict=PASS 与 DUT=ERROR 并记）、6 类缺口拒绝负例＋两类篡改负例各命中预期 exit/字段；退出码 0 不充分。消费者：C070（同一 run 窗口的 trace 审计）、C073（断网复跑重放）、C074（出口判据 5＝VER-22）。本卡验收闭合；W7-05 整体 DONE 以总入口关闭表为准（C069..C072 全落地＋端到端）。

<a id="c070"></a>

## C070 — Perfetto 单位审计、完整性与活动统计

- **任务**：W7-05（trace 审计半部）
- **前置**：C069
- **建议 commit**：`feat(coral-verify): perfetto unit/integrity audit and activity statistics extraction`
- **输入/先读**：`../components/W7_Verification_Trace.md` §3.1 单位注记（µs 仅展示视图；ns→cycle 仅 1000 MHz 时数值相等；排序/身份/偏序一律取原始 seq/cycle，禁止用展示 ts 排序或对拍）；`../implementation/W7_Verification_Trace_Plan.md` §11.1.3；C068 的 trace 产物与 raw JSON；`examples/artifacts/tools/trace_processor_shell`（现存于仓库，W0-02 容器必须携带）。
- **修改边界**：H。新增 `H/verification/trace_audit/{units.py,integrity.py,activity.py}`；在已有 `cn verify` handler（C007 注册骨架、W7-03 扩展）上新增 trace 审计段，`--trace-audit` 为本卡新增 flag，**parser＋consumer 同卡落地**（不造幽灵参数）；`cn unit --suite w7` 注册 `test_trace_audit_units.py`。V 不改。
- **实现步骤**：
  1. 单位审计：换算关系固定为 `cycle_ns=1000/clock_mhz`（1000 MHz→1 ns/cycle）；Chrome JSON trace 的 `ts` 是 µs、Perfetto SQL `ts` 是 **ns**，cycle＝`ts_ns×clock_mhz/1000`，仅 1000 MHz 时 SQL `ts` 数值与 cycle 相等。对每份 trace 的检查点 slice 用 SQL 取 `ts` 与 `extract_arg(arg_set_id,'args.cycle')` 按上式核对；非整数 ns 周期（如 800 MHz→1.25 ns/cycle）不强迫展示值零差——以**原始整数 cycle 字段或登记容差**对账，排序/身份/偏序一律取原始 seq/cycle。负例：800 MHz trace 若审计器按"ts 数值==cycle"直判 → 必须 FAIL（检出 ns 恒等 cycle 误用）。
  2. 完整性审计：查询全部非零 error/data_loss stats；解析错误、真实丢失或未解释错误拒绝通过。`slice_spill_overlapping_complete_event` 不能仅按 severity 断言丢数据：保留原始告警，逐项证明只是 track 迁移，并对账 raw slice、counter 的每个数值及需要的 flow 链；未证明完整仍失败。不静默丢告警或降低重放条件；trace processor 缺失为 BLOCKED。
  3. 从已审计 trace 与原始 CS1_LOG 提取 PERF-02 活动计数；写本卡派生 audit manifest，引用原 run manifest 的内容摘要。不得修改已归档/封窗的原始 manifest 来补 `activity_stats`，否则会破坏上游 seal。
- **验证**（`cn verify` 由 C007 提供、本卡扩展 `--trace-audit` parser＋consumer；suite runner C019）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  # cs1_manifest（C068 多 run 汇总 aggregate）路径取自本卡 inputs.json（§4.3）：先写清单再跑
  MANIFEST=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["inputs"]["cs1_manifest"])' "$EVIDENCE/inputs.json")
  cn verify --manifest "$MANIFEST" --trace-audit --out "$EVIDENCE/audit"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"
  ```

- **通过/交接**：每份 trace 的单位检查通过；无真实丢失、未解释错误或逐值对账差；已解释告警及其无损证据仍完整保留。800 MHz 的错误单位假设必须被检出。C071 只消费通过审计的派生 manifest，W7-05 整体关闭以总入口为准。

<a id="c071"></a>

## C071 — 性能矩阵、统计边界与门限预登记

- **任务**：W7-05（性能半部之登记：先 commit 再测量）
- **前置**：C070
- **建议 commit**：`feat(coral-verify): pre-registered perf gate matrix with sourced thresholds`
- **输入/先读**：`../implementation/W7_Verification_Trace_Plan.md` §11.2、§4（bench 退出码）；`../components/W7_Verification_Trace.md` PERF-01..07 度量行；C016 的 `perf_gates_contract_v1.yaml`（`pre_registration_sha256` 槽位）；C052 封印 workload 集合；C019 的 `cs1_c2.yaml`/`p0_coral.yaml`；C070 的 `activity_stats`。
- **修改边界**：H。新增 `H/verification/perf/{matrix.py,metrics.py,prereg.py}`、`H/verification/perf/perf_gates.json`（预登记实例，随本 commit 入库）、`H/verification/scenarios/performance.yaml`（bench matrix：显式 workload/binding/seed/延迟/背压，引用 C052 集）；`cn unit --suite w7` 注册 `test_perf_metrics_defs.py`。V 不改。
- **实现步骤**：
  1. 指标定义（`metrics.py`，每项含提取字段名＋统计函数＋原始来源）：总吞吐、makespan＝登记的测量起点事件到最终承诺完成事件的 cycle 跨度（终点含 Group 输出写回（out-store）与 root done，**不只最后一个 Task 的 end**；起点/终点事件名在本卡登记并供 C072 提取器消费）、单任务延迟、尾延迟（分位数定义写死）、P1 vs P0 单 Context 回归（同数值工作量/精度/内存/时钟/编译策略）、PERF-02 活动统计（消费 C070）、PERF-04 gather 有用/实际字节·合并率·outstanding（`gather_perf` workload 原始字节计数）、PERF-05 计时终点含输出写回＋完成语义（终点事件名逐个登记）、PERF-07 调度机会单列。禁项入校验器：validator 周期当 RTL 性能、PPA/Fmax、多 VME 利用率、任何固定加速比宣称。
  2. 比较配置矩阵：P1＝`cs1_c2` 单活跃 Context；PN＝同 `cs1_c2` 同 RTL hash 仅活跃 Context 数不同；P0＝`p0_coral`（另一 RTL hash，full RetirementBuffer 对齐）单列，不进同 RTL 列；PS 行仅 `not_run` 占位（带启动条件，不带阈值）。
  3. 预登记（本 commit 即登记快照，不跑任何测量）：`perf_gates.json` 每行含 `baseline_ref`、`pre_registered_threshold`、`threshold_source`（枚举：`config_budget[W0-04 参数路径]`／`design_budget[设计文档节]`／`registered_calibration[校准 run_id+方法]`／`zero_regression`）、`pre_registration_sha256`。禁止无来源数值与占位 0；W7 旧计划 §11.2 的"≤5% 回归"示例数字不可照抄（见文末澄清）。
  4. 登记校验负例：缺 `threshold_source` 的行被拒；PS 行带阈值被拒；`test_perf_metrics_defs.py` 用两类合成日志拒掉错误 makespan 口径——重叠任务（各 duration 之和＞span）与带空隙串行（之和＜span），并断言终点取 Group out-store/root done 而非最后 Task end。
- **验证**（suite runner C019；本卡不调用 bench——`cn bench` 由 C072 提供）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"   # prereg/matrix/metrics 定义用例
  ```

- **通过/交接**：`perf_gates.json` 全行有来源阈值并随本 commit 入库、其 sha256 可复算；`performance.yaml` 每行 pin 全 seed/延迟/背压；三类登记负例命中；消费者：C072（按登记测量并校验 `prereg_sha256` 与登记 commit 先于任何 bench 产物的 git 序）、C074（T-35 出口核对登记 hash）。

<a id="c072"></a>

## C072 — 同硬件 P1/PN 与原核 P0 度量

- **任务**：W7-05（性能半部之测量与判定）
- **前置**：C071
- **建议 commit**：`feat(coral-verify): p0/p1/pn measurement with causal controls and archived evidence index`
- **输入/先读**：W7 计划 §11.2；C071 已 commit 的 `perf_gates.json`＋`performance.yaml`；C005/C006 的 P0 baseline capture manifest；C040 的 cosim 入口与 C068 的 4-Tile RTL；C069 的窗口复用。
- **修改边界**：H。新增 `H/verification/perf/{runner.py,causal.py}`；`H/scripts/cn.sh` 注册 `bench` 子命令（本卡提供）；`cn unit --suite w7` 注册 `test_perf_metrics_causal.py`；新增 `H/verification/evidence_index/`（追加式 JSON 索引，本卡起逐 run 登记）。V 不改。
- **实现步骤**：
  1. `cn bench --matrix performance.yaml --out ...`：按矩阵执行 P1（`cs1_c2` 单活跃 Context）、PN（同 `cs1_c2`，先断言两 run `rtl_hash` 相同、仅 Context 数不同，否则 exit 3）、P0（`p0_coral`，断言 manifest 标注 full RetirementBuffer 对齐；P0 列单列，禁止同 RTL 表述；**P0 与 P1/PN 跑同数值工作量**——同形状/数据/精度，是可比性前提）。P0 行数值判定只用固定已知输出捕获（C006）或 KD-01 aware oracle profile，**不得拿 raw Spike 溢出行为当 oracle**（Spike F32AddOdd 溢出→±Inf 是已登记 ISA-reference 偏差，KD-01）。每次 run 唯一 `run_id`，输出目录互斥创建（已存在 → exit 3，不覆盖旧证据）；run 级 `manifest.json` 记录 `tree_digest`（C007 digest 函数）、`prereg_sha256`、`prereg_commit_receipt`（C071 的 commit id——事后引用先行 commit 允许，自引用本 commit 禁止）、config/seed/延迟/背压全量。
  2. 指标提取与判定：按 C071 定义从 CS1_LOG/PMU/原始字节计数计算写 `perf_report.json`（每指标含提取字段名＋统计函数＋原始 run 引用）；对照预登记阈值逐行判定，超阈值 → exit 1；本窗口数值/生命周期任一 FAIL → 不生成性能通过结论，输出 `NO_PASS_CONCLUSION(reason)`（bench 不给功能失败的数据背书）。
  3. 因果控制负例（防硬编码常量/文件存在性回归）：断言只在**隔离探针**上做——单 Context 串行独占关键依赖链 workload（每个访存响应门住下一请求，链长 N 由同 run CS1_LOG 计数），以 `--hw-override hbm_fixed_latency_cycles=<2×登记值>` 重跑（独立 run_id），断言 makespan 增量与 N×注入差**精确一致**（关键路径可算）。重叠/隐藏使延迟被吸收属合法现象：**普通 P1/PN run 不做"必须变差"断言**，无加速/无变化不判 FAIL。
  4. 证据归档：每 run 产物写 `out/coral_nexus/<run_id>/perf/`（运行目录本身 gitignored），run 级 `manifest.json` 的 sha256＋run_id＋`tree_digest`＋**真实可访问归档 URI** 追加进仓库内 `evidence_index/`（append-only，重写被拒）；可取回性由 C073 的 URI+sha256 重取验证落实；巨波形不入 repo。
- **验证**（`cn bench` 本卡提供；suite runner C019）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn bench --matrix hardware/coral_nexus/verification/scenarios/performance.yaml --out "$EVIDENCE/perf"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"   # test_perf_metrics_causal 等
  ```

- **通过/交接**：P0/P1/PN 三组 run 产物齐、P1/PN `rtl_hash` 相等断言 PASS、逐阈值判定表全 PASS、隔离探针因果断言精确命中（N×注入差）且普通 P1/PN 不强行变差、同 `run_id` 重跑被拒（exit 3）、`evidence_index` 逐条带归档 URI 可复核；负例：喂 bench 一个 `rtl_hash` 不匹配的 PN run → exit 3。消费者：C073（复现重跑对拍）、C074（出口判据 T-35/T-36/T-67）。本卡验收闭合；W7-05 整体 DONE 以总入口关闭表为准（C069..C072 全落地＋端到端）。

<a id="c073"></a>

## C073 — 清缓存构建、断网运行与归档复现

- **任务**：W0-05（双复现／许可维护终态收口）
- **前置**：C072
- **建议 commit**：`feat(coral-env): clean-cache build, offline execution reproduction and retrievable evidence archive`
- **输入/先读**：`../implementation/W0_Baseline_Environment_Plan.md` §T4（双复现定义、机械步骤、负例 a–d、退出码 22/23）、§4（`env export/verify`）；C002 的 image.lock/seed、C005/C006 的 P0 产物、C019 的 cs1 manifest、C072 的 perf 产物与 `evidence_index`。
- **修改边界**：H（新增 `env/repro/{exec_repro.sh,build_repro.sh}`，扩展 C004 已有 `MAINTENANCE.md` 与 notice；注册 suite w0 的实际复现负例）。V 不改。只使用已实现 env 参数面；需要额外 flag 时，本卡由 W0 同时实现 parser/consumer/帮助。复现使用专用可丢缓存，不删除用户已有 Docker/Bazel 缓存。
- **实现步骤**：
  1. 执行复现：`cn env export` → 空白 daemon 专用 workspace（不共享 docker 缓存）`docker load` → `cn env verify` → 断网容器内重跑 P0（`cn baseline --config "$P0"`）与 `cs1_c2` 构建＋验收子集＋C072 的一条 P1 行；判据是**内容比对**：`sources_manifest/target_manifest` 指纹字段序列一致、ELF/生成产物 byte-identical；有差异必须 diff 出唯一差异字节来源并写入 MAINTENANCE.md，无法解释 → 双复现 FAIL。
  2. 构建复现：`cn env build` 两次（第二次独立 docker context＋`--no-cache`），apt snapshot 解析、锁内 SHA、seed 内容一致，产物 byte-identical。
  3. G5 证据归档：把 C068 汇总 `cs1_manifest.json`、C069 replay 报告、C070 审计、C072 perf、本卡双复现产物登记进**真实可访问归档**（位置/URI 由 MAINTENANCE.md 登记的固定制品根路径给出；`out/` 是 gitignored 运行目录，不构成归档）＋仓库内 `evidence_index/`（逐条 sha256/run_id/`tree_digest`/归档 URI）；**取回验证**：按 URI+sha256 重取每类证据各一份并复核 hash，不只验本地 out/ 路径存在；`run_id` 目录互斥创建。commit receipt 在 `evidence_index` 外部后置登记，不为登记 receipt 强制新增代码提交。
  4. 负例（先失败后通过）：image.lock 篡改任一 SHA → `env verify` exit 22；seed 删一个归档 → 断网重放 exit 23（列缺失 label，不自动联网回落）；devcontainer.json 加 `build`/`privileged`/`mounts.docker.sock` → exit 22；PAR-08 未接线参数标 `consumed: rtl` → manifest bridge 校验 FAIL。
- **验证**（`cn env` 由 C002/C004 提供、`cn baseline` 由 C005 提供、`cn build` 完整 target 由 C040 提供；suite runner C019）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn env export      # C004 提供；产物路径按其 run 级 manifest.json 约定，repro 脚本消费
  bash hardware/coral_nexus/env/repro/exec_repro.sh "$EVIDENCE/repro-exec"   # 内部调 env verify/baseline/build 并断网
  bash hardware/coral_nexus/env/repro/build_repro.sh "$EVIDENCE/repro-build"
  cn unit --suite w0 --config "$C2" --out "$EVIDENCE/unit"
  ```

- **通过/交接**：执行/构建复现 byte-identical 清单、四负例各命中预期码、归档索引逐条带 URI+sha256 且取回验证通过、MAINTENANCE/notice 覆盖 E17/E24 锁内每一项；负例：删 seed 归档后断网重放必须 exit 23 而非联网续跑。消费者：C074（出口判据含复现证据）、C076（MAINTENANCE 交接）。本卡验收闭合；W0-05 整体 DONE 以总入口关闭表为准（最终裁决随 CN-G5 出口）。

<a id="c074"></a>

## C074 — G5 完整 CS1 产品证据门与过程 closure 分离

- **任务**：W7-06（最终门半部：聚合＋出口判据＋closure 表）
- **前置**：C073
- **建议 commit**：`feat(coral-verify): g5 final gate aggregating evidence with cs1 exit verdicts and closure ledger`
- **输入/先读**：W7 计划 §10.4（出口 6 判据）、§12、§5（非空 PASS／证据分级／允许 CS1_PY 的行清单）；总入口 §4.3（inputs.json 显式输入清单）与 §7.1 `gate` 行；C069..C073 全部产物与 `evidence_index`。
- **修改边界**：H。新增 `H/verification/gate/{g5.py,closure.py,write_inputs.py}`；`cn gate --stage G5`（C007 提供 gate 骨架/G0，本卡落 G5 判定）；`cn unit --suite w7` 注册 `test_g5_gate_negatives.py`。V 不改。
- **实现步骤**：
  1. `write_inputs.py` 从显式指定的 `evidence_index`/receipt 集读取路径、内容摘要和检查类型，写入本 gate 的 `--out` 目录下 `inputs.json`；不自动选择“最新成功 run”。gate 重验 hash 并重跑完整 C1–C7 判据；来源层级必须区分，不把 unit/mixed 当四 RTL。
  2. 六项产品出口：T-39/T-40 完整任务批与等价 Tile 语料；真实冷热/安装及 T-61～68 的 N=1 必需子项；所有适用场景资源闭合；VER-21 跨模型合同偏序/账本一致；VER-22 完整窗口故障重放；没有未裁决的 mandatory 项。Python 模型不执行算术的节点不参与数值对拍，RTL 数值由独立 oracle 判；真实字节搬运才比较双方输出 bytes。
  3. 产品与过程分账：全部适用 mandatory 产品证据在 C073 前已齐，缺任何一项不得用 pending 豁免；仅 T-59/T-60/N2 合法延期，PS 为批准的 `not_run`。T-49 可用 Python 基线；BUILD/LOADER 子项按自身类型，P0 性能必须有原核真实 RTL 证据，不得标为纯 Python。过程台账核对其余 41 任务；W7-06 的 C075/C076 SHA 不进入产品 gate，只由后续独立 receipt 关闭流程。
  4. 负例：产品侧任一行 FAIL → exit 1；BLOCKED → exit 2；缺证据/hash 不符 → exit 3；非允许行出现 `CS1_PY` 唯一证据 → C7 FAIL（`not_empty_pass` 拒空 PASS）；**产品 mandatory 缺席不得靠 pending 豁免放行**；closure 伪造负例：把 `pending` 改成 DONE 再跑 gate → closure 校验 FAIL。
- **验证**（`cn gate` 骨架 C007 提供、G5 判定本卡提供）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  python "$H/verification/gate/write_inputs.py" --index "$H/verification/evidence_index" \
    --out "$EVIDENCE/gate"
  cn gate --stage G5 --out "$EVIDENCE/gate"
  ```

- **通过/交接**：gate exit 0 的语义＝**完整 CS1 产品证据门通过**——六判据全 PASS＋全部适用 mandatory 产品证据 PASS（无缺席），不是过程 42/42 终裁；`task_closure.json` 如实记 41 任务 DONE＋W7-06 pending（C075/C076 SHA 分离在产品 gate 输入外）；`t_matrix` 快照 sha256＋`known_deviations`（KD-01）＋`open_items` 空表。负例：从 `inputs.json` 抽走一份 mandatory replay/数值证据 → exit 1/3 而非靠 pending 放行（退出码 0 不充分）。消费者：C075（拒绝路径与从零复现复用同一 gate 验产品）、C076（快照来源，最终外部 receipt 关闭工作流）。本卡验收闭合；W7-06 整体 DONE 以总入口关闭表为准。

<a id="c075"></a>

## C075 — 出口拒绝路径与从零复现实验

- **任务**：W7-06（最终门半部：拒绝边界＋从零复现）
- **前置**：C074
- **建议 commit**：`test(coral-verify): g5 gate rejection matrix and from-scratch reproduction experiment`
- **输入/先读**：W7 计划 §12、§5.4 禁项；C074 的 `g5.py`；C001/C002 的 sources/env 流程；C073 的归档与 seed 导出。
- **修改边界**：H。新增 `H/verification/gate/reject_matrix.py`、`H/env/repro/from_scratch.sh`；`cn unit --suite w7` 注册 `test_g5_rejection_paths.py`（runner C019）。V 不改。
- **实现步骤**：
  1. 拒绝矩阵：构造 ≥8 个破坏证据集，每出口判据至少一个——FAIL 行、BLOCKED 行、缺证据、hash 篡改、`CS1_PY` 冒充 `CS1_LOG`、quota 截断窗口宣称可完整重放、空 PASS 行（`NOT_FULLY_REPLAYABLE`）、deadline 到点当通过；逐个喂 `cn gate --stage G5`，断言 exit 码与 `reason` 字段精确命中，无一被 fallback 放行。
  2. 在新的专用 workspace 中从归档来源、锁和种子离线重建，实际重跑登记的验收子集。其余 mandatory 证据从 C073 归档逐项取回、重验完整内容；在本 gate 输出目录生成 `inputs.json`，再运行同一完整 G5 门。清楚区分“本次重跑”与“归档复核”，不把验收子集冒充全部重新执行。
  3. 负例：从零环境注入被篡改 seed → 断网 `env verify`/重放 exit 22/23 且无联网回落；从零 run 与归档 run 的 gate verdict 不一致 → FAIL。
- **验证**（`cn gate` 由 C074 提供 G5 判定；suite runner C019）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"        # 拒绝矩阵用例
  bash hardware/coral_nexus/env/repro/from_scratch.sh "$EVIDENCE/from-scratch"
  ```

- **通过/交接**：每个破坏证据集被预期原因拒绝；完整正例从新 workspace 重现相同产品判决。对 gate 做“放过缺失 mandatory 证据”的缺陷注入必须被回归检出，不能用“删测试断言后自测失败”这种源码结构检查代替行为证据。

<a id="c076"></a>

## C076 — 支持矩阵、维护手册和任务关闭交接

- **任务**：W7-06（交接半部）
- **前置**：C075
- **建议 commit**：`docs(coral-verify): cs1 support matrix, maintenance handover and closure ledger`
- **输入/先读**：W7 计划 §12 维护证据清单、§9 延期行；C073 的 MAINTENANCE.md、C074 的 `cs1_final.json`/`task_closure.json`、C075 的拒绝矩阵与从零流程。
- **修改边界**：H（`verification/maintenance/{support_matrix.md,closure_ledger.json,known_deviations.md,open_items.json}`，更新 `MAINTENANCE.md` 的升级/归档/回归入口）。最终 `closure_receipt.json` 作为提交后的独立归档产物，不写入包含其自身 SHA 的提交。实现文档已有缺漏时在所属范围修正并记清修复，不重写历史。
- **实现步骤**：
  1. 支持矩阵：CS1 支持面（功能、`cs1_c1`/`cs1_c2` 配置、运行模式、数值 profile、KD-01 偏差口径）与不支持/延期面（T-59/T-60 CS2/CS3、N2 多实例、PS、PPA/Fmax、多 VME 利用率），每行带 `defer_reason`＋启动条件（消费 `t_matrix.yaml`）。
  2. 关闭账本与最终外部 receipt：42 任务×76 commit×verdict×evidence sha256 聚合成 `closure_ledger.json`（从 `evidence_index` 聚合，不重算功能）；oracle/checker 版本＋hash；KD-01 登记；`not_empty_pass` 报告；允许 `CS1_PY` 行清单快照。**最终 42/42 裁决（外部 receipt）**：在文档同步审计与 `gate-recheck` 全过后，以 C074 的产品门结论＋过程 closure 快照＋C075/C076 自身 verdict 生成独立 `closure_receipt.json`（42/42 DONE；任一任务不满足则如实列 NOT_DONE 项与原因，**不提前伪造 DONE**）；`open_items.json` 应为空表（非空则逐条列 id 与归属）。
  3. 按卡片逐项核对组件文档、实际 CLI 帮助、支持矩阵与现实现一致；未实现不能写 supported。发现缺漏则由所属 owner 修复、重新验证相关闭包，再关闭本卡；不能以文档文件存在或固定字句通过。
  4. 负例使用真实 capability/target 与支持声明冲突，审计必须指出错误宣称；不把 Markdown 措辞、行数或文件存在性作永久回归。
- **验证**（`cn gate` 由 C074 提供；suite runner C019）：

  ```bash
  : "${EVIDENCE:?按总入口设置本卡新 attempt}"
  python "$H/verification/gate/write_inputs.py" --index "$H/verification/evidence_index" \
    --out "$EVIDENCE/gate-recheck"
  cn gate --stage G5 --out "$EVIDENCE/gate-recheck"   # 交接前最终重演
  cn unit --suite w7 --config "$C2" --out "$EVIDENCE/unit"
  ```

- **通过/交接**：支持说明与真实能力一致，完整产品 gate 通过，独立 receipt 核验 42/42 任务已关闭，才可称 CN-G5 完成。有 NOT_DONE 时如实报告具体阻断，不宣称交付完成。维护入口为 `MAINTENANCE.md`。

## 执行顺序澄清（相对旧计划的已定口径，非待裁项）

1. makespan＝登记测量起点到最终承诺完成（含 Group 输出写回/root done，不只最后 Task end）；两类定义级反例＝重叠任务（duration 之和＞span）与带空隙串行（之和＜span）。W7 计划 §11.2 的 "sum(start..end)" 表述错误，C071/C072 按本口径实现。
2. 重放等价：跨模型替换对拍（VER-21）允许仅偏序、周期可不等；同 RTL hash＋同 bundle seed/延迟/背压的单侧重放逐条 cycle 相等。已知故障场景级 verdict＝PASS 且 DUT run 终态＝ERROR/FAIL 并记；日志内 `payload_hash` 与 bytes 不符属完整性类拒绝（exit 3），与重放请求等价性 mismatch（exit 1）分开；响应只能在其配对请求实际匹配后注入。W7 计划 §11.1 的"周期不要求相等"仅指跨模型场景。
3. 性能阈值先 commit 预登记（`threshold_source` 枚举：config_budget/design_budget/registered_calibration/zero_regression）；W7 计划 §11.2 的"≤5% 回归"示例数字不可照抄，禁止无来源数值与占位 0。
4. 单位按 `cycle_ns=1000/clock_mhz`；Chrome JSON 用 µs，Perfetto SQL 用 ns。原始整数 cycle 是精确比较依据；任何非零错误统计都须解释并逐值对账，明确的无损 track 迁移不能误判为数据丢失，未知错误不能静默通过。
5. 因果断言只在单 Context 串行独占依赖链探针上做（delta＝N×注入差精确断言）；重叠/隐藏可吸收延迟，普通 P1/PN 不做"必须变差"断言，无加速不判 FAIL。
6. P0 与 P1/PN 同数值工作量、另 RTL hash、full RetirementBuffer 对齐、结果单列；P0 行数值判定不用 raw Spike 溢出行为当 oracle（KD-01 已登记偏差）。
7. 归档可取回＝仓库内 `evidence_index`（sha256/run_id/归档 URI）＋MAINTENANCE.md 登记的真实可访问归档位置＋按 URI+sha256 重取复核；`out/` 是 gitignored 运行目录不构成归档。run 级 manifest 只落 `tree_digest`，git commit receipt 外部后置登记（不为 receipt 强制新增代码提交），禁止 commit 内容自引用其 git hash。
8. C074＝完整 CS1 产品证据门（六判据＋全部适用 mandatory 产品证据，均产自 C073 及之前；任一缺 FAIL/BLOCKED，无"未到达"豁免）；过程 closure 账本分离——41 任务 DONE、W7-06 的 C075/C076 SHA 不在产品 gate 输入内、如实 pending；最终 42/42 由 C076 外部 `closure_receipt.json` 关闭。gate 输入按总入口 §4.3 `inputs.json` 显式清单（run 级 `manifest.json` → 多 run 汇总 `cs1_manifest.json` → gate），不按 mtime 猜最新。卡内示例命令均以 `: "${EVIDENCE:?…}"` 取总入口设置的新 attempt，不覆写旧证据。
9. W0 计划 §T4 的 `cn_g0_evidence` 段随 C007 服务 CN-G0 门；C073 只收口终态双复现/许可/归档，不重复实现。原 W 任务（W7-05/W0-05/W7-06）整体 DONE 一律以总入口关闭表为准，各卡"验收闭合"不等于原任务 DONE。
