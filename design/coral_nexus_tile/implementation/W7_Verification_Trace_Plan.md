# W7 验证／Trace／重放／性能 实施计划（W7-01..W7-06）

> **逐 commit 执行入口**：[execution_guide.md](../execution_guide.md)。先用该入口选择前置已满足的卡片；本页保留工作包细节，commit 顺序、工具可用时点、文件边界和原任务关闭条件以执行 guide 为准。架构语义仍遵守已批准设计，不按旧草稿中的占位命令或文字存在性检查宣称完成。

> 本文件是文件级实施计划，只写计划：全部目标（`hardware/coral_nexus/…`、`pipeline_validator/cosim/…`）均为**待创建**，截至 2026-11-05 仓库根没有 `hardware/` 目录（已核实），不得声称任何新路径、新命令已执行。共享合同见 `../plan.md`（parent 总计划 §2/§3，已裁决方案在此引用），组件设计见 `../components/W7_Verification_Trace.md`，需求来源为根目录 `Coral_NPU_to_Nexus_Tile_Requirements_v0.1.md`（r5）。未实测参数不伪称冻结。

## 0. 阅读顺序（给实施 agent）

1. 本文件 §1（已选方案/owner/非目标）→ §2（依赖表，确认无环）→ §3（未来命令/I-O）。
2. 对应任务小节（§4.1..§4.6）：输入→产物→步骤→验证→退出条件。
3. 写对应 T 行前先查 §6 表（68 行 traceability：owner/verify 任务/CS1 mandatory/证据类型/非空判据）。
4. 被消费的上游合同一律按 JF（联合冻结）记录的 hash 引用；hash 与设计正文冲突时升级 parent，不自行改。

## 1. 已选方案摘要（对齐 `../plan.md` §2、组件 W7 设计 §3）

1. **trace/检查/oracle 全部为 Python Host 侧资产**（CS1：Group/Device/L2/页授权/HBM 保持 Python，Tile 为 RTL）。W7 不写 RTL 执行路径，不做第二个执行 owner；C1 零泄漏等账本类检查消费 W1 冻结的 read-only ledger/snapshot API（derived only from RESOURCE_COMMIT_IND/RETIRE_ACK/TASK_COMPLETION 等真实消息），**禁止读未实例化的 Python Tile L1 或复制 bool/mock-echo 当 PASS**；CS1 final 的账本字段以**原始 endpoint 消息事实 + L2 收支交叉验证**为准。
2. **日志**：Host 被动捕获（VER-12）。窗口目录 = `cn_trace/v1` manifest + 段文件（JSON Lines：身份/时序整数、握手、payload 长度）+ 大 payload 的内容寻址 blob（SHA-256）+ 初始快照 blob + 封窗记录。逻辑 schema 按 W7 设计 §3.1；**二进制线编码不冻结**（W1-01 权威），文件容器格式为 Host 侧格式、可在计划冻结。配额（file quota）超限 → 停写 + 显式 gap。
3. **可重放性**：`"可完整重放"≠"任务成功"`；seq 连续 ∧ blob 可取 ∧ hash 匹配 ∧ 初始快照/映像在列 ∧ 封窗记录校验通过，才判可完整重放；尾部缺失、quota 截断、无封窗、缺快照、缺映像 hash → 显式 `NOT_FULLY_REPLAYABLE`（VER-22）。**单侧已知故障重放**（W7-05，CS1 出口(5)）；正常模式无日志不构成失败。
4. **校验链**：C1→C2(偏序)→C3(守恒/终态/账本)→C4(独立 oracle)→C5(终止)→C6(重放门)→C7(manifest 门)。checker 只消费窗口和冻结合同，**不比较全局 interleave、不要求周期一致**（VER-04/VER-20）；不使用未记录时间戳（不读 `ts`，避免 `cycle_to_us` 量化，见 W7-05）。
5. **oracle 分两层**（W7-03）：(a) **独立 ISA/数值 oracle**＝固定 pin `riscv-isa-sim@93a10ae6…`（经 W0-01 下载+校验+9 个 Coral 补丁，`spike_simulator`/`spike_cosim_dpi` API），可执行的指令级参考（标量/RVV FP32/VME tile），**不得假称其支持 Nexus 私有指令**；(b) **bitwise 数值 oracle**＝W7 写的 Python 整数位运算模型，按 W3-01 numeric profile 合同逐指令实现（BF16 乘-RNE/加-ROD/溢出不对称/OF·NV flags/掩码）。上游能力不足处是**精确生成门 + 显式适配（helper/CSR/region 声明扩展），不是模糊 TODO**：未覆盖类 → `ORACLE_NOT_INDEPENDENT`，禁止 ULP 放宽或换 FP16。
6. **CSR/region helper 需 oracle 扩展**：固定 Spike 只认识 Coral CSR（`spike_simulator.cc` 内嵌 `csr_map`/`ConfigureCoralNpuCsrs`：`mtype@0xC23` 在案），**Nexus 私有 declaration/region helper CSR、FIRST/CONTINUE/FINAL 绑定、START/END 状态机在固定 Spike 中不存在**；必须由 W7 在 harness 扩展（自定义 CSR/自定义指令 handler，复用 Spike `register_custom_insn`/CSR 注册机制），以 **W1-01 的 `csr_name_map.yaml` + W3-03/W4-04/W5-01 的 helper 语义**为唯一权威；适配语义冻结前 oracle 行为标 `EXPERIMENTAL`，不得作为 C4 依据。
7. **性能**：P0/P1/PN 用**同一负载、同一 RTL、同一 cs1_c2 config**（per parent 裁定：P0＝原 Coral 原核；P1/PN 均为同一改造硬件/同一 cs1_c2 target，仅活跃 Context 数 1 vs 2；**`cs1_c1.yaml` 只作退化配置功能回归，不得替代 P1 与 PN 对比**）。数值门槛**实验前登记**（PERF-06），不做事后挑选；PS 条件延期、不作本轮证据。
8. **出口清单**（W7-06）：68 行 T-matrix 机器可查（C7 判 status，非空 PASS）；单侧已知故障 replay 演示 + 拒绝负例（quota/drop/truncation）；Perfetto SQL 单位审计；P0/P1/PN 报告（PS 缺席）；全部在**固定容器+固定 RTL bit-accurate 仿真**中跑，不以 Python validator 调度模型（如 `l2_admission_wait` 的模型级 smoke）冒充 realbits/RTL 证据。

## 2. 本包 owner 表（唯一写者）

| 资产                                                                                             | owner                      | 边界                                             |
| ------------------------------------------------------------------------------------------------ | -------------------------- | ------------------------------------------------ |
| trace manifest/窗口/封窗/replay-bundle schema、`t_matrix`（68 行）、C1–C7 schema、verdict schema | W7-01                      | 引用 W1-01 字段/编码；不自造协议位编码           |
| 独立 ISA oracle 驱动器 + Python bitwise 数值 oracle、CSR/region helper 扩展                      | W7-02                      | 数值语义权威是 W3-01 profile；W7 只独立实现/比对 |
| C1..C7 checker 实现 + 负例注入（quota/drop/truncation/缺 blob/hash 不符）                        | W7-03                      | 判据按 r5 §18.3/18.4，不新造语义                 |
| CS1 场景执行/判定/manifest 生成（T-01..T-68）                                                    | W7-04                      | 逐 T 与 owner 包核对；不改他包产物               |
| 单侧已知故障重放、quota/drop/truncation 拒绝、Perfetto SQL 审计、P0/P1/PN 性能门                 | W7-05                      | 性能门槛登记在 `perf_gates.json`（先冻结后测量） |
| 最终出口 gate、维护证据、`cs1_manifest.json` 聚合                                                | W7-06                      | 消费 W7-04/W7-05 产物 hash，可重算、可复现       |
| 其他包场景、注入计划、binding/资源 manifest                                                      | W2/W3/W4/W5/W6 的 -01 产物 | W7 只校验/消费（见 §3.2）                        |

## 3. 任务依赖表（硬前置；本包内部无环）

| 任务                                                                                 | 硬前置（必须 DONE/冻结）                                                                                                                                       | 并行窗口                            | 备注（禁环）                                                                                                                                                  |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| W7-01                                                                                | 无任务级硬前置（可立即开工；**不等待任何其他 -01 DONE**；读取 W1-01/W3-01/W5-01 的**草稿/fragment 以 hash 引用，非 DONE 依赖**；实施期再以 JF 冻结 hash 输入） | 与所有 -01 并行                     | 输出 `contracts_lock`，供 JF（joint-freeze gate）校验；上游字段缺失 → 写 `field_requests/w7.yaml`，不阻塞他人                                                 |
| W7-02a spike 驱动器+能力探针                                                         | W0-01（submodule/下载 pin `riscv-isa-sim@93a10ae6…`）+ W0-02（容器/工具链）                                                                                    | 与 W1-02 并行                       | 只用 Coral 侧 SpikeSimulator API；**不声称支持 Nexus 私有指令/CSR**                                                                                           |
| W7-02b bitwise 数值 oracle                                                           | W7-02a + **JF**（W3-01 numeric profile 合同 hash + `cn_protocol_v1.yaml`/`csr_name_map.yaml` hash）                                                            | 与 W2-02/W3-02 并行                 | 消费 frozen hash，不依赖 W3-04 RTL                                                                                                                            |
| W7-02c ISA oracle 扩展（helper/CSR/region 声明注册+序列检查）                        | W7-02b + helper/region 语义合同片段冻结（W3-03/W4-04/W5-01 fragment + JF）                                                                                     | 与 W3-03 并行                       | 无法实现的类 → 显式 `ORACLE_NOT_INDEPENDENT`，不模糊 TODO                                                                                                     |
| W7-02d 独立核对/生成门（覆盖证明）                                                   | W7-02b/02c + W0-04（target/config/矩阵）                                                                                                                       | 与 W5-03 并行                       | 产出 `oracle_capability.json`/`coverage_matrix.json`；证据不足 → 不通过此门的行写 `BLOCKED(oracle 证据不足)`，不伪造 PASS                                     |
| W7-02e oracle 自测/单测                                                              | W7-02c                                                                                                                                                         | 随时                                | **不作为 G0/W0-03 前置**（W0-03 P0 用上游已支持的简单 ELF+上游 reference/固定已知输出，见 parent 裁定）                                                       |
| W7-03a checker 骨架/verdict schema（C1,C6,C7;无数据依赖部分）                        | W7-01                                                                                                                                                          | 与 W2-02 并行                       | 只消费 manifest/trace                                                                                                                                         |
| W7-03b checker 完整实现+负例套件                                                     | W7-03a + **第一批真实 CS1 窗口**（来自 W1-05 oneTile/`--rtl-tile-mask 0x1` smoke，含 `csir-run` 产物）                                                         | 与 W1-04/W2/W3/W4/W5/W6 的 -05 并行 | 禁环：**W2-05/W5-05/W6-05 不把 W7-03b 当硬前置**；他们以 W1-03/04 oneTile loop 自验；W7-03b 才在 oneTile 真窗口上跑 mutation（不得用 stub 或伪 log 冒充 CS1） |
| W7-04a 场景 harness/manifest 流（T 框架落地）                                        | W7-01/W7-02a/W7-03a                                                                                                                                            | 与 W1-05 并行                       | 不产最终判定，仅 harness+manifest                                                                                                                             |
| W7-04b oneTile 场景批（T-01..T-68 中 Tile 内可定标的行，见 §6 的 batchA..batchD 列） | W7-04a + W2-05/W3-05/W4-05/W5-05/W6-05 各自的 oneTile 就绪                                                                                                     | 与 W7-03b 并行                      | `--rtl-tile-mask 0x1` 真 loop                                                                                                                                 |
| W7-04c 最终 4 RTL Tile 场景批（全 mandatory 行）                                     | W1-05 final + W7-04b 全部通过 + W6-05 注入矩阵                                                                                                                 | 收尾                                | 最终 `cs1_manifest.json` 只认本批产物                                                                                                                         |
| W7-05a 重放驱动+负例拒绝                                                             | W7-01 + W1-03（C ABI tick 复用重放接口）+ W7-03b                                                                                                               | 与 W2-04 并行                       |                                                                                                                                                               |
| W7-05b 性能门                                                                        | W7-04b/04c + W0-03（P0 原核基线）+ W0-04（cs1_c2/c1 配置）+ W5 已封印 workload 集合                                                                            | 收尾                                | P0/P1/PN 同 cs1_c2 target+同 RTL hash                                                                                                                         |
| W7-06 出口 gate                                                                      | W7-04c + W7-05a/b + W0-05（环境复现/许可证据）                                                                                                                 | 最后                                | **唯一最终 gate**（CN-G5 概念由 W0-02 供给工具链，判定在 W7）                                                                                                 |

**禁环约束（parent 规则复述）**：

- W1-05(final) 不作为 W2-05/W5-05/W6-05 的前置；W7-04b 才消费 oneTile，W7-04c 消费 final：图中无环。
- oracle 冻结先 `experimental` 构建后测试再标 `supported`（W7-02e 先行），**不以 supported numeric_profile 作为首轮测试封印的前置**（W5-03 的算法合同在 JF 后即可读，`numeric_profile_id` 首版标 `experimental`）。
- 本表不把 W2-05/W3-05/W4-05/W5-05/W6-05 列为 W7-04c 硬前置以外的依赖：W7-04c 只依赖 W1-05 final（他们已把 RTL 就绪接口交给 W1-05 验收）+ 各包 manifest hash。
- 各包场景执行引用 W1-05 harness（`sim/cn.sh`）与 W1-02 TilePort 协议（`cosim/tile_port.py`）以 hash 引用，不复制内容。

## 3.2 被消费的跨包合同（JF 输入清单，W7 侧登记 hash）

| 上游产物                                                                       | 生产者 | W7 消费点                                                          |
| ------------------------------------------------------------------------------ | ------ | ------------------------------------------------------------------ |
| `cn_protocol_v1.yaml`（消息目录/身份/相位/幂等键）                             | W1-01  | C1 (protocol/枚举合法性)、C2 (身份匹配)、C6 (重放身份)             |
| `channel_budgets.yaml`                                                         | W1-01  | C3 (credit 守恒、账本余额)                                         |
| `lifetime_bounds.yaml`                                                         | W1-01  | C1 (seq 断档→窗口不完整)                                           |
| `csr_name_map.yaml`（新声明/region helper CSR 名→位地址）                      | W1-01  | W7-02c 的 Spike/独立 oracle CSR 扩展点                             |
| `w6_event_facility.contract.yaml`、`w6_terminal_reset.contract.yaml`           | W6-01  | C2 event/phase/取消/复位偏序、C5 终态唯一、T-51/52                 |
| `w3_region.contract.yaml`（START/END/FIRST/CONTINUE/FINAL 序列状态机合法集）   | W3-01  | W7-02c（region helper 语法 oracle）、T-62/T-68                     |
| `w5_binding.contract.yaml`（CoralTilePackage/binding/资源 manifest/hash 字段） | W5-01  | C7 manifest、W7-04 包装/冷热/idempotency 注入、T-30/31/33/53/61/63 |
| `w4_pagegrant.contract.yaml`（scope 页授权/epoch）                             | W4-01  | T-65/50 的 oracle（页授权检查）点                                  |
| `w0_config.contract.yaml`（cs1_c1/cs1_c2/csir-run 参数/延迟/seed/bp 配置）     | W0-04  | W7-04 manifest 填充、`verify`/`replay`/`bench` I-O                 |

> 以上产物在 JF（`PROTOCOL_FREEZE.json` 联合冻结记录）登记 hash 后才允许 W7 实施期读取；JF 之前 W7-01 草案阶段以 hash 占位（hash 待 JF 补），**不空手实施**。

## 4. 未来命令与本包 I-O（对照 parent §3；未创建不得声称已执行）

| 命令【未来，待 W0-02/W1-05 创建后 dispatch 到 W7】                                                                               | owner 任务                           | 输入                                                                                                 | 产物                                               | 语义/退出码                                                                               |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `bash hardware/coral_nexus/scripts/cn.sh unit --suite w7 --config <path> --out <path>`                                           | W7-01/02/03                          | `config`(cs1*c1/c2 yaml)、`w6*\*`/`w3_region`/`w5_binding`/`w4_pagegrant` frozen 产物、`oracle` 产物 | `out/coral_nexus/<run_id>/w7_unit/*.json`          | 逐 checker 单测；退出 0＝全 PASS，非 0＝失败明细                                          |
| `bash hardware/coral_nexus/scripts/cn.sh cosim --scenario <path> --backend rtl --rtl-tile-mask 0xf --config <path> --out <path>` | W7-04（harness 由 W1-05 提供执行侧） | `scenario yaml`（T 行、config、binding、注入计划 hash、oracle 引用、seed/延迟/背压）                 | 同上 `cosim/**/`：窗口目录、manifest、verdict      | 0＝窗口 sealed+all C1..C7 PASS；1＝FAIL；2＝BLOCKED/证据不足                              |
| `bash hardware/coral_nexus/scripts/cn.sh verify --manifest <path>`                                                               | W7-03/05/06                          | acceptance manifest（含逐 T verdict/evidence hash）                                                  | `.../w7_verify/verify_report.json`                 | C7 重放校验：re-hash+重跑 checker；0＝PASS，1＝FAIL，2＝BLOCKED，3＝证据 hash 不符/格式错 |
| `bash hardware/coral_nexus/scripts/cn.sh replay --bundle <path> --out <path>`                                                    | W7-05                                | 封窗 replay bundle（含 manifest/封窗/原始消息/blobs/injection 计划）                                 | `.../replay/report.json`                           | 单侧重放；0＝等价，1＝不等价，2＝BLOCKED，3＝不可重放（应拒）                             |
| `bash hardware/coral_nexus/scripts/cn.sh bench --matrix <path> --out <path>`                                                     | W7-05                                | perf matrix（P0/P1/PN workload 列表+参数+`perf_gates.json` hash）                                    | `.../perf/{cycles,pmu.json,perf_gates_check.json}` | 0＝全登记门槛满足，1＝任一门槛 FAIL，2／3 同上                                            |
| `bash hardware/coral_nexus/scripts/cn.sh gate --stage G5 --out <path>`                                                           | W7-06                                | 全部 `cs1_manifest.json`+`verify`+`replay`+`bench` 产物                                              | `.../cs1_final.json`                               | 0＝CS1 出口成立；1＝任一行 FAIL，2＝BLOCKED，3＝证据缺失                                  |

**当前可执行命令**（不是本轮产物，只作回归基线，在 plan 中引用不改）：

- `python -m pytest pipeline_validator/tests/ -q`（现有 Python 验证套件）
- `python -m pipeline_validator --workload pow --context-mode 2 --input-binding Y=0x100000:524288:rw --sim-override fidelity=full_memory --memory-trace --trace-json <p> --json --report <p>`（Design_Validation §2.1；T-49 的 Python 侧 cold/warm 对照）

## 5. 跨任务共享约定（W7 内部统一）

1. **产物路径**：逐 run 产物在 `out/coral_nexus/<run_id>/`（不进 repo）；`run_id`＝`csir-run-<utc>-<hash8(config + all 68 scenarios)>`，同一批次内重跑同一 config+同一 scenario 集**必须复用同一 run_id**，避免证据漂移。
2. **verdict schema（W7-01 冻结）**：`{schema:"cn_verdict/1", run_id, scenario_id, t_id, t_verdict∈{PASS,FAIL,BLOCKED,DEFERRED_NOT_APPLICABLE}, phase, checker:[{name:"C1..C7", verdict, notes}], evidence:[{kind, path, sha256, t_id}], t_scope:"mandatory"|"deferred"|"cs1_subset", notes}`。
3. **逐 T“非空 PASS”规则**：每个 T 行判定文件必须含 `kind=CS1_LOG` 的窗口 verdict（C1..C7 全 PASS）、`kind=REPLAY` 或 `kind=PERF`（T-35/67/36/39/40 类）、`kind=BUILD`（T-37 类）、**且 `t_verdict=PASS` 的行必须能被 `verify` 重放校验**；否则 C7 拒绝该行（写 `NOT_FULLY_REPLAYABLE`/`INSUFFICIENT_EVIDENCE`），**不空 PASS**。允许 `evidence.kind=CS1_PY`（Python 侧）作为唯一证据的仅限 T-49、T-35 的 P0 比较行、T-37、T-33 的构建期部分、T-53 的 loader 级部分。
4. **禁止事项**（复述 parent/design）：
   - **永不**把模拟器周期数当 RTL 周期（P0 除外单列）；**永不**用 Python `trace.py` 的 `cycle_to_us()` 换算做判定（见 W7-03 C1 单位审计）。
   - **永不**用 tolerance/U 改判 C4 数值比较；bitwise 合同不允许 ULP 放宽。
   - **永不**用 stub/mixed run 当最终判定（W7-04c 用 `--rtl-tile-mask 0xf`+`cs1_c2`）。
   - **永不**把上游 Spike/MPACT 的结果当 DUT PASS 证据；upstream mismatch 记 known_deviation，DUT 判定只看独立 oracle（KD-01）。
   - **永不**把 quota 截断/log 缺 blob/封窗缺失/缺快照的窗口当完整重放证据；`verify`/`replay`/`bench` 拒绝 + 退出码 3。
   - **永不**让 W7 checker 代写功能状态（只读）；**永不**写第二执行 owner。
   - **永不**用永久源码/非空检查当回归：性能/计数门必须绑定真实因果（见 W7-05）。
5. **T 行执行时的“检查门”前置**：W7-03a checker 未冻结前，任何 W7-04 场景判定都写入 `t_scope=harness_only`（不作为 CS1 证据）；W7-03b 冻结后，W7-04b/04c 产物才可标 `t_scope=mandatory`。
6. **逐 T 证据分级**：`CS1_LOG`（必）＞`CS1_PY`（仅限 §5.3 允许行）＞`BUILD`；判定顺序按此优先级，不允许 `CS1_PY` 覆盖 `CS1_LOG` 缺席。若某行确需只跑 Python 侧，须在 `t_matrix` 的 `cs1_scope` 列显式标 `validator_layer` 并在 notes 记录为何不需要 RTL（当前仅 T-49/T-37/T-33/T-53 部分、T-35 P0 部分、T-36 P0 对照）。

## 6. W7-01 日志／manifest／T 矩阵／C1–C7 schema（P0；硬前置：无任务级）

### 6.1 输入

- 组件设计 `../components/W7_Verification_Trace.md` §2/§3/§8.1（trace 逻辑 schema 逻辑字段）。
- r5 §16.3-16.4、§17-§19、§22（VER-12/13/14/19/20/21/22、OBS-01..18、PERF-01..07）。
- `pipeline_validator/trace.py`、`pipeline_validator/report.py:29-49`、`pipeline_validator/pmu.py:15-29`（引用为**现状证据**：`cycle_to_us` 位于 `trace.py:213-216`，`discard_profile_initialization` 位于 `trace.py:197-209`，`cycle_ns` 见 `config.py:366-368`，`l2_extent_release` 在 `trace.py:1050-1066`，`task_lease_acquire/release` 在 `tile_group.py:1281`/`1088,1711`，`group_task_done` 在 `tile_group.py:1815-1816` 与 `execution_ir.py:461`，`transfer_cancelled` 在 `trace.py:1282-1286`，`arena_reserve/retire`、`buffer_view_invalidate`、`ctx_switch`、`uce_launch_attempt/accepted`、`tile_signal` 均为现仓库事件名）。
- W1-01 草案的 `cn_protocol_v1.yaml` 字段命名/枚举（草稿只按字段名引用，JF 后以 hash 重读）。
- W5-01 草案 `binding/资源 manifest/hash` 字段。

### 6.2 产物（全部新文件；均待创建）

| 文件                                                                                                                      | 内容                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hardware/coral_nexus/verification/matrix/t_matrix.yaml`                                                                  | T-01..T-68 机器可读：`t_id`、`title`、`owner_packages`(组件设计 §8.1)、`impl_tasks`(各包计划 ID)、`verify_tasks`(批次 A..F)、`cs1_scope`(mandatory/c1_only/cs2_only/deferred/validator_layer)、`scenario_path`、`oracle`(kind/independent/entry)、`invariants`、`defer_reason`、`perf_gates`、`notes` |
| `hardware/coral_nexus/verification/contracts/trace_window_contract_v1.yaml`                                               | 窗口/封窗/manifest 逻辑字段+幂等键+枚举（schema_ver=1；二进制线编码**不在此**，由 W1-01 冻结）                                                                                                                                                                                                        |
| `hardware/coral_nexus/verification/contracts/log_record_contract_v1.yaml`（+ `derive.py`、`validator.py` 的 Python 实现） | Host 被动记录行结构（`cn_trace/v1` schema 的 Host 侧容器，不含线编码）                                                                                                                                                                                                                                |
| `hardware/coral_nexus/verification/contracts/checker_verdict_contract_v1.yaml`                                            | C1..C7 checker 的 verdict 结构+`not_empty_pass` 断言集                                                                                                                                                                                                                                                |
| `hardware/coral_nexus/verification/contracts/replay_bundle_contract_v1.yaml`                                              | 可重放 bundle：窗口+初始快照+blob 清单+oracle 引用+T 引用                                                                                                                                                                                                                                             |
| `hardware/coral_nexus/verification/contracts/perf_gates_contract_v1.yaml`                                                 | PERF 矩阵 schema（P0/P1/PN/PS 行、阈值登记字段、`pre_registration` hash 槽）                                                                                                                                                                                                                          |
| `hardware/coral_nexus/verification/schemas/*.py`（`cn_verify/schemas/`）                                                  | 上述 yaml 的 Python validator+`derive.py` 生成器（输入＝W1-01 cn_protocol/enums 的冻结 hash → schema_v1 草案）                                                                                                                                                                                        |

### 6.3 步骤（机械可执行；本轮只写计划文档，不创建文件）

1. 写 `trace_window_contract_v1.yaml`：窗口＝`{schema_ver="cn_trace/v1", run_id, scenario_id, t_id, window_kind∈{full, partial, fault_segment}, boundary_epoch, first_seq, last_seq, record_count, gap_status∈{none,quota,drop,truncate}, blob_manifest:[{sha256,len,blob_kind∈{payload,initial_memory,image_ref,prefetched_state}}], inject_plan_ref, seed_list, delay_bp_config_ref, manifest_ref, seal_ref}`。
2. 定义 68 行 `t_matrix.yaml`（与组件 §8.1 一一对应，列同 §6.2；见本文 §9 表）。
3. `cn_trace/v1` Host 侧容器 schema：`log_record`＝`{seq, cycle, phase(P0..P4), channel∈{CTRL,EVENT,PROGRAM,MEMORY,DEBUG}, dir∈{req,resp}, port, op_seq, message_seq, task_id/grid_id, ctx_gen, window_id, len, payload_hash, payload_len, payload_bytes, blob_hash?, t_id?}`；幂等键＝`(window_id, protocol, port, dir, cycle, phase)`（同拍同口 req/resp 各至多一条）。
4. `checker_verdict_contract_v1.yaml`：每个 checker 输出 `{name, status∈{PASS,FAIL,BLOCKED}, reason, evidence:[{log_ref, seq, cycle}], notes}`。
5. `replay_bundle_contract_v1.yaml`：窗口/初始快照/blob 清单 + `oracle 引用` + `T 引用` + `injection replay 计划 ref`。
6. `perf_gates_contract_v1.yaml`：P0/P1/PN/PS workload 行、阈值登记、`pre_registration_sha256` 槽位（**必须先登记后测量**，见 W7-05）。
7. Python validator 实现（`cn_verify/validate/…`）：校验 `t_matrix` 68 条不重复、每行 `cs1_scope`/`scenario_path`/`oracle` 非空、`manifest` 引用可解且 evidence hash 可重算（`verify` 时执行重 hash+重放，不只看文件存在）。
8. 与 W1-01 对齐：通过 `protocol/field_requests/w7.yaml` 申请 `log_record.payload_hash/payload_bytes/len/blob_hash/blob_ref`、`task_id/cycle/phase(P0..P4 枚举)/message_seq/op_seq/grid_id/ctx_gen`、`channel`、`port`、`dir`；JF 后由 `cn_protocol_v1.yaml` 生成 schema 草案，hash 校验。

### 6.4 验证（独立正/负例；当前命令 `python -m pytest pipeline_validator/tests/test_w7_contracts.py -q`【未来，待 W0-02/W1-05 提供测试容器】）

| 正例                                                                             | 负例（必须先失败后通过）                                                                         | 判据              |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------- |
| 68 行 t_matrix 全列非空/幂等 id                                                  | 注入重复 t_id / 缺 scenario_path / 缺 oracle → validator FAIL，报 `MISSING_FIELD`/`DUPLICATE_ID` | 退出码 2          |
| 窗口 manifest 逻辑字段可从 4 种窗口（normal/known-deadlock/known-fault）全量填充 | blob hash 篡改 → `verify` 报 `HASH_MISMATCH`，退出 3                                             | 0/1/2/3 语义见 §4 |
| 68 行每行至少 1 个 `kind`（`CS1_LOG`/`PERF`/`BUILD`，见 §5.3）                   | 去掉 kind 字段 → validator FAIL（不允许空 PASS）                                                 |                   |
| 68 行 `cs1_scope` ∈ legal 枚举                                                   | 标 `deferred` 但 `defer_reason` 空 → FAIL；`cs2_only` 但 T-59/T-60/明确 N2 之外 → FAIL           |                   |

### 6.5 成功/失败/取消/背压/旧代 不变量（本任务定义 W7 层判据，不改他包）

- Host 被动（VER-12）：W7 不驱动任何被测端口。
- 窗口完整性（VER-22）：`first_seq/last_seq/record_count` 与行内容一致；`payload_hash` 全部可验（SHA-256）；缺任意 blob/快照/映像 hash → 不可重放。
- **配额／drop／truncation 不作完整重放证据**（W7-05 负例专门验证拒绝路径）。
- 相位语义唯一（W1-01 冻结后引用；不自造第二种相位）。

### 6.6 消费者

W7-02a/b/c（oracle 引用窗口结构）、W7-03（C1..C7 checker 引用 trace schema/verdict schema）、W7-04（场景 manifest 填充器/t_matrix 引用）、W7-05（replay manifest/bundle、`perf_gates` schema）、W7-06（最终 manifest 聚合）。

### 6.7 T 追踪

本任务支撑全部 T 行的**判定基础设施**；T 行本身在 §9 表逐条。

## 7. W7-02 独立 ISA + bitwise 数值 oracle（P0；硬前置：W7-01 + JF）

### 7.1 任务拆分（子步 ID 供 §2 依赖表引用）

| 子步   | 主题                                                                                              | 硬前置                                                                                                     | 并行窗口                                            |
| ------ | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| W7-02a | Spike 驱动器 + per-instruction 能力证明（不假称支持 Nexus 私有指令/CSR）                          | W0-01/W0-02 提供 submodule/容器                                                                            | 与 W1-02 并行                                       |
| W7-02b | Python bitwise 数值 oracle（bf16/RV32 IEEE/ROD/溢出不对称）                                       | JF（`cn_protocol_v1.yaml`/`w3_region.contract.yaml`/`w5_binding.contract.yaml` hash + W3-01 数值合同文本） | 与 W3-01 后任何 W3 步并行的实现（不依赖 W3-04 RTL） |
| W7-02c | **oracle 扩展（Nexus 侧 CSR/helper/region 声明/页授权语义）**                                     | W7-02b + helper/region 合同 fragment（W3-03/W4-04/W5-01）冻结                                              | 与 W3-03 并行                                       |
| W7-02d | **生成门/覆盖证明**（`oracle_capability.json`、`coverage_matrix.json`、每类未覆盖项显式 BLOCKED） | 02a/02b/02c + W0-04                                                                                        | 与 W5-03 并行                                       |
| W7-02e | oracle 自测（固定已知输出/已知偏差对照表）                                                        | W7-02c（可与 02d 并行）                                                                                    | 随时；不作为 W0-03 P0 前置                          |

### 7.2 输入

- **已核实固定来源**（2026-11-05 实测，可引用）：
  - `riscv-isa-sim@93a10ae6…`：`insns/vtfmm_alt_tvv.h`（含 `bf16_t`/`DotConfig(2,1)`/`bulk_norm_dot_bf16`/`f32_add_odd`；`res = f32_add_odd(res, f32(out.out))`）、`bulknormdot.h`（guardBits/shift_right_jam/溢出不对称）、`insns/vfwmaccbf16_vf.h`/`_vv.h`（`f32_mulAdd`）、`insns/vfncvtbf16_f_f_w.h`（经 `VI_VFP_NCVT_FP_BF16_OFP8`/`VI_VFP_CVT_LOOP` → `f32_to_bf16`，frm 取自 `VFP_RM`=`STATE.frm->read()`，见 `v_ext_macros.h:1482`/`decode_macros.h:166`）、`insns/{msetmtype,msettk,vte*,vt0*,vtmv,vtmms,vtmmu}.h`。tarball sha256 `064f4c1f22005899fa5106b822bfdc7034ffc9babc2f5821771589047f028a66` 与 W0-01 声明一致（本轮实测校对）。
  - `mpact-riscv@e4f1e9c1…`（Coral `rules/repo_defs.bzl` pin）：`riscv_zvt_instructions.cc`。**本轮源码阅读证据：`RiscVZvtMatmul` bf16 分支的累加是**`float acc = GetElem<float>(kDestTile,...); acc += read_fp(...) * read_fp(...)`** 的顺序 FP32 累加，逐 K 无 ROD 处理**；Coral 打的 `mpact_riscv_zvt_alignment.patch`（patches.bzl 第 7 项）在 `sim/coralnpu_m4_instructions.cc` 引入 `BulkNormDotBf16`(guardBits=1)/`F32AddOdd`（`Zvt` per-op 路径仍按 `RiscVZvtMatmul` 顺序累加，未替换核心循环）。**结论：MPACT 的 BF16 matmul 与 W3 `ROD` 模型不保证 bitwise 同一致**，只能作**结构性/指令序参考**，不能单独作 bitwise oracle。
  - **KD-01（已登记、已通知 W3/parent）**：固定 Spike `f32_add_odd` 在 overflow 时 `res.v++` ⇒ FLT_MAX 饱和后跳到 ±Inf；MPACT 补丁版 `F32AddOdd` 同样 `res += 1; // FLT_MAX -> INF`。二者均**累加级溢出→∞**，而 Coral RTL（`fp_rounding.sv:96-104`：`ROD,RTZ→round_of_to_max_norm=1`；`zvt_pe_adder_fp_lane.sv:149-151` 由此选 max-normal；W3 §6.2 同）为**累加级 ROD 溢出→max-normal**。结论：**acc_overflow_policy 按用户已选 W3 固定 RTL 语义（max-normal）为 normative**；KD-01 仅作为 ISA-reference 已知不一致登记，**不把 Spike/MPACT 的 ±Inf 当 DUT 判定依据，也不把该类强制向量永久 BLOCKED**（由独立数 oracle 依 profile normative 判），DUT 实跑命中 KD-01 输入向量仍判 PASS/FAIL 由 bit oracle 决定。raw Spike mismatch 保留为诊断，**不得用于放宽比较**。
- 已核实 W3 合同（数值 normative 来源）：W3 §6.2/§8.1-§8.2/§8.4（ROD/limited width/jam、溢出不对称、OF·NV、mask/tail、`numeric_profile_id`）。**本轮已确认**：mask/部分 mask 语义是 W3 合同，固定 Spike 的 vtfmm 无 mask 语义（源码无 mask 输入），**mask 类必须由 bit oracle 覆盖**（与 parent 裁定一致）。
- W0-04（target/config/矩阵）、W0-01（固定 submodule/下载 hash）、W0-02（工具链）。

### 7.3 产物

| 文件                                                                                                                                                                    | 内容                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `hardware/coral_nexus/verification/oracle/spike_ref_driver.py`（+`BUILD.bazel`/容器内脚本）                                                                             | 用已核实 SpikeSimulator C++ API（`sw/coralnpu_sim/spike_simulator.{h,cc}`：`LoadProgram/Step/IsHalted/ReadRegister/WriteRegister/ReadMemory/WriteMemory/ReadVectorRegister/WriteVectorRegister/ReadCurrentMatrixTile/SetSwBreakpoint/Reset`）驱动固定 pin Spike：加载 sealed ELF、按 `instruction_list` 逐条 `Step`、`ReadMemory` 读输出/`ReadCurrentMatrixTile` 读 mt；**该驱动只执行上游 ISA 语义**                                                                    |
| `hardware/coral_nexus/verification/oracle/bitwise_*.py`（`bf16_f32.py`、`vme_matmul.py`、`rvv_fp32.py`、`vle.py`、`pagegrant.py`、`region_state.py`、`oracle_main.py`） | 不依赖 Spike/DUT 的独立实现：bf16 乘(带 RNE/ROD 备选开关，normative 按 W3)、f32 加(±mode)、`vfncvtbf16` 按 frm(RNE)、`vfwmaccbf16`、`f32_mulAdd`、`bulk_norm_dot_bf16` 语义**独立重建**（guardBits/jam/溢出不对称 per W3 合同）、`vtle/vtse/vtzero/vtmv/vtmm*` 的矩阵 tile 语义、i32 gather/scatter 地址表、页授权（scope/epoch）真值表、区域/链声明状态机（合法集/非法集），**用 Python 整数 bit-ops 重建、不用浮点、不用 numpy、不用 DUT RTL 源码复制**（parent 裁定） |
| `hardware/coral_nexus/verification/oracle/nexus_extend.py`（W7-02c）                                                                                                    | Spike 与 bit oracle 均不原生支持 Nexus 私有面时的显式适配：① CSR/region helper/声明指令 → 借鉴 Spike 已有 custom-instruction 机制实现同语义（不声称上游支持）；② 独立 bit oracle 里的 `region FSM`（FIRST/CONTINUE/FINAL 绑定、START/END）+ 页授权真值表；输入＝W3-03/W4-04/W5-01 合同 fragment（JF hash）                                                                                                                                                               |
| `hardware/coral_nexus/verification/oracle/oracle_capability.json`（生成）                                                                                               | 逐指令/逐语义类：`oracle∈{ISA_REF(独立), BIT_ORACLE(独立), BOTH, NONE}`；`NONE` 行列 `BLOCKED(oracle 证据不足)`；不模糊 TODO                                                                                                                                                                                                                                                                                                                                             |
| `hardware/coral_nexus/verification/oracle/coverage_matrix.json`（生成）                                                                                                 | W3 §8.2 恶性用例（ROD 边界、TK1/TK2 分组差、乘树有限宽 jam 边界、大指数差/异号大幅抵消、subnormal×subnormal、BF16→FP32 溢出、mask 先于 special 分类、masked-Inf/NaN 专项）逐类的向量覆盖数 + 参考来源                                                                                                                                                                                                                                                                    |
| `hardware/coral_nexus/verification/oracle/selftest/…`                                                                                                                   | 固定输出 ELFBIN/固定输出 vectors（固定已知输出对 P0-ELF 等简单 kernel 可直接给出，W0-03 P0 基线可直接消费；**P0 简单 ELF 的已知 bytes 不必等完整 oracle**）                                                                                                                                                                                                                                                                                                              |

### 7.4 步骤（机械可执行）

1. **先写 `bitwise_*`**（normative 依赖只有 W3 文本+W0 config），写“每个语义规则 → 对应整数位算法 + 已知偏差（KD-01 等）”表格并保留于 `known_oracle_deviation.md`。
2. `spike_ref_driver.py`：从已核实 API（§7.2 上游链接）写最小驱动：加载 sealed ELF（`LoadProgram`）、循环 `Step` 到 halt 或 instruction_limit、`ReadMemory` 拷出输出、`ReadCurrentMatrixTile` 抓 mt。
3. **能力探针**（W7-02a）：逐条 per-instruction 检查固定 Spike 是否可执行（不 trap、mt 与 memory 写符合 bit-级结果）；逐类写 `oracle_kind`。**不假称上游支持**；Nexus 私有面直接 `NONE/BLOCKED`，并走 W7-02c 扩展路径（非 TODO）。
4. **一致性差分**（W7-02d）：在覆盖矩阵中给每类语义至少 1 个 spike-vs-bit-oracle 一致向量；**KD-01 类（累加级 ROD 溢出）不强制二者一致**，写入 `known_oracle_deviation.md` 供 T-29 检查用。**若 bit oracle 与 Spike/MPACT 二者之一在某类持续不一致，不得把 DUT 结果判给二者之一**：按 W3 normative 由 bit oracle 决定、Spike 作为结构性/指令序第二读，且把该类写进 KD 清单、上报 parent/W3。
5. **独立核对（不抄 DUT）**：bf16 ROD 有限宽的独立核对来自**两个上游可查实现**：(i) 固定 Spike `bulk_norm_dot_bf16`（`bulknormdot.h`）＋ (ii) Coral 打了补丁的 MPACT `BulkNormDotBf16`（`F32AddOdd`，guardBits、溢出→Inf 与 Spike 同构）；两者与 `bitwise_*` 的 ROD 输出在 KD-01 以外的向量类**必须 bit 一致**（覆盖矩阵中给一致性断言），不一致即报警。**禁止**为凑一致而在 bit oracle 中抄 RTL 源码；如二参照互斥，取独立 bit oracle 为真值源（normative 来源＝W3 合同文本）。
6. `oracle_main.py` 提供 `cn.sh gate/verify` 可调 API：`oracle.check(t_id, scenario_run_dir) -> verdict`，并在 `cs1_manifest.json` 中登记 per-T `oracle` 结果（供 C4/W7-04/06 消费）。
7. **生成门**（W7-02d）：`cn.sh unit --suite w7 --out ...` 中逐条跑 `oracle_selftest`（固定向量）+ `capability` 生成 + `coverage_matrix` 生成；任何 `oracle_kind=NONE` 的 T 行必须在 `t_matrix` 里标 `cs1_scope∈{deferred, cs2_only}` 或在 notes 显式写 `BLOCKED(oracle 证据不足)`；**不允许只写“TODO 补 oracle”**。

### 7.5 成功/失败/取消/背压

- 成功＝`oracle_kind` 逐条非 NONE/BLOCKED 之外的类全部可从 `bitwise_*` 独立复现，且 KD-01 类被显式列进 `known_oracle_deviation.md`。
- 失败＝任意必检类 `NONE` 且其 T 行属 mandatory ⇒ 该 T 行判 `BLOCKED`，进 W7-06 汇总（非本轮 PASS）。
- **不做**：自动重试/超时当通过/ULP 放宽/借用 Python numpy 浮点重建/bit-approx。

### 7.6 不变量（写进 oracle 实现，作为后续 W7-02 单测断言）

- 只用整数位运算/有理数逻辑重浮点，不 import numpy/scipy 浮点路径。
- KD-01 之外与 Spike/补丁 MPACT bit 一致的类才可标 `BOTH`；KD-01 类必须按 W3 独立 bit oracle 判 DUT 结果。
- 不读 `zvt_pe_*.sv`/Zvt RTL 推导数值（“独立核对 bf16 ROD 有限宽不要 DUT 源码复制”），只能读上游 ISS/独立合同文本；如发现 DUT 实跑与 `bitwise_*` 不一致，按 C4 判 FAIL 并写 `RTL_VS_PROFILE`，不倒向 DUT。
- 逐类覆盖矩阵必须给 minimum 向量数（≥1 per rule/类；mask/TK 分组/溢出类 ≥3），不足＝`oracle_coverage=INSUFFICIENT`。

### 7.7 消费者/T

- C4（W7-03）唯一数值真值来源；T-01/T-29/T-39/T-61/T-68 的必过比较。
- T-29 的 mask/部分 mask 语义检查**只**由 bit oracle 判（Spike 无 mask，已核实）（T-29 的非 mask 语义由 ISA_REF+bit oracle 联合覆盖）。
- T-16/T-50/T-65 的 indexed/页授权数值真值（i32 搬运地址/字节）由 `vle.py`/`pagegrant.py` 输出，W7-04 场景直接消费。

## 8. W7-03 C1–C7 checker 与负例（P0；硬前置 W7-01；CS1 真窗口依赖 W7-04a 后）

### 8.1 输入

- W7-01 冻结 schema；`lifetime_bounds.yaml`（W1-01）。
- W6 合同 fragment（EVT-14 相位语义、`PHASE_SIGNAL` 不去重、`EVT` 幂等键、`_on_*` handler 名）作为 C2 规则输入。
- **数据源（二态）**：(a) 单元开发期用 `spike_ref_driver`/`bitwise` 输出/`csir-run`（W1-05 harness，`--rtl-tile-mask 0x1`）**真实 CS1 窗口**；(b) W7-03b 负例套件用**注入缺陷后的窗口**，不依赖 stub。**不拿现有 Python `trace.py` 输出当 CS1 窗口**（`l2_admission_wait` 的既有模型级 smoke 只作调度口径，不算 realbits/RTL 证据，见 parent 裁定）。

### 8.2 产物

- `hardware/coral_nexus/verification/checkers/c1_format.py`..`c7_manifest.py`（7 个文件）+`runner.py`+`verdict_io.py`+`not_empty.py`（“非空 PASS”公共实现）+`mutator.py`（负例注入驱动）。
- `hardware/coral_nexus/verification/checkers/known_deviations.py`（KD-01 声明表）。

### 8.3 步骤 + 判据（逐 checker）

1. **C1 格式/唯一性**：schema 合法、log 记录与 `cn_trace/v1` 匹配、seq 连续（`lifetime_bounds.yaml` 允许的 gap 除外）、`(window_id, protocol, port, dir, cycle, phase)` 唯一、R1 无缺。负例：**seq 断档、重复窗口记录、hash 不符**。
2. **C2 偏序**（VER-04/VER-14/20）：逐 Context 合法顺序（SCH 界内）；跨 Context 依赖边（依赖边来源＝`w3_region.contract.yaml`/`w4_pagegrant.contract.yaml`/`w5_binding.contract.yaml`）不倒置；生命周期规则（消费者不早于就绪、释放不早于最后读取、旧代不唤醒新代、终态不早于 store_done）；退休链（EXEC_DRAINED→RETIRE_READY→Group RETIRE 授权→RTL 本地回收→TASK_COMPLETION→剩余父 L2 pin/R 关闭→root 完成；TASK_COMPLETION 不等父 R free）；输入早释放链：`load await→Task signal→all_tasks 聚合→纯读 pin 解除→owner release 与 backing 引用闭合→物理 final-free`，**不强加在 Task completion 之后**；等待记录唯一 writer＝W6（region 等待＝W3、W2 只读）。**不比较全局 interleave、不要求周期相等**。负例：消费者早读、旧代唤醒新代、释放早于最后读取、终态早于 store_done。
3. **C3 守恒/账本**（VER-15；数据从 W1 snapshot API 读取，见 §1.1）：credit 守恒、R lease/Arena 计数守恒（每 bank `allocated+free==user_spm_per_bank`）、每已接受命令恰一终态、Profile generation 单调、每 phase 聚合恰一次、成功退出零泄漏（backing/claim/view/pin/in-flight）；L2 收支字段由 `l2_extent_release/blob` 与 `RESOURCE_COMMIT_IND/RETIRE_ACK/TASK_COMPLETION` 原始 endpoint 交叉验证，**不接受 mock bool/单侧 echo bool 当 PASS**。负例：双重回收、终态重复、credit 违反、Arena 分裂违规。
4. **C4 数值**：oracle（W7-02）与 DUT 输出 bit 对 bit。KD-01 类由 bit oracle 定 normative；mask 类由 bit oracle 判。**不允许 ULP 放宽**。负例：bit 翻转、精度放大、替换为 FP16、以 RVV/scalar 冒充 VME。
5. **C5 终态/终止**（VER-08）：显式有限环境假设集下全部已接受工作结算；外部永久阻塞时其他独立工作不被冻结；**超时＝FAIL 附重放包**；`"pending"≠"完成"`。负例：`pending冒充完成`、deadline 到点当成通过。
6. **C6 重放门**：§6.2 窗口完整性逻辑（`trace_window_contract_v1` 的 first/last/record_count/gap_status/blob_manifest/seal），**不完整窗口强制 NOT_FULLY_REPLAYABLE**。负例：**尾部缺失但前缀连续仍宣称可完整重放**、quota 截断、drop、blob 缺失。
7. **C7 manifest 门**：68 行全部有 status 只是必要条件；**所有适用 mandatory 行必须 PASS**；`DEFERRED_NOT_APPLICABLE` 仅限 T-59/T-60 与明确 N2 子项，且须注明启动条件；`BLOCKED` 须引用缺失环境假设或未冻结规格；**证据引用闭合无 dangling**；**不允许 deadline 到点当作通过**。负例：dangling 引用、非空 PASS 检查（`not_empty` 公共实现）。

### 8.4 独立验证（W7-03 自测）

- `python -m pytest pipeline_validator/tests/test_w7_checkers.py -q`（【未来】）
- 每个负例给出**期望 verdict 与命中字段**断言（如 `c6.check_truncated_window -> FAIL(first_seq mismatch)`），**先失败后通过**（先跑负例确认 FAIL，再跑修复后的正例 PASS，二者在报告里成对留证）。

## 9. T-01..T-68 追踪与 mandatory 矩阵（W7-04 承；owner 包列表见 t_matrix）

**列义**：`owner`＝组件设计 W7 §8.1 主责；`impl_t`＝主实现任务（各包内部 ID，W7 不改）；`verify`＝W7 验证任务（本次编号 W7-04.1/…/W7-04.8 批次）；`cs1`＝mandatory 判定（默认 mandatory；仅 T-59/T-60 与 N2 之外例外，见下）；`证据类型`＝`CS1_LOG+bit_oracle` 或 `CS1_LOG+PMU`、`CS1_PY`（仅 §5.3 允许行）、`BUILD`（仅构建期）。所有行默认 `cs1=MANDATORY`，**例外**＝`CS2`（T-59/T-60）/`N2`（T-28 多实例公平性行、T-68 的 N≥2 部分）/`PS 条件`（T-35 的 PS 子行）/`validator_layer`（T-49、T-33 拒绝边界部分、T-53 loader 级部分）/`build_time`（T-37）。

| T    | owner | impl_t                                              | verify 批次      | cs1                                        | 证据类型                                                                                                        |
| ---- | ----- | --------------------------------------------------- | ---------------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| T-01 | W2+W3 | W2-05, W3-04, W5-03, W7-02                          | W7-04.1（G2）    | M                                          | CS1_LOG+bit_oracle（ISA 矩阵定向操作数）                                                                        |
| T-02 | W2    | W2-02/02-05, W7-04                                  | W7-04.1（G2）    | M                                          | CS1_LOG（两 Context 同 RF 号隔离）                                                                              |
| T-03 | W4    | W4-03                                               | W7-04.1（G2）    | M                                          | CS1_LOG（A long-latency load 与 B 纯计算；B 先完成）                                                            |
| T-04 | W2    | W2-03, W6-02                                        | W7-04.1（G2）    | M                                          | CS1_LOG（A await+B 事件）                                                                                       |
| T-05 | W2    | W2-05, W3-02                                        | W7-04.1（G2）    | M                                          | CS1_LOG（切换瞬间在途 scalar/RVV/VME 无 flush/drain）                                                           |
| T-06 | W2+W3 | W2-05, W3-04, W5-03                                 | W7-04.1（G2）    | M                                          | CS1_LOG（发射时配置快照/无串写）                                                                                |
| T-07 | W2    | W2-01/02-04/02-05                                   | W7-04.1（G2）    | M                                          | CS1_LOG（重定向+未返回 load）                                                                                   |
| T-08 | W2    | W2-05, W6-01                                        | W7-04.1（G2）    | M                                          | CS1_LOG（slot 复用+gen 回绕）                                                                                   |
| T-09 | W2    | W2-05                                               | W7-04.1（G2）    | M                                          | CS1_LOG（人为延迟 retire 头）                                                                                   |
| T-10 | W4    | W4-03, W1-03                                        | W7-04.1（G2）    | M                                          | CS1_LOG（乱序/分片访存+错误归属 PMU）                                                                           |
| T-11 | W2    | W2-03                                               | W7-04.1（G2）    | M                                          | CS1_LOG（issue 口随机 ready+bundle 部分接受）                                                                   |
| T-12 | W2    | W2-01/02-04/02-05, W1-02                            | W7-04.1（G2）    | M                                          | CS1_LOG（C3 守恒零差+恢复曲线）                                                                                 |
| T-13 | W6    | W6-01/06-02                                         | W7-04.1（G2）    | M                                          | CS1_LOG（事件 wait 前/同拍/后+重复+错误码）                                                                     |
| T-14 | W6    | W6-02, W1-04, W4-01/04-05, W7-03                    | W7-04.2（G3）    | M                                          | CS1_LOG（早释放：多消费者输入，未聚合/仍有 claim/view/pin/in-flight 不可复用；闭合后可复用；L1/R/程序引用保留） |
| T-15 | W6    | W6-02, W4-02/04-05, W1-02                           | W7-04.2（G3）    | CS1 子集(M)+CS2 比较补充                   | CS1_LOG+CS2(validator 偏序对拍)                                                                                 |
| T-16 | W4    | W4-04                                               | W7-04.1（G2）    | M                                          | CS1_LOG+bit_oracle（gather/scatter 固定真值表）                                                                 |
| T-17 | W4    | W4-01/04-03/04-04                                   | W7-04.1（G2）    | M                                          | CS1_LOG（高地址/溢出/越界 fault 归属）                                                                          |
| T-18 | W1    | W1-04, W4-05                                        | W7-04.2（G3）    | M                                          | CS1_LOG（SAME/COMPATIBLE 并发+真实切档）                                                                        |
| T-19 | W1    | W1-04, W4-02/04-05, W4-01                           | W7-04.2（G3）    | CS1 子集(M)+CS2（写时序链全流程）          | CS1_LOG+CS2                                                                                                     |
| T-20 | W1    | W1-04, W4-01/04-02, W5-04                           | W7-04.2（G3）    | M                                          | CS1_LOG（两阶段准入+终态归零；N05 期望按本行；无跨 Tile gang）                                                  |
| T-21 | W1    | W1-04, W4-02                                        | W7-04.2（G3）    | M                                          | CS1_LOG（WAIT_CAPACITY 注入释放；trace 无轮询）                                                                 |
| T-22 | W1    | W1-04                                               | W7-04.2（G3）    | M                                          | CS1_LOG（可推进或启动前拒绝恰一）                                                                               |
| T-23 | W6    | W6-01/06-02, W2                                     | W7-04.1（G2）    | M                                          | CS1_LOG（completion queue/event table 灌满；终态计数=接受数）                                                   |
| T-24 | W6    | W6-03                                               | W7-04.2（G3）    | M                                          | CS1_LOG（已取消 store 的“真实完成或确认隔离”两出口）                                                            |
| T-25 | W6    | W6-03                                               | W7-04.2（G3）    | M                                          | CS1_LOG（C0 非法+C1 正常 fail-stop）                                                                            |
| T-26 | W6    | W6-03, W2                                           | W7-04.2（G3）    | M                                          | CS1_LOG（reset/halt 竞争；无 double-settle）                                                                    |
| T-27 | W6    | W6-02/06-04/06-05, W1                               | W7-04.2（G3）    | M                                          | CS1_LOG（全部 Context 等待；唤醒链闭合）                                                                        |
| T-28 | W3    | W3-03/03-05, W2                                     | W7-04.3（G3）    | CS1(M at N=1)+N2(多实例公平性)             | CS1_LOG+bit_oracle                                                                                              |
| T-29 | W3    | W3-04/03-05, W7-02, W5-03                           | W7-04.3（G3）    | M                                          | CS1_LOG+bit_oracle（mask 语义仅 bit_oracle；K 尾/SEED 3 来源/±Inf/NaN/subnormal/K 尾奇数/TK1/TK2 分组差）       |
| T-30 | W5    | W5-04                                               | W7-04.1（G2）    | M                                          | CS1_LOG（同 image 双 Context；可写区不相交）                                                                    |
| T-31 | W5    | W5-05                                               | W7-04.2（G3）    | CS1 子集(M)+CS2                            | CS1_LOG+CS2                                                                                                     |
| T-32 | W5    | W5-03/05-05                                         | W7-04.2（G3）    | M                                          | CS1_LOG（报告 kernel 映射与实际映像 hash/反汇编一致）                                                           |
| T-33 | W5    | W5-05, W0-04                                        | W7-04.1（G2）    | 拒绝边界=M（`BUILD` 证据）；CS1 执行子集=M | BUILD+CS1_LOG                                                                                                   |
| T-34 | W2    | W2-03, W3                                           | W7-04.2（G3）    | M                                          | CS1_LOG+PMU（服务间隔统计≤界；无饥饿）                                                                          |
| T-35 | W7    | W7-05                                               | W7-04.8（G5）    | P0/P1/PN=M；PS 条件延期                    | PERF（P0/P1/PN 报告+登记 gate）                                                                                 |
| T-36 | W7    | W7-05, W6-04                                        | W7-04.8（G5）    | M                                          | CS1_LOG+PMU（计数器定义表逐项核对；主周期/经过周期分列）                                                        |
| T-37 | W5+W0 | W5-01, W0-04, W7-02                                 | W7-04.6（build） | M（build 前置；非仿真）                    | BUILD                                                                                                           |
| T-38 | W4    | W4-04/06-04, W1                                     | W7-04.5（G4）    | M                                          | CS1_LOG（运行中容量耗尽；返回/取消/释放仍前进）                                                                 |
| T-39 | W1    | W1-05, W5-03                                        | W7-04.4（G3）    | M                                          | CS1_LOG+bit_oracle（含 P0 级语料 Tile 级运行）                                                                  |
| T-40 | W1    | W1-05                                               | W7-04.4（G3）    | M                                          | CS1_LOG（Device→Group→Tile 完整批）                                                                             |
| T-41 | W1    | W1-04                                               | W7-04.2（G3）    | M                                          | CS1_LOG（R=1 第二 Task；`tile.free/input_released/output_ready/取消` 后 lease 仍占用；WAIT_CONTEXT_LIMIT）      |
| T-42 | W1    | W1-05, W4-05                                        | W7-04.6（G4）    | CS1 子集(M)+CS2                            | CS1_LOG（单 Tile L1 占满/尾部 Task 部分放置/零 dispatch）                                                       |
| T-43 | W1    | W1-05, W4-05                                        | W7-04.6（G4）    | CS1 子集(M)+CS2                            | CS1_LOG（五类独立归因；空池→PERMANENT fault）                                                                   |
| T-44 | W1    | W1-04, W5-01                                        | W7-04.2（G3）    | M                                          | CS1_LOG（context=N 被占+越界静态钉住拒绝）                                                                      |
| T-45 | W1    | W1-04, W6-02                                        | W7-04.2（G3）    | M                                          | CS1_LOG（phase 重复/旧代/未知/漏发）                                                                            |
| T-46 | W1    | W1-04, W4-05                                        | W7-04.2（G3）    | CS1 子集(M)+CS2                            | CS1_LOG（Prepare/Commit ACK 全匹配才开 gate）                                                                   |
| T-47 | W4    | W4-06, W1                                           | W7-04.5（G4）    | CS1 子集(M)+CS2                            | CS1_LOG                                                                                                         |
| T-48 | W4    | W4-06, W1                                           | W7-04.5（G4）    | CS1 子集(M)+CS2                            | CS1_LOG（取消后“真实完成/隔离确认”两出口）                                                                      |
| T-49 | W5    | W5-04, W1（`program_table`/ensure_resident 侧接线） | W7-04.7（G4）    | M                                          | CS1_PY（Python 冷/热 epoch 侧基线，`TestRuntimeColdWarm`；RTL 侧冷热并到 T-58）                                 |
| T-50 | W4    | W4-04/04-06                                         | W7-04.5（G4）    | M                                          | CS1_LOG+bit_oracle（GAT-11..20 逐条；编译期拒绝三类）                                                           |
| T-51 | W6    | W6-03, W2                                           | W7-04.2（G3）    | M                                          | CS1_LOG（同拍多故障+first-fault；dropped/sticky）                                                               |
| T-52 | W6    | W6-03, W4-06/4-02                                   | W7-04.5（G4）    | CS1 子集(M)+CS2                            | CS1_LOG（复位×在途/MFE/Profile/DRAIN 阶段次序固定）                                                             |
| T-53 | W5    | W5-05                                               | W7-04.7（G4）    | M（loader 级=M；CS1 执行子集=M）           | BUILD+CS1_LOG                                                                                                   |
| T-54 | W1    | W1-02/01-03, W6                                     | W7-04.2（G3）    | M                                          | CS1_LOG（数据通道信用耗尽时控制消息注入）                                                                       |
| T-55 | W4    | W4-01, W4-02                                        | W7-04.5（G4）    | M                                          | CS1_LOG（逐 bank 碎片 Arena 序列）                                                                              |
| T-56 | W5    | W5-02/05-05                                         | W7-04.6（G4）    | M                                          | CS1_LOG（直线/受控循环/非法静态拒绝）                                                                           |
| T-57 | W1    | W1-04, W4-06                                        | W7-04.5（G4）    | CS1 子集(M)+CS2                            | CS1_LOG（私有 buffer 闭合；双 reader+producer 先退休 final-free 恰一次）                                        |
| T-58 | W4+W5 | W4-02/05-04, W1-04, W7-03                           | W7-04.7（G4）    | M                                          | CS1_LOG（冷→热→身份变更→group 复位→单 Tile 失效+装载中再提交等）                                                |
| T-59 | W1    | W1-05, W4/W5/W6                                     | W7-04.9          | **deferred(CS2 出口，D-24 评审后启动)**    | —                                                                                                               |
| T-60 | W1    | W1-05, W4/W5/W6                                     | W7-04.9          | **deferred(CS3 出口，D-25 评审后启动)**    | —                                                                                                               |
| T-61 | W5    | W5-01/05-05                                         | W7-04.7（G4）    | M                                          | CS1_LOG+bit_oracle                                                                                              |
| T-62 | W3    | W3-03/03-04/03-05, W7-02                            | W7-04.3（G3）    | M                                          | CS1_LOG（`compute_done≤state_committed≤output_visible≤state_free≤region_done≤event`）                           |
| T-63 | W5    | W5-04, W0-04                                        | W7-04.7（G4）    | M                                          | CS1_LOG（实际资源复核后封印）                                                                                   |
| T-64 | W1    | W1-05, W4/W6, W7-03                                 | W7-04.4（G3）    | M                                          | CS1_LOG（背压延迟+禁用 Python 执行路径）                                                                        |
| T-65 | W4    | W4-04/04-05, W6                                     | W7-04.5（G4）    | M                                          | CS1_LOG+oracle（scope 页授权/epoch/满表）                                                                       |
| T-66 | W4+W5 | W4-02/05-04                                         | W7-04.7（G4）    | M                                          | CS1_LOG                                                                                                         |
| T-67 | W6+W7 | W6-04, W7-05                                        | W7-04.8（G5）    | M                                          | CS1_LOG（正常零留存；溢出只丢诊断；完整窗口可单侧重放）+PERF（gate 拒绝缺口）                                   |
| T-68 | W3    | W3-03/03-05, W7-02                                  | W7-04.3（G3）    | M (N=1 全必)；N2 部分 deferred             | CS1_LOG（区域/链声明合法/非法全序列；`software 无法实例号可观察`）                                              |

> **批次分档**（verify 列）：1＝G2 标量/基础访存（W1-05 oneTile 后即可）、2＝G3 RVV/VME/事件/准入、3＝G3 VME 数值、4＝G3 出口（与 W1-05 同步）、5＝G4 内存/页授权、6＝G4 Group/包装/静态拒绝/build、7＝G4 程序/映像（含 T-66）、8＝G5 最终（W7-05 perf/重放/计数门），9＝延期。最终 **G5 出口只认 W7-04.8 的 CS1_LOG/报告**，前面批次结果作为进度证据继续被 W7-06 `verify` 重放校验。

## 10. W7-04 真实 CS1 定向验收（G2..G5 分批）

### 10.1 输入

- §6 `t_matrix.yaml`（由本任务填充 per-scenario 列）+ W0-04（cs1_c1/c2/c1-only 配置）+ W1-05（`csir-run` harness，TilePort C ABI）+ 各包 per-T 场景文件。
- **真实 CS1 负载**：W5-03 kernel 集（matmul(含 K 分块累加+accumulate 语义)、gather、scatter、**softmax 系**（含 online softmax）、norm、reduce、pow、add、**recurrence**（如 KV-cache 累加/状态机））。**禁止只做 matmul smoke**：每族至少 2 个真实 CS1 场景（`T-xx_a1`/`_a2`），在 `t_matrix` 场景列列名；W7-04 无权删族，族缺席＝该族 BLOCKED，须走 W7-06 报告。
- 既有 workload 侧证据锚点（复用现有 fixture 而非另造）：
  - `pipeline_validator/tests/test_runtime.py:628` `TestRuntimeColdWarm`（T-49 Python 验证基线）
  - `pipeline_validator/tests/test_tile_indexed.py:28` `FIXTURE = .../indexed_gather_scatter.mlir`、`:20` `ByteStore`（T-16/50 真字节来源）
  - `examples/workloads/gather_matmul.mlir`、`matmul_gather_add.mlir`、`transformer_*`（softmax/norm/recurrence 族来源）、`reduce_sum_*`。
- **T-14 早释放真实字节验证**：直接复用 `examples/scenarios/l2_admission_wait.mlir` 的**图形**，跑 `--ir-file ... --hw-config examples/configs/profile_l2_256k.yaml --hw-override num_dma_channels=2 --hw-override hbm_fixed_latency_cycles=10 --context-mode 2 --device-context-mode 2 --input-binding A_IN=0x100000:131072:rw --input-binding A_OUT=0x200000:131072:rw --input-binding B_IN=0x300000:131072:rw --sim-override fidelity=full_memory --memory-trace`（参照 Design_Validation §2.4 命令），**检查目标为 Python 模型的调度行为基线**（`task_lease_release` 均晚于 `l2_extent_release` 物理回收、`b_input.commit_cycle` 与已回收 extents 的逐 bank segment 一致；对照 `tile_group.py:1281/1088` 与 `memory/arena.py:1312` `l2_extent_release`）。**该 Python 侧结果只是 T-14 的必要条件（调度层面），不构成 realbits/RTL 证据**；realbits/RTL 证据由 W7-04.2 场景（同 scenario，CS1 真窗口）给。

### 10.2 分批步骤（对应 §9 verify 批次）

1. **W7-04.1（G2 标量/访存/事件）**：T-01..T-13, T-16, T-17, T-23, T-30, T-33。产出 `cosim/*.yaml` 场景+`t_matrix` 场景列+`cs1_manifest` 段。执行侧 W1-05/`csir-run`（TilePort+C ABI），`--rtl-tile-mask` 0x1 起。
2. **W7-04.2（G3 事件/准入/GROUP）**：T-14, T-18, T-19, T-20, T-21, T-22, T-24, T-25, T-26, T-27, T-31, T-41, T-44, T-45, T-46, T-51, T-54。
3. **W7-04.3（G3 VME/数值）**：T-28(N=1), T-29, T-32, T-62, T-68(N=1)。
4. **W7-04.4（G3 RTL/出口）**：T-39, T-40, T-64。
5. **W7-04.5（G4 内存/授权）**：T-38, T-47, T-48, T-50, T-52, T-55, T-57, T-65。执行侧 W4-06/W1。
6. **W7-04.6（G4 Group/包装）**：T-37(build), T-42, T-43, T-46, T-56, T-59(不跑，仅登记), T-60(不跑，仅登记)。**T-37 是 build-time 检查，不进 cosim 命令**。
7. **W7-04.7（G4 程序/映像）**：T-33(执行子集), T-49(Python 基线已跑可复查), T-53, T-58, T-61, T-63, T-66。
8. **W7-04.8（G5 最终 gate 聚合）**：T-35(P0/P1/PN), T-36, T-67（含“缺口窗口拒绝”和“正常模式零留存”）。`T-67` 还承担 replay 拒绝验证入口（配合 W7-05）。

每批次的场景文件逐 T 单独存放 `hardware/coral_nexus/verification/scenarios/<t_id>_<slug>.yaml`；场景 schema 由 W7-01 的 `scenario_contract.yaml` 冻结（含 oracle 引用/expected bytes/invariants）。

### 10.3 验证（每批独立）

- `cn.sh cosim --scenario <file> --backend rtl --rtl-tile-mask 0xf --config <c1/c2 yaml> --out <dir>`【未来】：退出 0＝所有 `t_id` 行 verdict=PASS 且 `not_empty_pass=true`。
- `cn.sh verify --manifest <cs1_manifest.json>`【未来】：重 hash 全部 evidence+重跑 C1..C7（不只查文件存在），退出 0＝manifest 有效。
- 每批负例（先失败后通过）：如 T-14 的“未聚合就复用”构造（`SIGNAL_PHASE` 不带全 `all_tasks` → C2/`w6` checker 报 `PREMATURE_RELEASE`）、T-24 的“已取消 store 被记作 DONE”→ C2/C3 报 `PREMATURE_TERMINAL`。

### 10.4 最终门（W7-06；对应需求 CS1 出口判据 §18.4.4）

- (1) T-39、T-40 在 P0 Tile 级语料通过；(2) T-58 真实取装与 T-61..T-68 简版计算/映像/区域 lease/消息 owner/页授权/装载隔离/日志策略通过（N=1 的 T-68 全必）；
- (3) VER-15 零泄漏（W7-06 复跑 C3 全场景汇总）；(4) VER-21 同语料替换一致（`--rtl-tile-mask 0x0` 纯 Python 基线 vs `0xf` RTL，合同级偏序/账本/bitwise 一致）；(5) VER-22 完整 Host 窗口已知故障单侧重放通过；(6) §23.7 无未裁决项（在 `cs1_final.json` 中逐条列出未决 id 或空表）。

## 11. W7-05 重放／quota／Perfetto／性能门（P0；硬前置 W7-01、W1-03、W7-03b、W7-04b）

### 11.1 重放与拒绝（单侧已知故障 + quota/drop/truncation 拒绝）

1. **`replay` 流程**：读窗口→C6 门检→失败→`NOT_FULLY_REPLAYABLE`；通过→按窗口日志逐条驱动 W1-03 的 `libcn_cosim.so` bp API（bp_begin→bp_tick→bp_collect；**不使用 libcn_cosim_stub.so**，复述 parent 决定），比对实际请求与日志请求逐条等价（协议/身份/payload hash 一致；周期不要求相等，仅偏序）。**重放失败＝FAIL 且逐条给出第一 mismatch 记录**，不静默。
2. **负例（每例独立可复现，含 verdict）**：
   - **quota 截断**：构造窗口 blob 超过 quota（场景用大 payload）→ `gap_status=quota/truncate`→ C6 FAIL，`verify` 退出 3。
   - **drop**：注入 R1 丢弃（删除 task/ctx/终态条目）→ C1 R1 缺失 FAIL。
   - **truncation**：尾 seq 缺断但 manifest 未记 gap → C6 FAIL（**前缀连续仍宣称可重放是关键负例**，复用组件 §8.3 专条）。
   - **缺 blob/缺快照/缺映像 hash**：C6 FAIL。
   - **已知故障重放**（单侧）：从 W6-03 注入计划的已知故障场景（C0 非法指令、双 SET、同拍故障）构造窗口，重放需产生**与首次运行相同的 FAIL verdict**（故障可复现性）。
3. **Perfetto SQL 单位审计**（组件 W7 设计 §3.1 单位注记）：
   - 用 `examples/artifacts/tools/trace_processor_shell`（存在，非新增依赖）执行只读 SQL：`select ts, extract_arg(arg_set_id,'args.cycle') from slice where name=...`；**`ts` 是 µs（由 `cycle_ns=1000MHz` 默认下与 cycle 相等）**——审计脚本必须验证 `ts_ns` 与 raw cycle 在每个检查点相等，并跑一个**已知 800 MHz 时钟**（`--hw-override clock_mhz=800`）的负例确认“SQL `ts` 数值≠cycle”会被检出（防“ns 恒等于 cycle”的误用）。
   - 每份 trace 跑必做的 stats 审计（`stats` 无非零 error/data_loss；slice/counter 数与 raw JSON 对账）——复用 perfetto-trace-analysis 套路但输出进入 `cs1_manifest`。

### 11.2 P0/P1/PN 性能门（对齐 parent 裁定 + 需求 PERF-01..07）

1. **固定 workload 矩阵**（`perf_gates.json`，W7-01 schema）：P0=原 Coral 原核（W0-03 交付）+ P1=改造硬件单 Context（`csir-run`+`cs1_c2.yaml`）+ PN=同一改造硬件多 Context（同 RTL hash，**必须同 `cs1_c2.yaml`，只改活跃 Context 数**）+ PS 仅作占位标记 `not_run`。
2. **先登记后测量**：`perf_gates.json` 里每行必须有 `baseline_ref`(P0/P1/PN 来源) 与 `pre_registered_threshold`（**数值阈值在实测数据出来前登记**，如“单 Context 吞吐较 PN ≤5% 回归”等允许在实测前登记的**阈值来源必须给登记理由**；无依据不得写死加速比，也不得用占位 0）；由 W7-05 在**测量前**填入，hash 记入 W7-06 manifest。
3. **口径**（PERF-01..07）：总吞吐/makespan/单任务/尾延迟/单 Context 回归（P1 vs P0，同工作量+精度+内存+时钟+编译策略）；Context 等待原因/发射/退休/引擎活动/访存占用（PERF-02）；**gather 有用/实际字节、合并率、outstanding**（PERF-04，用 `gather_perf` workload+`memory_trace` 原始字节计数，非模型层）；**计时终点包含输出写回+完成语义**（PERF-05）；**调度机会≠固定加速比**，报告分列（PERF-07）。**禁项**：把 validator 周期当 RTL 性能/PPA/Fmax/多 VME 利用率宣称。
4. **性能因果（不许源码文本/文件存在当回归）**：
   - 每项性能指标在 `perf_report.json` 里登记**测量方法+从原始日志/字节计数提取的字段名+统计函数**（如 “makespan=sum(任务在 Tile 的 start..end cycle)，取自 CS1_LOG”），并给**一个反向控制负例**（如注入 2× 延迟 → 对应指标必须变差才证明不是硬编码常量/文件存在性测试）。
   - `unit --suite w7` 里跑 `test_perf_metrics_causal.py`（未来）：改注入的 `hbm_fixed_latency_cycles`，核对 P1 makespan 增量与注入差值一致（T-35 因果链），证明指标测量是因果的、非占位。

## 12. W7-06 最终出口与维护证据（P0；硬前置全部前置批次）

**产物**：

- `out/coral_nexus/<run_id>/cs1_final.json`：出口 6 项逐条 verdict（§10.4）；`cs1_manifest.json` 聚合器；`maintenance/…`（每个 T 行 verdict 文件的 sha256 索引、known_deviations、open_items 表、`t_matrix` 场景列快照+校验）。
- **维护证据**：oracle/checker 的版本+hash、`t_matrix` 场景列快照、`perf_gates` 的 `pre_registration_sha256`、known_oracle_deviation（KD-01）、`not_empty_pass` 报告、T-59/T-60/N2/PS 的 deferred 原因+启动条件。
- **出口验证**：`cn.sh gate --stage G5 --out <dir>`【未来，待 W0-02 提供工具链后实现】：读入全部 `cs1_manifest.json`+replay/verify/bench 产物 hash，逐条重放 C1..C7 判据（不只看文件存在），核对 §5.4 列出的“允许 CS1_PY 的行”清单；输出退出码 0/1/2/3（§4）。

## 13. 给能力有限模型的手册

1. **只能写自己的任务文件**；先读 §2 依赖表，确认无环再动笔。
2. **每任务产出物落盘后必须跑自身验证条目**（如 `python -m pytest pipeline_validator/tests/test_w7_contracts.py -q`，或未来 `cn.sh unit --suite w7`），验证失败＝任务未完成，不得进 DONE。
3. **禁止**：修改 `pipeline_validator/*.py` 的业务逻辑（只允许新增文件/新增单测文件）、修改 W0..W6 各包产物/场景、引用未存在 hash、把 stub/伪 log 当 CS1 判定、用 Python trace.py 输出当 CS1 realbits 证据、ULP 放宽、写 “TODO oracle”、声称未来命令已执行。
4. **先失败后通过**：每次新写 checker/oracle/场景，先给至少一个负例断言（如 C6 断档、C3 双重回收、oracle NONE 行），再给正例。
5. **何时可标 done**：任务清单（§6..§12）的所有产物存在、所有验证条目 0 退出、相关 T 行在 `t_matrix` 里状态更新、无未关闭的 `defer_reason`（除 T-59/T-60/N2/PS 明示项）。
6. **接口缺失时**：在 `protocol/field_requests/w7.yaml` 登记所需字段，然后**向 W1-01 单点提请求**（不阻塞他人）；若阻塞自身则写 `BLOCKED(field_request pending)`，不自行造字段。

## 14. 风险、上报与待 parent/W1/W3/W4/W5/W6 决项

1. **KD-01（已报）**：累加级 ROD 溢出：固定 Spike/补丁版 MPACT→±Inf；Coral RTL（W3 §6.2/`fp_rounding.sv:96-104`）→max-normal；W3 已定 RTL 为 normative（parent 裁定 2026-11-05）。oracle 已实现方式：KD-01 登记为 ISA-reference known deviation，bit oracle 依 profile normative 判 DUT（不拿 Spike/MPACT ±Inf 作 DUT PASS/FAIL 依据）。**对 ISA oracle 的显式 profile arithmetic hook（parent 要求）**：在 `bitwise_*` 中保留 `acc_overflow_policy` 输入（normative/known_deviation 两档），由 `cs1_manifest` 与 `known_oracle_deviation.md` 强制显式选择；**不得**为凑一致而把该类向量降级 ULP 或永久 BLOCKED。
2. **mask 类语义**：固定 Spike/MPACT 无 mask；mask（vle/vtmm mask 输入、部分 mask）由 `bitwise_*` 覆盖（对齐 W3 §6.2/§8.1），`oracle_capability` 标 `BIT_ORACLE` 而非 `BOTH`。
3. **Nexus 私有指令/CSR（region helper、N-claim、页授权 CSR）**：固定 Spike 不支持（可证实）；W7-02c 通过显式 oracle 扩展（自定义 CSR/自定义指令模拟、独立 region FSM/页授权真值）承接，**不是 TODO**。
4. **T-39/T-40 P0 级语料**：`examples/scenarios/nest_subgraphs` 下 N06（event gen reuse）文件不存在，需 W7-04.4 构造 Tile 级等价场景并登记 WVER 引用（复用 spec §10 定义）。
5. **T-59/T-60 状态**：按 parent/需求裁定为 CS2/CS3 出口，W7 不在本轮跑；W7-06 维护 deferred 原因。
6. **T-49 证据层级**：Python 侧基线用现有 `TestRuntimeColdWarm`（可执行，不造假运行）；RTL 侧冷热并到 T-58（W7-04.7），避免重复测试。
7. **T-16/T-50 真字节来源**：现 `test_tile_indexed.py` 的 ByteStore fixture（`examples/workloads/indexed_gather_scatter.mlir`）为 Python 层真字节；CS1 realbytes 场景在 W7-04.5 由 W4/W5 提供真实 Gather/scatter kernel 侧字节。两层不混淆（C4 只信 CS1_LOG）。
8. **Perfetto 审计依赖**：`examples/artifacts/tools/trace_processor_shell` 当前在仓库内（已核实存在）；若 W0-02 容器中不携带，`verify` 判 `BLOCKED(trace_processor unavailable)` 并要求 W0-02 补——**不静默 skip**。
9. **W7-02a 能力探针执行依赖**：容器内需要 spike_simulator 构建产物；若 W0-02/W0-01 的构建矩阵缺项，`oracle_capability` 写 `BLOCKED(infrastructure)` 且 W7-04 相关行 BLOCKED；不冒 PASS。
10. **W7-05a replay 驱动依赖**：复用 `sw/coralnpu_sim/spike_cosim_dpi.cc` 的 DPI 模式不可行（需 VCS）；W1-03 的 `libcn_cosim.so` bp API 是唯一重放驱动入口（W1 已在 Plan 给出 `bp_begin/bp_layout/bp_drive/bp_tick/bp_collect/bp_status/bp_destroy` exports）。

## 15. 本包任务数汇总

- W7-01：7 产物文件（schema+Python validator+`t_matrix` 生成）+ 2 验证正/负例组。
- W7-02：5 子任务（02a..02e）、4 产物组、6 步骤、6 不变量。
- W7-03：7 checker+3 支撑件，逐 checker 正/负例。
- W7-04：8 批次（含 build-only 批次与登记批次），68 行均落位。
- W7-05：3 类（replay/负例拒绝/Perfetto 审计 + P0/P1/PN bench），全部带因果控制负例。
- W7-06：最终出口 6 判据维护+维护证据文件。

**总计任务/子任务数：7+5+10+8+3+1 = 34 任务级条目（不含逐 T 行落位）。**
