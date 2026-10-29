# P11 · C2-A：dataflow 选择与 FP32 PE

| 项   | 值                     |
| ---- | ---------------------- |
| ID   | P11（对应原方案 C2-A） |
| 前置 | P02、P03、P10          |
| 状态 | 未开始                 |

## 1. 目标与不做

**目标**：完成 dataflow 评审并冻结**一种**（WS 或 OS 之一，不要求两种）；在该映射下贯通 Gemmini PE 的 FP32 partial-sum 路径，使 PE 级 MAC 行为与 `bf16_fp32_mac_rne_v1`（§3.2）逐 bit 一致；用 P03 独立 oracle 的 golden vectors 在 PE 级对拍并留下可复跑记录。

**不做**：不实现完整 mesh/skew/drain 与不可暂停输出合同（P12）；不同时实现两种 dataflow；不改 tile_k 默认值 8（tile_k 是数值参数，任何改动走 P03/P06 的 fingerprint 与 cutover 规则）；不做 operand/transpose 适配（P13）；不建软件 RF/load 通知接口（已取消的需求）。

## 2. 前置与输入

- P02 通过：Chipyard/Gemmini/HardFloat/Chisel 6.5.0 锁定工具链可 elaborate（`hardware/gemmini/upstream/` subtree 与 `upstream.lock` 已由 P02 建立）。
- P03 通过：`hardware/boa/reference/` 独立 oracle 与 golden vectors 可用，NaN 编码、tie、signed zero、subnormal、overflow 特殊值已冻结；本任务直接消费这些向量，**C1 oracle probe 是本任务前置，不重复实现 oracle**。
- P10 通过：descriptor 中 shape/tile_k/transpose/acc_format 字段与容量字段已定义，PE 测试向量包装格式沿用其 schema。
- 原方案必读：§3.1（数据流与转置）、§3.2（逐 MAC FP32 RNE 契约全文）、§1.4 第 1/7 条（`spatialArrayOutputType=Float(8,8)` 未贯通、Gemmini `Float.mac` 用 HardFloat `MulAddRecFN`）、§7.2 C2、§7.4.4（DSP 预算）、§7.4.5（利用率推导）。
- 现有仓库证据（数值语义参照）：`pipeline_validator/boa_numeric.py:60 matmul_f32` 为 binary64 整步累加（SW-A3/P06 后按 profile 更新）；本任务以 P03 oracle 为唯一数值权威，不用该 Python 函数当 golden。

## 3. 必读文件与修改 ownership

- 必读：本文件、总则、`plan/evidence/P02.md`、`plan/evidence/P03.md`、`plan/evidence/P10.md`、上游 `PE.scala`、`Float` 算术与 `arithmetic.scala`（HardFloat 调用点）。
- 新增 `hardware/gemmini/src/test/scala/boa/BoaPESpec.scala` 与 `src/main/scala/boa/` 的适配器；必要上游PE改动直接编辑权威subtree并更新upstream.lock。测试不放src/main；配置为pe-vector-run.yaml。
- 注册并实现 `hardware/scripts/check` target `gemmini-pe`（本任务实现）。
- 不修改：`MeshWithDelays`/`AccumulatorMem`（P12/P16 范畴）；`pipeline_validator/**`。

## 4. 输入输出、接口与不变量

- PE 输入：BF16 A/B（精确扩展为 FP32，低 16 位补零）；输出：FP32 partial（不缩位）。每个输出位置按 k 递增 `p = fma_f32_rne(a, b, p)`；乘积不单独舍入，每 MAC 舍入一次；块起点 `p=+0`，块长 tile_k（默认 8），末块 valid_k≤tile_k。
- 无效 lane/k 不执行 MAC（真旁路），禁止 `fma(0,0,p)` 补零：`-0+ +0` 改变符号零、`0×Inf` 产生 NaN。
- 所选 dataflow 的块部分和位置：OS=PE 累加寄存器（每 tile_k 排空一次）、WS=列方向链（活跃 K 行链长，非活跃行旁路）；两种都必须能把本步 partial 交给共享 AM 更新路径（P16），未提交前不得声称 step_committed（该边界验收在 P16/P18）。
- FP32贯通须“类型/网表审计 + 真实PE逐bit数值”共同证明，保留c1/c2/propagate等正确性状态；不能仅assert一个输出类型就宣称路径正确。使用能区分BF16截断与FP32保真的反例穿过真实PE输出。
- HardFloat 语义（RNE、tininess after rounding、不 FTZ、溢出为 Inf）按上游实现；NaN 输出编码以 P03 冻结记录为准，本任务只对拍不重定义。

## 5. 实施步骤

1. **dataflow 评审矩阵**：对默认 shape 16×16×8 比较两种 dataflow 的：馈数拍数、四种 transpose 馈入路径（与 P13 接口预对齐）、填充/排空开销（§7.4.5：两者利用率上限约 50%）、FP32 路径改动点数量与 DSP 影响（§7.4.4 两种 MAC 方案）。输出选择结论 + 依据，冻结进 evidence；矩形能力只做映射可行性备注，实现归 P25。
2. **上游审计**：逐点列出 `Float.mac`、PE 累加寄存器、mesh 输出、`spatialArrayOutputType` 的当前类型与数据流；产出「FP32 断点清单」。
3. **贯通 FP32**：按清单修改（保留上游许可证与 `upstream.lock` 修改记录，可独立导出 diff）；elaboration 断言输出类型为 FP32。
4. **PE 级数值测试**：消费 P03 vectors——单 MAC（全部特殊值类）、tile_k 块序列（同位置 8 MAC 链）、尾块 mask 旁路、INIT 块覆盖语义中「块起点 +0」部分；每向量逐 bit 比对。
5. 交付真实PE→mesh接口（partial bits、valid lane、tag及块边界），字段采用P10 schema；本任务不留占位端口。最小不可暂停输出量由P12基于该可运行接口量化。
6. 在 `hardware/scripts/check` 注册 `gemmini-pe`（`--config hardware/configs/pe-vector-run.yaml --out DIR --seed N [--case ID]`，本任务实现），输出 manifest/summary：向量数、失败数、失败向量 hash。
7. 更新 `plan/evidence/P11.md`：dataflow 冻结记录、FP32 断点清单与修复、向量统计、上游 diff hash。

## 6. 验收表

| case | 构造                                      | 必须观测                                            | 失败特征 / 层                                   |
| ---- | ----------------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| PD01 | P03 单 MAC 全特殊值向量集                 | 每个 MAC 输出与 oracle 逐 bit 一致                  | 任一 bit 差（含 NaN 编码）即失败（R，RTL 仿真） |
| PD02 | 同输出位置 tile_k=8 MAC 链                | 块结果与 oracle 每块一致，非整步和一次舍入          | 双重舍入特征（binary64 习惯）即失败（R）        |
| PD03 | 尾块 mask + 0×Inf / -0 输入向量           | 无效 lane 不改变有效结果，无 NaN 污染               | 出现补零 MAC 伪 NaN 即失败（R）                 |
| PD04 | FP32路径审计 + 有BF16截断反例的真实PE执行 | 输出精确匹配FP32 oracle，低位信息保留，网表类型吻合 | 只改accType或仅类型assert（R）                  |
| PD05 | dataflow 评审记录                         | 一种映射冻结，含排空/馈数拍数与依据                 | 两种并存或无依据选择（F）                       |
| PD06 | 上游 diff 导出                            | `upstream.lock` 可独立解释全部修改                  | 隐藏 patch 即失败（F）                          |

反例边界：删除 square/doubleissue assert 不属于本任务（P12/P25 处理）；用 NumPy 容差比较替代逐 bit 即失败。

## 7. 可执行命令（契约；target 由本任务实现）

```bash
hardware/scripts/check gemmini-pe \
  --config hardware/configs/pe-vector-run.yaml --out artifacts/p11 --seed 1
# 底层等价（本任务交付的 runner）：
#   sbt "gemmini/testOnly boa.BoaPESpec"  精确 runner/类名按 P02 锁定 build graph
hardware/scripts/reproduce artifacts/p11/manifest.json --out artifacts/p11-repro --offline
```

预期：manifest 含 PD01–PD04 的真实断言与向量计数；PD05/PD06 为 evidence 审查项。工具链缺失时按总则记录阻塞，不伪造通过。

## 8. 产物、交接与停止条件

- 产物：dataflow 冻结记录、FP32 贯通修改（可导出 diff）、PE spec 测试、`gemmini-pe` target、`pe-vector-run.yaml`、evidence。
- 交接：P12 消费冻结的 dataflow 与 PE 排空接口；P13 消费馈数接口；P16 消费「partial→AM 更新」边界；P18 集成门核对 PD04。
- 停止条件：与 P03 oracle 有不可归因差异；需要改 tile_k 数值才能通过；FP32 贯通需要重写上游核心而未走 upstream.lock 记录。

## 9. 版本、cutover 与文档同步

- 本任务不改数值 profile/tile_k，故不触发 artifact cutover；一旦评审要求改 tile_k 或舍入点，必须回到 P03/P06 规则（fingerprint、严格版本升级、全部 typed fixture 重生成），不得在本任务内「顺手」改。
- dataflow 冻结后，原方案 §9.1「PE 数据流待 review」行的当前状态在 evidence 中更新（保留评审稿历史标签）；§7.4.5 的利用率推导数字保持「推导值」标注，不得改写为实测。
- 上游 Gemmini 修改全部经 `upstream.lock` 可导出；引用上游行为时以锁定 SHA 的源码为准，不引用浮动 master 文档。
