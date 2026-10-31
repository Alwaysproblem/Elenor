# P08 · C6-S 软件后处理与 im2col 软件 Conv

状态：通过（2026-10-31 独立复核补修后，822 passed / 1 skipped；见 [evidence/P08.md](evidence/P08.md)，证明边界见其 §4–§5）
前置：P06 通过（逐 MAC profile 与旧 artifact 拒绝机制落地；实施时以仓库实际 artifact 版本为基）；P03 冻结的 INT8/epilogue/零 K 语义与 vectors。
对应原方案：C6-S（[设计 v0.5](../design/proposal/BOA_Gemmini_Integration_Design.md) §3.2 后处理段、§7.2 C6 行）；总则 §3.2、§4.1、§5、§6。

## 1. 目标与不做

目标：把当前 “源码接受 i8、数值层显式拒绝” 的占位迁移为真实 INT8 输出支持；把 bias/scale/ReLU/clip 后处理参数以**全链方式**（dialect → execution_ir DTO → lowering → compiled artifact codec → loader/verifier → runtime 应用）接入 typed Matmul，只在 FINAL 应用一次；交付软件 im2col + typed Matmul 的 Conv 路径与展开流量记录，对照 P03 conv 参考。序列化新增字段触发又一次严格版本升级（以实施时仓库实际版本取下一版，不硬写 v6）。

不新增FP32/INT8输入、硬件Conv、per-channel quant或EVU数值功能。四转置的typed字段及软件数值路径必须在本任务/P09前全链可用，P13负责硬件实现，不得把软件能力缺口拖到不允许修改compiler的P13。

## 2. 前置与输入

- P06 evidence：现行 artifact 版本（以实际为准）、`boa_numeric` 新原语（`mul_f32_rne` 等）与 `boa_numeric_tile_k`/`boa_numeric_profile` 字段名。
- 消费P03实际冻结的epilogue舍入/阶段顺序、INT8 NaN/Inf政策和空工作规则，不预先把建议当已冻结。零K FINAL为epilogue(AM)：INIT为+0、ADD保持既有AM，只有其他阶段关闭时才简化为C=bias。
- 总则 §4.1：“P08 若新增序列化后处理字段，须再次按严格版本规则升级；不能向已冻结 ABI 静默加字段”。

## 3. 当前证据与修改文件

实施前历史快照（现行实现与复核结果见 evidence，不以本段描述当前状态）：

- `pipeline_validator/boa_numeric.py::convert_output`：`i8` 分支 raise "INT8 AM output conversion policy is not frozen"（本任务删除该占位）。
- `pipeline_validator/tests/test_boa_am.py::test_am_i8_output_is_accepted_by_source_but_rejected_numerically`（~626 行）：断言拒绝与本任务相反，需翻转为成功用例。
- `pipeline_validator/execution_ir.py::ExecBoaMatmulDesc`（286–304 行）：仅 a/b/output/am/m/n/k/seq/init/final，无 epilogue 字段。
- `pipeline_validator/dialects/elenor.py`：`tile.boa.async` op 的 attr 集；`pipeline_validator/compiler/lowering.py`：op → `ExecBoaMatmulDesc` 映射；`pipeline_validator/compiled_program.py`：DTO 序列化/校验/密封；`pipeline_validator/loader.py` / `execution_verifier.py`：加载期校验。
- 核对既有Conv/数据搬运入口是否可复用，区分estimate workload与真实字节im2col；不能因EVU有relu/copy名字就假定其写ByteStore。P08仅复用已证明有数值字节行为的通路。

修改 ownership：`dialects/elenor.py`、`execution_ir.py`、`compiler/lowering.py`、`compiled_program.py`、`loader.py`/`execution_verifier.py`、`boa_numeric.py`、`tile.py`（FINAL 应用点）、`examples/generators/`（Conv 生成器，拟新增）、`examples/workloads/`（Conv fixture，拟新增）、相关测试、`plan/README.md` 状态行。

## 4. 输入输出、接口与不变量

- 源attrs沿既有风格定义bias/scale/ReLU/clip/quant与A/B transpose，输出dtype以outs view为唯一来源，不另加重复acc_out。缺少阶段表示关闭；显式scale=1仍按冻结数值规则执行，不能仅按数学恒等省掉可能改变特殊值的操作。
- epilogue是**授予时不可变链配置**：若源语法附在FINAL上，lowering提前收集并写入AM allocation的executable链配置，所有STEP引用同一配置；静态拒绝中途变更。bias数据若为内存view必须有真实读取/生命周期；常量bias需计descriptor/mailbox容量，不能隐含无限向量。
- DTO/codec同时承载上述链配置、typed transpose及必要view信息；全链parser/verifier/lowering/codec/loader/runtime一致，新增字段升级到实际下一ABI，不静默扩格式。
- FINAL commit后才应用epilogue一次；中间不缩位/量化/激活。参数非法在副作用前拒绝，K=0仍按冻结规则初始化/保留AM后正常后处理。
- Conv：软件路径 = 真实字节 im2col materializer + typed Matmul + 可选 epilogue；不新增隐藏 conv op。**im2col 展开只允许经由真实字节服务**（`tile.load.async`/`tile.store.async` 的 MFE 事务或 Gather）完成：逐窗口 load 出重叠 patch、`tile.store.async` 将各 patch 行写入 im2col L2 视图区、再整体 load 进 matmul A 的 L1 缓冲（或生成器选择 Gather 按行组装，同为字节真实路径）。**EVU 不承载 im2col 字节**：`EVUEngine` 只计周期不写 ByteStore，timing-only 的 EVU copy 会让 matmul 读到从未写入的字节；若生成器用 EVU 做纯计数/调度展示，必须与字节路径分离且不进入正确性判据。若实施中确需新的 view/transpose 类字段，必须按同一条全链标准交付（dialect→DTO→lowering→codec→verifier→runtime），不允许只在 runner 或 oracle 侧存在。

## 5. 实施步骤

1. `boa_numeric.py`：实现 `quantize_i8(values, inv_scale, zero_point)`（按 P03 冻结式）、`apply_bias`/`apply_scale`/`apply_relu`/`apply_clip`（f32 RNE 一次）；`convert_output` 接入 i8；全部与 P03 INT8/epilogue 向量对拍。
2. IR/dialect：`tile.boa.async` 增加 §4 attrs 与 parser/printer/verifier（静态非法即 parse/verify 失败）；更新 `IR_SPEC.md` 对应节。
3. DTO/lowering：`ExecBoaEpilogue` 定义与 lowering 映射；参数非法在 lowering/verifier 报错（错误信息含参数名）。
4. codec/cutover：`compiled_program` 序列化字段 + 版本升到实际下一版（以 P06 落地版本为基）；上一版 artifact 负例（明确版本错误）；重新生成受影响 fixtures；IR16 replay/tamper 用例更新。
5. runtime预校验放受理前，真正epilogue在FINAL AM commit之后、output服务写回之前执行；\_make_boa_completion只准备数值partial/回调，不能提前写ByteStore或绕过分阶段结果服务。transpose消费typed view/stride得到真实A/B，四组合与独立oracle对照。
6. 测试翻转与新增：i8 成功链（对 P03 i8 vectors）；saturate/tie/NaN/Inf 注入用例（构造含 Inf/NaN 的 AM 值路径：溢出向量输入）；`inv_scale=0/负/NaN`、`zero_point=200`、bias 长度错 → 编译/加载期拒绝且 ByteStore 无部分输出；各阶段开关组合矩阵。
7. 软件 Conv：
   - 生成小shape、pad/stride/dilation完整case，经真实MFE load/store或gather组装im2col；padding必须显式写+0来源，不越界读，也不假设新L1默认全零。固定channel/kernel展平K顺序，与P03独立卷积参考一致，再执行typed Matmul及epilogue。
   - `examples/workloads/soft_conv_*.mlir` fixtures（拟新增）；`examples/run.sh` 登记新场景名与 `fidelity-of`（typed 数值 → full_memory；**不得在 run.sh 场景里硬写 `--sim-override fidelity=...`**）；
   - 字节真实性验证：断言 matmul 的 A 输入缓冲 bytes == 独立 im2col 参考（P03 conv 参考的展开矩阵）；审计 IR/trace 中不存在仅计时不写字节的 EVU copy 充当展开路径；
   - 逐阶段核算有效patch读、padding零数据搬运、im2col写/重载和最终输出字节，对照独立解析计数；明确padding/stride造成的实际有效窗口，不使用未定义“理想直接卷积流量”下界。
   - 正确性：输出 bytes == P03 conv 参考 golden（`gen-conv`）。
8. 全量回归 `pre-commit` + `pytest -q -n 4`；注册软件层 target `soft-conv`（P02 分派器，`--case`：int8-chain / param-reject / stage-matrix / conv-pad-stride-dilation / conv-traffic）。
9. 文档同步：IR_SPEC、`Limitation.md` 中 “INT8 数值未支持” 类陈述更新为新合同；旧拒绝行为描述加历史标签。

## 6. 验收表

| Case ID                | 构造                                       | 必须观测                                                      | 失败特征                       |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------- | ------------------------------ |
| E-P08-1 INT8 全链      | BF16 输入 + i8 输出 + inv_scale/zero_point | 输出 bytes == P03 i8 golden；饱和/tie 边界一致                | 与 golden 位差；仍走旧拒绝路径 |
| E-P08-2 特殊值量化     | Inf/NaN与量化边界输入                      | 完整bytes符合P03冻结表，未冻结则阻塞而非猜策略                | 擅自假定NaN编码/量化政策       |
| E-P08-3 参数非法拒绝   | 三组非法参数                               | 编译/加载期明确报错；ByteStore 无部分输出                     | 运行期才崩或写了脏输出         |
| E-P08-4 一次性应用     | 两 step 链 + bias                          | ADD 步无 bias；FINAL 恰好一次；顺序==冻结顺序                 | bias 作用两次或顺序错          |
| E-P08-5 阶段开关       | bias/scale/relu/clip 组合矩阵              | 每组合 bytes == 对应 golden                                   | 组合间串扰                     |
| E-P08-6 零K            | INIT/ADD零K，bias/scale等组合              | INIT epilogue(+0)、ADD epilogue(旧AM)，各阶段只一次           | 误清ADD或漏scale               |
| E-P08-7 Conv 正确性    | pad/stride/dilation 组合                   | 输出 bytes == conv 参考                                       | 仅算术近似通过                 |
| E-P08-8 Conv流量       | pad/stride/dilation与重叠窗口              | 逐阶段字节等于独立解析计数，A矩阵也逐byte正确                 | 仅最终C偶合正确或padding隐式零 |
| CUT-P08-9 版本拒绝     | 上一版 artifact                            | 明确版本错误、非零；新版本 replay 通过                        | 静默兼容                       |
| AUD-P08-10 无 EVU 数值 | split-K 用例审计                           | EVU combine 仍非数值；BOA-only fixture 承担 bytes 证明        | EVU 数值实现混入               |
| E-P08-11 四转置全链    | 非对称strided A/B四组合，经artifact replay | typed字段可序列化、view取样与golden一致，改原descriptor不串扰 | 只有harness参数有transpose     |

## 7. 可执行命令

- `conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_boa_am.py -q -n0`；本任务固定注册alias `soft-conv-pad1-stride1` 后执行 `bash examples/run.sh soft-conv-pad1-stride1 --json --report /tmp/p08/conv.report.json`。
- 本任务注册 `hardware/scripts/check soft-conv --out /tmp/boa-p08/conv --seed 20261028 --case conv-pad-stride-dilation`。
- P03向量入口：`hardware/boa/reference/cli gen-conv --seed 20261028 --out /tmp/boa-p08/vectors`。

## 8. 产物、交接与停止条件

产物：INT8/epilogue 全链实现、版本升级、Conv 生成器与 fixtures、流量记录、`plan/evidence/P08.md`（含新版本号决策、golden 对拍摘要、流量表）。交接：P17（C6-H）以本任务的参数语义与 codec 为硬件 epilogue 合同输入；P09 汇总；P13 落地四种 transpose 时不得回退本任务的 ABI 规则。停止条件：P03 若有 INT8/零 K 项未冻结（numeric_freeze 缺条目），本任务相应 case 记阻塞，不自行编策略；完成后更新 README 状态。
