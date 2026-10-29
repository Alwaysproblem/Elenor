# P06 · SW-A3 逐 MAC 数值 profile 与 artifact 迁移

状态：未开始
前置：P03 通过（冻结 vectors + 数值合同）、P05 通过（配置拼写与 fingerprint 字段状态）。
对应原方案：SW-A3（[设计 v0.5](../design/proposal/BOA_Gemmini_Integration_Design.md) §1.3、§3.1、§3.2、§7.2 SW-A 表）；总则 §3.2、§4.1、§5。

## 1. 目标与不做

目标：生产数值模型从 “binary64 累加整个 step 后一次舍入” 迁移为 `bf16_fp32_mac_rne_v1`：逐 tile_k 块、块内逐 MAC `fma_f32_rne`（从 +0 起）、INIT 首块覆盖 AM、后续块 `add_f32_rne`；`numeric_profile` 与 `tile_k` 进入配置与 fingerprint；完成统一 breaking artifact cutover（当前 v4/v4 起的下一版，实施时以仓库实际版本为准）并拒绝全部旧 artifact。生产实现**不得 import** `hardware/boa/reference`，与 oracle 的关系只通过 golden vectors。

不做：不实现 INT8 输出（P08）；不改 AM 协议/affinity/终态；不实现 EVU 数值 combine（split-K bytes 仍由 BOA-only fixture 证明）；不引入 numpy 等新依赖；不新增源级控制流。

## 2. 前置与输入

- P03：`hardware/boa/reference/vectors/*.json`（含 N-P03-11 零 K/零维组）与 `numeric_freeze.md`。
- P05 evidence：`boa_arrays_per_tile` 已入 fingerprint；拼写交接表（`tile_k`/`numeric_profile` 由本任务落地）。
- 阅读数值、空工作及严格版本合同。P06消费实际前序版本，不再假定自己一定是第一次升级；生产数值语义的breaking变更按原方案§9.1升级compiler ABI/schema并更新target fingerprint，所有生成/加载/校验入口一起切换。

## 3. 当前证据与修改文件

现状（符号定位，行号以当日为准）：

- `pipeline_validator/boa_numeric.py::matmul_f32`：binary64 全 step 累加、`k<1` 直接 raise（零 K 未定义）；`convert_output` 拒 i8（本任务不动）；`encode_f32/_float_to_bf16_bits` 舍入行为需对照 P03 转换向量核对。
- `pipeline_validator/tests/test_boa_am.py::_oracle`：与生产同构的 binary64 实现（P03 已判 “生产模型与 oracle 不能同构复用”）。
- `pipeline_validator/tile.py::_make_boa_completion`：整 step 调 `matmul_f32`，`lease.partial` 作为 ADD 输入。
- `pipeline_validator/compiled_program.py:986,1181`：`schema_version != 4 or compiler_abi != "v4"` 严格校验。
- `pipeline_validator/config.py` 映射表、`hardware_config.yaml engines.boa` 段、`execution_verifier.py::_HW_TARGET_FIELDS`。

修改 ownership：`boa_numeric.py`、`tile.py`（传 tile_k、块循环调用点）、`config.py`、`hardware_config.yaml`、`execution_verifier.py`、`compiler/lowering.py` 与 `execution_verifier.py`（零维/零 K 的 lower/静态校验）、`compiled_program.py`（版本 cutover）、全部 typed fixtures/artifacts、`tests/test_boa_am.py`（oracle 重写 + 版本负例）、`plan/README.md` 状态行。

## 4. 输入输出、接口与不变量

- 新配置 `engines.boa.numeric_profile`→boa_numeric_profile，唯一支持bf16_fp32_mac_rne_v1；`engines.boa.numeric_tile_k`→boa_numeric_tile_k，默认8，均进入fingerprint。设计中的tile_k映射到该专用键，不改WorkloadConfig.tile_k。改变块长可能改变bits，必须用已确认反例证明，不要求任意输入都有差异。
- 数值不变量（每输出 (i,j)）：按 step 序；每 step 按 k 递增切 `ceil(k/tile_k)` 块；块内 `p=+0.0f`，`p=fma_f32_rne(a,b,p)`；INIT 首块 `AM=p`（覆盖，不读旧值），其余块 `AM=add_f32_rne(AM,p)`（含同一 INIT step 的后续块）；尾块 valid_k<tile_k 只做有效 MAC；无中间缩位/量化。
- 空工作按P03冻结：INIT且K=0产生+0，ADD且K=0保留已有AM位型，FINAL仍做正常epilogue一次。m/n=0输出为空且不发mesh微块；选择最小合法lowering/执行表达，无源CFG，不允许用“skip”空实现丢掉需要的输出/事件/引用收尾。
- artifact cutover：升级到实际前序之后的新schema_version/compiler_abi，同时纳入numeric_profile/numeric_tile_k/几何target字段；更新compiler创建值、codec双入口、独立verifier、loader及所有fixtures。旧版本明确拒绝，新版但错误target也拒绝；这是两个独立负例，不以仅改fingerprint替代原方案要求的数值ABI升级。
- oracle 关系：测试侧 oracle 用独立实现（建议分数/整数路径，与生产代码结构不同），全部向量以 P03 golden bytes 为最终判据。

## 5. 实施步骤

1. 生产侧实现独立精确整数FMA/add及转换，按P03向量校验；matmul_f32按tile_k逐块求值。INIT的K=0返回+0，ADD的K=0原样保留partial，m/n=0返回定义的空输出；删除binary64路径，无隐藏fallback。
2. `tile.py` 完成回调传 `cfg.boa_numeric_tile_k`。**块更新的落点与合同**：逐块 AM 更新在 `_make_boa_completion` 的数值求值内部完成（`matmul_f32` 按块推进、INIT 首块覆盖、后续 `add_rne`），`lease.partial` 仍只在 step 边界被整槽替换一次；`shared_am.py::commit_step` 的整槽 payload 校验（长度==dims×4）与提交阶段划分不变，**不新增块粒度 commit API、不新增提交阶段**——设计 §1.3 映射表明确 “软件为 step 粒度，硬件为微块粒度”，块粒度可见性由 oracle 的中间值导出与向量承担，不进入 runtime 提交合同。禁止改成整 step 求和再一次相加。
3. 零工作全链审计source parser/dialect、workload verifier、lowering、DTO/codec、独立verifier、loader/runtime；当前shape>0限制若阻挡必须一致迁移。规范化空请求不进入mesh，K=0定义输出路径仍有正确完成/引用收尾；新增最小必要表达随本任务ABI升级，不只改lowering后让parser继续拒绝。
4. 记录真实wall-clock与数值向量规模，若有实现瓶颈再作等价局部优化；不设原方案没有的“必须≤2倍”性能门，也不靠float捷径、少算MAC或降低fidelity提速。
5. 配置与指纹：按 §4 加 `engines.boa.numeric_profile`/`engines.boa.numeric_tile_k` 与扁平字段 `boa_numeric_profile`/`boa_numeric_tile_k`（映射表 + `HardwareConfig` 字段 + `_HW_TARGET_FIELDS`）；非法 profile 报错负例。
6. 执行统一cutover并重生成全部受影响artifact/typed fixture，保留独立旧版拒绝与新版错target拒绝测试；source_hash必须来自权威generator/当前源，禁止手工改JSON版本或reseal旧数值产物。
7. oracle 重写：`test_boa_am.py::_oracle` 改为独立实现 + P03 向量消费；保留 “binary64 结果 ≠ profile 结果” 反例测试（引用 N-P03-8 向量），防止回归回旧实现。
8. 全量回归与文档同步：`pre-commit` + `pytest -q -n 4`；`pipeline_validator/IR_SPEC.md`/`README.md` 中数值与版本描述更新（旧数字保留历史标签）；注册软件层模块 target `numeric-profile`（P02 分派器，cases：vector-parity/zero-k/cutover-reject）。
9. typed 样例 bytes 终检：`boa-am-matmul`、`matmul-splitk-pipeline`（BOA 链部分）的输出 bytes 与 P03 对应向量逐 bit 相等；split-K 的 EVU combine 仍为非数值调度证据，不冒充 bytes 通过。

## 6. 验收表

| Case ID                   | 构造                                                         | 必须观测                                                              | 失败特征                    |
| ------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------- |
| V-P06-1 向量全对拍        | P03 matmul 向量（含特殊值/尾块/mask）经 full_memory 生产路径 | 输出 bytes 与 golden 逐 bit 相等                                      | 任一位差                    |
| V-P06-2 块长与fingerprint | P03已证明有舍入差异的定向输入，块长8/16                      | 两fingerprint不同，各自bits匹配golden且该反例确有差异                 | 指纹未变或不符合各自golden  |
| V-P06-3 块序语义          | k=64 链 8 块，中间 AM 导出                                   | INIT 仅首块覆盖；后续 add_rne；与 N-P03-5/6 一致                      | 整 step 求和再舍入          |
| V-P06-4 零K/零维          | INIT/ADD K=0、m/n=0                                          | 当前无epilogue时+0/保留AM/空bytes符合P03；bias/scale组合由P08/P09验证 | 依赖尚未完成P08或空工作挂死 |
| CUT-P06-5 旧版本拒绝      | P06前artifact按旧版本加载                                    | 明确版本错误，非零，不能自动升级旧数值语义                            | 仅更hash后继续执行旧版      |
| CUT-P06-6 指纹拒绝        | 旧 target_hash artifact                                      | `target fingerprint mismatch`                                         | 静默执行                    |
| V-P06-7 反例保留          | N-P03-8 输入                                                 | 旧 binary64 与现实现 bytes 不同的断言仍在                             | 反例被删                    |
| PERF-P06-8 性能记录       | 真实boa-am数值运行与向量规模                                 | 报wall-clock/配置，不放宽bits/fidelity                                | 无实测或偷降数值精度        |
| REG-P06-9 全量 gate       | 步骤 8                                                       | exit 0；报告新基线标签                                                | 旧数字混入新基线            |

## 7. 可执行命令

- `conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_boa_am.py -q` 先行，再全量。
- `bash examples/run.sh boa-am-matmul --json --report /tmp/p06/boa-am.report.json`，另存输出 C bytes 与 golden diff（evidence 存 hash 与首尾采样即可，完整 bytes 入 manifest）。
- 本任务注册：`hardware/scripts/check numeric-profile --out /tmp/boa-p06/vectors --seed 20261028 --case vector-parity`；与P03 oracle自检numeric不同，此处必须运行生产路径并比较独立golden。

## 8. 产物、交接与停止条件

产物：新数值实现与配置/指纹/artifact 版本、重生成 fixtures、`plan/evidence/P06.md`（含版本决策记录、性能对比、向量对拍摘要）。交接：P08 以实施时仓库实际版本为基加 epilogue 字段并按规则升版；P10 的 wire ABI 以本任务冻结数值语义为输入（软件 artifact 版本与 wire ABI v0 仍是不同版本域）；P11/P12 以同向量对拍 RTL。停止条件：若 P03 向量与生产实现在某向量上无法达成位级一致且归因不清（实现 bug 与向量 bug 无法区分），记录最小复现并交 P03 owner 复核，不得单方改向量迁就实现；完成后更新 README 状态。
