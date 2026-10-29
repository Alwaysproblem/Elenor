# P03 · C1 独立数值 Oracle 与数值合同冻结

状态：未开始
前置：P02 通过（需要锁定容器、Verilator、C 编译工具链；SoftFloat/HardFloat 源在容器内构建）。
对应原方案：C1（[设计 v0.5](../design/proposal/BOA_Gemmini_Integration_Design.md) §3.2、§7.2 C1 行）；总则 §3.2、§4.1。

## 1. 目标与不做

目标：在 `hardware/boa/reference/` 建立与生产实现完全隔离的独立数值 oracle：精确有理/整数运算求 `a×b+p` 后单次 RNE 到 FP32，按 tile_k 块做 AM 更新；用 Berkeley SoftFloat 3e 交叉校验，用 HardFloat primitive probe（Verilator 模型）在位级冻结 NaN 编码、tininess、符号零与 INT8/后处理舍入策略；产出 golden vectors 与冻结记录。**必须在 P11/P12 阵列存在前完成，禁止循环依赖完整阵列或以 DUT 作为 oracle。**

不改生产数值、不实现BOA阵列；本任务允许必要HardFloat primitive probe硬件/harness以冻结数值，不与P11完整PE混同。oracle不import生产模型，不引入numpy等生产依赖，不冻结wire编码。

## 2. 前置与输入

- P02锁定容器、可实际加载的SoftFloat libsoftfloat.so/headers和唯一pinned HardFloat源/构建引用；P03只消费，不重复获取第三方源码。
- 设计 §3.2（`bf16_fp32_mac_rne_v1` 伪代码、tile_k、mask 旁路、后处理顺序、INT8 建议），§1.4 第 7 条（HardFloat `MulAddRecFN`：RNE、tininess after rounding、不 FTZ、溢出为 Inf）。
- 总则 §3.2 数值合同、§5 “不能用 NumPy 容差”。

## 3. 当前证据与修改文件

生产现状（P06 将改，本任务只引用不改）：`pipeline_validator/boa_numeric.py::matmul_f32` 为 binary64 整 step 累加；`convert_output` 拒绝 i8；`_float_to_bf16_bits` 的舍入行为需与冻结转换向量对拍。仓库无任何独立 oracle。

拟新增（本任务 ownership）：

- `hardware/boa/reference/boa_reference/`：独立精确数值原语、块AM循环、epilogue；Conv golden用独立卷积索引循环，按声明的K展平次序/分块执行同一数值profile，不调用P08 im2col materializer生成答案。
- `softfloat_binding.py` 通过ctypes加载P02的PIC共享库libsoftfloat.so，定义明确的C ABI位型参数，不能加载静态.a。
- `probe/` 包含HardFloat primitive wrapper及Verilator harness：IEEE bits→recFN→MulAddRecFN/AddRecFN→IEEE bits转换完整，clock/组合settle按模块真实接口；不是把IEEE位型直接送recFN端口。
- `hardware/boa/reference/vectors/*.json`：golden vectors（bits 以 hex 字符串存，含 seed、生成器版本）。
- `hardware/boa/reference/numeric_freeze.md`：冻结记录（总则 §4.1 点名要求本文件）。
- `hardware/boa/reference/cli` 入口：`gen-matmul|gen-epilogue|gen-conv|selftest|crosscheck` 子命令。

## 4. 输入输出、接口与不变量

- 输入：BF16 bits 向量、shape/mask/tile_k、step 序（INIT/ADD/FINAL）、epilogue 参数。
- 输出：每个贡献块、AM 提交、最终 C 的精确 bytes（hex）；epilogue 各阶段中间值可选导出（供 P08 对拍）。
- 有限数/舍入/特殊值类别按独立oracle、SoftFloat、HardFloat交叉验证。NaN payload编码可能因SoftFloat specialization不同：先核对其配置，与HardFloat实际编码策略显式对齐并留证，不能忽略NaN或事后随意改DUT结果。tail不额外MAC；tile_k变化只保证可能影响数值，选定反例须确实显示差异，不能要求所有输入都变。

## 5. 实施步骤

1. 实现精确原语：bf16→精确值（整数/分数表示）、`fma_exact` 后单次 RNE、`add_f32_rne`、符号零/subnormal/overflow 处理（不 FTZ）、`f32→bf16` RNE、`round_to_int_rne`、`saturate_i8`。
2. 组装 §3.2 matmul 语义：对每个 (i,j)，按 step 序；每 step 的 K 按 k 递增切 tile_k 块（末块 valid_k≤tile_k）；`p=+0`，块内 `p=fma_rne(a,b,p)`；INIT 首块覆盖 AM，后续块 `AM=add_rne(AM,p)`；FINAL 读 AM 进 epilogue。
3. SoftFloat 交叉校验 runner：随机+定向向量，三方逐 bit 对拍；不一致即停止并记录（不许放宽 oracle）。
4. HardFloat probe：在 P02 交付的 pinned berkeley-hardfloat 源上编译 f32 `MulAddRecFN` 的 Verilator 模型；扫描特殊值矩阵（NaN payload 输入组合、±0、最小/最大 normal、subnormal 边界、溢出），记录输出位型。
5. 冻结记录逐项写证据：HardFloat NaN编码、signed zero、subnormal、不FTZ、overflow、RNE/tininess；epilogue每阶段的舍入与NaN/ReLU/clip语义、BF16最终转换；INT8建议采用NaN→zero_point、±Inf→饱和端点，但须在本任务明确review并冻结，不把HardFloat FMA probe当成它自动决定INT8政策的证据。非法inv_scale/zero_point/clip在输出前拒绝。
6. 生成非对称/tie/specials/mask/多step/binary64反例，以及空工作：INIT且K=0初始化+0，ADD且K=0保持已有AM位型不执行一次多余+0；FINAL应用正常epilogue一次（仅bias开启才简化为C=bias，scale等开启仍须应用）；m/n=0无数据输出，不构造等last微块。初始C若已有显式合同才支持，禁止引入未请求输入模式。
7. 注册numeric target：selftest、crosscheck、probe-nan和向量重生成；全部真实运行。matmul/Conv向量含flatten顺序、tile_k、阶段开关、seed与generator hash。

## 6. 验收表

| Case ID                 | 构造                                           | 必须观测                                                        | 失败特征                          |
| ----------------------- | ---------------------------------------------- | --------------------------------------------------------------- | --------------------------------- |
| N-P03-1 RNE tie         | 精确落在两可表示值中点的 fma/add/转 bf16 向量  | 三方 bits 一致且按 to-even                                      | 与 SoftFloat 差 1 ulp             |
| N-P03-2 符号零/非规格化 | `(-0)+(+0)`、subnormal 进/出                   | 与 probe 位型一致，无 FTZ；`-0 + +0 = +0` 表冻结                | 出现 FTZ 或符号零规则含糊         |
| N-P03-3 溢出/NaN        | 大数溢出、`0×Inf`、NaN payload 组合            | 溢出→±Inf；NaN 编码与传播规则逐条冻结并记录                     | “NaN 行为待定”留白                |
| N-P03-4 尾块旁路        | valid_k不足且无效槽含0×Inf/-0                  | 与相同块边界、仅执行有效k的独立计算一致                         | 补零MAC污染；改分组却要求结果一样 |
| N-P03-5 块分组          | 已选出有舍入差异的定向输入，tile_k=4/8         | 两套golden确实不同，普通无差异输入也正确                        | 用任意相同结果误判实现失败        |
| N-P03-6 INIT/ADD        | 两 step 链（INIT 覆盖 + ADD）                  | INIT 不读旧值；块间 add_rne 一次                                | INIT 读了旧 owner 残留            |
| N-P03-7 INT8            | saturate/tie/zero_point 边界/NaN/Inf/非法参数  | 量化策略逐条冻结；非法参数被拒                                  | 策略与设计建议矛盾却未记录        |
| N-P03-8 binary64 反例   | 存在一组输入使 sum-then-round ≠ 逐 MAC profile | 反例向量入 vectors，两者 bytes 不同                             | 无法证明差异（反例缺失）          |
| N-P03-9 conv 参考       | pad/stride/dilation 组合小形状                 | golden bytes + 解析流量公式                                     | 无独立参考供 P08 对拍             |
| N-P03-11 空工作         | INIT/ADD的K=0，bias/scale组合，m/n=0           | INIT +0、ADD保留原AM、FINAL epilogue一次、空输出与无MAC合同冻结 | K=0误清ADD旧值或漏scale           |
| N-P03-10 确定性         | 同 seed 两次 gen                               | bytes 相同                                                      | 依赖随机未播种                    |

## 7. 可执行命令

执行本任务注册的 `hardware/scripts/check numeric --out /tmp/boa-p03/all --seed 20261028` 和 `hardware/scripts/check numeric --case probe-nan --out /tmp/boa-p03/nan --seed 20261028`。需要再生成时：`hardware/boa/reference/cli gen-matmul --seed 20261028 --out /tmp/boa-p03/vectors`；runner创建输出目录并保存三方日志/向量hash。

## 8. 产物、交接与停止条件

交付reference包、vectors、numeric_freeze.md、numeric runner及P03 evidence；第三方来源由P02唯一拥有。本任务原样记录已冻结INT8/NaN政策，P06/P08/P11消费。无法归因的三方差异保留最小复现并阻塞，不挑有利一方或放宽bits。
