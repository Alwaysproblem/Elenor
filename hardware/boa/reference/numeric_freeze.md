# P03 数值冻结：bf16_fp32_mac_rne_v1

本文件冻结独立 reference 与后续 P06/P08/P11 的数值合同，不表示生产 validator 已切换，也不表示 Gemmini/BOA 阵列已实现。验收入口为 `hardware/scripts/check numeric`；完整运行与离线复现证据见 [P03 evidence](../../../plan/evidence/P03.md)。不冻结 descriptor、ISA 或 wire encoding。

## 1. 三个独立路径与证据边界

| 路径        | 实际实现                                                                                                                        | 比较对象                                                              |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| 精确 oracle | `boa_reference/scalar.py`，Python 整数与 `Fraction`；精确求值后单次 RNE                                                         | FP32 FMA/add/mul、FP32→BF16、FP32→i32 的 bits 与 flags                |
| SoftFloat   | P02 镜像内唯一 Berkeley SoftFloat 3e PIC `libsoftfloat.so`，RISC-V specialization；原生 ABI probe 后 ctypes 调用                | FP32 用真实库函数；BF16 使用下述独立适配器                            |
| HardFloat   | P02 pinned 源的 `MulAddRecFN`、`AddRecFN`、`MulRecFN`、`RecFNToRecFN(8,24,8,8)`、`RecFNToIN(8,24,32)`；Chisel→firtool→Verilator | IEEE→recFN→实际 primitive→IEEE；harness 保存实际输出，不做 NaN 后处理 |

SoftFloat 3e 没有 BF16 原生 API。本任务的 BF16 适配器将整数 significand 精确转为 FP32，乘精确的 `2^-16`，再调用 SoftFloat 的整数 RNE 决定目标 significand；整数处理 exponent carry、特殊值和 tininess。它不 import 精确 oracle，不冒充原生 BF16 库函数。BF16 第三方硬件参照为实际 HardFloat 格式转换器。

ABI probe 测量 `float32_t` 的大小/对齐、`uint_fast8_t`、`int_fast32_t`、`uint_fast32_t` 和 `bool`，而非假设 fast32 就是 32 位。锁定 Linux x86-64 镜像的观测分别为 4/4、1、8、8、1 字节。ctypes 简单整数返回 Python `int`。SoftFloat 状态在本构建中为 process-global，绑定只承诺单线程使用；每次运算重新设置 RNE/tininess-after 并清零 flags。

随机数完全由 seed 决定；特殊值笛卡尔矩阵含正负 qNaN/sNaN/payload、正负零、Inf、normal/subnormal 边界。原语三方比较同时检查 bits 和 flags，遇到第一个不一致即 FAIL 并保存输入与三方结果；不使用误差容限。`selftest.py` 中唯一的 Python float 运算是故意错误的 binary64 对照，绝不用于生成正确答案。reference 不 import `pipeline_validator`、NumPy 或生产数值函数。

## 2. 原语规则

- FP32 arithmetic：round-to-nearest, ties-to-even；FMA 的乘加只舍入一次，不拆成先乘再加。
- 保留 subnormal，禁止 FTZ/DAZ；RNE 溢出为同号 Inf。`bf16_to_f32` 是精确左移 16 位，不在这个位扩展步骤 quiet/canonicalize NaN。
- arithmetic 与 FP32→BF16 的 NaN 输出分别固定为正 `0x7fc00000` / `0x7fc0`。不保留输入 payload/sign。sNaN 置 NV；仅 qNaN 不置 NV；`0×Inf` 即使另一 FMA 输入为 qNaN 也置 NV；异号 Inf 相加置 NV。
- 相反符号精确抵消为 `+0`；`-0 + +0 = +0`；`-0 + -0 = -0`；乘法的零符号为异或；FMA 中同号的乘积零与累加零保留该符号。微小负值舍入为零时保留 `-0`。
- flags 为诊断数据：NV=16、DZ=8、OF=4、UF=2、NX=1。本任务无除法，DZ 不产生；不据此引入 runtime exception/trap ABI。
- FP32→i32 probe 使用 signed 32-bit RNE：越界或 Inf 为对应饱和端点，NaN 为 `0x7fffffff`，置 NV；合法非整数置 NX。HardFloat 的整数 invalid/overflow 合并为 NV，inexact 映射为 NX。这不是 INT8 特殊值政策的自动来源。

### 2.1 Tininess-after 不是检查最终结果的 exponent 字段

先按目标 significand 精度、**无界 exponent 范围**进行 RNE 来判断是否 tiny；再结合最终表示的 inexact 判断 UF。因此，最终编码恰好为最小 normal 仍可能有 UF。

| 操作/输入                     | 输出         | flags  | 精确解释                                                                                 |
| ----------------------------- | ------------ | ------ | ---------------------------------------------------------------------------------------- |
| mul `0x00800000 × 0x3f7fffff` | `0x00800000` | `0x03` | 精确值 `2^-126 - 2^-150` 在无界 FP32 精度下仍 tiny；最终 subnormal 网格舍入到 min-normal |
| mul `0x00800001 × 0x3f7ffffe` | `0x00800000` | `0x01` | 精确值 `2^-126 - 2^-172` 在无界 FP32 精度下已舍入为 min-normal                           |
| BF16 conversion `0x007f8000`  | `0x0080`     | `0x03` | 无界 BF16 精度下仍 tiny，最终编码 normal                                                 |
| BF16 conversion `0x007fc000`  | `0x0080`     | `0x01` | 恰在无界精度的进位 tie，to-even 到 min-normal                                            |

精确 oracle 用有理数在无界精度下再次计算 tiny 判定；SoftFloat BF16 适配器独立使用 subnormal FP32 significand 的整数阈值 `frac < 0x7fc000`。两者不共享舍入实现。

集成时曾错误地仅检查最终编码，三方对照真实失败。失败原件在 [tininess-before.json](../../../plan/evidence/P03/tininess-before.json)。归因核对了 pinned SoftFloat `s_roundPackToF32.c` 的 `isTiny` 判定，以及 pinned HardFloat `RoundAnyRawFNToRecFN.scala` 的 `unboundedRange_roundIncr` / `common_underflow`。修复的是独立 oracle 和 BF16 适配器的普适判定，没有修改 HardFloat 或 SoftFloat 输出；上述两侧边界均进入定向向量。

## 3. Matmul / AM / 空工作

输入为 BF16 原始位型；A 为 `[M, storage_k]`，B 为 `[storage_k, N]`，row-major。`MatmulStep(k,a,b,init,final,seq,storage_k=None,mask=None)` 的 seq 从 0 连续递增；仅首 step 为 INIT，仅末 step 为 FINAL。`matmul_reference(m,n,steps,tile_k=8,epilogue=None)` 返回 AM bits、每次贡献/提交、最终 C bytes 与阶段 trace。`epilogue=None` 表示关闭可选阶段并输出 f32，**不表示不输出**。

对每个输出单元：

1. 每 step 按原 k 下标切 `[0,tile_k)`、`[tile_k,2*tile_k)` 等贡献块，默认 `tile_k=8`。
2. 每个非空贡献块从 `p=+0` 开始，按有效 k 递增执行 `p=fma_f32_rne(a,b,p)`。
3. INIT 的首个非空块直接覆盖 AM，不读旧内容；同 step 后续块及 ADD step 的块各执行一次 `AM=add_f32_rne(AM,p)`。
4. FINAL 只在整条链末尾应用一次 epilogue。中间步骤不写最终 C。

`mask` 是 storage_k 长度的布尔序列。k 之后的 storage tail 必须 inactive；active 元素不 compact、不改变原贡献边界。无效 lane 不执行 MAC，整块 inactive 不执行 AM `+0`。oracle 内部用 poison NaN 证明 INIT 覆盖，但不增加公开的 initial-C 输入模式。

- INIT 且 K=0（或全 mask）将每个 AM cell 初始化为 `+0`。
- ADD 且 K=0（或全 mask）逐 bit 保留已有 AM，包括 `-0`。
- K=0 的 FINAL 仍正常执行启用的 bias/scale/ReLU/clip/conversion；例如 bias=2、scale=3 得到 6，不是 2。
- M=0 或 N=0 时仍校验参数，但无 MAC、贡献块或 C 数据，不等待不存在的 last 微块。
- `TensorResult.blocks` 含 `valid_indices`、各 MAC partial bits、提交后的 AM bits、FMA/ADD flags 和 INIT 标记；多字节输出均为 little-endian，C 为 row-major `[M,N]`。

分组反例：A 全 1，B 为 `[2^24,0,0,0,1,-2^24,0,0]`。tile_k=4 的 C 为 `0x3f800000`，tile_k=8 为 `0x00000000`；一次 binary64 step 累加的旧路径会得到 1。普通精确小整数控制组在不同 tile_k 下相同，不要求所有输入都变化。

真正的 binary64 双舍入反例另存于 primitives：FMA 输入 `a=b=0x3f800800, c=0x0d800000`，精确单次 FP32 RNE 为 `0x3f801001`，binary64 乘加再转 FP32 为 `0x3f801000`。selftest 实际运行故意错误的旧算法证明差异。

## 4. FINAL epilogue 与 INT8 的显式政策决定

顺序固定为 **每列 bias FP32 add RNE → 标量 scale FP32 mul RNE → ReLU → clip → 一次最终转换**。`None`/false 关闭阶段即不执行任何等价恒等运算；显式 scale=1 仍真实执行乘法，因此会 quiet sNaN 并置 NV。bias 必须恰有 N 个 FP32 words，跨行广播。

| 阶段        | 冻结规则                                                                                                               |
| ----------- | ---------------------------------------------------------------------------------------------------------------------- |
| ReLU        | NaN→canonical NaN，sNaN 诊断 NV；负有限数、-Inf、-0→+0；正值保持原 bits                                                |
| clip        | bounds 不能是 NaN，必须 lo≤hi，允许 Inf；严格比较，等于端点时保持输入 bits（包括 -0）；NaN→canonical NaN，sNaN 诊断 NV |
| f32 output  | 直接写当前 bits，无额外运算或 canonicalization                                                                         |
| BF16 output | 最后一次 RNE，包含 tie、subnormal、overflow、NaN 规则                                                                  |
| INT8 output | 下述 reviewed policy，不由 FMA probe 自动推出                                                                          |

INT8 冻结为：`z = mul_f32_rne(y, inv_scale)`，有限 z 用无界精确整数 `round_to_int_rne(z)`，再加整数 zero_point，最后饱和到 `[-128,127]`。不是先加 zero_point 再舍入，也不是先饱和再加。bytes 为 8-bit two's-complement；trace 分别保留 rounded、未饱和 biased、最终 after。

特殊值采用设计建议并在 P03 明确选定：**scaled NaN→zero_point，scaled +Inf→127，scaled -Inf→-128**。有限乘积溢出成 Inf 同样走端点。`inv_scale` 必须为有限、严格正的 FP32 位型；`zero_point` 必须是非 bool 的整数且在 `[-128,127]` 内。非法 quant 参数、无序/NaN clip、错误 bias 长度等均在产生任何输出前拒绝，包括空工作。参数与输出类型不匹配也拒绝。自检覆盖 ties、正负饱和、zero_point 两端、NaN/Inf、乘法溢出与非法参数。

## 5. 独立 Conv 与解析流量

`conv2d_reference(x,weights,shape,kernel,stride=(1,1),padding=(0,0,0,0),dilation=(1,1),tile_k=8,epilogue=None)`：x 为 NCHW、weights 为 OIHW；padding 顺序 top/bottom/left/right。stride/dilation 与 KH/KW 必须正，维度与 padding 非负。

`OH=max(0,(H+pt+pb-((KH-1)*dh+1))//sh+1)`，OW 同理。Matmul 行按 `(batch,oh,ow)` 展平，列为 output channel，K 按 `(ic,kh,kw)` 展平。`out_shape` 描述逻辑 `[N,O,OH,OW]`；**`c_bits/output_hex` 实际存储为 `[N*OH*OW,O]` row-major，不是 NCHW bytes**。

答案由直接卷积坐标循环生成，独立于导出的 A im2col 矩阵和 B 重排。空间 padding 是真实 +0 operand，会执行 MAC，故 padding 的 `0×Inf` 必须产生 NaN；这与 Matmul 的 masked/tail lane 旁路不同。K 分块与 AM/FINAL 规则同第 3 节。

设 `M=N*OH*OW`、`K=C*KH*KW`、`V=N*C*Σ(oh,kh)in_bounds_h*Σ(ow,kw)in_bounds_w`、`E=M*K`：

| 量                            | 公式                                                  |
| ----------------------------- | ----------------------------------------------------- |
| 有效 patch 读取               | `2*V` bytes                                           |
| padding 元素                  | `E-V` 个 BF16；写入 expanded A 的零为 `2*(E-V)` bytes |
| im2col A 写入 / Matmul A 读取 | 各 `2*E` bytes                                        |
| B 单次完整逻辑读取            | `2*K*O` bytes                                         |
| C 写出                        | `M*O*sizeof(output_dtype)` bytes                      |

这些是一次 materialization 的逻辑 payload，重复 patch 读取按展开次数计数；不计 cache/reuse、实际 burst、L1/L2 重叠、split-K 或额外多次读取，不冒充物理带宽/周期下界。每次参考运行还用独立坐标计数交叉验证 V；golden 保存解析与直接计数结果、A/B/C 的精确字节。

## 6. 使用、生成与后续消费者

```bash
# 推荐统一入口：锁定 P02 容器、断网、只读仓库，构建实际 primitive probe
hardware/scripts/check numeric --out /tmp/boa-p03/all --seed 20261028
hardware/scripts/check numeric --case probe-nan --out /tmp/boa-p03/nan --seed 20261028
hardware/scripts/reproduce /tmp/boa-p03/all/manifest.json --out /tmp/boa-p03/offline --offline

# 纯 stdlib reference 可使用已有 Python 3.11+，不安装任何包
hardware/boa/reference/cli selftest --seed 20261028 --out /tmp/boa-p03/reference
hardware/boa/reference/cli gen-matmul --seed 20261028 --out /tmp/boa-p03/vectors
hardware/boa/reference/cli gen-epilogue --seed 20261028 --out /tmp/boa-p03/vectors
hardware/boa/reference/cli gen-conv --seed 20261028 --out /tmp/boa-p03/vectors
```

可选 `--tile-k` 作用于生成器默认几何；明确测试 tile_k=4/8 的定向 case 仍保留自己的参数。统一 runner 的 `numeric.yaml` 支持 profile、tile_k、random_cases；配置、seed、fixture/source hash 全进入 manifest。`crosscheck` 子命令需要 runner 产出的 `--probe`、`--abi-json`，并消费镜像里的 SoftFloat 共享库，不用宿主库代替。

`vectors/{primitives,matmul,epilogue,conv}.json` 为 schema 1，含 profile、seed、generator_version、generator_hash、parameters、case inputs/expected。bits 采用 hex 字符串，最终 bytes 采用 little-endian hex；错误向量使用 `expected.error`，不把非法输入当成功数值。generator_hash 覆盖 package 全部 Python 文件及 cli 的路径和内容；probe 源与二进制另入 manifest。修改 reference 后显式重生成；`check numeric --case vectors` 只比较，不静默覆盖仓库 golden。生成器变化必须连同新 fixture hash 一起审查。

P06 消费逐 MAC/块提交与 FINAL 向量；P08 消费 epilogue、INT8、Conv 坐标/字节/流量；P11/P12 消费 primitive/PE vectors。当前 `check numeric` 不是“带 Gemmini RTL 的 validator”；完整阵列、AM 数据通路和生产数值切换仍由后续任务负责。
