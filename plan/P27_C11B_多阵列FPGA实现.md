# P27 · C11-B：VCU118 多阵列和矩形 FPGA 实现

- 状态：未开始。
- 前置：**P23、P26** 通过；P26已依赖单阵列板测P24与矩形P25。
- 依据：原方案 §6、§7.4、§8.1c F01/F02/F05、§9.1。
- 通用合同：[00](00_执行总则与共享合同.md)；索引：[README](README.md)。

## 1. 目标

在同一VCU118 harness上构建2/4阵列、方形/矩形配置，得到供P28最终板测的真实bitstream集合及post-route证据。扩展执行资源而不是复制四套私有AM；保持1 Tile、1 hart、本Tile总共8槽。

本任务不是ASIC PPA承诺。DSP/LUT/URAM与吞吐只报告真实综合/实现测量；原方案16×16×4的DSP数字是推导，不抄成结果。不同配置可用不同bitstream，不要求运行时改变物理阵列形状。

## 2. 必交配置

下列YAML由本任务在 `hardware/configs/` 新增，引用P26共同参数而非复制另一份ABI。每份保持默认tile_k=8、4 groups×2 slots、默认4contexts且可执行8context容量验收。

| 配置文件                    | arrays_per_tile | array_rows×array_cols | 用途                                                 |
| --------------------------- | --------------- | --------------------- | ---------------------------------------------------- |
| `vcu118_multi_array_4.yaml` | 4               | 16×16                 | V1主板级基线；两contexts各两链覆盖四engine           |
| `vcu118_multi_array_2.yaml` | 2               | 16×16                 | 2-engine对照；单engine超过两个live handles的可达验证 |
| `vcu118_rect_8x16_4.yaml`   | 4               | 8×16                  | 矩形板级F02，四transpose与尾块                       |
| `vcu118_rect_16x8_4.yaml`   | 4               | 16×8                  | 相反方向矩形板级F02，不只测长边一种方向              |

**为何包含2-engine板级构建：** least-live绑定、总8槽、4engines从空状态出发，不能合法期待某engine持有第3条链而其他engine都至少2条。F05“超过两个live handles”必须在1或2engines配置证实；2engine/8slots自然分配成4+4，既是多阵列又不伪造affinity。4engine配置继续证明真正四路并行。

若P23的baseline只支持4contexts，新增相应8context变体或重构成能在admission时选择1/2/4/8的容量配置；manifest精确标记静态容量与运行数。不得暗改compiler profile使其看似可支持8contexts。

## 3. 输入与修改范围

输入：P26 RTL/cycle/联仿完整证据，P25矩形微块输出量与skew/last合同，P23已通过的内存wrapper/时钟/约束，P24真实板测工具。检查这些输入的source/config hash来自同一可构建树。

修改 `hardware/fpga/vcu118/` 的配置与布局约束、上述YAML及 `fpga-synth` 的配置选择/报告解析；必要时扩展唯一BOA router/FPGA memory wrapper。不要重写allocator/固件，不能在板级配置直接指定链固定engine。

如物理实现要求给AmRouter加pipeline，更新P16 credit/互锁与P26 trace预期并重新验证；不能将布线寄存视作“纯物理无功能影响”。

## 4. 实施步骤

1. **参数贯通。** YAML→resolved config→Scala→manifest→validator target identity逐项比较；确认arrays数量、矩形行列、tile_m/n/k、profile、context和AM容量不丢字段。
2. **资源拓扑审计。** E份array/operand/result capture，唯一Tile allocator/ChainTable/events、4个AM groups；AmRouter为E到4路。metadata按8live handles配齐，不随E=1/2误收缩成2E硬上限。
3. **物理共享。** 独立group的RMW/读写端口真实存在；同group冲突由仲裁处理，不把4个array输出先全局串行化伪装安全。存储总量不得随engine数翻倍。
4. **矩形实现一致性。** 使用P25独立行列skew/transpose，检查选择的dataflow合法mapping、短边有效mask和首末结果路由。不能删assert后只看elaboration成功。
5. **同SLR布局。** 基线4array、router和4 AM group置同一SLR，检查资源位置与宽总线距离。若无法满足器件资源/布局，保留报告并停止；不得把共享AM拉到其他SLR同时仍宣称满足原合同。
6. **时序/容量迭代。** 对每配置完整synth/place/route。重点检查FP32 MAC、router扇出、URAM读后RMW与epilogue路径。加pipeline后按P11/P12/P16/P20/P25/P26重新验证受影响路径。
7. **DSP策略。** 默认使用已对拍HardFloat；若需要定制BF16 fused MAC降资源，先独立提出等价替换，重跑全部特殊值/RNE向量和RTL门；不得仅以资源更小替代数值正确性。
8. **报告比较。** 每配置导出实际LUT/FF/BRAM/URAM/DSP、WNS/TNS、频率、SLR分布、关键告警。区分相同频率的结构成本与不同频率结果，不将降低频率隐藏在性能改善中。
9. **干净复现。** 从无构建产物目录离线重跑至少主4engine配置；其余配置也须完整构建而非只参数推导。每个bitstream有自身hash和独立manifest。

## 5. 验收矩阵

| Case        | 构造与观察                            | 失败条件                                        |
| ----------- | ------------------------------------- | ----------------------------------------------- |
| F01-B-4     | 4×16×16完整实现，WNS≥0，明确器件/约束 | 无bitstream、负WNS、关键路径未约束              |
| F01-B-2     | 2×16×16真实构建，仍4组/8槽            | AM变成每engine2槽或总容量减半                   |
| F01-B-R1/R2 | 8×16与16×8各真实实现                  | 只改配置文件，未跑place/route                   |
| AM-SHARE    | 每配置综合网表的物理存储/端口         | 为每array复制private AM、共享adder吞吐被错算    |
| SLR-B       | array/router/AM在同一SLR              | 通过跨SLR宽结果总线规避拥塞而未满足约束         |
| RETIME-B    | 所有新增pipeline后的RTL/cosim         | tag错位、credit漏算在途数据、RMW hazard窗口过短 |
| CTX-B       | 8contexts容量和事件槽admission        | 静态事件容量不足但loader仍接受                  |

本任务只完成F01以及F02/F05的构建前提，数值和并行的B层通过属于P28。

## 6. 可执行命令

复用P23已实现的入口。配置可生成，但生成器与resolved内容都进入manifest；不能依赖shell手改Verilog参数。

```bash
hardware/scripts/check fpga-synth vcu118 \
  --config hardware/configs/vcu118_multi_array_4.yaml --out /tmp/boa-p27/e4
hardware/scripts/check fpga-synth vcu118 \
  --config hardware/configs/vcu118_multi_array_2.yaml --out /tmp/boa-p27/e2
hardware/scripts/check fpga-synth vcu118 \
  --config hardware/configs/vcu118_rect_8x16_4.yaml --out /tmp/boa-p27/r8c16
hardware/scripts/check fpga-synth vcu118 \
  --config hardware/configs/vcu118_rect_16x8_4.yaml --out /tmp/boa-p27/r16c8
hardware/scripts/reproduce /tmp/boa-p27/e4/manifest.json \
  --out /tmp/boa-p27/reproduce --offline
```

运行前补充P02要求的只读Vivado路径/license环境；不得在命令里提交license字符串。每个out目录必须包含真实完整工具报告及bitstream，不用一个main manifest掩盖其他配置未实现。

## 7. 交付、交接与停止条件

`plan/evidence/P27.md`逐配置列状态、bitstream路径/hash、时序与资源结果、对应firmware/ABI/输入能力、是否完成8context容量验证。交给P28明确的四个board targets；更新FPGA说明与配置表。

缺license/器件支持、任一必达配置布局/时序失败、数值等价回归未通过时不能关闭P27。可先保存已通过配置，但不得删除矩形或4array来缩小最终范围。PPA失败证据优于没有依据的性能保证。
