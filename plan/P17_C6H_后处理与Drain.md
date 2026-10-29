# P17 · C6-H：硬件后处理与自主 drain

| 项 | 值 |
| --- | --- |
| ID | P17（对应原方案 C6-H） |
| 前置 | P08、P13、P15、P16 |
| 状态 | 未开始 |

## 1. 目标与不做

**目标**：实现 FINAL epilogue 与自主 drain：FINAL 步骤 commit 后按 P03/P08 冻结规则做一次后处理（per-column bias → scalar scale → ReLU/clip → 输出转换 f32/BF16/INT8），经有限 L1 output credit 写出到授予时准入的 L1 目的区域，output_visible → 旧访问 quiescent → slot_free → chain_done 事件；L1 输出背压时链停在 FINAL_READY/DRAINING，下游恢复后**无固件动作**自主完成。

**不做**：不做中间步缩位/量化/激活（只 FINAL 一次）；不做硬件 Conv/im2col（软件 Conv 归 P08 已冻结）；不实现 credit 基础设施与端口仲裁（消费 P16）；不改 P03 冻结的 NaN/INT8 特殊值策略（NaN→zero_point、±Inf→饱和端点以 P03 冻结记录为准，本任务不重新决定）；不建 UCE/FREE 命令路径（正常释放无软件 FREE）。

## 2. 前置与输入

- P08 通过：后处理语义、参数校验规则与序列化字段已冻结（f32/BF16/INT8 golden bytes 及 im2col 展开流量记录）；P03 通过：数值 oracle 与特殊值冻结记录。
- P13 通过：L1 写出端口/byte mask 约定；P15 通过：chain_done 终态发射点与 quiesce 判定；P16 通过：L1 output credit、drain 端口仲裁、五时刻边界。
- 原方案必读：§3.2（后处理建议顺序与 INT8 公式）、§3.6（FINAL 自动释放与阶段次序）、§4.3（output_visible/slot_free/chain_done 定义）、§7.2 C6-H、§8.1 T07/T19/T22、§8.1b Q07、§8.1c F04（板级背压注入的合同来源）。

## 3. 必读文件与修改 ownership

- 必读：本文件、总则 §3.2/§3.3、`plan/evidence/P03.md`、`plan/evidence/P08.md`、`plan/evidence/P16.md`。
- 本任务新建：`hardware/boa/src/main/scala/boa/epilogue/EpilogueUnit.scala`（16-lane bias/scale/ReLU/clip/转换）、`DrainController.scala`（拟新增）；测试 `hardware/boa/src/test/scala/boa/epilogue/`；配置 `hardware/configs/epilogue-vector-run.yaml`（P05 格式，拟新增）。
- 注册并实现 `hardware/scripts/check` target `boa-epilogue`（`--case` 用 P02 统一分派）。
- 不修改：P08 冻结的软件规则与向量生成器；P14–P16 模块；`pipeline_validator/**`。

## 4. 输入输出、接口与不变量

- 处理顺序（各阶段可关闭）：`FP32 AM 读出一次 → 每列 bias → scalar scale → ReLU/clip → 输出转换`；仅 FINAL 应用一次，INIT/ADD 中间结果不经过任何后处理。
- 输出转换：f32 原精度直出；BF16 单次最终 RNE；INT8 = `saturate[-128,127](RNE(y*inv_scale)+zero_point)`，inv_scale 有限且为正、zero_point ∈ INT8；NaN/±Inf 策略按 P03 冻结记录。参数非法（inv_scale 非正/非有限、zero_point 越界）在可观察输出副作用前拒绝。
- 释放次序（成功 FINAL）：`compute_done ≤ step_committed ≤ output_visible ≤ slot_free ≤ chain_done ≤ UCE event`；output_visible 只表示 L1 目的区域可见（L2 output_ready/HBM 可见是后续 store 的事）；slot_free 前无待处理步骤、在途 AM/输出访问。
- 自主性：写回由设备自动完成，不等 UCE poll/ACK/reap，不实现软件 FREE；L1 output credit 有限（P16），credit 不可得时停在 FINAL_READY/DRAINING 并持有 lease，quota 不提前归还；背压解除后 credit 到位即自主完成（Q07/T22）；输出受阻不挤占旧链输出空间。
- drain 数据量与端口：按 §6.2 核算新 partial 写、ADD 旧值读、写回与 FINAL drain 的分别流量；drain 端口公平性由 P16 仲裁保证，本模块不绕过。

## 5. 实施步骤

1. EpilogueUnit 数据通路：16-lane FP32 输入，各阶段旁路开关；与 P08 冻结参数字段（bias 向量、scale、clip 上下界、量化参数）对接；每列 bias 的列索引生成。
2. 转换单元：BF16 RNE（单次）、INT8 饱和+RNE+zero_point；用 P03/P08 golden bytes 逐 bit 对拍（含 tie、符号零、饱和端点、NaN/Inf 冻结行为）。
3. DrainController：output credit 取得→写出（byte mask、目的地址=授予时准入区域）→output_visible→quiesce 查询（P15）→slot_free（P14）→chain_done 事件（P15，必须在 slot_free 之后）。
4. 背压注入：credit 不可得/下游 blocked 注入；验证停留状态、quota 保持、恢复后自主完成、全程无固件动作（对照软件 `set_output_blocked(until_cycle=...)` 行为，只作语义参照）。
5. 场景测试：T07（bias 一次、INIT 不读残留联动 P14/P16）、T19（drain 中 context 返回→FINISHING）、T22、Q07、关闭/开启各阶段矩阵、非法参数拒绝矩阵。
6. 注册 `boa-epilogue` target；更新 evidence（golden bytes 对拍统计、背压恢复 trace、五时刻单调校验）。

## 6. 验收表

| case | 构造 | 必须观测 | 失败特征 / 层 |
| --- | --- | --- | --- |
| PB01 | P03/P08 f32/BF16/INT8 后处理向量集 | 输出 bytes 逐 bit 一致；各阶段关闭开关有效 | 双重舍入/缺旁路（R） |
| PB02 (T07) | bias 向量 + 多链 ADD | bias 仅 FINAL 一次；中间步无后处理痕迹 | 每步重复 bias（R） |
| PB03 (Q07) | L1 output credit 不可得 | 停 FINAL_READY/DRAINING、持有 lease、quota 不减；恢复后无固件动作完成并 slot_free | 提前释放或需轮询（R） |
| PB04 (T22) | 背压+持续 ADD 长时间 | 下游恢复且仲裁公平后最终释放；无泄漏 | 永久饥饿（C/R） |
| PB05 (T19) | FINAL drain 中 context 返回 | FINISHING 自主收尾，终态完整 | 依赖已返回 context（R） |
| PB06 | 非法参数（inv_scale≤0、zero_point 越界） | 输出副作用前拒绝 | 先写后报错（F） |
| PB07 | 五时刻校验（全部成功 FINAL case） | `compute_done ≤ commit ≤ output_visible ≤ slot_free ≤ chain_done` 单调 | 次序颠倒即违反 SW-A4 对应硬件合同（C/R） |

反例边界：把 NaN→zero_point 策略在本任务重新发明（偏离 P03 冻结记录）即停止条件；用「drain 吞吐足够」论证不需要 credit 即违反有限资源合同。

## 7. 可执行命令（契约；target 由本任务实现）

```bash
hardware/scripts/check boa-epilogue \
  --config hardware/configs/epilogue-vector-run.yaml --out artifacts/p17 --seed 1
hardware/scripts/check boa-epilogue \
  --config hardware/configs/epilogue-vector-run.yaml --out artifacts/p17-pb03 --seed 1 --case PB03
hardware/scripts/reproduce artifacts/p17/manifest.json --out artifacts/p17-repro --offline
```

预期：manifest 含 PB01–PB07 断言、actual/golden bytes 记录、背压恢复 trace；数值全为 full_memory 真实字节对拍。

## 8. 产物、交接与停止条件

- 产物：EpilogueUnit/DrainController、`boa-epilogue` target、配置实例、evidence。
- 交接：P18 集成门核对 PB03/PB07；P19/P20 联仿消费 drain/事件时序；P23 板级 F04 背压注入寄存器以本模块接口为准。
- 停止条件：golden bytes 不可归因不一致；自主释放需要软件参与；次序不变量无法在全部注入下保持。

## 9. 版本、cutover 与文档同步

- 数值权威是 P03/P08 的当前冻结记录；两者若因 cutover 更新（如 NaN/INT8 特殊值重冻结），本任务向量为最小重验集。
- 后处理参数序列化若需新增字段（P08 未覆盖的硬件专用参数），按 P06 确立的严格版本规则升级软件 artifact，不向已冻结 ABI 静默加字段。
- 原方案 §3.6 引用的软件背压 smoke（`set_output_blocked(until_cycle=3000)`，5451 cycles）保留历史证据标签；本任务背压 trace 是新的 RTL 层证据，两者不互替。
