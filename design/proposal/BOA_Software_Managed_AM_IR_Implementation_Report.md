# 软件管理 AM 与 BOA Matmul IR 实施报告

- 状态：**已实现并验证硬件前置软件 gate；§3.4 源级控制流、低层 retry 与 RTL 明确不在本次范围。**
- 对应计划：[BOA_Software_Managed_AM_IR_Plan.md](BOA_Software_Managed_AM_IR_Plan.md)
- 当前产物：`schema_version=4` / `compiler_abi="v4"`；旧 v3 及更早 artifact 明确拒绝并要求重编译。

## 1. 已落地范围

| 计划项                          | 结果                                                                                                                                     | 主要文件                                                                                               |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| P0-01 源类型/操作/语法          | `!tile.am_buffer`、`tile.alloc memory_space="am" logical_slot/output`、四 operand typed `tile.boa.async`、独立 `tile.boa.estimate.async` | `pipeline_validator/dialects/elenor.py`                                                                |
| P0-02 源线性生命周期            | AM 定义/INIT/ADD/FINAL、seq、逻辑槽、output 绑定、AM 不可 free/load/store、缺少 FINAL 拒绝                                               | `pipeline_validator/workload_ir.py`                                                                    |
| P0-03 executable/codec/lowering | `ExecAmBuffer`、`ExecBoaMatmulDesc`、`ALLOC_AM`、typed/estimate lowering、v4 codec                                                       | `pipeline_validator/execution_ir.py`、`compiler/lowering.py`、`compiled_program.py`、`compiler/api.py` |
| P0-04 L1/AM 资源分离            | AM 容量与每 context live 峰值检查；AM 不进入 L1 Arena/Frame；`boa_am_slot_capacity_bytes` 等配置进入 target fingerprint                  | `compiler/resources.py`、`config.py`、`hardware_config.yaml`                                           |
| P0-05 独立 executable verifier  | 独立复核 AM 声明、typed operand、seq/mode、L1 生命周期、容量与 finish 条件；不信任 source verifier                                       | `pipeline_validator/execution_verifier.py`                                                             |
| P0-06 动态 AM manager           | 8 物理槽、每 context 2 live lease、固定 engine affinity、generation、terminal record、BUSY 无副作用                                      | `pipeline_validator/runtime/shared_am.py`                                                              |
| P0-07 UCE 接线                  | `ALLOC_AM` 非阻塞申请/重试、typed launch、提交回调、fault/reset 自动回收                                                                 | `pipeline_validator/tile.py`、`tile_group.py`                                                          |
| P0-08 BOA 数值/周期/自动输出    | BF16→FP32 Matmul、AM 分步 commit、f32/BF16 最终转换、compute/FIFO/commit/output-drain 分阶段、自动释放                                   | `pipeline_validator/boa_numeric.py`、`runtime/boa_result.py`、`engines.py`、`tile.py`                  |
| P1-09 调用方迁移                | 既有 timing 场景切 estimate；Matmul/split-K/gather 场景切 typed AM；generator/文档/输入数据同步                                          | `examples/**`、`examples/generators/**`、`pipeline_validator/IR_SPEC.md` 等                            |
| P0-10 软件 gate                 | `test_boa_am.py`、CLI smoke、独立 oracle、pre-commit、全量 pytest                                                                        | `pipeline_validator/tests/test_boa_am.py` 等                                                           |

## 2. IR01–IR17 验收状态

| ID   | 状态 | 证据                                                                                                                                                               |
| ---- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IR01 | 通过 | AM 声明不在 `l1_buffers`/`ArenaLayout`；`ALLOC_AM` 独立 opcode                                                                                                     |
| IR02 | 通过 | 64×256 f32 = 65536 B；容量 32768 拒绝，65536 通过                                                                                                                  |
| IR03 | 通过 | 首步 INIT 覆盖全部有效 AM 位置，最终 C 与独立 oracle 逐字节一致                                                                                                    |
| IR04 | 通过 | 非 FINAL 步只更新 AM，不写 L1 output；L1 output 仅在 FINAL 写入                                                                                                    |
| IR05 | 通过 | `accumulate output` 包含最后乘积；最终 f32 输出 oracle 一致，lease 自动释放                                                                                        |
| IR06 | 通过 | 单步 `output` fixture 由 IR03–05 同一路径覆盖（INIT+FINAL 一个槽）                                                                                                 |
| IR07 | 通过 | f32 直出保留 partial；BF16 只在最终转换；INT8 在数值后端明确拒绝（策略未冻结）                                                                                     |
| IR08 | 通过 | 重复 FINAL、跳/错 seq、AM 当 L1、AM free 均被 source/executable verifier 拒绝                                                                                      |
| IR09 | 通过 | 两链可表达；manager 第三 lease 返回 BUSY，槽/generation 无变化                                                                                                     |
| IR10 | 通过 | lease 的 bound engine 固定；一个 engine 现场可服务 3 个以上 handle                                                                                                 |
| IR11 | 通过 | launch 时原子预留有限 FIFO/O+R 槽；compute_done → commit → (FINAL) L1 output credit/write → drain → event；event 只在最终阶段触发                                  |
| IR12 | 通过 | lower 时冻结 typed views/config；输入最后访问后才允许 free                                                                                                         |
| IR13 | 通过 | 既有 await/旧 generation 隔离沿用；低层 retry 竞态明确延期                                                                                                         |
| IR14 | 通过 | 注入 L1 output backpressure 后 FINAL 停在 WAIT_OUTPUT/DRAINING，AM lease/quota 不提前释放；recovery 后自动写回、release、发事件，UCE 不 poll；terminal record 保留 |
| IR15 | 通过 | 无 FINAL 报错；正常路径 FINISHING/DONE；fault 路径 lease 回收                                                                                                      |
| IR16 | 通过 | v4 可 source-free replay；resealed 非法 seq/AM 绑定被拒                                                                                                            |
| IR17 | 通过 | 旧 operandless BOA 已清除；estimate 独立且不产生 AM/数值 C                                                                                                         |

## 3. T01–T22 对应关系

本次覆盖 T01/T02/T03/T04/T05/T06/T07/T08/T09/T10/T11/T12/T13/T14/T15/T17/T18/T19/T20/T21/T22 中与源语义、资源/数值/周期模型、事件和调用方相关的部分。

**明确延期、未验收**：T01/T13/T14/T16 中需要具体低层分支、TRY 状态、WAIT_ANY、check-and-register 或动态 retry 竞态的端到端部分；§3.4 源级受限控制流；RTL/Verilator/Gemmini 联合仿真；EVU/USE tensor 数值；INT8 量化策略。

## 4. 已执行验证

```bash
conda run -n elenor-validator pre-commit run -a
conda run -n elenor-validator python -m pytest pipeline_validator/tests/ -q   # 570 passed, 1 skipped (parallel -n 4)
conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_boa_am.py -n0
bash examples/run.sh boa-am-matmul --json
bash examples/run.sh matmul-splitk-pipeline --json
bash examples/run.sh matmul-splitk-multicontext-pipeline --json
bash examples/run.sh matmul-2048x512-boa256 --json
bash examples/run.sh matmul-pow-parallel --json
bash examples/run.sh matmul17-pow-tail-overlap --json
bash examples/run.sh gather-matmul --json
bash examples/run.sh gather-matmul-4tiles-2contexts --json
```

另以 Python 直接比对 C 字节与独立 oracle（f32 与 BF16 两条路径），并检查 AM lease 归零、terminal record、BUSY 无副作用，以及 `BoaResultService` 的 blocked-output recovery：`test_ir14_final_output_backpressure_defers_release_and_event` 与 `test_ir14_runtime_final_output_backpressure_recovers_without_polling`。

## 5. 已知边界

- `tile.boa.async` 只在 `full_memory` 下执行数值；`timing_only` 不读取/写入 tensor 字节。
- AM manager 当前按一个 Tile Group 的 8 个动态槽建模；物理 group/slot 选择是设备侧确定性首空闲策略，软件只给 logical slot。
- BOA 结果服务按 per-tile 有限 FIFO、O+R credit、commit 阶段和 L1 output credit/drain 建模；`set_output_blocked(..., until_cycle=...)` 是显式下游背压注入，不是隐式超时。
- split-K combine 的 EVU `add` 仍是 timing descriptor；其 `%acc` 由 q0 load 初始化，C 字节不代表 EVU 求和结果。本次不宣称 EVU/USE 整图数值正确。
- INT8 最终转换接口存在，但数值后端在量化 scale/clamp/异常值策略冻结前明确报错。
- 低层 retry CFG、显式 token、RTL 和系统级动态资源争用进展仍是后续硬件阶段的前置工作。
