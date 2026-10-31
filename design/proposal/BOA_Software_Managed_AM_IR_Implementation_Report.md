# 软件管理 AM 与 BOA Matmul IR 实施报告

- 状态：**已实现并验证硬件前置软件 gate；§3.4 源级控制流、低层 retry 与 RTL 明确不在本次范围。**
- 对应计划：[BOA_Software_Managed_AM_IR_Plan.md](BOA_Software_Managed_AM_IR_Plan.md)
- 当前产物（2026-10-31，P09 新合同基线）：`schema_version=6` / `compiler_abi="v6"`。新合同为 per-Tile AM（P04）、多 engine 固定 affinity（P05）、逐 MAC `bf16_fp32_mac_rne_v1`（P06）、终态偏序（P07）、FINAL epilogue/INT8/四转置/软件 im2col Conv（P08）。v5 及更早 artifact 一律拒绝。现状以 `plan/evidence/P09.md` 为准。
- **历史（S0，旧合同：Group 共享 AM + binary64 + 旧终态）**：本报告以下内容记录的是 S0 当时的 `schema_version=4` / `compiler_abi="v4"` 实施与数字，保留作对照。其中“INT8 明确拒绝”以及 IR07、§4、§5 的周期数字均已被 P06/P08 取代。

## 1. 已落地范围

| 计划项                          | 结果                                                                                                                                     | 主要文件                                                                                               |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| P0-01 源类型/操作/语法          | `!tile.am_buffer`、`tile.alloc memory_space="am" logical_slot/output`、四 operand typed `tile.boa.async`、独立 `tile.boa.estimate.async` | `pipeline_validator/dialects/elenor.py`                                                                |
| P0-02 源线性生命周期            | AM 定义/INIT/ADD/FINAL、seq、逻辑槽、output 绑定、AM 不可 free/load/store、缺少 FINAL 拒绝                                               | `pipeline_validator/workload_ir.py`                                                                    |
| P0-03 executable/codec/lowering | `ExecAmBuffer`、`ExecBoaMatmulDesc`、`ALLOC_AM`、typed/estimate lowering、v4 codec                                                       | `pipeline_validator/execution_ir.py`、`compiler/lowering.py`、`compiled_program.py`、`compiler/api.py` |
| P0-04 L1/AM 资源分离            | AM 容量与每 context live 峰值检查；AM 不进入 L1 Arena/Frame；`boa_am_slot_capacity_bytes` 等配置进入 target fingerprint                  | `compiler/resources.py`、`config.py`、`hardware_config.yaml`                                           |
| P0-05 独立 executable verifier  | 独立复核 AM 声明、typed operand、seq/mode、L1 生命周期、容量与 finish 条件；不信任 source verifier                                       | `pipeline_validator/execution_verifier.py`                                                             |
| P0-06 动态 AM manager           | 8 物理槽、每 context 2 live lease、固定 engine affinity、generation、terminal record；一次性申请队列、bundle 预留、FIFO 唤醒             | `pipeline_validator/runtime/shared_am.py`                                                              |
| P0-07 UCE 接线                  | `ALLOC_AM` 一次性申请 + `WAIT_AM` 阻塞 + slot_free 事件唤醒、typed launch、提交回调、fault/reset 取消/回收                               | `pipeline_validator/tile.py`、`tile_group.py`                                                          |
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
| IR09 | 通过 | 两链可表达；未授予的申请挂起且槽/generation 无变化；超出 bundle 的申请判为合同错误而非等待                                                                         |
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

另以 Python 直接比对 C 字节与独立 oracle（f32 与 BF16 两条路径），并检查 AM lease 归零、terminal record、BUSY 无副作用、Tile0/Tile1 同 ctx0 可同时获得 lease 且 fault 只隔离本 tile，以及 `BoaResultService` 的 blocked-output recovery：`test_ir14_final_output_backpressure_defers_release_and_event` 与 `test_ir14_runtime_final_output_backpressure_recovers_without_polling`。

### 4.1 FINAL 已受理、尚未释放时的同槽复用修正

`request()` 不再把物理配额满或同 logical slot 的任意存活 lease 一律视为永久错误。
同 logical slot 的旧链未受理 FINAL，或未结束链数已达到 bundle 上限时，仍拒绝申请；
旧链已经受理 FINAL、但仍在计算／输出／drain 时，新申请只挂起一次。
`_grantable()` 在首次申请和释放驱动的授予中都检查物理配额及 logical slot，直到旧链
`slot_free` 才授予新 lease，不能因还有其他预留空槽而提前复用同 logical slot。

- 回归 `test_am_same_slot_reuse_waits_for_final_release`：quota=1 与 quota=2 两种配置，
  修复前均失败，修复后通过；覆盖配额满、同槽仍占用，以及有额外预留槽时也不得提前授予。
- 本次 AM 测试：`32 passed`。
- full_memory runtime smoke：两条数值 Matmul 连续复用 logical slot 0，第二次 allocation
  放在第一条链 FINAL 之后、其 await 之前；物理槽数／quota 均为 1，注入输出背压至 cycle 3000。
  实测完成于 5451 cycles；2 次申请、2 次授予、1 次挂起、1 次唤醒，fault／pending／live／reserved
  均为 0；第二条链写出的 C 与独立 f32 oracle 逐字节一致。
- 修正后统一门禁：`pre-commit run -a` 全通过；
  `python -m pytest pipeline_validator/tests/ -q -n 4` 为 `579 passed, 1 skipped`。

## 5. 已知边界

- `tile.boa.async` 只在 `full_memory` 下执行数值；`timing_only` 不读取/写入 tensor 字节。
- AM manager 当前按一个 Tile Group 的 8 个动态槽建模；物理 group/slot 选择是设备侧确定性首空闲策略，软件只给 logical slot。lease owner 身份包含物理 tile_id、context_name/launch_generation、hardware ctx_id 和 epoch，避免不同 tile 的同 ctx_id 被误判为同一 owner；`abort_owner` 只影响同一物理 tile 的 lease。
- BOA 结果服务按 per-tile 有限 FIFO、O+R credit、commit 阶段和 L1 output credit/drain 建模；`set_output_blocked(..., until_cycle=...)` 是显式下游背压注入，不是隐式超时。
- split-K combine 的 EVU `add` 仍是 timing descriptor；其 `%acc` 由 q0 load 初始化，C 字节不代表 EVU 求和结果。本次不宣称 EVU/USE 整图数值正确。
- INT8 最终转换接口存在，但数值后端在量化 scale/clamp/异常值策略冻结前明确报错。
- AM 等待（2026-10-27 修订）：`ALLOC_AM` 只发一次；授予不了就挂入 manager 队列（无槽/配额/generation 副作用），context 进入 `WAIT_AM`，不再每次调度重发，由 `slot_free` 事件唤醒并直接拿到 lease。防死锁依据：每个 context 的 bundle = 其 Tile Program 静态最大同时存活链数，compiler 与独立 verifier 都要求 `<= boa_am_slots_per_context`(2)；首链授予时须整个 bundle 都能放进未被预留的空槽，余下槽为该 context 预留；后续链只用自己的预留或等待自己已发出 FINAL 的链 drain；新 context 申请 FIFO、不插队。所有未结束链都还需 UCE 指令才能 FINAL 时再申请，判为合同错误。验证：`test_am_request_parks_once_and_wakes_on_slot_free`、`test_am_bundle_reserves_continuation_and_prevents_hold_and_wait`、`test_am_continuation_waits_only_for_own_draining_final`、`test_am_cancelled_request_is_never_woken`、`test_runtime_alloc_am_blocks_once_and_wakes_on_slot_free`。
- 该进展论证依赖 §9.5 的其他前提（输入生产终会完成、ready 步骤与 drain 公平、最终输出通路可推进）；源级 TRY/BUSY 分支、RTL 和系统级动态资源争用进展仍是后续硬件阶段的前置工作。
