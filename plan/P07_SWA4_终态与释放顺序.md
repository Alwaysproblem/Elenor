# P07 · SW-A4 终态时序与安全释放

状态：通过（2026-10-31，见 [evidence/P07.md](evidence/P07.md)）
前置：P04通过。P05/P06没有语义前置关系，但会改shared_am.py/tile.py，不能并发写；默认依索引顺序执行。
对应原方案：SW-A4（[设计 v0.5](../design/proposal/BOA_Gemmini_Integration_Design.md) §1.3、§3.3、§3.5、§4.3、§7.2 SW-A 表）；Q07（§8.1b）；总则 §3.3。

## 1. 目标与不做

目标：修正成功链终态记录时序，使每个成功 FINAL 满足偏序 `compute_done ≤ step_committed ≤ output_visible ≤ slot_free ≤ chain_done ≤ UCE event`；失败/取消终态在 “quiesce → 回收 → 发布终态” 之后记录；终态环仍是有界诊断环（可覆盖、计溢出），不承载 correctness。补齐运行时偏序校验与 trace 证据。

不做：不改事件硬件语义（P15）；不改 WAIT/await 协议、不加源级 CFG/`await_any`；不改 commit/output/drain 的阶段划分与周期数；不改 Q09 场景本身的顺序（只允许其断言字段名随终态记录迁移更新）。

## 2. 前置与输入

- P04 evidence（per-Tile snapshot、Q09 永久回归的位置）。
- 设计 §3.3 状态机（FINAL_READY→DRAINING→slot_free→FREE、FINAL 受理置 final_issued）、§3.5 结果接收与 commit、§4.3 事件/释放合同（事件在 release 之后）、§8.2 不变量（`slot_free 时无待处理步骤/在途 AM 访问/未完成 drain`）。
- 总则 §3.3（成功 FINAL 偏序、非 FINAL 完成为 step_committed 不写 L1、故障先结清再发布失败终态、旧响应不污染新 generation）。

## 3. 当前证据与修改文件

现状（符号定位，行号以当日为准）：

- `pipeline_validator/runtime/shared_am.py::commit_step`：FINAL 提交即 `self._record_terminal(lease, seq, "chain_done", True, "", cycle)`——早于 output_visible 与 slot_free（总则点名的 SW-A4 问题）。
- `shared_am.py::release_final`：`FINAL_READY → DRAINING` 后立刻 `_free_slot(lease, cycle, "chain_done")`；`_free_slot` 释放 slot、`_drop_idle_bundle`、`slot_free_count += 1`、trace `am_release`、`_grant_pending(cycle)`。
- `shared_am.py::abort_lease`：先 `_record_terminal(..., "fault", False, ...)` 再 `_free_slot`（失败终态早于回收）。
- `pipeline_validator/runtime/boa_result.py`：阶段推进 `commit_cycle → output_cycle → drain_end → _finish(event_cycle)`；release 回调由 `tile.py` 在 drain 后调用（实施时以 `advance()` 实际顺序为准核对并写入 evidence）。
- `shared_am.py::AmTerminalRecord`：字段 `(lease_id, owner, chain_id, generation, seq, kind, success, detail, cycle)`，`terminal_capacity=256`，溢出丢弃最旧。
- 测试：`test_am_slot_reuse_increments_generation_and_keeps_terminal`、`test_ir11_typed_boa_commit_is_after_compute_and_drain`、Q07 两个背压用例。

修改shared_am终态记录位置、必要结果服务/trace接线与相关测试；优先复用BoaResultService现有阶段记录，不为诊断另建事件回填API或第二份可变状态表。

## 4. 输入输出、接口与不变量

- FINAL accepted即在begin_step/受理处置final_issued，保持原语义；commit只置FINAL_READY/更新AM，不写成功终态。成功terminal在实际slot/quota回收后记录；不得把final_issued延到commit。
- 偏序审计通过既有结果阶段记录、am_release、terminal和UCE事件按完整身份关联；确缺slot_free时刻才最小补记录/trace字段，不在UCE消费事件后再回填manager终态，诊断环溢出不影响正确性。
- 失败/取消：`abort_lease`/`abort_owner`/reset 路径改为 quiesce（在途访问/引用结清）→ `_free_slot(reason="fault"/"reset")` → 记录失败终态；失败终态的 cycle ≥ 回收完成 cycle。
- 非 FINAL 步骤：只有 `am_commit` trace（step_committed），不产生终态记录、不写 L1（现状保持并补断言）。
- terminal_capacity沿现有可配置有界环，满覆盖并计溢出；旧事件依靠独立EventTable保留，不依靠诊断环无限保留。测试按实际配置容量边界，不能把默认256钉成协议上限。

## 5. 实施步骤

1. 列出已有结果记录/trace能提供的时刻，只补缺失的释放/终态身份字段，不无条件新增四个None字段。
2. 删除FINAL commit处终态；在free的线性化点清region/quota后、重新评估pending之前记录旧链成功terminal，随后允许wake新owner。同cycle顺序可用trace序号核验，不人为插空拍。
3. fault/reset/abort先确保在途访问已结清，再free再发布失败完成。诊断环reset是否清空不影响正确性事件；不得在发布给等待者前把唯一失败记录清掉。
4. 复用既有完成通知，不新增event回填API；join trace/结果记录验证event不早于terminal/free。
5. 运行时校验：`release_final` 前断言 “无待处理步骤、无在途 AM 访问、drain 已结束”（引用现有 BoaResultService 状态）；`_finish` 事件调度处断言 `event_cycle ≥ slot_free_cycle`。
6. trace：`am_release`、`boa_output`、UC 事件 trace 已含各时刻，补一条 `am_chain_terminal` trace（终态发布时刻 + 全字段），供 `shared-am-cycle` target 机检。
7. 测试更新与新增：
   - 背压拉开compute/commit/output，断言规范的非严格偏序≤；free/terminal/event允许同cycle，另核其顺序，不强迫严格递增。
   - `test_am_slot_reuse_increments_generation_and_keeps_terminal`：终态 cycle ≥ slot_free、generation 不变断言保持；
   - Q07 两用例：恢复自主写回→释放→终态→事件顺序断言；
   - 失败终态：fault 注入后失败终态 cycle ≥ 回收完成；旧响应不进新 generation（引用 P04 隔离用例补跨 Tile 断言）；
   - P04 的 Q09 回归：仅更新断言引用的字段名/来源，顺序语义不变。
8. 周期重录：typed 样例 `--json --report` 与 `--trace-json` 重跑，终态字段进入报告；旧报告数字保留历史标签。
9. 交付可直接pytest/CLI运行的偏序/背压/失败smoke；P09统一注册shared-am-cycle，P07无需P02环境即可验收。
10. 全量回归 `pre-commit` + `pytest -q -n 4`。

## 6. 验收表

| Case ID                 | 构造                                        | 必须观测                                                                         | 失败特征                   |
| ----------------------- | ------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------- |
| S-P07-1 成功偏序        | 背压拉开阶段的 FINAL 链                     | commit ≤ output_visible ≤ slot_free ≤ chain_done ≤ event 全部成立且有 trace 证据 | chain_done 早于 slot_free  |
| S-P07-2 事件不早于终态  | 正常与背压运行                              | UCE 事件 trace ts ≥ chain_done 发布 cycle                                        | 事件先于终态发布           |
| S-P07-3 失败收尾顺序    | 输入/AM 访问注入 fault                      | quiesce→回收→失败终态；终态 cycle ≥ 回收；无旧响应泄入新 generation              | 先发终态后回收             |
| S-P07-4 非 FINAL 无终态 | 多 step 链中途快照                          | 只有 step_committed trace；无 L1 写；无终态记录                                  | 中间步骤发布终态           |
| S-P07-5 环有界          | 配置小terminal_capacity后产生超过容量的完成 | bounded环/overflow正确，旧程序事件仍可观察，不因环满挡grant                      | 默认值字符串测试或环满死锁 |
| S-P07-6 Q07 恢复        | 背压解除                                    | 无 UCE 轮询下自主 写回→释放→终态→事件                                            | 需要固件动作才恢复         |
| REG-P07-7 全量 gate     | 步骤 10                                     | exit 0                                                                           | 终态断言散漏导致旧测试红   |

## 7. 可执行命令

- `conda run -n elenor-validator python -m pytest pipeline_validator/tests/test_boa_am.py -q`；全量 `pytest -q -n 4`。
- `bash examples/run.sh boa-am-matmul --json --report /tmp/p07/boa-am.report.json --trace-json /tmp/p07/boa-am.trace.json`（用 trace 机检偏序；1 ts 单位=1000 cycles @1GHz，注意换算）。
- P09注册后：`hardware/scripts/check shared-am-cycle --out /tmp/p09/cycle --case terminal-order-backpressure`；当前任务先运行上述真实smoke与pytest。

## 8. 产物、交接与停止条件

交付终态位置修正、必要trace字段、偏序/故障回归及P07 evidence，给P09/P15/P19/P20交接完整身份与阶段记录。无法确定output/free时刻先修边界，不以估计时刻替代；不引入额外诊断回填API。
