# L2 显式共享与提前释放：分批执行索引

本目录把 [`L2_显式共享_PLAN.md`](../design/proposal/L2_显式共享_PLAN.md) 拆为三个**可验收的实施批次**。原计划是语义、边界条件和验证目标的依据；若实施时发现源码与计划不符，先核实源码并同步修订相关批次，不通过省略安全检查求通过。批次 I/II 的实现和验证已完成；批次 III 的独立 CLI 场景与 Perfetto/全量 corpus 验收仍待执行。

| 批次                | 执行文件                                        | 对应原计划                                                                                     | 完成门槛                                                                                                                                                     |
| ------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0：变更前基线       | [01 私有释放 §0](01_private_l2_release.md)      | 验证方案“先量化真实 workload 的容量等待”                                                       | 对 `run.sh list` 当前 Runnable workloads 收集 runtime/full_memory 两档的容量等待和输入哈希；不重复运行用户已报告的旧失败以“确认”它。                         |
| 1：私有 L2 提前归还 | [01 私有释放](01_private_l2_release.md)         | 第一步、第二步；第五步的 transaction hooks、私有 cancel/reset；第八步的物理占用及 extent trace | 不回绑的 padded extent、accepted transaction 的物理引用、原子 release、真实 final-free 后 FIFO 准入与安全 drain **一起**完成；私有路径的测试与实际场景通过。 |
| 2：显式只读共享     | [02 显式共享](02_explicit_l2_sharing.md)        | 第三步、第四步；第五步的共享 claim/容量分类；第六步第 1–3 条；第八步共享计数                   | 源 IR→编译/独立 load 验证→published backing→借用/释放→profile/故障闭合；共享路径的行为与负例通过。                                                           |
| 3：场景与全量验收   | [03 场景与验收](03_scenarios_and_acceptance.md) | 第六步第 4–7 条、第七步；第八步 trace/文档收尾；完整 CLI、Perfetto 与回归                      | 两种 fidelity 的四个端到端场景及当前全部示例入口已运行并审计，跨 profile 的完整 store→switch→load 顺序与共享单份物理 backing 都有新证据。                    |

批次 II 的 in-process ByteStore、runtime/full_memory trace、编译产物、
命令及 SHA-256 证据见
`examples/artifacts/l2-sharing-release/20260925T131034Z/batch-II/`。
`sharing`、`nest.publish`、`nexus.shared.ref` 已由当前 parser 支持；各计划中的草图
不是批次 III 的完整可运行场景。

## 必须保持的接口与交接

1. **批次 0 → 1：**在任何代码修改前取本次 workload 基线：从当前 `bash examples/run.sh list` 枚举 Runnable workloads（不是 NEST 名称），分别运行 `runtime`、`full_memory`，记录 `context_admission_wait`、wait reason、区间并集/makespan、WAIT_SLOT/WAIT_CONTROL_RESOURCE、source/config SHA-256。旧失败是已知现象；不拿过期 trace 作新实现证据。成果归档 `examples/artifacts/l2-sharing-release/<run-id>/`，含命令、输出与报告。
2. **批次 1 → 2：**L2 每个本地 allocation 的 whole-stripe padded span 永久不交叠；`commit_exact` 一次提交 backing 和 slack 的 exact units；`release_exact` 只归还原样提交的 units。`AllocationHandle.backing_id` 是物理字节域和最终归还身份，`allocation_id`/`ContextBufferOwner` 是逻辑视图身份。L1 布局、free-map、tile.free 不改。accepted L2 事务的 src/dst 引用直到 terminal 字节/owner 后处理和 ack 后才撤销；成功公共 release 与内部 cancel 的语义分开。所有物理归还只经 `_try_release_l2_backing`，先完整 preflight 后 mutation。
3. **批次 2 → 3：**生产者发布后只读；消费者只能引用同次 model 调用、同 Tile Group、同 L2 profile generation 的特定 producer submit callsite + slot。`DECLARED/BOUND/RELEASED/CANCELLED` 组成唯一 claim 状态，容量由 backing + slack 去重；尚未 submit/WAIT_CAPACITY 的 claim 也保活，不能用裸整数 refcount。消费者从 producer **成功完成**后开始，不能以 publish event 自创跨层 phase ABI。L2 改档等全部 root frontier 和旧 epoch 已物化对象排空；L1-only 改档不清 L2 sharing。新增 `ExecSharedInput`、`ExecPublishRequest`、导入 bind/发布动作及 codec/ABI 切换后才能落地可运行共享案例。
4. **终态交接：**每批采用自己文档的行为性测试与真实路径 smoke，不把测试通过当作动作时序证据。最终 `assert_l2_closed` 同时检查 backing、claims、views、pins、accepted transactions、slack 与守恒；终态泄漏先进入 controller 可见 fault，再推进 reset drain；cap 后独立 drain 到 DONE，否则 poison 并保留物理占用。不能交付中途可 overlap 但会 late write、错误 free 或死锁的状态。

## 覆盖与验收导航

| 目标                                       | 落点                                                                  | 可观察证据                                                                                                                                                               |
| ------------------------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R3-6：同 profile 私有输入释放即为 B 腾容量 | [01](01_private_l2_release.md)、[03](03_scenarios_and_acceptance.md)  | A input 的 `l2_extent_release` 与 B **port** `active_cycle` 同拍；B 的真实 prefetch 与 A `EVU:pow` 区间相交，A 尚未完成。不得拿 device admission 或当拍 DMA issue 代替。 |
| R3-5：跨 context 只读 W/X 扇出             | [02](02_explicit_l2_sharing.md)、[03](03_scenarios_and_acceptance.md) | W 只做一次 HBM→L2 prefetch（对照两次），X 不经过 HBM，B/C 输出逐字节正确；直到 C 的最后安全 release 才唯一一次 physical free。                                           |
| 不同 L2 profile 的隔离                     | [02](02_explicit_l2_sharing.md)、[03](03_scenarios_and_acceptance.md) | A HBM store 完成→A context 完成→L2 切档命令开始/完成（OPEN_ISSUE）→B admit→首次 prefetch；跨 L2 epoch 静态拒绝。                                                         |
| 边界与负例                                 | [01](01_private_l2_release.md)、[02](02_explicit_l2_sharing.md)       | compiled artifact 篡改拒绝、非法 release 零 mutation、accepted late access 不写新 owner、FIFO 同类队头不被绕过、reset/cycle cap 实际 drain 或隔离失败 poison。           |
| 可复现的最终产物                           | [03](03_scenarios_and_acceptance.md)                                  | 新一轮 CLI 双档矩阵、Perfetto SQL 与 raw JSON 审计、完整回归和文档更新；每条命令/退出码/输入哈希/compiled hash 可追溯。                                                  |

**范围不变：**不修改 HBM 通道交织、L1 释放语义或 device slot 调度；不做跨 run、跨 Tile Group、跨 L2 generation 的共享，不加 spill/自动重载/任意读写共享。源文档原样保留作为完整规格，三个文件只规定如何分步安全实施。
