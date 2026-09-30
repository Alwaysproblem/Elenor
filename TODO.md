<!-- 1. 当前的 BOA MFE 等各个模块并未提供大部分文档中的描述的功能，例如 BOA 由 reduce 模式，post scale 等等，MFE 并没有layout transformation等等
2. MFE 的queue 需要重新设计来保持 intra tile program 的IO pipeline

--- -->

## 限制

不可以改动 除了当前项目之外的任何文件，如果实在需要需要征求用户同意

## R2 需求 （已完成）

1. 当前的内存系统完全是摆设
2. 需要增加输入 nest.context, 还有需要将 prefetch，store load，都需要有 目标地址和原地址的，当前完全没有
3. 后续加入对于 HBM 的模拟等支持，gather
4. memory 以参考 /home/yongxiy/Desktop/multicontext 和 /home/yongxiy/Desktop/dockerVolumn/Elenor 的对模型的建模，对runtime的建模基本上已经符合我的预期了
5. 当前的 nest.release 没有对应的实现，具体实现思路为硬件等待所有tile.signal后再进行释放
6. tile.signal 并没有加入 context id 或者说 唯一的ID 让 L2 可以清楚的知道什么时候 进行内存的释放
7. 当前 IR 并没有对与 传入参数进行实质性的处理，也就是说，当前的 IR 并没有 输入参数的概念，需要加入输入参数的概念，并且在 IR 中进行处理，具体可以参考 reference.mlir， /home/yongxiy/Desktop/multicontext 和 /home/yongxiy/Desktop/dockerVolumn/Elenor 的对模型的建模。
8. 当前的trace 已经非常直观了，并且很好的了解当前的运行情况，memory 大小相关的，可以参考当前 queue 的状态来表示，memory latency 相关的也可以加入，但是需要注意，tile 的 memory 需要在tile 那一栏里面，L2 的 cache 需要和 L2 一起
9. 当前的 trace 中 indices_ready 的状态是 轮训的，这个需要改一下
10. 当前的 nest.dispatch.tasks.async 的 ins 和 outs 表示的并不正确，需要增加类似于 function args 这种表示，ins 和 outs 只是明确定义的输入输出。
11. 需要加入 L1 级别的 free 操作，用于释放不再使用的内存资源，确保内存的高效利用。
12. 当前的 nexus.program 需要采用 depends 机制来明确各个 tile program 之间的依赖关系，确保使用 ready-action 策略而不是wait这种策略来进行触发。
13. 需要询问 当前 next.context 的运行是不是 ready action 这种模式。

> 状态更新于 2026-09-13，基于 ready-action 改造后的代码逐条核对（全量 342 tests 通过、CLI 实测）。
> 每条含：原始意图（想做的事）→ 当前状态 → 剩余工作。编号沿用原列表。
> 2026-09-25 复核：基于批次 I/II/III 落地后的代码（全量 444 tests、pre-commit、4 focused 场景 × 双档 + 动态 corpus 270 run 全过）。

14. **IR 对 nest.context 的资源显式配置与管理**

- 想做：像 reference.mlir 那样在 nest.context 上声明编译器生成的资源合同（`#nest.context_resources<logical_tasks / l2_scratchpad_bytes / tile_l1_bytes_per_context / requested_contexts_per_tile>`、execution_model、epoch_model），供 admission 直接消费，而不是把 context 当普通 tile、资源需求靠推导。
- 状态：❌ 未开始。当前 `NestContextOp` 只有 `placement / context(pin) / completion_event`；L2 需求从 `nest.alloc` 列表求和推导（`try_admit_l2_buffers`），无 context 级声明包络、无 per-context L1 envelope、无 requested_contexts_per_tile；epoch_policy 只是仿真配置（`GroupSchedulerConfig`），不在 IR。
- 剩余工作：定义 `resource_contract` 属性 + verifier 一致性检查（声明值 vs 推导值）+ admission 改为消费声明包络（可在 load 期做静态拒绝，早于运行时 fault）。
- ✅ 2026-09-25 标注：已完成（上方 ❌ 为 2026-09-13 旧状态）。`#nest.context_resources<placement / l2_mode / allowed_profiles / logical_tasks / l2_spm_bytes / requested_contexts_per_tile>` 已在 IR 声明并由 verifier 与 admission 直接消费，超包络在编译/load 期静态拒绝（`workload_ir.py::_resource_contract`、`execution_verifier.py`、`try_admit_context_task`）；epoch_policy 仍留在 `GroupSchedulerConfig`，未进 IR。

15. **内存不足时的等待与 release 驱动重试**

- 想做：L2/L1 容量不足时下一个 context 等待而非直接失败；不能每 cycle 盲重试，要依据编译器给出的内存信息（大小、cache/scratchpad 模式）判定，只在有 release 时才重试。
- 状态：⚠️ L2 已完成，L1 未完成（合同上明确没有 L1 等待队列）。
  - L2 ✅：bundle 三态 admission（INVALID/PERMANENT 立即 fault、TEMPORARY 入 FIFO ticket 且等待零持有）；重试仅由 release final-free 通知触发（pool_version + capacity_change_cycle 双去重，无忙轮询）；event_table 容量不足走同一机制。
  - L1 ❌：TEMPORARY 容量不足在调度器层返回 BACKPRESSURE、action 留在候选表被逐周期重扫描，但这只是调度器行为，不是等待队列——IR_SPEC §4.2.1 明确 "does not change eager admission or add an L1 wait queue: software must order a capacity-dependent dispatch after the relevant free"，即合同仍是 eager admission + 编译器负责排序；cache/scratchpad 模式比较也没有（cache 是 fidelity 层 metadata-only，不参与 admission 决策）。
- 剩余工作（若推进）：L1 版 WAIT_CAPACITY + tile.free 通知驱动的重试（对齐 L2 的 PR3.5 模式），以及编译器内存模式元数据进 admission；或显式决策维持现合同（编译器静态排序解决），把本项 L1 部分标记为"不进运行时"。
  comments：
  不必要，而且当前这种不对称是合理的设计，不是欠账。 理由五条：

1.  粒度和等待时长不匹配。 L2 的 release 驱动 ticket 服务的是 context 级准入——粗粒度、等待长（等别的 context 整个跑完）、等待者少。L1 背压是 dispatch 级——等待时长由 resident 程序的 engine job 决定（几十到几千 cycle），同时等待者被 dispatch_capacity=8 / 候选表 16 上限约束。规模小一个量级。
2.  重试延迟可量化且是噪声。 扫描 cursor 每周期推进 scan_width=4，候选表内每条记录约每 len/scan_width ≈ 4 cycle 被重访一次。容量归还（terminal cleanup / tile.free）后，被阻塞的 dispatch 最多多等 ~4 cycle 才重试成功——相对数百上千 cycle 的等待本身，收益空间接近零。L2 则不同：ticket 唤醒保证
    release 后下一周期就能 admit，对 context 级长等待这 1 cycle 差异虽也小，但 L2 的动机更多在“不做无谓重试”的合同清洁性。
3.  FIFO ticket 会和 ready-action 语义打架。 L2 严格 FIFO 合法，因为 context 准入本来就是串行决策点（一个 context 整体进或整体等）。dispatch 现在是 s1/s2 乱序发射——若给 L1 等待加严格 FIFO ticket，等于在 dispatch 之间重新引入队头阻塞（N04 讨论过的那类 HOL），要么丢掉 ready-action 收益、要么再设计 bypass 规则。扫描重探测天然满足“阻塞自己、不阻塞别人 + RR 公平”。
4.  唤醒源异构，挂钩成本大于收益。 L2 只有一个唤醒源：release_l2 的 final-free（单一通知点 + pool_version 去重，干净）。L1 容量回来有多条路：每 tile 的 tile.free（高频、per-buffer）、grid terminal cleanup、跨 placement 多个 tile 的分配器状态变化。要做成通知驱动需要 per-tile pool version +
    多源挂钩 + 调度器唤醒路径——为一个 ~4 cycle 的延迟优化复制整套 PR3.5 机制，不值。
5.  真正的解在上游，不在运行时。 L1 footprint 静态已知，L1 等待本就该被编译器消灭掉大部分：IR_SPEC §4.2.1 的合同（容量依赖的 dispatch 排在相关 free 之后）+ R3-1 的 resource_contract（声明 per-context L1 envelope，load 期就能静态检查峰值并发 footprint 是否超 tile_l1_bytes）。先做 R3-1 把 L1
    等待变成编译期知识，比给运行时兜底路径加精密等待机制的 ROI 高得多。
6.  唤醒源异构，挂钩成本大于收益。 L2 只有一个唤醒源：release_l2 的 final-free（单一通知点 + pool_version 去重，干净）。L1 容量回来有多条路：每 tile 的 tile.free（高频、per-buffer）、grid terminal cleanup、跨 placement 多个 tile 的分配器状态变化。要做成通知驱动需要 per-tile pool version +
    多源挂钩 + 调度器唤醒路径——为一个 ~4 cycle 的延迟优化复制整套 PR3.5 机制，不值。
7.  真正的解在上游，不在运行时。 L1 footprint 静态已知，L1 等待本就该被编译器消灭掉大部分：IR_SPEC §4.2.1 的合同（容量依赖的 dispatch 排在相关 free 之后）+ R3-1 的 resource_contract（声明 per-context L1 envelope，load 期就能静态检查峰值并发 footprint 是否超 tile_l1_bytes）。先做 R3-1 把 L1
    等待变成编译期知识，比给运行时兜底路径加精密等待机制的 ROI 高得多。

什么时候值得重访这个决定（建议写进 TODO 备注而不是现在做）：

- trace 里实测 release→issue 间隔显著大于扫描周期（比如候选表常年满 16 条、group_action_backpressure 占比高）；
- dispatch 并发规模大幅增长（action_capacity 扩到几十上百）；
- 出现需要严格 FIFO 准入次序的确定性合同；
- 若真要做，最小版本是 per-tile pool_version 门控重试（复用 L2 的去重思想、跳过明显无变化的周期），而不是全套 FIFO ticket。

4. **bank 模式进 IR**
   - 想做：硬件上一个 bank 支持多种模式，需要暴露给编译器做优化。
   - 状态：❌ 未开始（IR 侧）。仿真内部已有 bank 建模：`BankedFreeExtentAllocator` 的 bank/alignment profile 参与 `can_ever_fit_bundle` 永久性判定；但 IR 无任何 bank 属性，编译器不可见、不可指定。
   - 剩余工作：`nest.alloc` / `tile.alloc` 增加 bank 模式属性 + verifier 校验 + allocator 按模式放置/对齐。

5. **L2 shared memory（多 tile 共享 weight，减少重复 memory IO）**
   - 想做：在 L2 开 shared memory 概念，允许多个 tile 共享一份 memory，共享 weight 放 L2 中，减少重复 prefetch 的 memory IO 占用。第一版默认私有分配，显式允许只读共享；跨 context 中间结果传递作为受控能力，而不是默认任意共享读写。
   - 状态：⚠️ 部分完成（同 context 共享已具备，显式/跨 context 共享合同缺）。
     - ✅ 同 context 多 tile 共享：dispatch 把同一 `(generation, slot)` L2 handle 映射进 placement 内每个 tile（tile_group.py:1223-1250），配合不带 task 维的 `tile.subview`，weight 式广播读共享同一份 L2 allocation，只需 prefetch 一次。
     - ✅ 单 context 内 buffer 跨 dispatch 复用（shared_A 测试：B 写 → D 读 → 最后读者释放）。
     - ❌ 显式共享分配合同（跨 context）：每个 context 的 L2 bundle 仍独占（`ContextBufferOwner(context, generation, slot)`），IR 无共享 alloc/引用语法——多个 context 用同一份 weight 仍需各自 alloc + 各自 prefetch，重复 IO 未消除；admission 无引用计数，释放无"最后使用者"合同。
   - 剩余工作：IR 共享分配/引用语法 + L2 admission 扩成引用计数（共享不重复计容量）+ 释放合同（最后使用者释放，需扩 release preflight）。
   - ✅ 2026-09-25 标注：剩余工作已全部完成（plan/02_explicit_l2_sharing.md 批次 II，同日场景验收见 plan/03）。IR 新增 `nest.alloc sharing="readonly"`、`nest.publish`、`nexus.shared.ref`；admission 以 run-scoped claim（DECLARED→BOUND→RELEASED/CANCELLED）推导引用，共享不重复计物理容量；最后安全 borrower release 唯一 final-free，producer release/retirement 不释放仍被 claim 持有的 backing。证据：`pipeline_validator/tests/test_l2_sharing.py`、`tests/test_l2_sharing_source.py`、`tests/test_compiler.py::TestIndependentSharedArtifactVerification`、`tests/test_memory_invariants.py`（claim 台账/原子性/保留容量），批次 II 实测 `examples/artifacts/l2-sharing-release/20260925T131034Z/batch-II/`（W 8192 B vs 私有 16384 B、延迟 C 后唯一 physical free）。
     comments：
   - 分支复用和独立任务复用，不总能自然转化成局部 fusion, 如下图这种，A的结果就需要被同时供给 B 和 C，难以在局部 fusion 中消除重复IO。

   ```
         → B → ...
   A → X
         → C → ...
   ```

6. **L2 buffer 粒度释放（sub-arena extent 提前归还），恢复 release 驱动的提前准入**
   - 想做（2026-09-22 用户拍板，为原始初衷）：单笔 `nest.release` 释放即把该 buffer 的 striped extent 归还 L2 free-map 并推进 pool_version，使 WAIT_CAPACITY 的后续 context 在前一个 context 的 input_released/release 当拍就 FIFO 重试准入、提前启动 load，而不是等前一个 context 整体退休。l2-admission-wait 的预期行为（B 在 A 仍算 pow 期间准入并开始 load）靠此恢复。
   - ✅ **批次 I 已完成（2026-09-25，plan/01_private_l2_release.md）**：L2 编译期 no-rebind 不重叠 padded 布局；唯一 finalizer `_try_release_l2_backing` 在最后一个私有引用（view forfeit / pin / accepted transaction）安全排空后原子归还整 stripe-rounded backing span 并推进 pool_version；WAIT_CAPACITY 同类队头由 admission_version 当拍唤醒。`nest.release` 即永久 forfeit，`nest.barrier` 不恢复权限；L1 合同不变；profile 切换仍等完整 root frontier。证据：`test_runtime.py::TestRootArenaAdmission::test_pending_root_admits_on_input_extent_release` 两档通过（ctx_b port `active_cycle == a_input l2_extent_release release_cycle`：runtime 804 / full_memory 3522，早于 ctx_a completion 17977 / 22052，prefetch×pow overlap >0）；全量回归 395 passed（含台账/泄漏/恢复负例测试与切档顺序回归）；smoke 见 `examples/artifacts/l2-sharing-release/20260925T012727Z/private-smoke/`（含 acceptance-summary.json）。批次 II（跨 context 只读共享 claims）挂接同一 finalizer，尚未实施。
   - 历史诊断（2026-09-22，实施前背景，行号对应旧代码）：当时为整 Arena 原子预留/释放——invalidate_view 不还 extent、pool_version 仅 commit/retire/reconfigure 推进、port 重试门 `(admission_version, slot_version)`、`_l2_capacity_change_cycle` 仅在 retire_context_arena 设置；实证 ctx_b 进 WAIT_CAPACITY 后零重试直到 ctx_a 完成同拍（runtime 17977 / full_memory 22052）。
   - ~~合同冲突点（2026-09-22）~~：已于 2026-09-25 解决——IR_SPEC §3.8/§8 改写为 L2 forfeiture 合同，execution_ir 表述随实现同步。
   - ~~改动面预估（2026-09-22）~~：已全部落地——exact-unit commit/free、pool_version 下移 final-free、release preflight 复用、test_t09_t10_t14 重写为两档参数化的 `test_pending_root_admits_on_input_extent_release`、l2_admission_wait.mlir 文件头更新、全量回归 + 两档 trace 验证完成。
   - [x] 需要增加 mlir 的 example，对 memory profile 变化的时候, 就不以将下一个的 op 的 load 提前到当前 context 的 release 当拍，而是在当前的 context 跑完了也就是 HBM store 完成之后 profile 切换完成后再进行 load。
     - ✅ 2026-09-25 已完成：`examples/scenarios/l2_profile_switch_load_ordering.mlir`（配置 `examples/configs/profile_l2_switch.yaml`，L2 mode0 全 SPM → mode1 带 cache）。ctx_b 连续提交、无源级 await——顺序完全由编译器生成的 L2 profile command（frontier = done_a）保证。两档 trace 完整链条 `release < store_done <= A_done <= switch_start < switch_end <= B_admit < prefetch <= tile_load` 成立（runtime：804 < 17975 <= 17977 <= 17979 < 18052 <= 18054 < 18058；full：3522 < 22050 <= 22052 <= 22054 < 22127 <= 22129 < 22133）。回归：`test_runtime.py::TestL2ProfileSwitchOrdering` 两档通过。
   - 风险与代价（2026-09-22 更新；批次 I 实施时已逐条闭环——滞后访问由引用台账+drain 覆盖、forfeiture 由 no-rebind 布局保证、记账按 backing 粒度、碎片分类保持 WAIT_CAPACITY/WAIT_FRAGMENTATION、回归面全量 395 passed、R3-5 耦合留待批次 II 同一 finalizer）：
   - ✅ 2026-09-25 标注：①批次 II（跨 context 只读共享 claims 挂接同一 finalizer）已完成，见 R3-5 的标注；②风险第 8 条的量化已完成——改代码前 13 个真实 workload × 双档共 26 run 的容量等待为 0（`examples/artifacts/l2-sharing-release/20260925T012727Z/baseline/root_wait_metrics.json`），与预期一致：R3-6 收益集中在 l2-admission-wait 类容量阻塞场景，不声称恢复 matmul-pow-free-slot / matmul17-pow-tail-overlap 的 overlap；③批次 III 场景与端到端验收已完成（4 focused × 双档 + 动态 corpus 270 run + 270 trace 交叉审计，`examples/artifacts/l2-sharing-release/20260925T135828Z/batch_III/acceptance-summary.json`，全量回归 444 passed）。
     1. **正确性：释放后的滞后访问**。现模型下 extent 在 run 内永不回收，退休时有全量 drain 屏障（routes/leases/jobs 全零），天然免疫"释放后仍有慢访问"。buffer 级归还后，quiescence 证据只剩 release preflight 的 pin/inflight 计数——需审计所有可能滞后触碰该 extent 的路径（cache dirty line 延迟写回、MSHR drain、profile/maintenance 命令、其它 transaction 名下的在飞 prefetch 腿），漏一条就是新占用者 B 的数据被写坏。
     2. **正确性：A 的回绑（release 即 forfeiture）**。A 的 Arena 布局编译期固定，若 A 后续 view（同 context 内 offset 复用）绑入已归还、且已被 B 占用的 extent → 跨 context 数据损坏。合同必须明确 released extent 对 A 永久 forfeited，编译器/verifier 保证 released 区域之后无任何 bind；当前 resources.py 的 L2 复用窗口以 retirement fence 为界，需重新推导。
     3. **原子性与记账**：Arena 出现"部分释放"新状态——`l2_reserved_bytes`/live bytes/per-bank 占用、report 的 arenas 段、PMU 计数都要改按 extent 粒度记账；cancel/reset 的 drain 路径须防 double-free；`arena_lifetime` trace 语义不再是"容量占用窗口"，需重定义。
     4. **碎片与等待分类**：逐 buffer 归还的 striped extent 可能与剩余空闲 extent 拼不出 B 的 striped 几何——字节够但放不下，B 继续等；`can_ever_fit` 的 TEMPORARY/PERMANENT 判定需重审，避免"永远放不进却反复重试"。
     5. **静态合同削弱**：IR_SPEC §940-943 的 dominating barrier 是编译期可验证的复用安全证明；移除后安全证明下沉为运行时动态检查（pin/inflight），编译期可验证性变弱；profile reconfig 的 quiescence 证明（按 arena 粒度 owner 假设）同样要重审。
     6. **回归面**：`test_t09_t10_t14` 语义反转重写；凡含 `nest.release` 的场景两档时序全变（136 corpus 场景 + 全部 trace artifacts 需重生成）；port 重试频次随 pool_version 变化频率上升，需确认 FIFO 队头阻塞行为不变。
     7. **与 R3-5 耦合**：共享 L2 的引用计数与 buffer 级 extent 归还都操作 free-map，必须协同设计（共享 buffer 被一方 release 时，extent 归还受 refcount 门控）。
     8. **收益预期管理**：R3-6 只救"L2 容量阻塞"的流水线（l2-admission-wait 类）。matmul-pow-free-slot 的 pow 是 device slot 阻塞、matmul17 的 pow 是数据依赖 + HBM 往返阻塞——R3-6 对这两条 example 的 overlap 无直接帮助。实施前应在真实 workload 上量化 L2 容量等待占比，再决定优先级。

- 当前 matmul17_pow_tail_overlap 里面中 tile program pow 中， 并不是只有一个 pow，当前是两个 pow 串行， trace 显示它们依次被调度执行。
  - ✅ 2026-09-25 标注：现状已变——文件现为 @pow_np_lo（C 块 0..7）/ @pow_np_hi（C 块 8..15）两个数据并行 halves，各配独立生产者，不再是一先一后串行；此条作为旧观察保留。
- 当前 l2-admission-wait 这个后面的 load 行为并没有和 前一个 的 pow 进行 overlap ，理论上 input release 过后下一个可以立即开始加载，而不必等待前一个 pow 完全结束。
  - 2026-09-22 诊断：调度/依赖均正确，根因是整 Arena 粒度预留释放（见 R3-6）。已拍板走 R3-6 的 buffer 粒度释放方案。
  - 2026-09-25 已实施（plan/01_private_l2_release.md 批次 I）：L2 编译期 no-rebind 不重叠布局 + buffer 粒度 padded backing 物理释放 + accepted transaction 引用台账 + fault/cap drain/poison 闭环。l2-admission-wait 在 runtime/full_memory 两档均验证 ctx_b port `active_cycle == a_input l2_extent_release release_cycle`（runtime 804 / full_memory 3522），严格早于 ctx_a completion（17977 / 22052），B 的真实 HBM→L2 prefetch 与 A 的 `EVU:pow` 窗口相交（>0 cycle）。证据：`examples/artifacts/l2-sharing-release/20260925T012727Z/`（baseline、phase1-matrix、gate4-probe、private-smoke）。
- 当前 matmul_pow_free_slot 并没有连起来起来
  - 2026-09-22 诊断：调度无 bug（first-free-slot 两档一致，pow 均在 matmul 完成 +1 拍接管 slot）。不 overlap 主因是 HBM 整腿单通道（transfer.py:1096-1104，`(addr//64)%8`）+ 该场景地址全 64KB 对齐塌缩到 Ch:0。run.sh 地址已错开（总周期 45088→33496，gap 6216→2611），但 C store 512KB 等距步距使 4 笔 store 结构性锁同通道，彻底恢复 overlap 需 per-burst 通道条纹——**是否实施待定**。matmul17-pow-tail-overlap 同根因（另叠加 pow 消费路径绕 HBM、无 L2 直通）。
  - ✅ 2026-09-25 标注：per-burst 通道条纹已实施——transfer.py 现按 `(address // hbm_burst_bytes) % hbm_channels` 选通道（transfer.py:1148），不再整腿单通道。实测 runtime 总周期 22222（旧基线：地址错开前 45088、错开后 33496），completed=True。matmul17 的“pow 消费路径绕 HBM、无 L2 直通”子项仍未改。
- [x] 需要增加 mlir 的 example，对 memory profile 变化的时候, 就不以将下一个的 op 的 load 提前到当前 context 的 release 当拍，而是在当前的 context 跑完了也就是 HBM store 完成之后 profile 切换完成后再进行 load。
  - ✅ 2026-09-25 已完成：`examples/scenarios/l2_profile_switch_load_ordering.mlir`（配置 `examples/configs/profile_l2_switch.yaml`，L2 mode0 全 SPM → mode1 带 cache）。ctx_b 连续提交、无源级 await——顺序完全由编译器生成的 L2 profile command（frontier = done_a）保证。两档 trace 完整链条 `release < store_done <= A_done <= switch_start < switch_end <= B_admit < prefetch <= tile_load` 成立（runtime：804 < 17975 <= 17977 <= 17979 < 18052 <= 18054 < 18058；full：3522 < 22050 <= 22052 <= 22054 < 22127 <= 22129 < 22133）。回归：`test_runtime.py::TestL2ProfileSwitchOrdering` 两档通过。

1. 每次 profile 切换都必须进行完整的等待，插入等待这一工作由编译器来做，但是这部分需要在 IR 中显示声明
   - ✅ 2026-09-25 标注：已完成。IR 以 `l2_mode / allowed_profiles` 显式声明，编译器 `bind_profiles` 自动生成完整 root completion frontier 的 device await + L2 ProfileReconfigDesc（源码不手写 await）；切走再切回同 mode 视为新 generation，共享 backing 必须同 epoch。证据：`examples/scenarios/l2_profile_switch_load_ordering.mlir`、`examples/scenarios/l2_admission_profile_switch.mlir` 及 `plan/03` 批次 III 验收。

- 需要查看 L1 的 memory 是否需要 类似于 context-local 的管理，也就是说需要检查 L1 的buffer是不是也是 必须 store 才可以释放，还是可以直接复用。

## R3

- 当前的 input release 或者 output release 在 tile program 中粒度是在太粗是不是适合硬件设计需要考量，还有考虑是否需要按照 buffer 的名字进行 release

## example

## 需要调研的问题 （2026-09-25：第 1、2 条已由实现闭环 ✅ ｜ 第 3 条编译器无法预知调度/SPM 占用 ❌ 仍开放）

- 当前的 context wait，只负责自己的 context wait。理论上如果 profile 切换的时候，理论上 profile 之间切换的时候也需要一个 await，需要详细地考虑当前的 await 是不是阻塞所有的 context，还是只管自己的 context 的 await。然后需要将多个 context await，就是这种 context 之间的 await，交给 device 去做。这个 idea 需要什么，需要详细地去做论证。（**注释：IR 上我决定不允许 一个 context 有多个 profile phase 来切换 memory的 profile**）
  - ✅ 2026-09-25 标注：已按“交给 device”落地——编译器 `bind_profiles` 生成普通 device await + ProfileReconfigDesc，frontier 为完整 root completion history；context 内不允许多个 profile phase（IR 合同，见 plan/03 §0 边界）。证据：`examples/scenarios/l2_admission_profile_switch.mlir` 两档因果链验收。
- 这个 await 理论上应该是由编译器去做，但是呢，编译器并不知道实际 runtime 的调度模型，所以说这个 await 需要在什么时候加，理论上应该是取决于 runtime 才对。当然这个的话后续也需要详细地去讨论和调研。如果是由编译器加，如何该去加，在什么状态下去加这个 await。因为理论上来说，编译器是只知道 allow profile，但是它是不知道整个的这个具体的调度信息的。
  - ✅ 2026-09-25 标注：已裁决并由实现验证——编译器只依据静态 `allowed_profiles` 变化点插入 await/reconfig，等待对象是“完整 root completion history”（调度不可知也安全）；运行时仅在 profile 命令完成/OPEN_ISSUE 后才开放下一档 admission（issue gate），不做调度预测。证据：plan/02 §H、plan/03 切档因果链两档通过。
- 还有一个问题理论上 matmul 在所有的 tile context 打满的时候 不一定 占满 整个SPM 所以 SPM = 100 的时候其实可能存在部分空闲，所有并不是所有的 matmul 就是 SPM = 100 就性能最好，但是 编译器是不知道调度信息的，编译器怎么知道什么时候 SPM 不能占满呢？也是就是说假设有 17 个 matmul 的 context 和一个 gather， 编译器不知道调度信息，那么就有可能说：
  - 编译器没法判断 SPM 不是 100的时候 可能没办法跑满 16 个， 还是不满的时候就可以跑满 16 个。
  - 编译器不知道是不是 这 16 个 matmul context 能够同时跑
  - 编译器无法准确预测 SPM 的实际占用情况

## 未来需要考虑的问题暂时先不考虑

以下问题暂不实施，仅记录方向与当前思考，待 R3 或后续轮次再评估。

### 1. Transpose / Layout 变换的硬件归属

当前思路是设一个专门器件做 Transpose 和 Layout 变换。另一个候选方案：把 Transpose 这类 Layout 变换作为子器件嵌入 BOA 或 EVU 内部。该决策与第 3 条（BOA register file pipeline）耦合——若采用"transpose input → BOA → transpose output"的链路，会直接影响 BOA 的面积与功耗预算，需要一起评估。

### 2. 编译器需求清单（待整理成正式文档）

编译器侧需要做的事，后续要整理成一份明确清单：

- 拓扑排序；
- liveness / non-liveness memory 分析；
- peak memory 分析（为后续 memory admission 相关机制做准备）；
- tiling 策略与 context 切分；
- L1 并发超内存的静态可见性：L1 容量不足的现行合同是 eager admission + 编译器负责排序（IR_SPEC §4.2.1：无 L1 等待队列，容量相关的 dispatch 必须排在相关 free 之后；调度器的 BACKPRESSURE 只是仿真层行为）。因此"编译器必须静态知道哪里会超内存、把容量依赖排好序"这一需求是刚性的，不是优化项。

### 3. BOA register file load 与指令级 pipeline

BOA 这类计算单元的 register file 装载需要硬件级流水设计。理论上 multi-context 要把计算单元利用率打满，BOA 内部需要指令级 pipeline。若嵌入 transpose（第 1 条），寄存器装载链路变长，需重新衡量 BOA 整体大小与功耗。

### 4. 子图 fusion / auto partition

从 trace 观察到部分 fusion 能显著提升利用率。候选子图条件：子图内无分叉、前后为线性连接、I/O 时间占比与计算占比差异大——计算时间超过 I/O 时间的子图最能发挥当前调度模型的优势。当前只考虑静态图，if-else 等动态 branch 不在范围内。

后续需要做自动切分（auto partition）：按"易于 fusion + 计算占比 > I/O 占比"的规则切分子图，再交给调度。注意：group 级已默认启用 ready-action（S1，独立分支可越过慢等待），fusion 的相对收益结构可能与早期 trace 观察时不同，制定 partition 规则前应基于新调度器重新测量。

### 5. 真正的底层 ISA 的设计与实现

### 6. load/store 拆分为 L2↔L1 与 L1↔Register 两类

- 想做：当前 `tile.load.async` / `tile.store.async` 只有 L2↔L1 语义（简化设计）；后续增加 L1↔Register 一类 load/store，更好模拟真实硬件行为。
- 状态：❌ 未开始。方言无任何 register 级 op；BOA/EVU 描述符直接引用 L1 buffer，无寄存器级传输。
- 剩余工作：新 op（如 tile.rload.async / tile.rstore.async）+ 引擎描述符操作数模型改动 + 寄存器端口/带宽性能模型。依赖"未来问题"清单里 BOA register file load 的硬件 pipeline 结论。

### 7. 后续版本可能需要考虑在 L2 group 级别进行从 collection 和 简单的 reduction 操作，以提升数据复用率和整体性能。

### 8. L2 group 级别的 SRAM 需要虚拟内存地址映射来防止碎片化
