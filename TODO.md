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

## R3

> 状态更新于 2026-09-13，基于 ready-action 改造后的代码逐条核对（全量 342 tests 通过、CLI 实测）。
> 每条含：原始意图（想做的事）→ 当前状态 → 剩余工作。编号沿用原列表（无 4 号项）。

1. **IR 对 nest.context 的资源显式配置与管理**
   - 想做：像 reference.mlir 那样在 nest.context 上声明编译器生成的资源合同（`#nest.context_resources<logical_tasks / l2_scratchpad_bytes / tile_l1_bytes_per_context / requested_contexts_per_tile>`、execution_model、epoch_model），供 admission 直接消费，而不是把 context 当普通 tile、资源需求靠推导。
   - 状态：❌ 未开始。当前 `NestContextOp` 只有 `placement / context(pin) / completion_event`；L2 需求从 `nest.alloc` 列表求和推导（`try_admit_l2_buffers`），无 context 级声明包络、无 per-context L1 envelope、无 requested_contexts_per_tile；epoch_policy 只是仿真配置（`GroupSchedulerConfig`），不在 IR。
   - 剩余工作：定义 `resource_contract` 属性 + verifier 一致性检查（声明值 vs 推导值）+ admission 改为消费声明包络（可在 load 期做静态拒绝，早于运行时 fault）。

2. **内存不足时的等待与 release 驱动重试**
   - 想做：L2/L1 容量不足时下一个 context 等待而非直接失败；不能每 cycle 盲重试，要依据编译器给出的内存信息（大小、cache/scratchpad 模式）判定，只在有 release 时才重试。
   - 状态：⚠️ L2 已完成，L1 未完成（合同上明确没有 L1 等待队列）。
     - L2 ✅：bundle 三态 admission（INVALID/PERMANENT 立即 fault、TEMPORARY 入 FIFO ticket 且等待零持有）；重试仅由 release final-free 通知触发（pool_version + capacity_change_cycle 双去重，无忙轮询）；event_table 容量不足走同一机制。
     - L1 ❌：TEMPORARY 容量不足在调度器层返回 BACKPRESSURE、action 留在候选表被逐周期重扫描，但这只是调度器行为，不是等待队列——IR_SPEC §4.2.1 明确 "does not change eager admission or add an L1 wait queue: software must order a capacity-dependent dispatch after the relevant free"，即合同仍是 eager admission + 编译器负责排序；cache/scratchpad 模式比较也没有（cache 是 fidelity 层 metadata-only，不参与 admission 决策）。
   - 剩余工作（若推进）：L1 版 WAIT_CAPACITY + tile.free 通知驱动的重试（对齐 L2 的 PR3.5 模式），以及编译器内存模式元数据进 admission；或显式决策维持现合同（编译器静态排序解决），把本项 L1 部分标记为"不进运行时"。
   comments：
    不必要，而且当前这种不对称是合理的设计，不是欠账。 理由五条：

    1. 粒度和等待时长不匹配。 L2 的 release 驱动 ticket 服务的是 context 级准入——粗粒度、等待长（等别的 context 整个跑完）、等待者少。L1 背压是 dispatch 级——等待时长由 resident 程序的 engine job 决定（几十到几千 cycle），同时等待者被 dispatch_capacity=8 / 候选表 16 上限约束。规模小一个量级。
    2. 重试延迟可量化且是噪声。 扫描 cursor 每周期推进 scan_width=4，候选表内每条记录约每 len/scan_width ≈ 4 cycle 被重访一次。容量归还（terminal cleanup / tile.free）后，被阻塞的 dispatch 最多多等 ~4 cycle 才重试成功——相对数百上千 cycle 的等待本身，收益空间接近零。L2 则不同：ticket 唤醒保证
    release 后下一周期就能 admit，对 context 级长等待这 1 cycle 差异虽也小，但 L2 的动机更多在“不做无谓重试”的合同清洁性。
    3. FIFO ticket 会和 ready-action 语义打架。 L2 严格 FIFO 合法，因为 context 准入本来就是串行决策点（一个 context 整体进或整体等）。dispatch 现在是 s1/s2 乱序发射——若给 L1 等待加严格 FIFO ticket，等于在 dispatch 之间重新引入队头阻塞（N04 讨论过的那类 HOL），要么丢掉 ready-action 收益、要么再设计 bypass 规则。扫描重探测天然满足“阻塞自己、不阻塞别人 + RR 公平”。
    4. 唤醒源异构，挂钩成本大于收益。 L2 只有一个唤醒源：release_l2 的 final-free（单一通知点 + pool_version 去重，干净）。L1 容量回来有多条路：每 tile 的 tile.free（高频、per-buffer）、grid terminal cleanup、跨 placement 多个 tile 的分配器状态变化。要做成通知驱动需要 per-tile pool version +
    多源挂钩 + 调度器唤醒路径——为一个 ~4 cycle 的延迟优化复制整套 PR3.5 机制，不值。
    5. 真正的解在上游，不在运行时。 L1 footprint 静态已知，L1 等待本就该被编译器消灭掉大部分：IR_SPEC §4.2.1 的合同（容量依赖的 dispatch 排在相关 free 之后）+ R3-1 的 resource_contract（声明 per-context L1 envelope，load 期就能静态检查峰值并发 footprint 是否超 tile_l1_bytes）。先做 R3-1 把 L1
    等待变成编译期知识，比给运行时兜底路径加精密等待机制的 ROI 高得多。
    6. 唤醒源异构，挂钩成本大于收益。 L2 只有一个唤醒源：release_l2 的 final-free（单一通知点 + pool_version 去重，干净）。L1 容量回来有多条路：每 tile 的 tile.free（高频、per-buffer）、grid terminal cleanup、跨 placement 多个 tile 的分配器状态变化。要做成通知驱动需要 per-tile pool version +
    多源挂钩 + 调度器唤醒路径——为一个 ~4 cycle 的延迟优化复制整套 PR3.5 机制，不值。
    7. 真正的解在上游，不在运行时。 L1 footprint 静态已知，L1 等待本就该被编译器消灭掉大部分：IR_SPEC §4.2.1 的合同（容量依赖的 dispatch 排在相关 free 之后）+ R3-1 的 resource_contract（声明 per-context L1 envelope，load 期就能静态检查峰值并发 footprint 是否超 tile_l1_bytes）。先做 R3-1 把 L1
    等待变成编译期知识，比给运行时兜底路径加精密等待机制的 ROI 高得多。

    什么时候值得重访这个决定（建议写进 TODO 备注而不是现在做）：
    - trace 里实测 release→issue 间隔显著大于扫描周期（比如候选表常年满 16 条、group_action_backpressure 占比高）；
    - dispatch 并发规模大幅增长（action_capacity 扩到几十上百）；
    - 出现需要严格 FIFO 准入次序的确定性合同；
    - 若真要做，最小版本是 per-tile pool_version 门控重试（复用 L2 的去重思想、跳过明显无变化的周期），而不是全套 FIFO ticket。

3. **load/store 拆分为 L2↔L1 与 L1↔Register 两类**
   - 想做：当前 `tile.load.async` / `tile.store.async` 只有 L2↔L1 语义（简化设计）；后续增加 L1↔Register 一类 load/store，更好模拟真实硬件行为。
   - 状态：❌ 未开始。方言无任何 register 级 op；BOA/EVU 描述符直接引用 L1 buffer，无寄存器级传输。
   - 剩余工作：新 op（如 tile.rload.async / tile.rstore.async）+ 引擎描述符操作数模型改动 + 寄存器端口/带宽性能模型。依赖"未来问题"清单里 BOA register file load 的硬件 pipeline 结论。

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
   comments：
   - 分支复用和独立任务复用，不总能自然转化成局部 fusion, 如下图这种，A的结果就需要被同时供给 B 和 C，难以在局部 fusion 中消除重复IO。
    ```
          → B → ...
    A → X
          → C → ...
    ```


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
