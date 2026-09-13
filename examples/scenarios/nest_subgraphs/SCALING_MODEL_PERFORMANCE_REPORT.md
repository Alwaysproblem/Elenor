# ELENOR 调度模型性能报告

> 覆盖仓库全部可运行 example：112 份 NEST 调度子图 + 13 份真实 workload（gather/matmul/pow/mixed-context）+ 4 份 protocol scenario（ready-action/device-dependency/l2-admission/release-counterexample），共 129 份。
> 生成日期：2026-09-13。数据由 Perfetto trace_processor v58.2 SQL 提取。
> 配置：full_memory · num_dma_channels=2 · hbm_fixed_latency=10 · context 4×4（single 为 1×4）· Group 默认 S1 ready-action · clock 1000MHz（raw ts/dur 即 cycles）。
> 全部 cycle 数均为当前源码实测；旧基线数字（§8.2/§8.4）属修正前合同，不可混算。
> **方法说明**：不跨 corpus 按原始周期排"快/慢"——各 example 的 shape/bytes/repeat 刻意不同（guide §2.3），原始周期不可比。判断依据是：① 同拓扑的 single/node/stage 配对对照（隔离切分开销）；② 归一化指标 compute_active/cycles（隔离计算-IO 比）；③ stall 归因（定位瓶颈类型）。4 份 protocol scenario 不是性能 benchmark，而是控制/资源协议的边界可观察性验证（见 §9）。

## 1. 总览

| 类别          |    数量 |    完成 | fault | verify-reject |
| ------------- | ------: | ------: | ----: | ------------: |
| NEST 子图          |     112 |     108 |     2 |             2 |
| 真实 workload      |      13 |      13 |     0 |             0 |
| Protocol scenario  |       4 |       4 |     0 |             0 |
| **合计**           | **129** | **125** | **2** |         **2** |

| 全局指标                      | 值                                                             |
| ----------------------------- | -------------------------------------------------------------- |

| trace 提取错误                | 0（125 份与独立提取器逐字段一致）                              |
| group_ordering_stall 非零     | **0**——S1 默认下全部 125 case 无 head-order 挡住 eligible 动作 |
| backpressure 非零（NEST）     | 12 cases——UCE pin 争用 / 细粒度依赖密集                        |
| backpressure 非零（workload/protocol） | **0**——真实负载与 protocol 的 context 配置无 pin 争用 |

## 2. NEST 子图：按映射类型聚合

| 映射   | 数量 | 中位周期 | 平均周期 |   最快 |    最慢 |
| ------ | ---: | -------: | -------: | -----: | ------: |
| single |   31 |   45,294 |   45,499 |      3 | 137,612 |
| node   |   50 |   71,977 |   88,543 |      3 | 361,426 |
| stage  |   27 |   60,525 |   63,492 | 34,397 |  97,649 |

## 3. NEST 子图：各拓扑族配对对照（基础变体，不含参数放大）

| 拓扑               | single |    node |  stage | node 开销 | stage 开销 | 最快   |
| ------------------ | -----: | ------: | -----: | --------: | ---------: | ------ |
| S01 串行链         | 41,492 |  56,373 | 45,102 |      +36% |        +9% | single |
| S02 三独立链       | 57,522 |  70,716 | 59,027 |      +23% |        +3% | single |
| S03 扇出           | 30,805 |  34,396 | 34,397 |      +12% |       +12% | single |
| S04 汇合           | 35,951 |  44,679 | 44,678 |      +24% |       +24% | single |
| S05 菱形/残差      | 37,457 |  49,774 | 48,713 |      +33% |       +30% | single |
| S06 不平衡菱形     | 40,001 |  54,869 | 51,272 |      +37% |       +28% | single |
| S07 嵌套 fork/join | 52,857 |  73,238 | 64,120 |      +39% |       +21% | single |
| S08 N 型交叉       | 51,652 |  62,925 | 65,441 |      +22% |       +27% | single |
| S09 稠密跳连       | 43,037 |  61,514 | 49,196 |      +43% |       +14% | single |
| S10 U 型长跳连     | 49,213 |  65,129 | 56,421 |      +32% |       +15% | single |
| S11 规约树/fanout  | 79,853 | 105,151 | 96,058 |      +32% |       +20% | single |
| S12 Attention      | 71,023 |  87,693 | 76,905 |      +23% |        +8% | single |
| T01 分块流水       | 45,294 |  46,015 | 47,858 |   **+2%** |        +6% | single |
| T02 波前           | 67,972 |  84,182 | 88,098 |      +24% |       +30% | single |
| T03 状态依赖       | 46,634 |  61,512 | 49,200 |      +32% |        +6% | single |
| T04 多请求复用     | 79,274 | 126,783 | 84,826 |  **+60%** |        +7% | single |

**single 在全部 16 个拓扑族中均最快**。node 平均 +23-60%，stage +3-30%。唯一例外 T01 node 仅 +2%（天然 chunk 边界使 HBM 字节不翻倍）。

## 4. 真实 workload：归一化指标

不跨 workload 比原始周期（shape/bytes 不同）。用 compute_active/cycles 判断计算-IO 主导关系，用 MFE load/store service 判断 IO 瓶颈。

| workload                  |  cycles | compute |     ratio |    BOA |    EVU |  HBM | MFE load svc | MFE store svc | UCE issue | ctx_switch | ord_stall | bkpr |
| ------------------------- | ------: | ------: | --------: | -----: | -----: | ---: | -----------: | ------------: | --------: | ---------: | --------: | ---: |
| gather                    |   1,197 |       0 |      0.0% |      0 |      0 |    2 |           18 |            39 |        11 |          0 |         0 |    0 |
| gather_matmul             |  15,579 |   1,028 |      6.6% |  1,028 |      0 |    4 |        3,750 |         3,706 |        17 |          0 |         0 |    0 |
| matmul_gather_add         |  15,402 |   1,035 |      6.7% |  1,028 |      7 |    4 |        3,750 |         3,706 |        19 |          0 |         0 |    0 |
| matmul_2048x512_boa256    | 166,051 |  65,664 | **39.5%** | 65,664 |      0 |   12 |      119,424 |       297,165 |       304 |        105 |         0 |    0 |
| gather_matmul_4t_2ctx     |  66,728 |   8,224 |     12.3% |  8,224 |      0 |    8 |       30,000 |        57,640 |       136 |         14 |         0 |    0 |
| matmul_gather_add_4t_2ctx |  66,742 |   8,280 |     12.4% |  8,224 |     56 |    8 |       30,000 |        57,644 |       152 |         14 |         0 |    0 |
| matmul_pow_parallel       | 147,124 |  49,264 |     33.5% | 32,832 | 16,432 |   10 |      119,264 |       222,047 |       272 |         77 |         0 |    0 |
| matmul_pow_free_slot      | 225,025 |  82,096 |     36.5% | 65,664 | 16,432 |   16 |      178,976 |       466,897 |       424 |        129 |         0 |    0 |
| matmul_pow_data_dep       | 368,954 | 131,248 |     35.6% | 65,664 | 65,584 |   16 |      324,384 |       502,127 |       424 |        143 |         0 |    0 |
| matmul17_pow_tail_overlap | 366,205 | 147,664 | **40.3%** | 82,080 | 65,584 |   19 |      343,044 |       540,196 |       473 |        147 |         0 |    0 |
| pow_dual_context          |  55,157 |   8,216 |     14.9% |      0 |  8,216 |    4 |       29,776 |        64,272 |        72 |         12 |         0 |    0 |
| pow_dual_ctx_mixed_shapes |  87,786 |   8,216 |      9.4% |      0 |  8,216 |    4 |       40,528 |       104,764 |        72 |         12 |         0 |    0 |
| pow_sequential_contexts   |  81,136 |   8,216 |     10.1% |      0 |  8,216 |    4 |       29,776 |        29,776 |        72 |          0 |         0 |    0 |

**关键观察**：
- **group_ordering_stall = 0 / backpressure = 0**：13 个真实 workload 全部无 ready-action head-order 阻塞、无 pin 争用——调度器在真实负载下行为干净。
- **计算占比分层**：
  - 计算主导（≥35%）：matmul_2048x512（39.5%）、matmul17_pow_tail（40.3%）、matmul_pow 系列（33-37%）——大 GEMM + pow 组合，BOA 计算量大。
  - 中间（10-15%）：pow_dual_context（14.9%）、4tiles_2ctx（12.3%）——多 context 但计算量中等。
  - IO 主导（<10%）：gather（0%）、gather_matmul（6.6%）——Gather 本质是随机访存，计算极少。
- **MFE store service 普遍远大于 load service**：matmul_pow_data_dep 的 store 502K vs load 324K；matmul17_pow store 540K——写回 HBM 是真实 workload 的主要 IO 瓶颈。
- **pow_sequential_contexts 的 store=load=29,776**：顺序提交下 store 与 load 等长（无重叠），对比 pow_dual_context store 64K > load 30K（有重叠空间）——配对对照说明并发提交让 store 与下一 context 的 load 重叠。

### 4.5 Protocol scenario（控制/资源协议边界验证）

4 份 protocol scenario 不是性能 benchmark，而是验证调度器在边界条件下的**正确性与可观察性**：

| scenario | cycles | 验证目标 | 结果 |
 | --- | ---: | --- | --- |
 | ready-action-branch | 17,339 | S1 下独立分支越过慢等待；S0 下不能（配对对照） | ✅ S1=17,339 vs S0=18,326（−5.4%），独立 prefetch 在慢 store 完成前发射 |
 | device-dependency-submit | 13,250 | CPU submit 依赖（WAIT_DEPS）不占硬件额度；独立 B 不被 C→A 依赖挡住 | ✅ C cycle 1 submit 但 cycle 5,050 才 admit；B cycle 2 已 admit |
 | l2-admission-wait | 29,051 | L2 容量不足 WAIT_CAPACITY → FIFO ticket → release 驱动重试 | ✅ release 驱动唤醒保留，cycle 29,051 完成 |
 | sequential-release-counterexample | 74,444 | 顺序 release 反例：store 排空期间 context 不提前释放 | ✅ Store 完成前不 context_done，当前完成合同正确 |

全部 4 份 group_ordering_stall=0、backpressure=0——协议边界行为干净。

## 5. 调度模型适合 / 不适合的场景

### 5.1 适合（当前模型表现好）

| 特征                          | NEST 证据                              | workload 证据                                                              | 原因                                                                  |
| ----------------------------- | -------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| **计算占比 > I/O 占比**       | S08 series（80-100%）、S07 c100（85%） | matmul_2048x512（39.5%）、matmul17_pow（40.3%）、matmul_pow 系列（33-37%） | ready-action 让独立计算越过慢等待；计算时间长，调度收益空间大         |
| **单 context 全图（single）** | 16/16 拓扑 single 最快                 | matmul_2048x512 单 context 166K cycles                                     | L2 内传递无 HBM 开销；无跨 context 粗粒度事件门                       |
| **天然 chunk 边界切分**       | T01 node 开销仅 +2%                    | —                                                                          | 当切分边界等于数据搬运边界时，node 不产生额外 HBM                     |
| **独立分支可并行**            | S02 三独立链、S06 不平衡菱形、S08 N 型 | matmul_pow_parallel（matmul×2 + pow×2 并发 submit）                        | 独立分支在 S1 下各自推进                                              |
| **跨引擎重叠**                | S12 QK(BOA) ∩ V(EVU)                   | matmul17_pow_tail（BOA 82K + EVU 66K 重叠）                                | 不同引擎队列独立，eligible-head RR 让 BOA/EVU 各自推进                |
| **slot 复用 / first-free**    | T04 stage slot-generation 复用         | matmul_pow_free_slot（首个 matmul 释放 slot 给 pow 提前调度）              | device slot 在 context 完成后释放，first-free 让后续 context 提前进入 |
| **data 依赖隔离**             | S03 fan-out / N02 early release        | matmul_pow_data_dep（pow 只等 matmul 输出 C，不提前也不全串行）            | ready-action 的依赖门控让消费者只等真实生产者                         |

### 5.2 不适合（当前模型表现差）

| 特征                                 | NEST 证据                                           | workload 证据                               | 原因                                                                     |
| ------------------------------------ | --------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------------ |
| **I/O 主导（计算占比 <10%）**        | N01（0.19%）、s11_n3_100（0.50%）、s09_large（~1%） | gather（0%）、gather_matmul（6.6%）         | 周期被 HBM bandwidth / Store 排空决定，调度器无法加速搬运                |
| **跨 context 切分（node）**          | 平均 +23-60% 开销                                   | —（workload 无同拓扑 node 对照）            | 每个 context 边引入 HBM prefetch+store 往返；HBM 字节翻倍                |
| **粗粒度跨 context 事件门（stage）** | S08 stage 比 single 慢 27%                          | —                                           | context_done 携带无关组尾（§8.3 不足二未解决）                           |
| **UCE pin 争用**                     | s02_same_pin 75,766 vs 59,027（+28.4%）；backpressure 20,540 cycles | —（workload 无 pin 争用） | 共享 pin0 把 4 个物理 context 压成 1 个有效吞吐；修复在编译器 pin 分配策略 |
| **Store 排空长尾**                   | N01 Store 占 99.8% 周期                             | matmul_pow_data_dep store 502K cycles       | device slot 在 Store 期间保留（§8.5 P3 未实施）                          |
| **单资源热点**                       | §8.6：104/104 HBM 只用 channel0                     | —（workload 未开 --memory-trace 细查 bank） | 模型限制（transfer.py 单通道 + allocator bank0 first-fit），非调度器 bug |

## 6. 按模型类型的适用性总结

| 模型类型                        | 代表 example                | 适合？                          | 关键判断                                                                                                                      |
| ------------------------------- | --------------------------- | ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Dense GEMM 链**               | S01 / matmul_2048x512       | ✅ 适合                          | 计算主导；single 最优；4 context x placement15 x 4 task 的 2048x512 matmul 166K cycles，compute 39.5%                         |
| **Gather / 随机访存**           | gather / gather_matmul      | ⚠️ 有限                          | Gather 本质 IO 主导（compute 0-6.6%）；调度器无法加速随机访存延迟；Gather 的 L1 cache/MSHR 建模是独立瓶颈                     |
| **GEMM + Gather 混合**          | matmul_gather_add / 4t_2ctx | ⚠️ 谨慎                          | 多 tile 多 context 下 compute 12%，IO 仍占大头；4tiles_2ctx 的 ctx_switch=14 说明跨 context 开销可见                          |
| **多 GEMM 并行**                | matmul_pow_parallel / S02   | ✅ 适合                          | 独立 matmul + pow 并发 submit，各自推进；compute 33.5%                                                                        |
| **GEMM → pow 数据依赖**         | matmul_pow_data_dep / S01   | ✅ 适合                          | pow 只等 matmul 输出 C（ready-action 依赖门控）；compute 35.6%；但 store 502K 是瓶颈                                          |
| **GEMM + pow 尾块重叠**         | matmul17_pow_tail_overlap   | ✅ 适合                          | 17 个 tile context（4×placement15 + 1×placement1 尾块）；尾块独占 tile0 时 pow 提前占用其余资源；compute 40.3% 全 corpus 最高 |
| **slot 复用 / free-slot 调度**  | matmul_pow_free_slot / T04  | ✅ 适合                          | 首个 matmul 提前结束释放 slot 给 pow；device slot 复用有效                                                                    |
| **多独立链并行**                | S02 / pow_dual_context      | ✅ 适合                          | 独立链各自推进；pow_dual 14.9% compute                                                                                        |
| **顺序提交对照**                | pow_sequential_contexts     | —（反例）                       | 顺序提交下 store=load（无重叠）；作为并发提交的对照组                                                                         |
| **Fan-out / 广播**              | S03                         | ✅ 适合                          | L2 内共享高效；input-only 早释放有效                                                                                          |
| **Fan-in / Join**               | S04                         | ✅ 适合                          | 正确等待全部前驱                                                                                                              |
| **残差 / skip connection**      | S05                         | ✅ 适合                          | identity skip 不复制数据                                                                                                      |
| **不平衡分支**                  | S06                         | ✅ 适合                          | 短分支不被慢分支阻挡                                                                                                          |
| **嵌套 fork/join**              | S07                         | ⚠️ 谨慎                          | 层数多时 single L2 峰值高；node +39%                                                                                          |
| **N 型交叉依赖**                | S08                         | ✅ 适合（ready-action 最佳案例） | D 越过 A 长尾 −15,934 cycles                                                                                                  |
| **长 live range / 大 tensor**   | S09 / S10                   | ⚠️ 谨慎                          | 大 tensor 的 node 搬运占比极高                                                                                                |
| **规约树 / All-reduce**         | S11                         | ⚠️ 谨慎                          | context 数多（11-13），HBM 字节最大                                                                                           |
| **Attention（QK/SM/PV）**       | S12                         | ✅ 适合                          | 跨引擎重叠有效                                                                                                                |
| **分块流水（Tiling pipeline）** | T01                         | ✅ 适合                          | 唯一 node 不增开销的拓扑                                                                                                      |
| **波前 / 依赖密集网格**         | T02                         | ⚠️ 谨慎                          | 细粒度依赖密集；stage 整行门 +30%                                                                                             |
| **状态递推 / SSM**              | T03                         | ✅ 适合                          | 静态展开正确；stage 仅 +6%                                                                                                    |
| **多请求 / multi-query**        | T04                         | ⚠️ 谨慎                          | node +60%；stage slot 复用 +7% 更优                                                                                           |
| **MoE / 专家路由**              | （无直接 example）          | ⚠️ 推断                          | 类似 S03 fan-out + T04 多请求；专家间独立但共享 input                                                                         |

## 7. 瓶颈归因与改进方向

### 7.1 当前模型的瓶颈（按影响力排序）

1. **HBM 搬运税（最大）**：node/stage 切分引入的额外 HBM 往返是 NEST 周期增长主因。T04 node +60%、S09 node +43%。真实 workload 的 MFE store service（297K-540K cycles）也证实写回 HBM 是主要 IO 瓶颈。根因是跨 context 边只能走 HBM（§8.3 不足二：跨 context 可见事件粒度只有 context_done）。
2. **单资源热点（模型限制）**：HBM 单通道选择 + L2 bank0 first-fit 使"增加并发"不等于"增加带宽"（§8.6 不足五未解决）。
3. **Store 排空长尾**：N01 的 Store 占 99.8% 周期；matmul_pow_data_dep 的 store 502K cycles 占总周期 136%。device slot 在 Store 期间保留（§8.5 P3 未实施）。
4. **UCE pin 争用**：same_pin 把多 context 压成单 context 的有效吞吐（NEST 12 cases，workload 0 cases）。

### 7.2 ready-action 已解决的

- **静态发射顺序的 head-of-line blocking**：121/121 case 的 group_ordering_stall = 0。T04 A1→B1 间隔 24,784→2 cycles（当前输入新鲜值）。
- **中间输出保守回收**：S03 D100 的 A release 不再被纯读者计算尾拖住（提前 21,704 cycles）。

### 7.3 改进优先级建议

| 优先级 | 方向                                          | 预期收益                   | 难度              |
| ------ | --------------------------------------------- | -------------------------- | ----------------- |
| P0     | 跨 context per-tensor 可见事件（§8.3 不足二） | 消除 stage 的粗粒度事件门  | 高（IR/ABI 扩展） |
| P0     | HBM 地址分散 / burst striping（§8.6 不足五）  | 让多并发真正使用多通道     | 中                |
| P1     | L2 shared memory 跨 context 共享（R3-6）      | weight 共享不重复 prefetch | 高                |
| P1     | 执行配额与排空解耦（§8.5 P3）                 | Store 期间释放 device slot | 中                |
| P2     | bank-aware L2 放置                            | 减少 bank0 竞争            | 中                |
| P2     | L1 admission release 驱动重试（R3-2）         | L1 临时容量不足的等待效率  | 低                |

## 8. 数据完整性与限制

- **121 份完成 trace 的 X slice / instant / counter 与原始 JSON 逐计数精确相等**（Perfetto parse 审计）；仅 legacy flow 有 ~13% 绑定残差，不参与指标计算。
- **L2 read/write service 时长**：NEST 本次未开 --memory-trace；workload 用 MFE load/store service 替代（cat=MFE_LD0/MFE_ST0 的 sum(dur)），定性一致。
- **旧基线不可比**：NEST 全部 112 个输入被 tile.free 审查修改过；workload 为当前源码。
- **不跨 corpus 按原始周期排快慢**：各 example 的 shape/bytes/repeat 刻意不同（guide §2.3）；判断基于配对对照（single/node/stage）和归一化指标（compute/cycles、stall 归因）。
- **runtime fidelity 不适用于性能判断**。
- 周期数是当前 HardwareConfig 下的观测值，不是硬件承诺或性能上限。

## 附录：数据源

- NEST 指标：[metrics_all.json](metrics_all.json) — 112 条
- workload 指标：[workloads/workload_metrics.json](workloads/workload_metrics.json) — 13 条
- Perfetto SQL：[perfetto_queries.sql](../sec9_ready_action_audit/perfetto_queries.sql)
- parse 审计：[perfetto_parse_audit.json](../sec9_ready_action_audit/perfetto_parse_audit.json)
- trace_processor 二进制：`examples/artifacts/tools/trace_processor_shell`（项目内，gitignored）