# NEST 子图库：运行结论、结构与测试目的

> 范围：`examples/scenarios/nest_subgraphs/` 下全部 **112 份独立 MLIR**。本文原有周期表、§8 诊断和 `full_memory` / `runtime` 链接均为 **2026-09-10 修正前合同基线**，不是迁移后源码的实测结果。迁移前 SHA-256 已与原 summary 全量核对；原文/hash 保存在 [migration_inventory.json](../../artifacts/nest_subgraphs/l2_access_contract/migration_inventory.json)。当前源码已切换 bindings + 真实 ins/outs；新结果见 §8.4 的独立证据目录。周期数不是硬件承诺或性能上限。
>
> 本次全量 `tile.free` 审查已改变部分输入。本文 §8.4 的2026-09-11测量同样是审查前快照；
> 当前逐文件决定、原始/新 hash 和安全释放点见 [全量审查](../../artifacts/tile_free/full_mlir_audit-20260912-110535Z/file_decisions.json)。
> 不用旧 trace 的周期或 source hash 冒充当前输入的结果。
>
> [本轮重新生成的 trace、完整配置与执行结果](../../artifacts/tile_free/full_mlir_audit-20260912-110535Z/execution/summary.json)
> 已覆盖239次调用；两种 fidelity 的112例仍各为108 completed、2 verify 拒绝、2容量 fault。
>
> Ready-Action 切换后，上述数字也只作为历史基线：当前默认是共享 S1，
> CPU submit 通过独立 pending/message port，Tile context 不再由 Device 数量自动扩充。
> `nest.await/barrier` 的显式控制边仍保留，L2 FIFO 与最后使用点 release 仍保留。
> 请重新运行需要比较的子图；不要把历史多 sequencer 结果当作等带宽 S0。
> 本轮实现与独立验证见 [实施记录](../../../design/proposal/03_NEST_Ready_Action_Implementation_Plan.md)。

## 1. 运行结论与证据边界

### 1.1 两种模式的结果

| 项目                       | full_memory | runtime |
| -------------------------- | ----------: | ------: |
| 实际执行输入               |         112 |     112 |
| 正常完成，exit 0           |         108 |     108 |
| 预期容量 fault，exit 1     |           2 |       2 |
| 预期 verifier 拒绝，exit 2 |           2 |       2 |
| 真实 trace JSON            |         110 |     110 |
| 运行报告 JSON              |         110 |     110 |
| 日志                       |         112 |     112 |
| 非预期失败                 |           0 |       0 |

四份非零退出是测试本身要求的结果，不应改写为成功：

- `s09_single_short`、`n09_impossible`：永久 L2 admission capacity fault，保留 fault trace 和 report。
- `n08_invalid_zero`、`n08_invalid_count`：verifier 在仿真开始前拒绝，因此没有 trace/report，只保留具体错误日志。

本次导出验证确认：文件清单完整；输入 hash 在两种模式下相同；报告完成状态/退出码符合上述分类；全部 220 份 trace 通过 `Tracer.assert_well_formed()`；有 transfer 的 runtime trace 使用单腿路径，full_memory 使用多腿路径。

**这不等于数值正确性或全图最优调度认证。** 下文“主要目的”描述案例要检验的行为，不是把每一项都自动标为 PASS。外部 MLIR 的通用 CLI 检查不能替代逐条 tensor use-def、阶段事件和调度质量检查。

**修正前调度诊断（详见 §8）**：静态发射顺序不能主动绕开无关等待、跨 Context 完成事件偏粗、中间 Buffer 的旧双绑定与回收过于保守，以及内存布局/性能指标限制了优化判断。它们不能统称为调度器错误。旧分析覆盖两模式全部 220 份基线 trace；本次只修正访问与释放合同，不改变调度、成本或 Context partition。

### 1.2 可从当前 trace 直接观察的代表结论

以下事件周期使用本次默认 `clock_mhz=1000` 换算成整数 cycle；S08、S07、S12 的计算活动均限定在 **Tile0**，不累加跨 Tile 交集，也不用 `uce_issue` 或 device slot run 窗口冒充计算重叠。

| 案例                 | 本次 full_memory 观测                                                                                        | 可以支持的结论                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `s08_single`         | D.compute_start=16976 < A.compute_end=35318；总周期 58227                                                    | D 在 A 尚未完成计算时已有实际计算活动。                                                                    |
| `s08_single_barrier` | D.compute_start=44308 > A.compute_end=35318；总周期 79906                                                    | 等待/提交顺序把 D 推迟；正常完成不代表 Scheduling Quality 合格。                                           |
| `s08_node`           | D.compute_start=19924 < A.compute_end=37882；总周期 62820                                                    | D 在 A 尚未完成计算时已有实际计算活动。                                                                    |
| `s08_node_barrier`   | D.compute_start=53034 > A.compute_end=37882；总周期 83769                                                    | 等待/提交顺序把 D 推迟；正常完成不代表 Scheduling Quality 合格。                                           |
| `s07_node_c100`      | F 的 Tile0 compute 为 [48985,49245)，与 C 的实际 BOA compute 交集为 259 cycles                               | 在当前 C repeat1000 诊断中，F 无需等待 C；不是只凭 C 的 Store 排空证明重叠。                               |
| `s12_node_v100_evu`  | QK 的 Tile0 compute 为 [47923,48183)，与 V 的实际 EVU compute 交集为 259 cycles                              | 隔离 BOA 竞争后，QK 与 V 可实际并行；当前 V repeat1000。                                                   |
| `n02_delayed_read`   | last_X_load_complete=45854 ≤ release_X=45863 < D.EVU_end=71955                                               | 共享 input X 可在最后真实读取结束后、D 全部计算结束前释放。D 前段 BOA 是 Gate 工作，不是这里比较的结束点。 |
| `n03_wait_capacity`  | Waiter 在 cycle0 排队；admitted=2138，first_action=2139；Waiter context_done=3870，Holder context_done=29695 | 容量等待后重新接纳，且首动作在接纳的下一周期；仅这些事件本身不替代全部资源隔离检查。                       |
| `n04_fifo_hol`       | Head/Tail 都在 cycle0 排队；均在 cycle35024 admitted，事件顺序 Head→Tail；首动作均为35025                    | FIFO 排队重试的接纳顺序得到保留，同周期不等于没有先后。                                                    |
| `t04_stage`          | R1 在50153完成；R3 在50154接纳到 slot1，generation 1→3；R0 到79453才完成                                     | 默认长尾版本确实复用 R1 的槽位，且 R0 仍在飞；未注入旧通知。                                               |
| `t04_stage_uniform`  | R0 在57595完成，R1 在62723完成；R3 在62724接纳到 slot0、generation3                                          | first-free 复用的是先释放的 R0 槽，而不是固定要求 R1 的 slot1。                                            |

初始 repeat100 的 S07/S12 诊断未观察到要求的重叠，两份具名文件此前已按批准分支改为 repeat1000；`diagnostic_fallback` 元数据注明。原周期表使用修正前112份输入对应的导出，不与本次新合同结果混算。

### 1.3 证据与导航

- [调度子图原始规范](../../../NEST_Context_Scheduling_Test_Spec.md)：测试目标与边界来源。
- [导出验证记录](../../artifacts/nest_subgraphs/verification.json)：逐例 hash/输出检查所对应的验证结果、trace 结构检查和 transfer-leg 分类；hash 与完整命令见下方 summary。
- full_memory：[输出目录](../../artifacts/nest_subgraphs/full_memory/)、[summary.json](../../artifacts/nest_subgraphs/full_memory/summary.json)、[index.csv](../../artifacts/nest_subgraphs/full_memory/index.csv)。
- runtime：[输出目录](../../artifacts/nest_subgraphs/runtime/)、[summary.json](../../artifacts/nest_subgraphs/runtime/summary.json)、[index.csv](../../artifacts/nest_subgraphs/runtime/index.csv)。
- [模拟器限制](../../../pipeline_validator/Limitation.md)。

本次 full_memory 导出为 UTC 12:33:21–12:37:39，runtime 为 UTC 12:33:18–12:34:53。`examples/artifacts/` 是 gitignore 中的本地生成目录；未生成这些产物的工作区需要重新运行，文中的 trace 链接才有对应文件。

## 2. 如何理解这些子图

### 2.1 组织与计数

| 类别    | 数量 | 组成                                                           |
| ------- | ---: | -------------------------------------------------------------- |
| S01–S12 |   74 | 36 份 single/node/stage 基础映射 + 38 份参数变体               |
| T01–T04 |   23 | 12 份基础映射 + 11 份时间展开/请求变体                         |
| N 边界  |   15 | 正常边界、容量 fault 与 verifier 反例；不额外伪造 N05/N06 文件 |
| 总计    |  112 | 48 基础映射 + 49 参数变体 + 15 N 类案例                        |

### 2.2 Context 映射与数据可见性

- **single**：全图在一个 Context，节点间主要通过 L2 与 `output_ready` 关联；不同 dispatch 的 UCE pin 明确区分。Context 仍有线性 PC，等待放置会影响可发射工作。
- **node**：S 图通常每节点一个 Context，跨 Context 边经 HBM；T01 按完整 chunk 切分，T03 按 step，T04 按请求内 A/B/C 节点。
- **stage**：按分支或阶段合并。下文 `{A,B}/{C,D}` 表示两个 Context，不表示阶段事件可以跨 Context 导出。
- 当前 IR 的跨 Context 可见事件粒度是 `context_done`：消费者 submit 前等待所需 producer Context，包含其必要 HBM 写回。这是当前 IR 的保守实现，不是架构上“所有边都必须等 context_done”的结论。
- 当前 dispatch 用 `bindings` 保留唯一 positional actual 列表，`ins`/`outs` 精确声明真实读/写集合。Store 只等真实写者；所有 role 的 release 等全部 reader.input_released、prefetch 与 Store completion。旧双绑定造成的保守等待仅保留为 §8.4 修正前基线。
- Device slot 和 Tile-local UCE context 是两层资源。默认 4×4 指 **device-context-mode × context-mode**；single 是 1×4，只有 `t01_single_one_uce` 是 1×1。共享 pin0 不会把物理配置自动变成 Nx1。

### 2.3 数据、成本与两种 fidelity

- 默认 tensor `4×64×64xbf16`，记为 **U=32768 B**；每 task 处理连续 `1×64×64`。常规 placement=15、range0..4、alignment=256；N08 等边界以各文件实际形状/范围为准。
- `large` 通常把指定输出扩大到 `4×256×64xbf16`，即 **4U=131072 B**。读取方的 view/L1/bytes 同步扩大；BOA 输出 M 扩大时，其第一输入与 `ops=2*M*N*K` 也同步调整。
- 通常 BOA 为 m=n=k=64、ops=524288；EVU 默认 ops=16448。repeat 是静态重复 compute 并 await 的模拟成本扫描，不是“真实 tensor GEMM 变成更大 shape”。Copy 节点只做 load/store，不凭空增加 EVU 计算。
- `load10` 是重复真实同址 load/await；`store10` 是串行重复对应 HBM Store，最后一次才允许最终释放/完成。`n3_100` 重复的是 Copy 的 load，而非 compute。
- **full_memory** 模拟 HBM/NoC/DMA/SRAM-bank 多阶段竞争；**runtime** 保留真实地址、分配和生命周期，transfer 折叠成单腿时延。两者均开启 memory trace，但 runtime 不是详细内存性能模型。
- 两模式 cycle 差异来自 fidelity；single/node/stage 还改变 L2/HBM 路径、Context 数和驻留数据量。因此不能仅凭总周期较小就断言 scheduler 更聪明。

## 3. 全部 S 类子图：静态数据依赖

每个文件占下表中的一行。**F/R 链接分别为 full_memory/runtime 的 trace**；cycle 数是对应报告的全程 `cycles`，不是某一 engine 的累加 active cycles。`完成` 仅指两模式均正常完成；非法输入的 `—` 表示没有开始仿真。

### 3.1 S01：纯串行链

**结构**：`A → B → C → D`。

Matmul → BiasAdd → Activation → Matmul；B 额外读取独立 Bias。

**目的**：检查串行真依赖、阶段事件和中间数据生命周期；不能把 compute 完成当作 HBM Store 完成。

| 子图（源码，省略 .mlir）                    | 大概信息 / 切分与参数                             | 主要目的                                                                      | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                          |
| ------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s01_single`](s01_single.mlir)             | 1 Context，全图；按节点固定 UCE pin               | 在 纯串行链 中检查组内 L2/output_ready 依赖与线性 PC 等待。                   |              41886 |           2108 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s01_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s01_single.trace.json)             |
| [`s01_node`](s01_node.mlir)                 | 4 Context，每节点独立，dispatch 随 device slot    | 在 纯串行链 中检查逐节点 HBM 传递和 producer context_done 可见性。            |              56376 |           2902 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s01_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s01_node.trace.json)                 |
| [`s01_stage`](s01_stage.mlir)               | {A,B}/{C,D}                                       | 比较 纯串行链 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。   |              46161 |           2566 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s01_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s01_stage.trace.json)               |
| [`s01_node_a10`](s01_node_a10.mlir)         | A compute repeat10，其他节点不变；4 个 Context    | 观察串行链首段长尾，后继必须仍只消费自己的真实输入。                          |              58727 |           5253 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s01_node_a10.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s01_node_a10.trace.json)         |
| [`s01_node_b100`](s01_node_b100.mlir)       | B compute repeat100，保留 Bias 输入；4 个 Context | 观察链中段长尾，区分前驱完成与本节点完成事件。                                |              82266 |          28792 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s01_node_b100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s01_node_b100.trace.json)       |
| [`s01_node_store10`](s01_node_store10.mlir) | A 的 HBM Store 顺序重复 10 次；4 个 Context       | 检查下游跨 Context 读取必须等最后一次 Store，而非 A 的 compute/output_ready。 |              84024 |           4432 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s01_node_store10.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s01_node_store10.trace.json) |

### 3.2 S02：三条独立链

**结构**：`A0 → A1 → A2；B0 → B1；C0 → C1 → C2`。

三条链之间没有数据边；基础配置混合 BOA 与 EVU。

**目的**：检查一条链等待或变慢时其他链能否推进，并区分同引擎或共享 UCE pin 的资源争用。

**注意**：三根先发射；共享 pin 等待属于实际 UCE 争用，不是数据边。

| 子图（源码，省略 .mlir）                              | 大概信息 / 切分与参数                                                          | 主要目的                                                                      | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s02_single`](s02_single.mlir)                       | 1 Context，全图；按节点固定 UCE pin                                            | 在 三条独立链 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              68112 |           3400 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s02_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s02_single.trace.json)                       |
| [`s02_node`](s02_node.mlir)                           | 8 Context，每节点独立，dispatch 随 device slot                                 | 在 三条独立链 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              70718 |           3369 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s02_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s02_node.trace.json)                           |
| [`s02_stage`](s02_stage.mlir)                         | {A0,A1,A2}/{B0,B1}/{C0,C1,C2}                                                  | 比较 三条独立链 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              58571 |           2010 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s02_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s02_stage.trace.json)                         |
| [`s02_stage_a100`](s02_stage_a100.mlir)               | A0 compute repeat100；切分 {A0,A1,A2}/{B0,B1}/{C0,C1,C2}                       | 检查 B/C 独立链在 A 长尾期间仍有实际 service。                                |              64696 |          27752 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s02_stage_a100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s02_stage_a100.trace.json)               |
| [`s02_stage_c100`](s02_stage_c100.mlir)               | C0 compute repeat100；切分 {A0,A1,A2}/{B0,B1}/{C0,C1,C2}                       | 交换慢链，检查隔离结论不依赖固定 A 链。                                       |              60372 |          27942 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s02_stage_c100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s02_stage_c100.trace.json)               |
| [`s02_stage_homogeneous`](s02_stage_homogeneous.mlir) | 全链改为同成本 EVU；切分 {A0,A1,A2}/{B0,B1}/{C0,C1,C2}                         | 减少引擎类型差异，观察同构资源竞争；不保证所有链同周期完成。                  |              44728 |           2696 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s02_stage_homogeneous.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s02_stage_homogeneous.trace.json) |
| [`s02_stage_same_pin`](s02_stage_same_pin.mlir)       | 所有 dispatch 固定 pin0；设备仍为 4×4 配置；切分 {A0,A1,A2}/{B0,B1}/{C0,C1,C2} | 制造 UCE 争用对照；不能把共享 pin 称为物理 Nx1。                              |              72135 |           3462 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s02_stage_same_pin.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s02_stage_same_pin.trace.json)       |

### 3.3 S03：单生产者扇出

**结构**：`A → B、C、D`。

A 为 BOA；B/C/D 分别进行 pow timing 工作并真实读取 A。

**目的**：检查完成事件能唤醒所有消费者，共享输出在最后读取前不能释放或覆盖。

**基线注意**：下表旧合同 intermediate out/inout 曾保守持有至读者 output_ready/最终 Store；当前输入已切换真实访问合同，新测量与旧值分开见 §8.4。

| 子图（源码，省略 .mlir）                  | 大概信息 / 切分与参数                          | 主要目的                                                                        | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                        |
| ----------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s03_single`](s03_single.mlir)           | 1 Context，全图；按节点固定 UCE pin            | 在 单生产者扇出 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              33854 |           2134 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s03_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s03_single.trace.json)           |
| [`s03_node`](s03_node.mlir)               | 4 Context，每节点独立，dispatch 随 device slot | 在 单生产者扇出 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              34399 |           1952 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s03_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s03_node.trace.json)               |
| [`s03_stage`](s03_stage.mlir)             | {A}/{B,C}/{D}                                  | 比较 单生产者扇出 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              34403 |           1956 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s03_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s03_stage.trace.json)             |
| [`s03_single_d100`](s03_single_d100.mlir) | D compute repeat100；1 Context                 | 检查全部消费者仍执行；中间 out/inout 的保守存活不同于 N02 input-only 早释放。   |              59693 |          27973 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s03_single_d100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s03_single_d100.trace.json) |

### 3.4 S04：多生产者汇合

**结构**：`A、B、C → D`。

A/B/C 是独立根，D 实际读取三个输出。默认同 service-cost 不意味着实际同周期到达。

**目的**：检查 Join 等待全部 distinct producers；任何一个前驱完成都不能替代其他输入。

**注意**：abc/cab/bca 表示成本配置，不承诺实际到达顺序，更不承诺三根同周期完成。

| 子图（源码，省略 .mlir）            | 大概信息 / 切分与参数                          | 主要目的                                                                        | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                  |
| ----------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s04_single`](s04_single.mlir)     | 1 Context，全图；按节点固定 UCE pin            | 在 多生产者汇合 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              40047 |           2078 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s04_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s04_single.trace.json)     |
| [`s04_node`](s04_node.mlir)         | 4 Context，每节点独立，dispatch 随 device slot | 在 多生产者汇合 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              44679 |           1906 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s04_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s04_node.trace.json)         |
| [`s04_stage`](s04_stage.mlir)       | {A,B}/{C}/{D}                                  | 比较 多生产者汇合 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              45492 |           2009 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s04_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s04_stage.trace.json)       |
| [`s04_node_abc`](s04_node_abc.mlir) | A/B/C compute repeat 为 1/10/100；4 个 Context | 扫描名义长尾次序，检查 Join 等三份真实输入；实际到达序以 trace 为准。           |              61774 |          30062 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s04_node_abc.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s04_node_abc.trace.json) |
| [`s04_node_cab`](s04_node_cab.mlir) | A/B/C compute repeat 为 10/100/1；4 个 Context | 改变名义最快/最慢前驱，检查 fan-in 对不同完成次序的适应。                       |              59469 |          27779 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s04_node_cab.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s04_node_cab.trace.json) |
| [`s04_node_bca`](s04_node_bca.mlir) | A/B/C compute repeat 为 100/1/10；4 个 Context | 继续置换成本，避免 Join 仅对固定前驱顺序有效。                                  |              69480 |          27724 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s04_node_bca.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s04_node_bca.trace.json) |

### 3.5 S05：菱形与残差

**结构**：`A → B、C；B、C → D`。

基础是 fork/join；两份 residual 变体改为 X → F0 → F1 → Add，并加入 identity 或 Copy skip。

**目的**：检查分支独立推进、最终汇合，以及原地址跳连与真实数据复制的区别。

| 子图（源码，省略 .mlir）                                          | 大概信息 / 切分与参数                                                              | 主要目的                                                                      | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                                                |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s05_single`](s05_single.mlir)                                   | 1 Context，全图；按节点固定 UCE pin                                                | 在 菱形与残差 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              40016 |           1909 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s05_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s05_single.trace.json)                                   |
| [`s05_node`](s05_node.mlir)                                       | 4 Context，每节点独立，dispatch 随 device slot                                     | 在 菱形与残差 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              51277 |           2369 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s05_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s05_node.trace.json)                                       |
| [`s05_stage`](s05_stage.mlir)                                     | {A}/{B,C}/{D}                                                                      | 比较 菱形与残差 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              50594 |           2381 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s05_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s05_stage.trace.json)                                     |
| [`s05_stage_residual_identity`](s05_stage_residual_identity.mlir) | X → F0 → F1 → Add；Add 同时读原 X 地址；切分 {X}/{F0,F1}/{Add}                     | 检查 identity skip 不复制数据，原 tensor 的地址和生命周期保持正确。           |              48166 |           2696 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s05_stage_residual_identity.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s05_stage_residual_identity.trace.json) |
| [`s05_stage_residual_copy`](s05_stage_residual_copy.mlir)         | 新增 Copy：load X → store 独立 Skip；Add 读 Skip/F1；切分 {X}/{F0,F1}/{Copy}/{Add} | 检查真实 Copy 的独立地址与 Store，不把 identity 依赖冒充数据复制。            |              53306 |           2860 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s05_stage_residual_copy.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s05_stage_residual_copy.trace.json)         |

### 3.6 S06：不平衡菱形

**结构**：`A → B、C；C → D；B、D → J`。

B 是单节点分支，C/D 是双节点分支；用不同成本放大 load、compute 或 Store 长尾。

**目的**：检查无关慢分支不阻止短分支推进，J 仍需等待全部真前驱。

| 子图（源码，省略 .mlir）                            | 大概信息 / 切分与参数                                            | 主要目的                                                                      | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                                  |
| --------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s06_single`](s06_single.mlir)                     | 1 Context，全图；按节点固定 UCE pin                              | 在 不平衡菱形 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              43093 |           2402 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s06_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s06_single.trace.json)                     |
| [`s06_node`](s06_node.mlir)                         | 5 Context，每节点独立，dispatch 随 device slot                   | 在 不平衡菱形 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              52734 |           2826 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s06_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s06_node.trace.json)                         |
| [`s06_stage`](s06_stage.mlir)                       | {A}/{B}/{C,D}/{J}                                                | 比较 不平衡菱形 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              50291 |           2658 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s06_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s06_stage.trace.json)                       |
| [`s06_stage_compute100`](s06_stage_compute100.mlir) | B compute repeat100；切分 {A}/{B}/{C,D}/{J}                      | 检查短分支 C/D 在 B 仍有实际 task/compute 活动时推进。                        |              69451 |          28221 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s06_stage_compute100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s06_stage_compute100.trace.json) |
| [`s06_stage_load10`](s06_stage_load10.mlir)         | B 的真实 input load/await 顺序重复 10 次；切分 {A}/{B}/{C,D}/{J} | 制造读取长尾，检查 input_released 不早于最后一次实际 load。                   |              97645 |           3029 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s06_stage_load10.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s06_stage_load10.trace.json)         |
| [`s06_stage_store10`](s06_stage_store10.mlir)       | B 的 HBM Store 顺序重复 10 次；切分 {A}/{B}/{C,D}/{J}            | 制造写回长尾；J 的跨 Context 可见性不能使用 compute_done 代替。               |              76410 |           3861 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s06_stage_store10.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s06_stage_store10.trace.json)       |
| [`s06_stage_reversed`](s06_stage_reversed.mlir)     | D compute repeat100，B 恢复默认快分支；切分 {A}/{B}/{C,D}/{J}    | 反转慢支路，检查分支隔离不固定依赖 B 慢/D 快假设。                            |              75065 |          28546 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s06_stage_reversed.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s06_stage_reversed.trace.json)     |

### 3.7 S07：嵌套 fork/join

**结构**：`A → B、C；B → D、E；D、E → F；F、C → G`。

C 为 BOA，其余节点为 EVU；F 的真前驱只有 D/E，不能增加 C → F。

**目的**：区分局部 Join 与最终 Join，暴露把局部等待扩大成全层 Barrier 的问题。

**注意**：c100 文件已按诊断分支加长为 C repeat1000；不能按文件名解释最终工作量。

| 子图（源码，省略 .mlir）              | 大概信息 / 切分与参数                                                      | 主要目的                                                                          | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                    |
| ------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s07_single`](s07_single.mlir)       | 1 Context，全图；按节点固定 UCE pin                                        | 在 嵌套 fork/join 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              56468 |           3161 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s07_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s07_single.trace.json)       |
| [`s07_node`](s07_node.mlir)           | 7 Context，每节点独立，dispatch 随 device slot                             | 在 嵌套 fork/join 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              73241 |           3821 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s07_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s07_node.trace.json)           |
| [`s07_stage`](s07_stage.mlir)         | {A}/{B,D,E,F}/{C}/{G}                                                      | 比较 嵌套 fork/join 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              66662 |           3489 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s07_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s07_stage.trace.json)         |
| [`s07_node_c100`](s07_node_c100.mlir) | 当前 C 实际 repeat1000；D/E 各 repeat5；名称保留 c100 参数族；7 个 Context | repeat100 初测未观察到目标重叠后的加长诊断；验证 F 不依赖 C，而 G 仍需 C。        |             307406 |         263673 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s07_node_c100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s07_node_c100.trace.json) |

### 3.8 S08：N 型交叉依赖

**结构**：`A、B → C；B → D`。

默认 A/B/C/D compute repeat 分别为 100/5/5/90；正例先推进只依赖 B 的 D。

**目的**：比较正确依赖下的发射顺序：D 不应被无关的 A/C 等待阻挡。

**注意**：stage 为 {A}/{B,D}/{C}，C 只能等待整个 BD Context 完成，是明确的非最优 coarse-event 映射。barrier 变体只改变等待/提交顺序，不制造 C 的额外输出数据依赖。

| 子图（源码，省略 .mlir）                        | 大概信息 / 切分与参数                                                 | 主要目的                                                                        | full_memory cycles | runtime cycles | 两模式结果（exit）  | Trace / 日志                                                                                                                                              |
| ----------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s08_single`](s08_single.mlir)                 | 1 Context，全图；按节点固定 UCE pin                                   | 在 N 型交叉依赖 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              58227 |          28590 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s08_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s08_single.trace.json)                 |
| [`s08_node`](s08_node.mlir)                     | 4 Context，每节点独立，dispatch 随 device slot                        | 在 N 型交叉依赖 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              62820 |          28568 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s08_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s08_node.trace.json)                     |
| [`s08_stage`](s08_stage.mlir)                   | {A}/{B,D}/{C}                                                         | 比较 N 型交叉依赖 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              66642 |          28568 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s08_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s08_stage.trace.json)                   |
| [`s08_single_barrier`](s08_single_barrier.mlir) | C 前先 await A/B ready，再 issue C、D；不添加 C → D 完成边；1 Context | 故意让局部线性 PC 等待挡住 D；作为正确性可过但调度质量差的对照。                |              79906 |          51900 | 完成（0）；质量反例 | [F](../../artifacts/nest_subgraphs/full_memory/s08_single_barrier.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s08_single_barrier.trace.json) |
| [`s08_node_barrier`](s08_node_barrier.mlir)     | 把 C 所需的 A/B Context await/submit 放在 D submit 前；4 个 Context   | 与 node 正例保持工作/地址/硬件一致，测量人为发射顺序造成的延迟。                |              83769 |          52055 | 完成（0）；质量反例 | [F](../../artifacts/nest_subgraphs/full_memory/s08_node_barrier.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s08_node_barrier.trace.json)     |

### 3.9 S09：稠密跳连

**结构**：`A → B；A、B → C；A、B、C → D`。

A 为 BOA，B/C/D 为 EVU；D 仍需实际读取 A/B，不能用依赖的传递化简删除 tensor read。

**目的**：检查长 live range、多输入地址、大小传播，以及完整 L2 bundle 的容量边界。

**注意**：short 少的是明确的 256 B 测试步长，不是已冻结的硬件 allocation quantum。

| 子图（源码，省略 .mlir）                    | 大概信息 / 切分与参数                                                        | 主要目的                                                                    | full_memory cycles | runtime cycles | 两模式结果（exit）  | Trace / 日志                                                                                                                                          |
| ------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -----------------: | -------------: | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s09_single`](s09_single.mlir)             | 1 Context，全图；按节点固定 UCE pin                                          | 在 稠密跳连 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              44620 |           2436 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s09_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s09_single.trace.json)             |
| [`s09_node`](s09_node.mlir)                 | 4 Context，每节点独立，dispatch 随 device slot                               | 在 稠密跳连 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              61516 |           3102 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s09_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s09_node.trace.json)                 |
| [`s09_stage`](s09_stage.mlir)               | {A,B}/{C,D}                                                                  | 比较 稠密跳连 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              50248 |           2600 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s09_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s09_stage.trace.json)               |
| [`s09_node_a_large`](s09_node_a_large.mlir) | A 输出与 BOA 第一输入同步改为 4×256×64 bf16；消费者实际读取 4U；4 个 Context | 检查大输出的 M/ops、view、L1 输入和 transfer bytes 同步传播。               |             147310 |           6588 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s09_node_a_large.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s09_node_a_large.trace.json) |
| [`s09_node_b_large`](s09_node_b_large.mlir) | B 输出改为 4×256×64 bf16；C/D 的 B 输入同步扩大；4 个 Context                | 检查中间大 tensor 的多消费者和长跳连，不删除冗余完成边对应的真实读。        |             110764 |           4666 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s09_node_b_large.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s09_node_b_large.trace.json) |
| [`s09_single_exact`](s09_single_exact.mlir) | 完整 bundle 6U=196608 B，L2 容量恰为 196608 B；1 Context                     | 验证 exact-fit 可接纳并完成。                                               |              22692 |           2436 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/s09_single_exact.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s09_single_exact.trace.json) |
| [`s09_single_short`](s09_single_short.mlir) | 完整 bundle 仍为 196608 B，L2 容量降为 196352 B；1 Context                   | 预期永久 admission capacity fault；不是等待将来释放，也不是 cycle cap。     |                  8 |              8 | 预期容量 fault（1） | [F](../../artifacts/nest_subgraphs/full_memory/s09_single_short.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s09_single_short.trace.json) |

### 3.10 S10：U 型长跳连

**结构**：`E0、E1、E2 独立；E2 → D2；D2、E1 → D1；D1、E0 → D0`。

E2 为 BOA；E0/E1 要保留至对应 decoder 真正读取。

**目的**：检查跨多层的 skip 生命周期和大 tensor 内存压力，不能只按当前节点估算存活数据。

**注意**：stage 为整个 encoder/decoder 两组；跨组只能等待 encoder Context 完成，不能导出假的单节点 ready。

| 子图（源码，省略 .mlir）                        | 大概信息 / 切分与参数                                                    | 主要目的                                                                      | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s10_single`](s10_single.mlir)                 | 1 Context，全图；按节点固定 UCE pin                                      | 在 U 型长跳连 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              54393 |           2922 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s10_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s10_single.trace.json)                 |
| [`s10_node`](s10_node.mlir)                     | 6 Context，每节点独立，dispatch 随 device slot                           | 在 U 型长跳连 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              65133 |           3068 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s10_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s10_node.trace.json)                     |
| [`s10_stage`](s10_stage.mlir)                   | {E0,E1,E2}/{D2,D1,D0}                                                    | 比较 U 型长跳连 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              57616 |           2796 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s10_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s10_stage.trace.json)                   |
| [`s10_node_e0_large`](s10_node_e0_large.mlir)   | 仅 E0 输出扩大为 4U，D0 输入同步扩大；6 个 Context                       | 施压最长 skip 存活区间，检查 E0 不能在 D0 最后读取前丢失。                    |              94125 |           3800 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s10_node_e0_large.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s10_node_e0_large.trace.json)   |
| [`s10_node_e2_large`](s10_node_e2_large.mlir)   | 仅 E2 输出扩大为 4U，BOA 第一输入/M/ops 与 D2 输入同步调整；6 个 Context | 对照近端大 tensor 与远端大 skip 的不同内存压力。                              |             115912 |           5564 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s10_node_e2_large.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s10_node_e2_large.trace.json)   |
| [`s10_node_all_large`](s10_node_all_large.mlir) | E0/E1/E2 输出均为 4U；所有实际读方同步扩大；6 个 Context                 | 检查多个大 live range 叠加时的容量与字节一致性。                              |             152525 |           6714 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s10_node_all_large.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s10_node_all_large.trace.json) |

### 3.11 S11：规约树与 fanout

**结构**：`P0、P1 → R0；P2、P3 → R1；R0、R1 → R；R → N0…N3`。

P* 为 BOA，R* 为 EVU reduce timing 节点，N\* 是真实 load/store Copy；不是数值 Collective 基准。

**目的**：检查局部/全局 Join、广播消费者、奇数尾项，以及等待集合只包含真实 participant。

**注意**：p0/p1/p3/p5 表示归约树根数量，不是单 grid 的 task 数；不要与 N08 的参与 Tile 数混淆。没有用无 tensor operand 的 1-cycle Collective 冒充数值规约。

| 子图（源码，省略 .mlir）                  | 大概信息 / 切分与参数                                                              | 主要目的                                                                           | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s11_single`](s11_single.mlir)           | 1 Context，全图；按节点固定 UCE pin                                                | 在 规约树与 fanout 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              87383 |           4101 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_single.trace.json)           |
| [`s11_node`](s11_node.mlir)               | 11 Context，每节点独立，dispatch 随 device slot                                    | 在 规约树与 fanout 中检查逐节点 HBM 传递和 producer context_done 可见性。          |             105158 |           3988 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_node.trace.json)               |
| [`s11_stage`](s11_stage.mlir)             | {P0,P1,R0}/{P2,P3,R1}/{R}/{N0,N1,N2,N3}                                            | 比较 规约树与 fanout 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              96919 |           4190 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_stage.trace.json)             |
| [`s11_node_p3_100`](s11_node_p3_100.mlir) | 4 个 participant roots 不变，只有 P3 compute repeat100；11 个 Context              | 检查右侧 partial 长尾及最终 R 的完整等待集合。                                     |             124412 |          29827 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_node_p3_100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_node_p3_100.trace.json) |
| [`s11_node_n3_100`](s11_node_n3_100.mlir) | N3 是 Copy：真实 input load/await 重复100次，Store 仅一次；11 个 Context           | 制造慢广播消费者；不能为 Copy 凭空增加 compute。                                   |             361433 |           7629 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_node_n3_100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_node_n3_100.trace.json) |
| [`s11_node_p0`](s11_node_p0.mlir)         | participant roots=0，提交一个空 Context；1 个 Context                              | 检查无 reducer/consumer/phase waiter 的合法空路径，不产生伪数值结果。              |                  2 |              2 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_node_p0.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_node_p0.trace.json)         |
| [`s11_node_p1`](s11_node_p1.mlir)         | participant roots=1，无需 reducer；P0 输出 identity fanout 给 4 个 N；5 个 Context | 检查单 participant 的退化路径，而非等待不存在的另一半。                            |              39539 |           1408 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_node_p1.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_node_p1.trace.json)         |
| [`s11_node_p3`](s11_node_p3.mlir)         | participant roots=3，相邻配对规约，奇数尾项直达下一层，再 fanout；9 个 Context     | 检查非二次幂根数的真实 partial 等待集合。                                          |              87426 |           3302 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_node_p3.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_node_p3.trace.json)         |
| [`s11_node_p5`](s11_node_p5.mlir)         | participant roots=5，逐层相邻配对，奇数项直达，再 fanout；13 个 Context            | 检查超过基础 4 根的归约树；根数不等于某个 grid 的 Tile participant 数。            |             125309 |           4725 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s11_node_p5.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s11_node_p5.trace.json)         |

### 3.12 S12：Attention 式错位汇合

**结构**：`X → Q、K、V；Q、K → QK；QK → SM；SM、V → PV；PV → O`。

Q/K/V/QK/PV 为 BOA，SM 为 softmax，X/O 是 Copy；Q/K/V 是独立 Tile Program。

**目的**：检查 QK 只等 Q/K、PV 才等 SM/V；把真实 BOA 资源等待与错误 V → QK 依赖分开。

**注意**：v100_evu 的最终 V 为 EVU repeat1000；普通 v100 仍是 BOA repeat100。两者不是等工作量性能比较。

| 子图（源码，省略 .mlir）                      | 大概信息 / 切分与参数                                          | 主要目的                                                                                | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                            |
| --------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`s12_single`](s12_single.mlir)               | 1 Context，全图；按节点固定 UCE pin                            | 在 Attention 式错位汇合 中检查组内 L2/output_ready 依赖与线性 PC 等待。                 |              73448 |           3613 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_single.trace.json)               |
| [`s12_node`](s12_node.mlir)                   | 8 Context，每节点独立，dispatch 随 device slot                 | 在 Attention 式错位汇合 中检查逐节点 HBM 传递和 producer context_done 可见性。          |              93461 |           4198 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_node.trace.json)                   |
| [`s12_stage`](s12_stage.mlir)                 | {X}/{Q,K,QK,SM}/{V}/{PV,O}                                     | 比较 Attention 式错位汇合 的组内 L2 与组间 HBM 路径，观察 coarse-event 带来的额外等待。 |              81060 |           3793 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_stage.trace.json)                 |
| [`s12_node_v100`](s12_node_v100.mlir)         | V 保持 BOA，compute repeat100；8 个 Context                    | 观察 V 长尾；QK 可能受 BOA 资源竞争影响，不能据此添加 V 数据依赖。                      |              97372 |          29124 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_node_v100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_node_v100.trace.json)         |
| [`s12_node_q100`](s12_node_q100.mlir)         | Q compute repeat100；8 个 Context                              | 检查 QK 真正等待 Q/K 全部输入。                                                         |             107868 |          30347 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_node_q100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_node_q100.trace.json)         |
| [`s12_node_k100`](s12_node_k100.mlir)         | K compute repeat100；8 个 Context                              | 对称置换慢投影，检查 QK 不误把 Q ready 当双输入 ready。                                 |             115274 |          30297 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_node_k100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_node_k100.trace.json)         |
| [`s12_node_sm100`](s12_node_sm100.mlir)       | SM softmax compute repeat100；8 个 Context                     | 检查 PV 必须等待 SM 与 V。                                                              |             119349 |          30086 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_node_sm100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_node_sm100.trace.json)       |
| [`s12_node_v100_evu`](s12_node_v100_evu.mlir) | V 改为 EVU，当前实际 repeat1000；Q/K/QK 保持 BOA；8 个 Context | 隔离 BOA 争用并加长 V，用实际跨引擎重叠诊断 QK 不依赖 V；不是原名暗示的100次最终成本。  |             312242 |         264010 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/s12_node_v100_evu.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/s12_node_v100_evu.trace.json) |

## 4. 全部 T 类子图：时间展开与请求复用

### 4.1 T01：分块 Load/Compute/Store 流水

**结构**：`每个 chunk：Load_Ai、Load_Bi → Compute_i → Store_i；共 4 chunks`。

Compute_i 为 repeat10 BOA；single 用 A/B/O 两套 L2（共 6 allocations），0/2 和 1/3 分别复用。

**目的**：检查独立 A/B transfer event、chunk 无丢失/重复、真实流水，以及输入/输出覆写的不同保护条件。

**注意**：node 以 chunk 切分而非逐 DMA/compute 节点切分；serial 是质量反例，one_uce 是物理资源对照。

| 子图（源码，省略 .mlir）                        | 大概信息 / 切分与参数                                                     | 主要目的                                                                         | full_memory cycles | runtime cycles | 两模式结果（exit）  | Trace / 日志                                                                                                                                              |
| ----------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`t01_single`](t01_single.mlir)                 | 1 Context；2 套 A/B/O 共6个 L2 allocations；chunk0/2 pin0，1/3 pin1       | 观察同 Context 双缓冲流水：输入覆写等 input_released，输出覆写等上次全局 Store。 |              45304 |          10933 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/t01_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t01_single.trace.json)                 |
| [`t01_node`](t01_node.mlir)                     | 4 Context，每个封装一个完整 chunk 的 Load_A/Load_B/Compute/Store          | 观察 chunk 粒度并发；不把需要共享 L2 的 load/compute/store 硬拆成独立 Context。  |              46006 |          11207 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/t01_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t01_node.trace.json)                     |
| [`t01_stage`](t01_stage.mlir)                   | 2 Context：偶数 chunks0/2 与奇数 chunks1/3；各复用一套 A/B/O              | 比较两条独立复用流水的重叠和 buffer 生命周期。                                   |              43168 |          11251 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/t01_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t01_stage.trace.json)                   |
| [`t01_single_serial`](t01_single_serial.mlir)   | 四 chunks 逐个 grid+Store await；仍保留两套 buffer 与 repeat10；1 Context | 人为串行对照，不能满足不同 chunk 的流水重叠目标。                                |              70944 |          12368 | 完成（0）；质量反例 | [F](../../artifacts/nest_subgraphs/full_memory/t01_single_serial.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t01_single_serial.trace.json)   |
| [`t01_single_one_uce`](t01_single_one_uce.mlir) | 两套 buffer 不变，全部 dispatch pin0，device/UCE 配置均为1；1 Context     | 对照 UCE 资源受限的情况；不要把只有一个 UCE 当作只处理一个 logical task。        |              54557 |          11321 | 完成（0）           | [F](../../artifacts/nest_subgraphs/full_memory/t01_single_one_uce.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t01_single_one_uce.trace.json) |

### 4.2 T02：3×3 波前

**结构**：`Vij 只依赖上方 V(i−1)j 和左侧 Vi(j−1)，边界省略不存在的邻居`。

V00 读外部 input；全部为 EVU add。逻辑坐标记录在节点与数据边上，不是物理 Tile ID。

**目的**：检查局部依赖而非整行/整列 Barrier，并观察角落长尾对无关波前的影响。

| 子图（源码，省略 .mlir）                    | 大概信息 / 切分与参数                                         | 主要目的                                                                     | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                          |
| ------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`t02_single`](t02_single.mlir)             | 1 Context，9个 EVU dispatch；pin=(i+j)%4                      | 观察细粒度上/左依赖与静态发射顺序，不使用整行完成事件。                      |              73869 |           4585 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t02_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t02_single.trace.json)             |
| [`t02_node`](t02_node.mlir)                 | 9 Context，每个逻辑坐标独立                                   | 检查每个节点只等待真实上/左 producer 的跨 Context HBM 可见性。               |              89317 |           4765 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t02_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t02_node.trace.json)                 |
| [`t02_stage`](t02_stage.mlir)               | 3 Context，每行一个                                           | 作为 coarse-event 对照，记录整行 context_done 造成的额外等待；不称最优波前。 |              88353 |           4896 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t02_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t02_stage.trace.json)               |
| [`t02_node_v02_100`](t02_node_v02_100.mlir) | V02 compute repeat100；使用基础顺序优先推进左下；9 个 Context | 检查右上角长尾不应成为不相关左下节点的数据依赖。                             |              96650 |          30439 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t02_node_v02_100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t02_node_v02_100.trace.json) |
| [`t02_node_v20_100`](t02_node_v20_100.mlir) | V20 compute repeat100；采用转置顺序优先推进右上；9 个 Context | 对称检查左下角长尾，避免被静态提交顺序伪装为数据依赖。                       |              96650 |          30439 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t02_node_v20_100.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t02_node_v20_100.trace.json) |

### 4.3 T03：有限展开的状态依赖

**结构**：`State_i、Input_i → Step_i → State_(i+1)`。

默认展开 4 步，每步 EVU add；State0…State4 是不同 Buffer 和 arena 区间。

**目的**：检查状态地址/版本、迭代事件身份和 0/1/2/4 步边界；这是静态展开，不是运行时循环或 early-exit。

**注意**：zero 无 dispatch；one/two 是静态截断而非动态分支，所有状态 tensor 的 family-wide arena index 保持稳定。

| 子图（源码，省略 .mlir）                  | 大概信息 / 切分与参数                                        | 主要目的                                                     | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                        |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`t03_single`](t03_single.mlir)           | 1 Context，4步静态展开；Step_i pin=i%4                       | 检查组内 State0→State4 的不同地址/版本和 output_ready 依赖。 |              47028 |           2146 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_single.trace.json)           |
| [`t03_node`](t03_node.mlir)               | 4 Context，每 step 一个                                      | 检查四次跨 Context 状态传递与每次最终 HBM Store。            |              61516 |           2940 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_node.trace.json)               |
| [`t03_stage`](t03_stage.mlir)             | 2 Context：{Step0,Step1}/{Step2,Step3}                       | 比较组内状态 L2 传递与组间 HBM 完成事件。                    |              50252 |           2604 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_stage.trace.json)             |
| [`t03_single_zero`](t03_single_zero.mlir) | 0 步：空 Context，无 dispatch，State0 保持为结果；1 Context  | 检查零迭代不生成不存在的任务或 phase waiter。                |                  2 |              2 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_single_zero.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_single_zero.trace.json) |
| [`t03_single_one`](t03_single_one.mlir)   | 1 步：读 State0/Input0，写 State1；1 Context                 | 检查同 Context 最小非空状态更新路径。                        |              15379 |            735 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_single_one.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_single_one.trace.json)   |
| [`t03_single_two`](t03_single_two.mlir)   | 2 步静态截断，保留不同 State0/1/2；1 Context                 | 检查有限状态链；不称为运行时 early-exit。                    |              25126 |           1302 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_single_two.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_single_two.trace.json)   |
| [`t03_node_zero`](t03_node_zero.mlir)     | 0 步：仍只有一个合法空 Context；1 个 Context                 | 检查 node 映射的零工作入口，无伪造逐节点实例。               |                  2 |              2 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_node_zero.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_node_zero.trace.json)     |
| [`t03_node_one`](t03_node_one.mlir)       | 1 步，一个 step Context；1 个 Context                        | 检查最小状态实例的 HBM 输入/输出与完成语义。                 |              15379 |            735 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_node_one.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_node_one.trace.json)       |
| [`t03_node_two`](t03_node_two.mlir)       | 2 步，每 step 独立 Context，State1 经 HBM 传递；2 个 Context | 检查跨 Context 读取正确状态版本并等待写回可见。              |              30758 |           1470 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t03_node_two.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t03_node_two.trace.json)       |

### 4.4 T04：多请求共享程序与实例复用

**结构**：`四请求各自 A_r → B_r → C_r，A_r 共享只读 W`。

默认 R0.A 为 BOA repeat100，其余 A 为短 EVU；所有请求共享 B/C Tile Program，但请求数据互不别名。

**目的**：检查共享代码不共享完成事件；stage 中 R1 完成后提交 R3，观察 first-free slot 与 launch_generation 隔离。

**注意**：局部 event tag 可复用，但 nexus done_Rr 唯一；当前 trace 没有注入旧通知。uniform 下 R3 可能使用 R0 先释放的 slot0。

| 子图（源码，省略 .mlir）                      | 大概信息 / 切分与参数                                                                                             | 主要目的                                                                                      | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                            |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`t04_single`](t04_single.mlir)               | 1 Context，4请求；每个 request 固定 pin=rid，共享只读 W                                                           | 观察共享 Tile Program 的多个 UCE 实例；单 device Context 不证明 device-slot generation 复用。 |             107443 |          31026 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t04_single.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t04_single.trace.json)               |
| [`t04_node`](t04_node.mlir)                   | 12 Context，每请求的 A/B/C 各独立；先发 A0/A1/A2，优先 R1→R3                                                      | 检查逐节点 HBM 依赖与共享 B/C 程序的实例隔离。                                                |             126796 |          28167 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t04_node.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t04_node.trace.json)                   |
| [`t04_stage`](t04_stage.mlir)                 | 4次 request Context 提交；R1/R2/R3 重用 ctx_request_fast 符号                                                     | 主要 slot-generation 复用案例：先 submit R0/R1/R2，await R1 后 submit R3。                    |              85091 |          27660 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t04_stage.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t04_stage.trace.json)                 |
| [`t04_stage_uniform`](t04_stage_uniform.mlir) | R0 的 A 恢复短 EVU；保留共享程序和 await R1 后 submit R3 的序列；切分 {A0,B0,C0}/{A1,B1,C1}/{A2,B2,C2}/{A3,B3,C3} | 观察 first-free 实际旧 owner；R3 不必复用 R1 的 slot，不能硬编码槽号结论。                    |              90920 |           4566 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/t04_stage_uniform.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/t04_stage_uniform.trace.json) |

## 5. 全部 N 类子图：Admission、生命周期与非法输入

N 类不是额外的计算算子覆盖，而是专门让控制/资源协议的边界变得可观察。默认 EVU；BOA 的例外是 N02 私有 Gate，以及 N03/N07 的 X×X timing 工作。

### N01：完成阶段分离

| 子图（源码，省略 .mlir）                | 大概信息 / 切分与参数                                                  | 主要目的                                                                            | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                      |
| --------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n01_slow_store`](n01_slow_store.mlir) | 单个快 EVU expand；独立 input/output，output 为 4U，HBM Store 顺序10次 | 分离 compute_end、output_ready、tile_done 与 context_done；最后写回前不得完整完成。 |             137631 |           7123 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/n01_slow_store.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n01_slow_store.trace.json) |

### N02：延迟读取与早释放

| 子图（源码，省略 .mlir）                    | 大概信息 / 切分与参数                                                               | 主要目的                                                                                                | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                          |
| ------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n02_delayed_read`](n02_delayed_read.mlir) | 同 Context 的 B/C/D 共读 role=in 的 X；D 先对私有 Gate 做 BOA100，再读 X、做 EVU100 | 检查 last_read(X) ≤ release(X) < D.compute_end；B/C 读完不能提前覆盖 X，也不必等 D 全部计算结束才回收。 |              75982 |          52830 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/n02_delayed_read.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n02_delayed_read.trace.json) |

### N03：暂时容量不足

| 子图（源码，省略 .mlir）                      | 大概信息 / 切分与参数                                      | 主要目的                                                                                                            | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                            |
| --------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n03_wait_capacity`](n03_wait_capacity.mlir) | L2=2U；Holder input U+output U 占满，Waiter 仅申请 input U | 检查占 device slot 的 WAIT_CAPACITY 不占执行资源；input final-free 唤醒后才启动 Waiter，且不增加虚假的输出/weight。 |              29696 |          26623 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/n03_wait_capacity.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n03_wait_capacity.trace.json) |

### N04：Strict FIFO HOL

| 子图（源码，省略 .mlir）            | 大概信息 / 切分与参数                                          | 主要目的                                                                               | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                  |
| ----------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n04_fifo_hol`](n04_fifo_hol.mlir) | L2=6U；Holder input2U+output4U，Head 需4U、Tail 需2U，依次排队 | 只释放2U时 Head 不够、Tail 虽够也不得绕过；验证 strict FIFO 的 head-of-line blocking。 |              43411 |          28496 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/n04_fifo_hol.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n04_fifo_hol.trace.json) |

### N07：局部 wait 隔离

| 子图（源码，省略 .mlir）                | 大概信息 / 切分与参数                                           | 主要目的                                                                                     | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                      |
| --------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n07_local_wait`](n07_local_wait.mlir) | ctx0：BOA100 A → 显式 await grid_A → 短 EVU B；ctx1：独立 EVU C | 检查 ctx0 的 PC await 只约束本 Context 后续 dispatch，ctx1 仍可进行真实 load/compute/store。 |              45875 |          27153 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/n07_local_wait.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n07_local_wait.trace.json) |

### N08：空/尾 task 与非法范围

| 子图（源码，省略 .mlir）                      | 大概信息 / 切分与参数                                                          | 主要目的                                                                        | full_memory cycles | runtime cycles | 两模式结果（exit）      | Trace / 日志                                                                                                                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- | -----------------: | -------------: | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n08_p0`](n08_p0.mlir)                       | 合法空 Context，不发 task.range/dispatch                                       | 检查 count0 是省略工作，不是放行非法 zero-range。                               |                  2 |              2 | 完成（0）               | [F](../../artifacts/nest_subgraphs/full_memory/n08_p0.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n08_p0.trace.json)                 |
| [`n08_p1`](n08_p1.mlir)                       | 一个 grid，placement=1，task.range 0..1，shape=1×64×64                         | 检查单 participant 的真实 task 集合和聚合计数。                                 |               3747 |            457 | 完成（0）               | [F](../../artifacts/nest_subgraphs/full_memory/n08_p1.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n08_p1.trace.json)                 |
| [`n08_p3`](n08_p3.mlir)                       | 一个 grid，placement=7，task.range 0..3，shape=3×64×64                         | 检查不足四 Tile 的尾参与者与恰好3份输出。                                       |               8075 |            617 | 完成（0）               | [F](../../artifacts/nest_subgraphs/full_memory/n08_p3.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n08_p3.trace.json)                 |
| [`n08_p4`](n08_p4.mlir)                       | 一个 grid，placement=15，task.range 0..4，shape=4×64×64                        | 作为四 participant 基线，检查每个真实 task 完成一次。                           |              10239 |            697 | 完成（0）               | [F](../../artifacts/nest_subgraphs/full_memory/n08_p4.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n08_p4.trace.json)                 |
| [`n08_p5`](n08_p5.mlir)                       | 两个同时提交的 Context/grid：placement15+1，对应4+1 tasks；第五份 HBM 输出独立 | 检查总共5项由两个合法 grid 表达；不能生成“单 grid 5参与者”或假装有5个物理 UCE。 |              10853 |            837 | 完成（0）               | [F](../../artifacts/nest_subgraphs/full_memory/n08_p5.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n08_p5.trace.json)                 |
| [`n08_invalid_zero`](n08_invalid_zero.mlir)   | 唯一指定错误：task.range from=0,to=0；placement=1                              | 预期 verifier exit2，命中 from < to；不是用任意 ParseError 冒充拒绝。           |                  — |              — | 预期 verifier 拒绝（2） | [F日志](../../artifacts/nest_subgraphs/full_memory/n08_invalid_zero.log) / [R日志](../../artifacts/nest_subgraphs/runtime/n08_invalid_zero.log)   |
| [`n08_invalid_count`](n08_invalid_count.mlir) | 唯一指定错误：placement=15，却 task.range 0..3                                 | 预期 verifier exit2，命中 placement popcount 与 task 数不一致。                 |                  — |              — | 预期 verifier 拒绝（2） | [F日志](../../artifacts/nest_subgraphs/full_memory/n08_invalid_count.log) / [R日志](../../artifacts/nest_subgraphs/runtime/n08_invalid_count.log) |

### N09：永久容量不足

| 子图（源码，省略 .mlir）                | 大概信息 / 切分与参数                      | 主要目的                                                                       | full_memory cycles | runtime cycles | 两模式结果（exit）  | Trace / 日志                                                                                                                                      |
| --------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------ | -----------------: | -------------: | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n09_impossible`](n09_impossible.mlir) | 一个 Context input U+output2U=3U，L2只有2U | 预期永久容量 fault，不进入无限 WAIT_CAPACITY；保留真实非零退出及 fault trace。 |                  8 |              8 | 预期容量 fault（1） | [F](../../artifacts/nest_subgraphs/full_memory/n09_impossible.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n09_impossible.trace.json) |

### N10：HBM 可见性

| 子图（源码，省略 .mlir）                            | 大概信息 / 切分与参数                                                                      | 主要目的                                                                                  | full_memory cycles | runtime cycles | 两模式结果（exit） | Trace / 日志                                                                                                                                                  |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | -----------------: | -------------: | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`n10_hbm_visibility`](n10_hbm_visibility.mlir)     | 不同 Context：A 先 output_ready 后 HBM Store10；B 从 A 的同一 HBM 地址 prefetch            | 检查 B 必须 await A.context_done 后提交，首个 prefetch 不早于最后 HBM Store completion。  |              48126 |           2924 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/n10_hbm_visibility.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n10_hbm_visibility.trace.json)     |
| [`n10_local_visibility`](n10_local_visibility.mlir) | 同 Context：A Store 到 HBM 后显式 nest.await store_done，再 prefetch 同址到另一个 L2 input | 检查真实 HBM 可见性；不能用 output_ready 或不存在的 prefetch depends_on 替代 Store wait。 |              20476 |           1392 | 完成（0）          | [F](../../artifacts/nest_subgraphs/full_memory/n10_local_visibility.trace.json) / [R](../../artifacts/nest_subgraphs/runtime/n10_local_visibility.trace.json) |

## 6. 明确未覆盖、不能从运行成功推出的结论

| 项目                                   | 当前结论 / 替代材料                                                                                                                                                                                        |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 真实 tensor 数值                       | 未建模；BOA/EVU descriptor、隐式 L1 output timing 与搬运字节仅用于时序。不能声称模型数值正确或浮点规约误差合格。                                                                                           |
| Scheduling Quality 全面最优            | 未证明。S08 barrier、T01 serial 是故意保留的质量反例；stage 常受粗事件粒度限制。                                                                                                                           |
| N05：late commit/bind failure rollback | 没有能注入这种故障的独立 MLIR；现有 `pipeline_validator/tests/test_runtime.py::TestAtomicDispatchAdmission` 包括 `test_late_context_bind_failure_rolls_back_all_tiles`。静态容量拒绝不等于部分提交后回滚。 |
| N06：旧通知注入                        | T04 stage 覆盖正常实例复用；`TestGridSignalAggregation.test_stale_launch_signal_ignored` 是已有测试中的注入路径。本次两模式导出没有注入旧通知，不能标为 MLIR 已覆盖。                                      |
| 非法 Context 切分环                    | 原图 A→B→C 切成 `{A,C}/{B}`，在 context_done-only 下收缩为 ctx0→ctx1→ctx0。没有生成冒充合法的 MLIR，也没有用 dangling SSA/ParseError 冒充 compiler partition-cycle reject；合法重切可参考 S01 node。       |
| 物理 Nx1 / SRAM bank pin               | 共享 pin0 仅制造争用；没有交付物理 Nx1 或显式 same/separate-bank placement 实验。                                                                                                                          |
| 强制同周期 Event                       | 同 service-cost 不等于实际同周期到达；此次没有事件注入来强制同时完成。                                                                                                                                     |
| 动态循环、分支、数量                   | T01–T03 是有限静态展开；D01–D04 按原规范暂不考虑。                                                                                                                                                         |
| 完整数值 Collective / 多 Group         | S11 为 EVU timing reduction-tree 与 Copy fanout，不能代替数值 Collective；当前模拟范围为单 Tile Group。                                                                                                    |
| 通用 DAG fuzz / reference scheduler    | 不在此次子图库范围；没有新增此类 framework。                                                                                                                                                               |

上述引用测试是现有覆盖入口，本次为编写说明未重新运行它们；本文件对本次 trace 导出的结论以 §1 的实际证据为准。

## 7. 查找、查看与复现

在仓库根目录运行。`nest-` 后面的名字由文件 stem 的 `_` 替换为 `-`；例如 `s08_node_barrier.mlir` 对应 `nest-s08-node-barrier`。

```bash
bash examples/run.sh list

# 选择一个可用输出目录；这里用 /tmp，避免覆盖上文的已验证快照。
bash examples/run.sh nest-s08-node \
  --sim-override fidelity=full_memory --memory-trace \
  --trace-json /tmp/s08_node.full_memory.trace.json \
  --json --report /tmp/s08_node.full_memory.report.json

bash examples/run.sh nest-s08-node \
  --sim-override fidelity=runtime --memory-trace \
  --trace-json /tmp/s08_node.runtime.trace.json \
  --json --report /tmp/s08_node.runtime.report.json
```

将 `.trace.json` 导入 Perfetto 查看。每个已导出目录内还有：

- `<stem>.report.json`：报告数组，含 `completed`、`cycles`、`reason` 和资源统计。
- `<stem>.log`：真实 stdout/stderr；verifier 负例在这里查具体拒绝原因。
- `index.csv`：112 例的文件索引、cycle 和 exit code。
- `summary.json`：完整运行命令、输入 source SHA-256、每例结果与导出时间。

入口保留每例的绑定和硬件设置：2 DMA channels、HBM fixed latency=10、通常 cycle cap=2000000；single 仅 device slot 数降为1，UCE仍4。`t01_single_one_uce` 两者均1；N03/N09 的 L2=65536 B，N04/S09 exact=196608 B，S09 short=196352 B；两个 repeat1000 诊断的 cap=8000000。T04 使用 R0/R1/R2/R3/W 五个 binding，而不是 arena。用户附加参数位于默认参数之后。

建议阅读顺序：**S01 → S03/S04/S05 → S08 正反例 → N02/N03/N04 → T01/T03/T04 → S09/S10/S11/S12**。先理解事件和数据可见性，再比较资源争用与调度质量。

## 8. Trace 调度分析与改进建议

### 8.1 分析范围、归因规则与总判断

本节在 §1 的运行结果之上进一步分析**已有 trace**，没有修改 MLIR、调度器、配置或重新运行反事实实验。原始报告中的 108 例正常完成不等于 108 例调度最优；四份非零退出仍是预期边界。

数值明细见 [scheduling_analysis.json](../../artifacts/nest_subgraphs/scheduling_analysis.json)：包含两模式全部 220 份 trace 的计算区间并集、HBM 搬运量、L2 bank 分布、16 个基础拓扑的映射比较，以及下文关键事件的完整身份与时间。文件保留对应输入 hash；它是本地分析产物，不是新调度器或通用 checker。

采用以下归因规则：

1. `cycles` 取 report，事件时刻取 trace；最终 `context_done` 时刻可能比 report 的总周期少 1，二者不混用。
2. 计算占用只计 BOA/EVU/USE 的实际执行区间，按 Tile 分别求并集；不能把 MFE、UCE 或多个 engine 的累加计数当作 MAC 占用。
3. 精确的“计算与搬运 service 重叠”使用另一 chunk 的 transfer-leg `accepted_cycle → completion_cycle` 区间，先求并集再取交集；不把两腿之间的排队空隙计作搬运 service。
4. **`group_transfer.start_cycle` 是首腿被接受的时刻，不是提交到 transfer manager 的时刻**。`ready → start` 同时可能包含 PC 顺序、资源排队等因素，不能直接标成“调度器浪费”。见 [transaction 时间定义](../../../pipeline_validator/memory/transfer.py#L190-L210)。
5. 不同 Context 映射会改变 HBM/L2 路径、驻留容量、pin 和并发资源。只有核对了相同工作与约束的已有对照，才能报告相应的实测差值；新算法的收益一律待验证。

**总判断：优先改进生成的调度顺序、成本模型和观测能力，再考虑需要修改 ABI 或公平性合同的调度策略。不是先增加 Context 数，也不是提前发 `context_done`。**

| 当前不足 / 限制                                   | 主要归属                                 | 已有证据                                                        | 改进方向                                                    |
| ------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| 已 ready 的独立节点仍排在慢节点等待之后           | MLIR 发射顺序、编译器；执行器遵守线性 PC | S08 正反例、T04 single 的 A1→B1 长等待                          | 先做依赖合法的静态重排；再评估显式 ready-action 调度        |
| Context 级完成事件携带无关节点与写回的尾部        | Context 切分、IR 可见性合同              | S08 stage 的 `{B,D}`；N01 长 Store 排空                         | 更细的合法切分；需要时设计带地址空间/代际的 tensor 可见事件 |
| 读者被纳入中间输出的 producer 集合                | dispatch 绑定 ABI、verifier、lowering    | S03 D100 中 A 的 tile 读取已结束，仍等待 D.output_ready         | 分离实际读写集合；回收必须同时满足最后读取与必要 Store      |
| 容量够给小请求，严格 FIFO 仍不允许绕过队头        | Admission 公平性策略                     | N04 有 2U 空闲却等待 31455 cycles                               | 可选有界绕过/aging，但保留默认 FIFO 与防饥饿约束            |
| 当前数据地址和 first-fit 放置使带宽集中于少数资源 | 工作负载布局、allocator、transfer 模型   | 104/104 有 HBM 活动的案例仅 channel0；98/104 的 L2 活动仅 bank0 | 补充地址分散、bank-aware 与 burst-striping 对照，先校准模型 |
| “更多重叠”“更高 utilization”不一定对应更短总周期  | 性能指标、优化目标                       | T01 single 的局部重叠更多，但 stage 总周期更短                  | 以关键路径、makespan、请求延迟、峰值存储联合评估            |
| 固定 pin 与 device-slot 跟随可能集中 UCE 争用     | 编译器 pin 分配、资源绑定策略            | S02 same_pin 在相同 4×4 配置下变慢                              | 为并行支路做活跃区间感知的 pin 分配，不迁移未结束的 task    |

### 8.2 不足一：静态发射顺序没有主动绕开无关等待

#### 已有正反例能量化的代价

S08 的正反例已核对 Tile Program 相同、HBM 总字节数相同；关键区别是等待/提交顺序。下表“增加”以对应正例为分母。

| 对照       | full_memory 正例 → barrier |   增加 | runtime 正例 → barrier |   增加 |
| ---------- | -------------------------: | -----: | ---------------------: | -----: |
| S08 single |              58227 → 79906 | 37.23% |          28590 → 51900 | 81.53% |
| S08 node   |              62820 → 83769 | 33.35% |          28568 → 52055 | 82.21% |

这说明**合法但不好的顺序确实有成本**，不说明执行器违反了依赖合同。S08 barrier 本来就是人为质量反例。

#### 基础映射中也存在类似机会：T04 single

`t04_single` 并非 barrier 命名反例，但源码顺序是先发 A0/A1/A2，然后 B0/C0，再到 B1/C1、B2/C2。B1 只读 A1，不读 R0 的结果：

| 模式        | A1.output_ready | B1 role dispatch | ready→dispatch 间隔 | 完整事件                       |
| ----------- | --------------: | ---------------: | ------------------: | ------------------------------ |
| full_memory |           17505 |            42289 |               24784 | `s0l0_ready_A1 → s0l0_grid_B1` |
| runtime     |             981 |            26808 |               25827 | `s0l0_ready_A1 → s0l0_grid_B1` |

此处是同一 launch：`ctx_all_requests / slot0 / generation0`；A1、B1 使用 request1 的 pin1，数据边是 A1→B1。B1 前面的 B0/C0 等待 R0，使独立请求没有被及时考虑。证据：[发射顺序](t04_single.mlir#L305-L318)、[F trace](../../artifacts/nest_subgraphs/full_memory/t04_single.trace.json)、[R trace](../../artifacts/nest_subgraphs/runtime/t04_single.trace.json)。

**归因**：线性 PC 下的 head-of-line blocking，首先是调度生成/映射问题。[Tile Group Sequencer](../../../pipeline_validator/tile_group_sequencer.py#L129-L174) 在当前 pending wait 未完成时不会越过它去发射后续 action。不能要求它无条件违反已生成程序的顺序。

**改进建议**：

- 低风险先做静态 list scheduling：在真实数据依赖与 buffer hazard 允许时，将已经可发射的独立分支放在慢分支等待之前；结合预计成本与关键路径，而不是固定按请求编号执行。
- 保留 S08 barrier 作为反例，不为了“全 PASS”删掉它；另用 T04 的等工作量合法重排验证编译器改进。
- 对成本难以静态预测的场景，再设计显式的 ready-action 集合或多个独立 sequencer 域；这需要 IR/执行器合同支持，不能暗中把现有顺序语义变成乱序。

**验收**：同一工作、地址和硬件配置下，B1 的无关等待缩短，R1/R3 请求完成时间改善；所有真实 A→B→C 边、pin/帧绑定、HBM 可见性与 generation 隔离仍成立。上述 24784/25827 是观察到的间隔，**不是承诺能全部消除的周期数**，因为实际 engine/内存竞争仍需重测。

另外，T04 中 R3 在 R1 完成后才引入是本案例指定的请求序列；不能把删除这个 gate、让 R3 从 cycle0 就可执行的结果称为同负载优化。重排应针对已可用请求的无关等待，并保持请求到达/引入规则一致。

### 8.3 不足二：编译器切分策略与粗粒度事件边界

S08 stage 使用 `{A}/{B,D}/{C}`：C 的真实输入是 A/B，但跨 Context 只能等待 B 所在的整个 BD Context，所以 D 的完成与必要写回也被带入了 C 的等待条件。S10 的 encoder/decoder 两组、T02 的逐行 stage 也存在类似粗粒度边界。

**责任归属澄清：这首先是切分策略／编译器调度生成的问题，不应归为 runtime 执行错误。** 当前只有 `context_done` 可供跨 Context 等待时，runtime 等整个 BD Context 完成，是正确执行了生成的程序，并不是 runtime 自己添加了错误的 D→C 数据依赖。编译器决定怎样切分；IR/runtime 接口决定可以表达多细的依赖与可见事件。

| 优化层面                                       | 主要责任                                                                                      | 是否需要改变 runtime 语义                            |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| 在现有接口内避免无关组尾等待                   | 编译器根据消费者的真实依赖选择切分，例如 S08 改为 `{A}/{B}/{D}/{C}`，即现有 node 映射         | 通常不需要；runtime 继续遵守现有提交、等待和完成合同 |
| 保留粗分组，又希望 B 的输出可见后让 C 提前推进 | 编译器与 IR/runtime 协同，提供更细的 tensor 可见事件，并维护 ownership、generation 和错误传播 | 需要扩展合同，不能由编译器单方面生成当前不支持的事件 |

runtime 不能自行拆分已提交的 Context 或提前发出 BD 的 `context_done`。仅把它换成 B 的 `output_ready` 也不合法：当前 C 可能从 HBM 读，B 的 L2 输出 ready 不等于 HBM 已完成必要写回。

更严格地说，**这批 MLIR 直接指定了切分，本次定位的是工作负载的切分／调度生成策略不足，还不是某个自动编译器实现的 bug。** 应优先优化现有接口内的合法切分；只有表达能力确实不足、且保留粗分组有必要时，再扩展 IR/runtime。更细切分也可能增加 HBM 搬运和 Context 开销，因此不能统一改成“一节点一 Context”。

S08 stage 的当前量化证据：

| 模式        | A/B 中较晚的 HBM Store 完成 | BD Context 完成 | C Context submit | 数据可见→submit 间隔 |
| ----------- | --------------------------: | --------------: | ---------------: | -------------------: |
| full_memory |                       47643 |           50218 |            50219 |                 2576 |
| runtime     |                       26781 |           25543 |            26789 |                    8 |

full_memory 中，A/B 都已在 HBM 可见后，C 仍被 BD 的组完成事件挡住；runtime 中 A 自身的尾更长，这一额外组门没有形成长等待。这个现象具有参数和 fidelity 依赖性。S08 node/stage 的 HBM 字节分别为327680/294912 B，**并非等搬运量实验**；两者总周期差不能全部归到单一事件门。

也有应明确保留的负结果：S10 stage 当前基线在两模式下的 `decoder submit − E2 Store completion` 都只有16 cycles（F：25465−25449；R：1136−1120）。它有结构性的 coarse-event 边界，但当前 E2 正好是关键输入/尾部，**本组没有证实大额无关 encoder 等待**。不能把未来“慢 E0 拖住 D2”的风险写成本次已经发生的故障。

同时，node 映射增加的成本不仅来自事件调度，还有真实 HBM 搬运。以 S09 为例：

| S09 映射 | full_memory cycles | HBM prefetch+Store 字节 | HBM transfer 数 | 峰值 L2 字节 |
| -------- | -----------------: | ----------------------: | --------------: | -----------: |
| single   |              44620 |                  196608 |               6 |       196608 |
| node     |              61516 |                  393216 |              12 |       131072 |
| stage    |              50248 |                  262144 |               8 |       131072 |

single 较快，但其峰值 L2 更高；node 慢的一部分是搬运量翻倍。不能把这组差异都记为“调度开销”，也不能简单决定所有图都融合成 single。

**改进建议**：

1. 编译器对分组做多目标选择：关键路径、预计 HBM 字节、峰值 L2、UCE 活跃区间，以及组完成事件带来的额外等待。
2. 当前 ABI 内先做合法重切，避免把无关长尾合入关键 producer 的完成域。
3. 若确实需要跨 Context 提前推进，再设计**带 tensor 身份、地址空间、版本/代际和错误传播的可见事件**。HBM 读取等待所需 tensor 的最终 Store；L2 直接共享还需要新的 handle ownership、pin 和存活协议。仅增加一个阶段 event 名称不够。

**验收**：在 S08/S10/T02 上逐条核对所有 fan-in 的真实地址与最早可见时刻；新映射的等待减少不能以增加未说明的 DMA、扩大 SRAM 或放松 release 为代价。未运行新 ABI，故不报告它的预计加速比。

### 8.4 修正前基线：中间输出的读写绑定与回收过于保守

修正前 `s03_single_d100` 的 B/C/D 都读取 A，但完整双绑定使它们也进入 A 的 `outs_producers`。旧合同要求 A 最终 Store 等这些 dispatch 的 output_ready，再由 Store 完成门控 release。以下数值仅属于旧输入 hash。

| 模式        | A 最后一次 tile.load 完成 / D.input_released | D.output_ready | A 最终 Store 首腿接受 → 完成 | A release | release−最后 tile.load |
| ----------- | -------------------------------------------: | -------------: | ---------------------------- | --------: | ---------------------: |
| full_memory |                                        20342 |          47383 | 47386 → 50454                |     50455 |                  30113 |
| runtime     |                                          685 |          27271 | 27274 → 27440                |     27441 |                  26756 |

身份：A 为 `allocation_id=l2:1:3, generation=1`；D 为 `(ctx_0, slot0, launch_generation0, dispatch_ordinal3)`，事件 `s0l0_read_D / s0l0_ready_D`；最终 Store 为 `s0l0_store_A_0`。这里使用完整 **tile.load 完成**，不是较早的单条 L2 read leg 结束。

对照 N02：共享 input-only X 在 full_memory 的最后 load 完成 45854 后，于 45863 释放，而 D 的 EVU 到 71955 才结束。说明当前系统已经有按 input_released 释放的机制；不足主要是中间 out/inout 的访问集合表达与法定回收路径，而不是“所有 Buffer 都不会早释放”。

**旧合同根因**：release verifier 按旧 `ins/outs` full-list 收集消费者/生产者，纯读者的 output_ready 因而成为 A 最终 Store 前提。顺序式 sequencer 还会让后面的 release action 等待前面无关的 action；本次不改变该 PC 调度语义。

**修正范围与保留边界**：

- 先将已经合法的 input-only release 放到其全部读者事件已定义、且不阻挡独立发射的较早位置，减少单纯的 PC 延迟。
- 当前合同分离实际读者、实际写者和 positional `bindings`；verifier、lowering、pin 与两类 dispatch ordinals 一并切换。
- 必要 Store 由真实 writer.output_ready 门控；release 精确等全部 reader.input_released、全部 prefetch completion 和全部 Store completion。运行时在任何 writer unpin 前验证两阶段和未结束 transfer，readwrite 的两个 phase 不依赖固定先后。

**验收**：S03 D100 的 A 不再因 D 后续纯计算而被额外持有；但最后 Store 未结束时仍不可释放。30113/26756 包含现有法定等待和 Store 时间，不能直接宣称是可回收的全部时间或可节省的 makespan。不得把产出 Buffer 改成 input-only 来绕过合同。

#### 新合同实测：2026-09-11

独立输出：[full_memory summary](../../artifacts/nest_subgraphs/l2_access_contract/full_memory/summary.json) /
[runtime summary](../../artifacts/nest_subgraphs/l2_access_contract/runtime/summary.json) /
[生命周期与调度对照](../../artifacts/nest_subgraphs/l2_access_contract/lifecycle_comparison.json)。
两模式各112例，均为 **108 completed、2 verify exit2、2 capacity fault exit1**；
各110份真实 trace 全部通过 `Tracer.assert_well_formed()`。原结果目录未覆盖。
另13个 workload 和2个 protocol 入口全部完成，见 [extras summary](../../artifacts/nest_subgraphs/l2_access_contract/extras/summary.json)。

访问合同修正时的 source SHA-256（本次全量 free 审查前）：

- `s03_single`：`5d3996ae26989ab915459a2dda878f45c3e70cae9244dbfbe262a331562cbf9d`。
- `s03_single_d100`：`f37b36fc792b48223a19b6e30b8f731ef57b904ae38a41d6bded522f983d2e1c`。
- 其余输入、实际命令、退出码与运行时间见原 summary；它只记录传给 runner 的参数，
  **未展开全部配置默认值**。[roundtrip](../../artifacts/nest_subgraphs/l2_access_contract/roundtrip.json)
  对应访问合同切换时的126份合法非空输入、两份指定 N08 拒绝及保留的空文件。
  新增 `tile.free` 后的独立复跑在执行前重新捕获了
  [完整配置快照](../../artifacts/nest_subgraphs/l2_access_contract/configured-run-20260911-120503Z/execution/configuration/manifest.json)：
  包含最终 HardwareConfig、SimConfig、global bindings、context 数、trace 开关、
  展开后的直接 CLI 命令及配置/source 指纹，不再依赖未来 runner 的默认值。
  [完整配置复跑结果](../../artifacts/nest_subgraphs/l2_access_contract/configured-run-20260911-120503Z/execution/summary.json)
  确认239次调用均符合预期、235份 trace 通过检查；执行前后全部源码指纹一致。

| 输入       | 模式        | A Store 首腿接受 → 完成 | A 最后实际 load 完成 | A release | D 后续 EVU 结束 | 相比旧基线 release 提前 |
| ---------- | ----------- | ----------------------: | -------------------: | --------: | --------------: | ----------------------: |
| S03 single | full_memory |           14896 → 17964 |                22906 |     22910 |           23168 |                    1706 |
| S03 D100   | full_memory |           14896 → 17964 |                22906 |     22910 |           49007 |                   27545 |
| S03 single | runtime     |               580 → 746 |                  685 |       747 |            1395 |                     855 |
| S03 D100   | runtime     |               580 → 746 |                  685 |       747 |           27234 |                   26694 |

以上单位为 cycle。A 是 `allocation_id=l2:1:3, generation=1`；
Store 是 `transaction_id=0:s0l0_store_A_0`、`event_id=s0l0_store_A_0`；
B/C/D 的真实读取按 `role_event_id=s0l0_grid_B/C/D` 与逐 task transaction 关联，
D 的阶段事件仍为 `s0l0_read_D / s0l0_ready_D`。JSON 保留完整 allocation、
phase、transaction、逐腿地址和端点记录，不以 UCE issue 或 context slot 占用替代真实 service。

结论：A Store 不再等待纯读者 B/C/D 的 output_ready；A release 同时晚于全部
真实读取和必要 Store，却早于 D100 的后续计算完成。full_memory 的读取端点也因
真实内存竞争改变，不能把 release 提前量直接当成整个 workload 的加速量。

保留合同的实测检查：

- **N02**：X 最后读取／release／D EVU 结束分别为 F `45854 / 45867 / 71955`，
  R `26507 / 26520 / 52608`；input-only 仍可早释放。
- **N03**：holder final-free 与 waiter admit 同 cycle（F `2140`，R `312`），
  first_action 在下一 cycle（F `2141`，R `313`）；等待期间不启动该 launch 的搬运。
- **N04**：Head/Tail 的排队、接纳顺序仍均为 `ctx_head → ctx_tail`。
- **T01**：四 chunks 的12个 HBM 传输、32个 Tile load、16个 Tile Store 均核对
  tensor 地址与 bytes；输入覆写不早于上次 input_released，输出覆写不早于上次
  HBM Store completion。两套六个 allocations 保持不变。
- **T04**：single 的 R3 prefetch 仍晚于 R1 的 C1 grid 完成；node 的 A3 submit
  仍晚于 C1 context_done；stage/uniform 的 R3 submit 仍晚于 R1 context_done。
  重用 context/program 名称不混淆 device slot 与 launch generation。

证据口径：runtime 的单腿 `local_dma` trace 沿用 stage 的 nominal space 标签，
因此 Tile Store 方向由 `op=tile_store` 确认，并核对实际 destination_address/bytes，
不将该腿的 nominal `destination_space` 当成真实传输方向。此次未改动 trace schema。

### 8.5 不足四：Admission 公平性会放弃部分并发机会，但正确性基线成立

#### Strict FIFO 的可测代价

N04 的 Head/Tail 在 cycle0 都已经排队。Holder 先释放 2U，Head 需 4U 不够，Tail 只需 2U 也不能绕过：

| 模式        | 空闲 2U 的窗口 | 窗口长度 | Head/Tail 接纳                   | 首动作    |
| ----------- | -------------- | -------: | -------------------------------- | --------- |
| full_memory | [3569,35024)   |    31455 | 同周期 35024，事件顺序 Head→Tail | 均为35025 |
| runtime     | [464,27346)    |    26882 | 同周期 27346，事件顺序 Head→Tail | 均为27347 |

身份分别是 `ctx_head/slot1/gen1` 与 `ctx_tail/slot2/gen2`。这是 strict FIFO **政策的机会成本**，不是当前实现违反了公平性合同。Tail 提前接纳也会与 Holder 争用计算/内存资源，因此窗口长度不等于最终可节省的执行时间。

**改进建议**：若业务更重视短请求延迟，可评估可选的有界绕过、等待年龄或队头容量预留策略。只限制“一次释放最多绕过一个”仍不足以保证长期不饥饿；还需限制队头整个等待期的总绕过次数/时间，或明确保留容量的规则。默认 strict FIFO 应保留为基线。

**验收**：在 N04 类负载中证明 Tail 获得更早的接纳机会，同时在持续小请求到达的压力下给出 Head 的等待上界；保留原子分配、final-free 唤醒与失败零副作用，报告请求延迟和 makespan，不能只看平均利用率。

#### 不应误改的现有行为

- N03 的 final-free 唤醒已经有效：full_memory 在2138释放、同周期 retry/admitted、2139首动作；runtime 为310/310/311。该路径不是需要轮询加速的瓶颈。
- N03 只有 Holder/Waiter **2 个 Context，配置却有4个 device slots**。它证明 WAIT_CAPACITY 会保留一个 slot，但没有证明槽位耗尽或因此发生吞吐损失。slot reservation 在此只能列为**扩展性风险**；若要主张解耦 Admission 与 slot reservation 有收益，必须增加固定资源预算的饱和槽位对照，证明所有 slots 被占用时确有独立、可满足资源条件的请求被阻塞，再测量请求延迟与吞吐。
- N09/S09 short 的永久容量 fault 正确。S09 的完整 bundle 是 6U，后续某时刻只剩 4U 驻留，不意味着可以在准入时只预留 4U；改变原子 bundle 合同需要另外证明动态资源循环和失败回滚。
- N01 的全局 Store 排空尾很长：full_memory 的 role_complete=15705，最后 Store completion=137623，context_done=137630；runtime 为617/7115/7122。此时 UCE/帧已结束，device slot 仍保留是当前完成合同的一部分。**N01 本身只有一个 Context，并未证明其他请求因 device slots 全满而被阻塞。** 可先增加 execution-active/drain-only 两类观测；只有在饱和负载证明瓶颈后，才考虑执行配额与 DMA 生命周期解耦，绝不能提前宣布 HBM 完成。

### 8.6 不足五：内存布局与模型耦合使“调度性能”结论容易失真

#### 单资源热点不是增加 Context 就能消除的

在 full_memory 的 110 份 trace 中，104 例确有 HBM/L2 搬运；其余是 4 个合法空例和 2 个永久容量 fault。

- **104/104** 个有 HBM 活动的案例只观察到 channel0。
- **98/104** 个有 L2 活动的案例只观察到 bank0。
- T01 single 的 60 条 L2 read/write leg 全落在 bank0；按方向计，read service 共30864 cycles、write service 共30816 cycles。当前模型中 read/write 是独立 stage，不能把两者简单相加后当作一条严格串行时间线。

这有明确的构造/模型原因：

1. 默认有8个 HBM channels，burst=64 B；[channel 选择](../../../pipeline_validator/memory/transfer.py#L778-L795) 使用 `(起始地址 // 64) % 8`，并把整个 transfer 的字节交给这个 channel。该路径没有按每个 burst 跨 channel 分拆。图库的 arena 基址与 262144 B tensor 保留步长都使起始 channel 为0。**不能仅凭这些 trace 推断真实芯片的8个 HBM 通道也只会使用一个。**
2. [allocator first-fit](../../../pipeline_validator/memory/allocator.py#L288-L328) 从 bank0 开始填充。默认 L2=8 MiB、16 banks，每 bank 容量524288 B；许多小 bundle 全部装进第一个 bank，增加独立 dispatch 仍竞争同一个 L2 bank。

**改进建议**：先补充固定容量/带宽下的地址分散和 bank-aware 放置对照；如目标硬件按 burst 条带化，则校准 transfer model 的分拆、并行、完成聚合和资源释放。不能靠增大通道数或 SRAM 容量就宣称算法优化，也不能承诺“8通道就是8倍加速”。

#### 容量测试当前还改变了 bank 几何

S09 single 与 exact 的 Tile 工作、HBM 字节和实际分配峰值都是相同的；但更小的 L2 配置反而更快：

| 案例               |   配置 L2 | 当前每-bank容量 | full_memory cycles | runtime cycles | full_memory 中使用的 L2 banks |
| ------------------ | --------: | --------------: | -----------------: | -------------: | ----------------------------- |
| `s09_single`       | 8388608 B |        524288 B |              44620 |           2436 | 仅0                           |
| `s09_single_exact` |  196608 B |         12288 B |              22692 |           2436 | 0–15                          |

[allocator 初始化](../../../pipeline_validator/memory/allocator.py#L203-L221) 令每-bank容量等于总容量除以 bank 数，因此修改 `group_sram_bytes` 同时改变了分段和 bank 并行性。**这不是“小 SRAM 调度更高效”的证据。**

容量正负例仍然有效；但若要单独研究 Admission 的容量敏感性，应固定 bank/interleave/bandwidth 几何，另设逻辑可接纳容量或使用明确的占位分配压力，再与当前配置分开报告。此建议尚未实现。

### 8.7 不足六：优化目标不能只看重叠量或 report.utilization

#### T01 已有流水，但仍有内存服务和编排限制

这些 T01 对照已核对 Tile Program 相同；每例 HBM 搬运都是393216 B、12次，BOA 总 active 都是41600 tile-cycles。下表重叠仅统计 **Tile0 的某 chunk 计算与另一 chunk 已接受的 tile-load/store transfer-leg service**，不包含 group HBM DMA，也不包含腿间等待空隙。

| T01 映射 / 对照 | full_memory cycles | runtime cycles | F：Tile0跨chunk service重叠 | R：同口径重叠 |  峰值 L2 |
| --------------- | -----------------: | -------------: | --------------------------: | ------------: | -------: |
| single          |              45304 |          10933 |                        1129 |           239 | 196608 B |
| serial          |              70944 |          12368 |                           0 |             0 | 196608 B |
| one_uce         |              54557 |          11321 |                           0 |             0 | 196608 B |
| stage           |              43168 |          11251 |                         933 |           209 | 196608 B |
| node            |              46006 |          11207 |                         933 |           314 | 393216 B |

可以得出的结论：

- single 相对 serial 的全程耗时减少 **36.14%（F）/11.60%（R）**；这是现有对照的实测收益，不是新算法预测。
- 相对 one_uce，single 耗时减少 **16.96%（F）/3.43%（R）**，但这是 UCE 资源配置对照，不能记为纯软件重排收益。
- single 的局部重叠1129大于 stage 的933，却仍比 stage 慢2136 cycles；node 也没有因预留更多 L2 而成为最优。必须看关键路径、整段内存服务和请求推进，而不是只最大化某一种 overlap 指标。
- 若使用整个已接受 transfer 的首腿到末腿窗口，single 的同类重叠会得到3535而不是1129；其中包含排队/腿间等待。两种口径必须明确区分。

**改进建议**：按输入预取、计算发射、输出 Store、回收四类 action 做依赖合法的编排；先发独立就绪工作，在不覆盖输入/输出的前提下移动预取与回收位置。流水深度、prefetch 提前量与 pin 分配需要结合 bank 竞争和存储峰值选择，而不是固定增加 buffers 或 Context。

**验收**：同样4 chunks、同样 compute repeat10、相同总字节和资源预算；所有输入覆写不早于 input_released，输出覆写不早于前次 Store 完成；既报告 makespan，也报告关键请求/阶段延迟和峰值内存。更早 prefetch 导致的额外 bank 排队不能被“重叠更多”掩盖。

#### utilization 不是计算阵列占用率

| 案例（full_memory） | report.utilization | 本分析 BOA/EVU/USE 区间并集占4个Tile总周期 |
| ------------------- | -----------------: | -----------------------------------------: |
| S01 single          |             50.72% |                                      2.48% |
| T01 single          |             87.50% |                                     22.96% |
| S12 V-EVU 长尾诊断  |             95.09% |                                     83.44% |

[现有 PMU 公式](../../../pipeline_validator/pmu.py#L82-L91) 汇总 engine/UCE 的 active 计数后除以 tile-cycles 并截断到1；它不是对计算阵列执行区间求并集。MFE 的 active 还可能涵盖等待内存完成的 transfer 生命周期。

右列也不是“可避免空闲”的补集：合法等待、Copy 工作、尾 task、没有 ready 工作都可能使计算覆盖率低。**不足是指标不足以独立解释瓶颈，而不是看见低计算覆盖率就认定调度器失效。**

### 8.8 不足七：成本模型、pin 策略与观测接口需要联合改进

#### runtime 不能单独决定性能映射

16 个基础拓扑、各3种映射中，full_memory 下 single 最快13个、stage 最快3个，node 没有成为最快。这个样本小且默认容量充足，不能外推为“node 永远不好”。

其中6个拓扑的最快映射集合与 runtime 不同：

| 拓扑 | runtime 最快映射 / cycles | full_memory 最快映射 / cycles |
| ---- | ------------------------- | ----------------------------- |
| S03  | node / 1952               | single / 33854                |
| S04  | node / 1906               | single / 40047                |
| S08  | node 或 stage / 28568     | single / 58227                |
| S10  | stage / 2796              | single / 54393                |
| S11  | node / 3988               | single / 87383                |
| T01  | single / 10933            | stage / 43168                 |

runtime 适合较快检查控制/生命周期行为，但不能直接拿其周期训练或选择详细内存条件下的最优切分。full_memory 的选择也应注明 §8.6 的当前地址热点和模型限制；推荐使用经校准的内存成本与多目标 Pareto 比较，而不是只保留一个 cycle 数。

#### 不应误判为错误依赖：S12 的共享 BOA 可以按算子交错

`s12_node` 与 `s12_node_v100` 的 QK role dispatch 都是45556（`s0l4_grid_QK`）；Tile0 的 QK compute 起点分别为49090/49147，只相差57 cycles。v100 的同一 BOA lane（pid3/tid143）上，V 的一次 compute 为 [48887,49147)，随后 QK 为 [49147,49407)，再继续 V 的 [49407,49667)。QK 没有被 V 的100次计算整体挡住。

这支持“当前仲裁可以在重复算子之间服务另一任务”，不支持给 QK 添加 V→QK 数据边。应保留该正面基线，并在未来改变 engine 队列/仲裁粒度时重新检查；不能把 V 的整个 program 窗口当作一条不可打断的 BOA job。这里的同 lane 交错也不同于 §1.2 中 BOA/EVU 的同时 service。

#### 相同物理配置，pin 仍会影响可发射机会

S02 stage 默认与 same_pin 的 Tile Program/HBM 总字节相同，物理配置均为4×4：

- full_memory：58571 → 72135，增加23.16%。
- runtime：2010 → 3462，增加72.24%。

这是明确的 UCE pin 争用对照，不是缺少数据依赖或实现了物理 Nx1。编译器可对并行支路做资源活跃区间感知的 pin 分配；更动态的 UCE 分配必须保持 per-Tile 原子 bind、L1 frame、事件实例与结束回收的一致性，不能直接迁移未结束任务。

#### 目前仍缺少足够清晰的等待归因

建议新增或汇总以下观测点，而不是抑制现有等待计数：

- dispatch 的全部真实数据依赖 ready、程序/帧资源 ready、开始尝试发射、被接受、首个实际 service，各自的 cycle。
- group transfer 的提交到 manager、首腿接受、各腿接受/完成；补齐 `submitted_cycle` 后才能拆开 PC 延迟与首腿排队。
- `dependency_wait / PC_not_reached / UCE_pin_wait / engine_queue_wait / L2_admission_wait / memory_stage_wait / drain_only` 分层统计，并明确多 Tile/engine 累加与墙钟并集的不同口径。
- 每个请求的 submit、首次工作、结果可见、context_done，以及 ready-but-not-issued 的持续时间；保留 launch_generation、dispatch_ordinal、task_id 和 transaction_id。

**验收**：选择 T04 single 的 A1→B1、T01 的 Store 等待、N04 的 Tail 排队各一条路径，使报告能区分数据未就绪、PC 未走到、资源未接受，而不是全部归到一个 WAIT_EVENT。不能用 `uce_issue` 次数替代 accepted work。

### 8.9 建议实施顺序与退出标准

以下是**尚未实施的改进清单**，优先级表示后续工程顺序，不是声称已发现新的正确性故障。

| 优先级                                  | 工作                                                                            | 首要责任层                          | 可执行的验收标准 / 风险边界                                                                            |
| --------------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------ |
| P0：建立可信优化基线                    | 补齐 ready/submit/accept/drain 时间，区分 service 与排队窗口；记录地址/bank配置 | PMU/Tracer、性能分析                | T04/T01/N04 能逐段归因；同一次运行的字段可回溯到完整事件身份，计数与并集不混用                         |
| P0：校准内存敏感性                      | HBM 地址分散/burst-striping、L2 bank-aware 对照；隔离逻辑容量与 bank 几何       | 模拟器模型、allocator、基准输入     | 容量与带宽不暗改；N09/S09 short 仍按明确合同拒绝；布局改变后的收益重新实测                             |
| P1：先优化合法静态发射顺序              | 独立 ready 分支优先于无关长 wait；较早的合法 input release；成本感知 pin        | 编译器/MLIR 调度生成                | T04 快请求等待缩短、T01 稳定重叠；完整 tensor 边、覆写保护、峰值内存与请求尾延迟一起检查               |
| P1：优化 Context partition 成本         | 联合最小化关键路径等待、HBM 字节和 L2 峰值                                      | 编译器/运行时选型                   | S09/S12 三映射报告时间/容量 Pareto 结果；不把减少 DMA 说成纯 scheduler 收益                            |
| P2：细化访问与可见性合同                | 分开实际读写集合；需要时导出 per-tensor 可见事件                                | IR/ABI、verifier、lowering、runtime | S03 D100 不再被纯读者计算尾部保守持有；N10 地址空间可见性、generation、取消/错误、Store 后释放全部保留 |
| P2：可选非 FIFO 接纳或动态 ready-action | 有界绕过/aging/容量预留；显式动态调度语义                                       | Device/Group scheduler              | N04 Tail 可早接纳且持续到达负载下 Head 不饥饿；不引入资源循环等待或部分绑定泄漏                        |
| P3：执行配额与排空生命周期解耦          | 仅在 slot 饱和 trace 证明收益后评估                                             | Runtime、DMA ownership              | 必要 Store 完成前不得 context_done；旧通知/取消路径不能错误释放新实例资源                              |

**应保留的正面基线**：N02 的 input-only 早释放、N03 的 release-driven 次周期启动、N04 的默认 FIFO 顺序、T04 的正常 slot-generation 复用，以及 N09/S09 short 的永久容量拒绝。优化应在这些合同之上进行，而不是绕过它们换取更小的周期数。
