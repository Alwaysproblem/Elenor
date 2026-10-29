# Scala BOA / Gemmini BF16 Matmul 与 pipeline validator 集成设计

- 状态：**总体设计评审稿 v0.5；硬件/FPGA 实现尚未开始。软件前置 gate 已实现并验证（见[实施报告](BOA_Software_Managed_AM_IR_Implementation_Report.md)），包括 2026-10-27 的 AM「一次申请 + 阻塞等待 + slot_free 唤醒」、bundle 预留和分阶段结果服务；本版把这些已实现的软件合同映射为 FPGA 硬件合同。**
- 需求优先级：用户最新修改 > [共享 AM V1 需求规格](../../history/nexus_shared_am_v1_requirements_and_design.md) > [prompt.md](../../prompt.md) 与此前确认中仍适用的内容。
- 本版新增确认（2026-10-28）：共享 AM 位于 **Compute Tile 内**，由该 Tile 的 1→4 个阵列共享；数值改为逐 MAC FP32 RNE profile `bf16_fp32_mac_rne_v1`；首个 FPGA 平台为 **AMD UltraScale+**，参考板 VCU118。
- 阅读顺序：§1 范围 → §4.1 AM 申请协议 → §5.2 进展证明 → §3.5/§3.6 结果与释放 → §7.4 FPGA 映射 → §8 验收。
- 本轮只修改本文与两张图，不修改实现代码。§7.2 的 SW-A1–SW-A4 是 FPGA 联调前必须完成的软件对齐项，尚未实施。源 IR 仍不增加分支/重试控制，软件计划 §3.4 仍延期。

## 1. 定位、目标和 First Silicon cutline

### 1.1 已经由用户确认的选择

| 议题             | 确认结果                                                                                                                                                                                                                      |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实现语言         | BOA、Gemmini 适配和 UCE 接口使用 Scala/Chisel，生成 RTL；不是 Scala 数值软件代替硬件                                                                                                                                          |
| Gemmini 复用边界 | 复用阵列核心、PE 和浮点运算；BOA 自主管理共享 AM、链表、有限队列和仲裁，不套用完整 Gemmini 控制器                                                                                                                             |
| BOA 命名         | 沿用 BOA 历史名称；本设计中指 Gemmini 脉动阵列组，不沿用旧文档对 outer-product primitive 的实现限定                                                                                                                           |
| 初始规模         | 先一个阵列；默认物理阵列 16×16，逻辑计算块 M×N×K = 16×16×8                                                                                                                                                                    |
| 形状演进         | 先方形单阵列，再实现并验证矩形；矩形是必达能力，不是默默删除的需求                                                                                                                                                            |
| context          | 默认验证配置仍为 4 个、可配置；每 context 最多两个 live AM lease，跨本 Tile 所有 engine 统计；同 context 两条独立输出链可并行                                                                                                 |
| engine affinity  | 每条输出链在 AM lease **授予**时固定本 Tile 内一个 engine，直到 slot_free 不迁移；不是把整个 context 永久绑定到一个 engine                                                                                                    |
| 累加存储         | **Tile 内阵列组共享**（2026-10-28）：每 Tile 的阵列共享本 Tile 一个逻辑 AM 池；V1 基线每 Tile 4 engines、4 AM groups、每组 2 slots，共 8 个动态物理槽；Tile 之间不共享                                                        |
| 输入边界         | 取消“必须配置 BOA RF 并向 UCE 报告内部 load 完成”的要求；只保留普通 producer/consumer 输入生命周期                                                                                                                            |
| 软件管理 AM      | `tile.alloc memory_space="am"` 返回一个 `!tile.am_buffer`，Matmul 返回一个完成事件；`ALLOC_AM` 只发一次，不能授予则挂起、context 阻塞于 `WAIT_AM`，由 slot_free 唤醒并直接取得 lease；源 IR 不写分支/重试；最终写回后安全归还 |
| 防死锁           | 每 context 的 bundle = Tile Program 静态最大同时存活链数（≤2）；首个 lease 须整个 bundle 放进未预留空槽，余下槽为其预留；续链只用自身预留或等自身已受理 FINAL 的链；fresh 申请 FIFO、不插队                                   |
| 数值             | BF16 A/B 输入，**逐 MAC FP32 fused RNE**（profile `bf16_fp32_mac_rne_v1`，§3.2，2026-10-28）、FP32 AM；默认 BF16 输出，支持 f32 原精度直出及可选 INT8 最终转换，不增加 FP32/INT8 输入 Matmul                                  |
| 后处理           | 简单 bias、scale、ReLU/clip；另含 INT8 输出量化，带 scale、zero-point、RNE、饱和；不增加 INT8 输入 Matmul                                                                                                                     |
| 转置             | 支持硬件 A/B transpose；需计入装载延迟、缓冲和资源成本                                                                                                                                                                        |
| 验证顺序         | 软件 gate（已完成）→ SW-A 对齐 → 单阵列 BOA RTL → 联仿 → 真实 UCE → 单阵列上板 → 多阵列 → 多阵列上板                                                                                                                          |
| FPGA 平台        | AMD UltraScale+（2026-10-28）；首个参考板 VCU118（XCVU9P，Chipyard 已有 `vcu118` SUB_PROJECT）；Alveo U55C 等 HBM 板只作后续选项                                                                                              |
| 真实 UCE         | Rocket + RoCC；保持 Scala/Chisel 生态，不选择 PicoRV32                                                                                                                                                                        |
| 环境与仓库       | monorepo，Gemmini 与 BOA 分目录；Docker/dev container；不在宿主机执行 apt install 或修改系统配置                                                                                                                              |

“微型 UCE”是控制职责目标，**Rocket 并不自动满足微控制器级面积/功耗目标**。首个真实 UCE 用裁剪配置验证功能；最终微型化需有综合证据，不能用名字代替 PPA 结论。

### 1.2 必达边界与非目标

完整交付必须同时具备：软件显式 AM allocation、带 A/B/L1 output/AM acc 的 Matmul、accumulate 与 accumulate output 模式、BF16 输入与 f32/BF16/INT8 最终输出、动态两槽 quota、一次申请 + 挂起 + slot_free 唤醒与 bundle 预留、链级固定 affinity、同 context 两链独立推进、普通输入安全释放、硬件转置、矩形、Tile 内多阵列共享 AM、Rocket 自定义指令、validator 联仿、容器复现，以及 AMD UltraScale+ 板上的数值与协议验收（§8.1c）。

**软件/IR gate 已完成，仍是硬件实现的前置条件。** gate 覆盖 AM/Matmul 源语义、资源和数值/周期模型以及 AM 挂起/唤醒，不把源级 CFG 作为前置。之后按「SW-A 对齐 → 单阵列实现/正确性 → 联仿 → 真实 UCE → 单阵列上板 → 多阵列 → 多阵列上板」推进。4-context 仍只是默认配置，不是全局上限。

不纳入本轮：K-step 跨 engine 迁移、同 C 的多 engine split-K、跨 Tile 共享 AM、AM 抢占/spill/eviction/搬迁、任意地址 atomic add、驻留 context 动态切换 memory profile、不计配额的旁路 AM、通用 Tile Program DAG 执行器、EVU 长期直接持有 AM 引用，FP32/INT8 输入 Matmul、硬件 Conv/im2col、复杂非线性、物理宏冻结和 PPA 保证，以及 FPGA 上的多 Tile、PCIe/XDMA host 链路、HBM 与 partial reconfiguration。Conv 继续采用**软件 im2col + BF16 Matmul**；已有无关功能不因本规格而删除。

输入仍必须在最后读取及全部读者结束后才允许复用，但不增加专门的 BOA load 进度事件、软件可见 RF slot 或强制独立 RF 模块。必要的馈数/转置寄存器是内部实现细节，不恢复已取消的要求。

### 1.3 当前仓库事实，不等同于拟议硬件

| 事实                                                                                     | 当前源码证据                                                                                                                                                                                   | 设计影响                                                                         |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| typed Matmul 已有 A/B/output/AM、seq、INIT/FINAL descriptor                              | [execution_ir.py](../../pipeline_validator/execution_ir.py) `ExecBoaMatmulDesc`；[lowering.py](../../pipeline_validator/compiler/lowering.py)；artifact `schema_version=4`/`compiler_abi="v4"` | 硬件 descriptor 由该合同派生；estimate descriptor 不进入数值 backend             |
| AM manager 按 Tile Group 建一个，8 槽由全部 Tile 共享                                    | [tile_group.py](../../pipeline_validator/tile_group.py) 263–270 行 `SharedAmManager(..., tile_id=0)`，并传给每个 `ComputeTile`                                                                 | 与 Tile 内共享的新确认不一致 → SW-A1                                             |
| AM 申请为一次发出、挂起、slot_free 唤醒，带 bundle 预留与 FIFO                           | [shared_am.py](../../pipeline_validator/runtime/shared_am.py) 13–29、206–400 行 `request`/`_grantable`/`_grant_pending_locked`；[tile.py](../../pipeline_validator/tile.py) 925–1032 行        | 硬件 allocator 直接实现同一规则（§2.3、§4.1、§5.2）                              |
| bound_engine 取物理 group 号；每 Tile 只有一个 BOA 实例                                  | [shared_am.py](../../pipeline_validator/runtime/shared_am.py) 427 行 `bound_engine=slot.group`；[tile.py](../../pipeline_validator/tile.py) 1400–1402 行 `_engine_for_queue`                   | affinity 目前只是元数据，且隐含每 engine ≤2 个 live handle → SW-A2               |
| 数值在 binary64 内累加整个 step，commit 时一次舍入为 FP32                                | [boa_numeric.py](../../pipeline_validator/boa_numeric.py) 60–93 行 `matmul_f32`；[test_boa_am.py](../../pipeline_validator/tests/test_boa_am.py) `_oracle` 同构                                | 与逐 MAC FP32 RNE profile 不同，FPGA 无法逐 bit 对拍 → SW-A3                     |
| 结果服务分阶段：launch 原子预留 FIFO 项与 2 credit，再 commit/output/drain/release/event | [boa_result.py](../../pipeline_validator/runtime/boa_result.py) 112–243 行；[hardware_config.yaml](../../pipeline_validator/hardware_config.yaml) 104–112 行                                   | 对应 ResultCapture/AmCommit/Drain 硬件；软件为 step 粒度，硬件为微块粒度（§3.5） |
| `chain_done` 终态记录在 FINAL commit 时写入，早于 slot_free                              | [shared_am.py](../../pipeline_validator/runtime/shared_am.py) 511–514 行 `commit_step`、532–539 行 `release_final`                                                                             | 与 §4.3「chain_done 包含 slot_free」不一致 → SW-A4                               |
| UCE 是 Python Tile Program 解释器                                                        | [tile.py](../../pipeline_validator/tile.py) `TileUCE`                                                                                                                                          | 没有真实 RISC-V 指令执行；RoCC 是新增验证层                                      |
| 数值 Matmul 只在 full_memory/ByteStore 下执行                                            | [tile.py](../../pipeline_validator/tile.py) 1344–1349 行                                                                                                                                       | 联仿与上板对拍都以 ByteStore 字节为外部内存权威                                  |
| 硬件配置仍是 8 个 16×16 OPA 峰值几何参数                                                 | [hardware_config.yaml](../../pipeline_validator/hardware_config.yaml) 83–86 行                                                                                                                 | 阵列数、阵列形状、numeric profile 需要新配置键（§6.1）                           |

旧 [BOA 设计](../elenor_boa/ELENOR_BOA_Design.md) 只参考 ownership、事件身份和故障边界；其 dtype、OPA、reduce tree 等内容不覆盖本需求。

共享 AM 概念到当前仓库对象的映射如下；右列是 FPGA 硬件需要满足或改造的边界：

| V1 概念              | 当前真实对象/入口                                                                                                              | 当前状态                                                                       | FPGA 改造边界                                                                 |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| tile context / epoch | [tile.py](../../pipeline_validator/tile.py) `_UCEContext`；`AmOwner(context_name, launch_generation, ctx_id, tile_id, epoch)`  | owner 含物理 tile 与 epoch；fault/reset/rebind 经 `abort_owner` 取消挂起并回收 | 硬件 owner 为 (ctx_id, epoch)，Tile 由 allocator 实例隐含；trace 保留 tile_id |
| GEMM / K-chain       | `ExecBoaMatmulDesc`；`SharedAmManager.begin_step/commit_step`                                                                  | seq/INIT/ADD/FINAL 已校验；FINAL 受理置 `final_issued`                         | ChainTable 维护 submitted/issued/committed 与 final_issued                    |
| engine 选择          | `TileUCE._engine_for_queue`；`BOAEngine`（pipeline_depth=1）                                                                   | 每 Tile 一个时序 BOA；bound_engine = group                                     | SW-A2 后每 Tile N 个 engine，按 §4.1 独立选择并只在 bound_engine 发射         |
| AM / 部分结果        | `SharedAmManager`、`BoaResultService`、`boa_numeric`                                                                           | 数值 partial、generation、终态、FIFO/credit/commit/drain 已建模                | URAM group + RMW（§7.4.3）；数值按 §3.2 profile                               |
| event / wait         | `_UCEEventRef`、`EventTable`、`WAIT_AM` 唤醒回调                                                                               | BOA 事件在 release 之后发出；唤醒是同周期回调                                  | wake mailbox + 事件位图（§4.5）；不能把零延迟回调当作硬件握手                 |
| 输入与输出可见性     | [execution_verifier.py](../../pipeline_validator/execution_verifier.py) launch/WAIT/FREE_L1 检查；`TileGroup._on_phase_signal` | Group phase 按全部 Task 聚合                                                   | typed 输入按真实读者结束，L1 output_visible 与 L2 output_ready 分离           |
| admission / 退休     | `TileGroup._step_task_admission`、`TileUCE._complete_context`                                                                  | 缺 FINAL 报错；FINISHING/DONE；fault 回收 lease（IR15）                        | 硬件沿用；Task lease 不是 AM lease                                            |

`EventTable.wait` 当前读取终态，不是所有资源域的原子 check-and-register 硬件实现。源码复用和硬件等待协议仍需分别验证。

### 1.4 Gemmini 调研与依赖基线

上游调研版本是 `ucb-bar/gemmini@8c3f9923a44a2fe2c7930587be297d6d4f8c09ca`。为避免随意拼接 Rocket/Chisel 版本，**拟采用一套 Chipyard 锁定组合做构建基线**：

| 依赖                   | 候选不可变来源                             | 证据/状态                                                                                            |
| ---------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| Chipyard               | `e02074414158c85059c1dcb231ff2f8941254b9a` | 来自 Gemmini 的 [CHIPYARD.hash][up-chipyard-hash]；尚未在本轮构建                                    |
| Gemmini 基线           | `25809f78323a729ef76fb68f3cedd8a24da2942b` | 上述 Chipyard 的 [实际 submodule pin][up-parent-gemmini]，不是调研 master 的 SHA                     |
| Rocket Chip            | `c7aabd2823d3d001a7a9e8a982bb65d6cb7c9bdf` | 同一父版本的 [实际 submodule pin][up-parent-rocket]                                                  |
| Scala / 主 Chisel      | `2.13.12` / `6.5.0`                        | 上述 [Chipyard build.sbt][up-chipyard-build]，`commonSettings`/`chiselSettings`                      |
| Gemmini 独立 build.sbt | 调研版本写 `2.13.10` / `chisel3 3.6.0`     | [上游独立构建声明][up-gemmini-build]；**不得混入前一行构建**，更不能沿用 `rocketchip 1.2.+` 浮动依赖 |

这些 SHA 是可复核的候选组合，不是「已完成 BF16 BOA 构建」的声明。构建 gate 必须解析并锁定所有 transitive dependencies、HardFloat、sbt、JDK、firtool、Verilator 和 RISC-V 工具链；失败时提交证据再调整整个组合，不能暗中换 latest。

已核实的复用限制：

1. [基线 BF16 配置][up-bf16] 为 `Float(8,8)` 输入、`Float(8,24)` AM 类型，但 `spatialArrayOutputType` 仍是 `Float(8,8)`；[PE][up-pe] 的 MAC 输出类型依赖 array output type。**必须贯通 FP32 partial-sum 路径，不能只改 accType。**
2. [MeshWithDelays][up-mesh-delay] 断言总行数等于总列数；矩形不能只删除 assert。馈数、skew、transpose、last、tag、drain 都要重新验证。
3. 同一模块的输出为 `Valid`，没有 `ready`。BOA 必须在最小不可暂停微块发射前证明其全部在途结果有接收保障；不能凭当前 FIFO 空位或直接添加下游 ready 假定阵列可停。
4. `n_simultaneous_matmuls` 有最小值约束，PE 有 `c1/c2`、propagate 等内部状态；“禁用 double issue”不是把这个参数设成 1，更不是直接删寄存器。
5. [上游 AccumulatorMem][up-accumulator] 的外部存储接口不等于 V1 的跨 engine 所有权和配额；输入规格 §17 另指出共享 adder、AM 写流水线和局部地址互锁的复用边界。不能把增加 bank 数等同增加 adder 吞吐，也不能把 mesh 最后输出等同 AM commit。
6. Gemmini/Chipyard 是源代码依赖图，不是一个已验证可单独 Maven 引入的稳定硬件库。保留 Gemmini 许可证及修改记录；Rocket、HardFloat 等分别保留其许可证。
7. [Gemmini Float `mac`][up-arith] 先用 `RecFNToRecFN` 把输入扩到累加格式，再用 HardFloat `MulAddRecFN`（RNE、tininess after rounding）做 fused multiply-add，与 §3.2 的逐 MAC profile 一致；前提是第 1 条的 FP32 partial 路径已贯通。
8. 上述 Chipyard 版本的 [VCU118 Configs.scala][up-vcu118] 提供 `WithVCU118Tweaks`（UART、SPI SD 卡、DDR backing memory、JTAG，默认 100 MHz）和 `RocketVCU118Config`，是 C11 板级 harness 的起点；BOA 只叠加配置，不复制另一套 harness。

## 2. 职责、非职责和 ownership

### 2.1 硬件划分

![BOA 硬件组件和共享 AM 布局](../image/boa_gemmini_hardware.svg)

本稿的一个 matrix engine 对应一份 Gemmini 阵列执行资源。每个 Compute Tile 的 BOA 含 1（bring-up）到 4（V1 基线）个 engine，经带身份的结果路由访问**本 Tile** 的 4 AM groups；每组两个固定物理区域，由同一个 Tile 级动态分配与所有权域管理。group 不对应 engine，也不对应 context；不同 Tile 的 AM 与 allocator 相互独立。

术语：本文 “AM group” 指 Tile 内 AM 池的物理分组（每 Tile 4 个、每组 2 slot），与 “Tile Group”（一个 Group 内的 4 个 Compute Tile）无关。当前 `engines.boa.am.group_count` 由 Tile Group 级 manager 使用，并被兼作 engine 编号（`bound_engine=slot.group`）；SW-A1 后它按每 Tile 解释，SW-A2 后不再隐含 engine 编号。FPGA 实现不得把 8 槽池放到跨 Tile 层。

| 组件                           | 拥有的职责/状态                                                                             | 不拥有                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Compiler                       | 逻辑 AM 位置 0/1、静态 bundle（≤2）、有限 K-chain、输入依赖、INIT/ADD/FINAL、两槽软件流水   | 物理 bank、固定 engine 编号、轮询共享 SRAM                       |
| Group / validator runtime      | TaskIdentity、context admission、memory profile、L1/L2 引用、phase 聚合                     | 把 Task lease 计作 AM lease，或在 admission 时永久预留两个 AM 槽 |
| UCE / Rocket firmware          | Tile Program PC/寄存器、`ALLOC_AM` 一次发出、WAIT_AM/WAIT/context switch、已有链续步        | AM bank 仲裁、重发或轮询 ALLOC、内部 load 进度、触发 slot_free   |
| SharedAmAllocator / ChainTable | ctx quota、bundle 预留、挂起请求 FIFO 与唤醒、动态区域、generation、不可变 affinity、链状态 | 可迁移的 engine 选择、每 engine 两槽私有限额、跨 Tile 分配       |
| StepMailbox / ReadyArbiter     | 每链当前记录和一个 next-step mailbox、在 bound_engine 上 ready 仲裁                         | 未提交的未来步骤、跨链无依据的程序顺序依赖                       |
| Operand / Transpose Adapter    | 声明 view 的合法读取、四种 A/B transpose、有效 lane 馈数                                    | 软件可见 RF 分配和专用内部 load 通知                             |
| ResultCapture / AmRouter       | launch 时原子预留结果 FIFO 项与 O/R credit、在途身份、合法 group 路由                       | 以 compute_done 冒充 step_committed                              |
| AM group / RMW                 | SRAM 端口、bank、局部 adder、物理地址互锁和 writeback                                       | 通用软件 atomic、隐含无限吞吐                                    |
| Epilogue / Drain               | FINAL 后处理、有限 L1 output credit、自动输出到已准入 L1、可见性和安全释放                  | 等 UCE 发 FREE、等事件被 poll/reap 才归还 AM                     |
| EventTracker                   | 程序事件槽（context 绑定时分配）、AM 唤醒交付、等待者、有界 terminal 诊断环和安全退休       | 让 AM 授予等待事件 await/reap；将事件生命周期绑死到物理 AM slot  |

这些是逻辑职责，可复用或组合已有模块；不要求为名字一致而复制第二套 scheduler/event 系统。

### 2.2 身份分层与链级 affinity

软件语义为 `ctx_key=(tile_context_id, context_epoch)`、`logical_am∈{0,1}`、`acc_handle=opaque(entry_id,generation)`。ctx_key 的作用域为 Physical Tile；allocator 按 Tile 实例化，所以硬件 owner 只需 (ctx_id, epoch)，跨 Tile 消息与 trace 仍须带 tile identity（软件 `AmOwner` 已含 tile_id）。`TaskIdentity` 继续标识原有 Grid/Task，不与 ctx_key、物理 slot 或 engine 混同。

硬件 handle 表至少记录：owner_ctx_key、logical_am、output_chain_id、physical_am_group/region、bound_engine、shape/acc_format/layout、submitted/issued/committed 进度及生命周期。event token 还须区分 context、entry、generation、步骤和事件种类，不能只携带裸 slot 或 generation 数值。

- affinity 在 lease 授予的线性化点固定（申请时即授予，或挂起后由 slot_free 驱动授予），直到 slot_free 不变。原 engine 忙时等待它，不能转给空闲 engine。
- affinity 不是 engine 的整链独占预约；某链等输入/下一 descriptor 时，同一 engine 可服务其他绑定到它的链。
- 同 context 两条链可绑定不同 engine，独立 ready；一个 engine 可服务超过两个全局 handle，不能继承隐含的两条本地 AM 元数据上限。
- context switch 不释放 AM、不改变 owner/affinity；只在所有命令和引用安全收尾后复用 context epoch。

ID/寄存器/事件位宽和二进制 ABI **由后续规格冻结**。必须有有限回绕和 quiescence 规则；较大的 generation 不能代替旧访问收尾。

### 2.3 动态两槽配额

令 L(ctx) 为某 context 尚未安全释放的 AM lease 集合；下文“全局”均指**一个 Compute Tile 的 AM 池**：

```text
used_am_slots(ctx) = |L(ctx)|，0 <= used_am_slots(ctx) <= 2
sum_{ctx ∈ Tile} used_am_slots(ctx) = tile_live_leases <= 8
```

配额跨该 context 使用的本 Tile 所有 engines 统计。RESERVED、WAIT_STEP/WAIT_INPUT、RUNNING、COMMITTING、FINAL_READY、DRAINING、FAULT_QUIESCING 均计一个槽；one-shot 也使用同一池、计一个短生命周期槽。只有 slot_free 才减一，授予成功加一；挂起的申请和 ERROR 申请不留下 lease、slot、generation 变化或半初始化资源。

**bundle 预留不是 lease。** owner 的 bundle(ctx) 在首个 lease 授予时登记，取值为 Tile Program 的静态最大同时存活链数（1 或 2）。预留量 `reserved(ctx) = bundle(ctx) − |L(ctx)|` 不计入 used_am_slots，但从 fresh owner 可用的空槽中扣除：

```text
fresh_grantable(ctx)        = free_slots − Σ_owner reserved(owner) >= bundle(ctx)
                              AND 没有更早挂起的 fresh 请求（FIFO，不插队）
continuation_grantable(ctx) = reserved(ctx) >= 1 AND |L(ctx)| < 2
                              AND 同 logical_am 没有任何 live lease（含已受理 FINAL、尚未 slot_free 的链）
```

owner 没有 live lease 且没有挂起请求时，bundle 归还池中。

两个逻辑位置不是永久物理预留。4 contexts 各以 bundle=2 占满，或在 UCE/memory 容量允许时 8 contexts 各以 bundle=1 占一槽，都可以填满八槽。同 context 两条链都未受理 FINAL 时再申请第三条，是 compiler/verifier 静态拒绝的合同错误；运行时出现则 ERROR 并 fault 该 context，**不挂起**（挂起会自锁）。若其中一条 FINAL 已受理，新申请挂起至该链 slot_free。释放后可以获得不同物理区域和新的 engine affinity。

完成事件与 AM 分离：每次 typed Matmul 的完成事件占用其 Tile Program 的一个静态事件槽（程序内事件名唯一，verifier 拒绝重复），在 context 绑定/admission 时按程序事件数分配，保留到被 await 或 context 安全退休。**AM 授予与唤醒从不等待事件 await/reap**，不存在 `event_record_full` 这类授予失败原因。AM 自身的 terminal record 只是有界诊断环（满则覆盖最旧并计溢出，与软件 `terminal_capacity` 一致），不参与正确性判断。

### 2.4 内存与输入生命周期

Tile L1 SPM、Group L2 和阵列组内部共享 AM 是不同存储域。每条链在授予时校验并关联已准入的 L1 输出区域；不因等 AM 的新任务预取而挤占旧链的输出空间。一个驻留 context 保持一个 memory profile。

本轮**不规定独立 BOA RF、RF 容量/槽位或向 UCE 通知内部装载完成**。仍然必须满足 R14：指定层级的输入缓冲只有在最后实际读取、所有声明读者和在途访问结束后才能复用；STEP 接受仅固定 descriptor，不代表 A/B 已读取。

普通 input_usage/ref tracking 可保守地在已证明没有后续读的步骤完成边界释放，不必追求内部装载后的提前释放，也不能仅按 MAC done 推断。多个读者共享输入时必须聚合全部读者。L1 `tile.free` 使 view 失效，不自动释放整个 Task Arena 的物理预留；Group `input_released` 仍按原 L2 输入和 Task 聚合合同处理。

## 3. 微架构和状态机

### 3.1 固定 tile、阵列数据流和转置

配置仍分开表达 `ArrayRows/ArrayCols` 与 `TileM/TileN/TileK`；默认物理阵列 16×16、逻辑块 16×16×8。先方形，再验证 8×16、16×8 等矩形，合法形状由实际 adapter 和分块规则校验。

**不冻结 WS，也不未经 review 改为 OS。** 新规范固定的是共享 AM 和链级 affinity，不规定 PE 数据流；此前以 K-step 跨阵列迁移支持 WS 的前提已取消。两种内部数据流都必须把本步骤 partial 交给共享 AM，只有实际提交后才能发布 step_committed。不能用 OS 将跨步骤状态一直留在 PE、却把尚未更新 AM 的步骤报告为已提交。

数据流选择需比较默认 shape、馈数、transpose、填充/排空和 FP32 路径；实施前冻结一种已验证映射，不强制同时实现两种。两种数据流都必须实现 §3.2 的块内顺序：OS 中块部分和是 PE 累加寄存器（每 tile_k 个 k 排空一次），WS 中块部分和是列方向链（链长等于活跃 K 行，非活跃行旁路）。WS 的 K/N 空间映射约束不能当作所有 dataflow 的通用形状限制；FPGA 吞吐推导见 §7.4.5。

硬件 A/B transpose 继续在阵列输入适配层实现：A 正常读 `[m,k]`、转置读 `[k,m]`；B 正常读 `[k,n]`、转置读 `[n,k]`。支持四种组合，并验证 view/stride/mask。所需流水寄存器或重排缓冲按实际接口建模，不形成新的 UCE 可见 RF 管理协议。

边界块携带 valid_m/n/k；无效位置不能越界读写，无效 MAC 不得因 `0×Inf` 或额外舍入污染有效结果。零维/零 K 由 compiler 明确 lower 为合法空工作或数学定义的输出路径，不向硬件发射永远等不到 last beat 的工作。

### 3.2 数值契约 v0：`bf16_fp32_mac_rne_v1`（逐 MAC FP32 RNE）

保留 BF16 A/B、FP32 partial 和 AM、RNE，输出支持默认 BF16、f32 原精度直出及可选 INT8。本 profile 冻结每次舍入的位置和顺序，使软件模型、独立 oracle、RTL 与 FPGA 可以**逐 bit**对拍：

```text
a = bf16_to_fp32(A[i,k]); b = bf16_to_fp32(B[k,j])     # 精确扩展，低 16 位补零
对每个输出 (i,j)：按 seq 顺序处理各 step；每个 step 的有效 K 按 k 递增切成长度 tile_k 的块（末块 valid_k ≤ tile_k）
  p = +0.0f
  for k in 块内 k 递增:  p = fma_f32_rne(a, b, p)    # 乘积不单独舍入，每个 MAC 舍入一次
  if (INIT step 且为本 step 第一块):  AM[i,j] = p      # 覆盖，不读旧值
  else:                                AM[i,j] = add_f32_rne(AM[i,j], p)
FINAL：AM 读出一次，进入 epilogue 与输出转换（BF16 RNE / f32 原样 / INT8 规则）
```

- `tile_k` 是 profile 参数（默认 8，对应逻辑计算块 16×16×8），进入 target fingerprint；改 tile_k 会改变数值结果，不是纯性能参数。
- seq=0 为 INIT，后续为 ADD；one-shot 为 INIT|FINAL。INIT 必须覆盖全部有效输出，不能读上一 owner 的残留值；同一 INIT step 的后续块按 ADD 处理。
- 无效 lane/k（尾块、mask）不执行 MAC，不能用 `fma(0,0,p)` 代替旁路：`-0 + +0` 会改变符号零，`0×Inf` 会产生 NaN。
- 舍入与特殊值以 [Gemmini Float `mac`][up-arith] 所用 HardFloat `MulAddRecFN` 为准：RNE、tininess after rounding、不 FTZ、溢出为 Inf。NaN 输出编码由 C1 对 HardFloat 位级检查后冻结，冻结前不得宣称。
- 当前软件 `boa_numeric.matmul_f32` 在 binary64 中累加整个 step 后一次舍入（§1.3），不满足本 profile；SW-A3 必须同时修改模型与测试 oracle。Python 3.11/3.12 没有 `math.fma`，binary64 先求和再转 FP32 有双重舍入，oracle 须用精确有理/整数运算再单次 RNE，并以 Berkeley SoftFloat `f32_mulAdd`/`f32_add` 交叉校验。
- shape/layout/acc_format 在同一链中不变；部分 mask、未初始化位置、同地址写后读及 FINAL read 均进入 hazard 检查。bias/初始 C 仅按数学合同应用一次；中间结果不缩位、不激活、不量化；FINAL 提交后才执行 epilogue。

后处理建议顺序为 `FP32 AM → 每列 bias → scalar scale → ReLU/clip → 输出转换`，各阶段可关闭。f32 保持宽度写出，BF16 最终转换一次；INT8 为 `saturate[-128,127](RNE(y * inv_scale) + zero_point)`，inv_scale 有限且为正，zero_point 在 INT8 范围内。NaN→zero_point、±Inf→饱和端点仍待 review；不引入 FP32/INT8 输入 Matmul 或默认 per-channel quant。

软件 Conv 继续先 im2col 再 Matmul；展平布局、padding/stride/dilation 和额外流量必须显式验证，不实例化硬件 LoopConv。

### 3.3 链状态机与生命周期

```text
(无 lease) --AM_REQ，可授予--> RESERVED --> WAIT_STEP
(无 lease) --AM_REQ，不可授予--> PARKED      # 请求而非 lease：不计 quota，无 slot/generation 变化
PARKED --slot_free/CANCEL 后重评估，可授予--> RESERVED   # 恰好一次唤醒并交付 lease
PARKED --owner fault/reset/rebind--> CANCELLED          # 永不唤醒
WAIT_STEP --STEP accepted--> WAIT_READY --ready/issue--> RUNNING
RUNNING --全部结果已接住--> COMMITTING
COMMITTING --非FINAL且全部提交--> WAIT_STEP / WAIT_READY
COMMITTING --FINAL且全部提交--> FINAL_READY --取得L1 output credit--> DRAINING
DRAINING --output_visible且旧访问结束--> slot_free --> FREE   # 随即重评估挂起请求

任一live状态 --终止性故障--> FAULT_QUIESCING
FAULT_QUIESCING --旧访问安全结束且资源回收--> ERROR/CANCELLED terminal
```

每链一个当前执行记录和一个 next-step mailbox；RUNNING/COMMITTING 时可接收下一步，但同 handle 同时最多一个 issued 且未 committed 的步骤。mailbox 已就绪时，前步 commit 后可以直接转 WAIT_READY，不必人为插空。FINAL 被受理即置 final_issued：此后该链无需 UCE 指令即可完成，因此同 owner 的新申请可以等待它（§4.1）。

必须区分：

1. AM lease 从授予持续到安全 slot_free，包括等待输入和 FINAL drain。
2. engine 只在步骤实际执行和必要交接时占用。V1 保守在 step_committed 且配置切换安全后释放该步骤占用；不等整条链 drain，也不因链等输入而锁住 engine。
3. 完成事件槽在 context 绑定时按 Tile Program 静态分配，保留到被 await 或 context 退休，与 AM 授予和 slot 复用无关。
4. 挂起请求从 AM_REQ 到唤醒或取消，不是 lease、不占 quota；bundle 预留从首个授予持续到 owner 没有 lease 且没有挂起请求。

### 3.4 Ready 仲裁、有限队列和进展路径

STEP 的发射条件为：handle/owner/seq 合法、前步 committed、输入 ready、**bound_engine** 可接收、结果/AM 服务保障已取得、配置切换处于安全边界。调度器只处理已提交工作；同 context 两条链不因共享 context 自动建立依赖。

AM 申请走独立的请求 FIFO，与已有链的 continuation mailbox/提交路径隔离。不可授予的申请在设备内挂起（每 owner 至多一个），只阻塞发出它的 context，不占 UCE 执行，不挡旧链 STEP/FINAL/供数/drain。mailbox 或结果 credit 不足时 STEP 不被受理（FULL），submitted 游标不增加，该 context 进入 `WAIT_ENGINE_QUEUE`，credit 归还后重新提交同一步；这与当前软件 `_UCEHeadStatus.WAIT_ENGINE_QUEUE` 一致，不是源级重试分支。

目标水位只适用于已有合法工作和可用 credit；不复制工作、不以 NOP 填满、不假设 UCE 尚未提交的 FINAL 已存在。各 command/mailbox/result/event 容量独立建模，不用一个 pipeline_depth 替代所有资源。

仲裁在 ready 候选内公平轮转/按年龄选择，优先保障可释放状态的 drain 与已有链；不能空等未 ready 的高优先级候选。多个 group/端口允许的重叠需保留，不以全局串行化掩盖竞争。挂起请求的授予顺序：continuation 在其预留出现空槽时即可唤醒；fresh 严格 FIFO、不插队，避免大 bundle 饿死。

### 3.5 结果接收、AM group 和 commit

每组独立核算 SRAM read/write port、bank、adder lanes/latency 和请求服务。增加 bank 不自动增加共享 adder 的吞吐。固定大小 slot 完整位于一个 group 内；越界 shape 报 unsupported/size error，不能隐式借邻槽或增加不计配额的临时槽。

局部更新为 `partial → AmRouter → group read old C（ADD）→ FP32 adder → writeback`；旧 C 不回送矩阵阵列。INIT 跳过对应位置的旧值读取。同一物理行的 RMW 从读旧值至写回保持互锁或等价 forwarding；控制位不同不能绕过地址 hazard。

对每个实际接收 FIFO 分别定义 O=已到达占用、R=已预留但尚未到达、Q=容量：

```text
O + R <= Q
发射：R += 最坏不可暂停输出量
到达：R -= 到达量；O += 到达量
安全出队：O -= 出队量
```

按**最小不可暂停物理微块**预留，计入所有在途输出。不能把不同层级不可互换的容量简单相加，不能因为数据到达就释放总 credit；下一级必须有独立接收保障。微块超过保证容量时，在合法边界进一步分块或拒绝发射；采用保证 bank 服务代替缓存时必须证明最坏冲突窗口。

**软件模型的对应粒度**：`BoaResultService` 在 launch 时原子预留 1 个结果 FIFO 项和 2 个 credit（A/B 读取与 AM RMW 写回各 1；默认 `result_fifo_depth=4`、`am_credit_capacity=8`），compute_done 后等 `commit_cycles` 再提交。这是 step 粒度的保守模型；硬件按最小不可暂停微块预留。C5/C7-B 只要求 step 级边界顺序（compute_done ≤ commit ≤ output ≤ event）与资源不变量一致，不要求微块级周期相同。

```text
step_committed =
    end_of_step_seen
    AND all_expected_results_received
    AND pending_am_updates == 0
```

pending_am_updates 覆盖 result capture、route、queue、RMW 和待写回，而非仅 SRAM 内部。它与 FIFO 的 O/R credit 含义不同；在阵列还会产出结果时，计数暂为零不能触发 commit。

group 服务资格可按微块获取，不为等待输入的整条链永久锁 group；多资源同时获取需全成或全败。drain 必须公平取得读端口，不能被持续 ADD 永久饿死。

### 3.6 FINAL 自动释放与 double issue

FINAL 步骤 commit 后，设备自动 epilogue/drain 到授予时确定的已准入 L1 目的区域。阶段与软件模型一致：compute_done → commit → 取得有限 L1 output credit → 写出 → drain → slot_free/quota-- → chain_done 事件。输出可见、没有待处理步骤、所有旧 AM/输出访问 quiescent 后才记录终态并 slot_free；L1 output 受阻时链停在 FINAL_READY/DRAINING 持有 lease，下游恢复后自主完成（软件 IR14 已用 `set_output_blocked(..., until_cycle=3000)` 验证）。**正常释放不需要软件 FREE，不等待 UCE poll/ACK/reap。**

V1 每 engine 最多一个执行中的步骤，保留 Gemmini 正确性所需的 propagate/c1/c2/tag pipeline；不能盲目将 n_simultaneous_matmuls 改成 1 或删 PE 状态。多 engine 并行、L1 ping-pong 和已接收下一步的 mailbox 均不等于单 engine double issue。

context 返回时，已提交 FINAL 尚在 drain 则进入 FINISHING；没有 FINAL 的活链是协议错误。终止性错误停止该 context 新提交，对其余已拥有链和访问受控收尾，再报告失败终态并退休；不等待失败链永远不会发生的成功 chain_done，不按 timeout 直接清 quota。若只能受控 reset 恢复，应显式暴露该限制。

## 4. 接口、descriptor、寄存器和协议

### 4.1 AM_REQUEST：一次发出，授予或挂起，slot_free 唤醒

以下是共享 AM V1 的**下层协议合同**，与已实现软件 `SharedAmManager.request` 同语义；源 MLIR 仍只有一条 `tile.alloc memory_space="am"`，不出现返回码或分支。

```text
AM_REQUEST(owner=(ctx_id, epoch), logical_am, bundle_slots,
           output_shape, acc_format, layout, output_destination, chain_config)
  -> GRANTED(lease_handle, bound_engine, completion_token)
   | PARKED(request_id)     # 之后恰好一次 WAKE(request_id, lease_handle, bound_engine, completion_token)
   | ERROR(reason)          # 合同错误：context fault，不挂起
CANCEL(owner)               # fault/reset/rebind：丢弃挂起请求（永不唤醒），回收该 owner 的 lease
```

ERROR 与软件 `request()` 的永久错误一致：logical_am ∉ {0,1}；字节数超过 slot 容量；bundle_slots ∉ [1, slots_per_context]；该 owner 已有挂起请求；同 logical_am 有尚未受理 FINAL 的 live 链；持有 lease 期间 bundle 改变；尚未受理 FINAL 的链数已达到 bundle（自锁）。这些条件靠等待无法满足，所以不能挂起。

授予线性化点必须同时完成：检查 §2.3 的 grantable 条件，选择空闲 slot，取得 ChainTable 项、一个 next-step mailbox 和必要元数据，绑定已准入 L1 输出目的地，设置 owner/generation/不可变 bound_engine，置 RESERVED 并 quota++；首个 lease 同时登记 bundle。不能先返回 handle 再等待 AM 资源；授予时不要求物理清零，但有效数据 INIT 前禁止读取。授予**不取得、也不等待**完成事件记录：step/chain 事件写入发起 STEP 的 Tile 事件槽（§4.3），因此旧链事件尚未被 await 不会阻止同一 logical_am 的新授予。

**挂起与唤醒不丢失**：不可授予的判断与入队在同一个 allocator 串行化点完成；每次 slot_free 和 CANCEL 都在同一串行化点重新评估挂起队列（continuation 只要其预留出现空槽即可授予；fresh 严格 FIFO）。唤醒交付的是**已经授予的 lease**，不是重试许可，因此旧稿 `BUSY + retry_event_snapshot` 中“判失败之后、读快照之前”的竞态不再存在。每个 owner 至多一个挂起请求，请求队列容量等于每 Tile context 数，不会溢出。挂起只阻塞发出请求的 context，不能挂住 RoCC 或其他 context。

slot 与 engine 选择（V1 默认；必须确定性并进入 target fingerprint，使 validator 与 RTL 选同一个）：slot 取 `(group, index)` 序的首个空闲槽，与当前软件一致；bound_engine 与 group **独立**选择，取本 Tile 中已绑定 live handle 最少的 engine，平局取最小编号。可以绑定暂时忙的 engine，这不占其执行时段。软件不指定物理 bank 或固定 engine 编号；当前软件的 `bound_engine=slot.group` 由 SW-A2 改为本规则。

请求身份 `(owner, request_seq)` 保证互连重放不会重复授予。bundle_slots 由 compiler 写入 Tile Program 元数据；当前实现由 `_am_peak_live` 与独立 verifier 校验 ≤ slots_per_context，运行时 `_program_am_bundle` 复算。硬件 loader 拒绝元数据与程序不一致的 Tile Program。

### 4.2 SUBMIT_STEP 与 immutable descriptor

```text
SUBMIT_STEP(lease_handle, seq, A_descriptor, B_descriptor,
            input_ready_tokens, input_usage_handles,
            k_extent, INIT_or_ADD, FINAL)
  -> ACCEPTED(step_token)
   | FULL                   # mailbox 或结果 credit 不足：无任何状态变化
   | ERROR(reason)
```

接收前验证 owner/context_epoch/handle generation、bound_engine、shape/布局/范围/mask 和 seq。seq 从 0 连续，拒绝重复、跳号、重复 INIT 和 FINAL 后提交；seq=0 为 INIT，后续 ADD，最后一步带 FINAL；one-shot 为 INIT|FINAL，不能绕过配额。

ACCEPTED 说明 descriptor 和所有影响执行的配置已快照或完整转存到受保护存储，软件可改写原 descriptor 本体；**不说明 A/B 数据可以释放**。FINAL 被接受即关闭此 handle 新步骤入口并置 final_issued（§4.1 的等待条件依赖它）。submitted、issued、committed 游标分开维护，仅 ACCEPTED 增加提交游标。

允许接受尚未 input-ready 的步骤，但 token 必须对应已经 ready 的输入或已由 MFE/producer 受理的生产操作。不能提交等待任意未来软件信号的步骤，再把其生产者放到可能阻塞的新申请之后。FULL 时 UCE 将该 context 置 `WAIT_ENGINE_QUEUE` 并调度其他 context，credit 归还后重新提交同一步；同 context 后续指令因 Tile Program 程序序而等待，compiler 应交错两链步骤，硬件不额外增加链间依赖。

源 IR 的 `tile.boa.async "matmul"` 必须显式携带 `ins(A,B)`、`outs(L1 output)`、`acc(AM)` 和 seq。lowering 校验 outs 与 AM allocation 绑定一致后，映射为 INIT/ADD 和 FINAL；硬件协议不必每步重复传完整目的地。`accumulate output` 包含本次乘法，再触发最终写回和自动释放，不是单纯拷贝旧 accumulator。

逻辑 descriptor v0 至少包含：

| 字段组     | 字段与 owner                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------- |
| 版本/身份  | descriptor_version、size、patch_mask、request identity、ctx_key、handle/generation、chain_id、seq |
| 不变链配置 | output shape、acc_format、layout、已准入 output_destination、epilogue 参数；授予时固定            |
| 输入       | A/B typed view、allocation/generation、offset/extent/stride、dtype/权限、transpose、有效 mask     |
| 步骤       | k_extent、INIT/ADD、FINAL、input_ready_tokens、input_usage_handles                                |
| 内部状态   | bound_engine、physical group/region、结果 credit；由设备维护，不由 compiler 猜测                  |

运行时 patch 仅允许 patch_mask 声明的字段，且发生在 ACCEPTED 前；成功接受后不能再受软件配置寄存器改写影响。参数非法在可观察 AM 副作用前拒绝；中途故障则走安全收尾，不承诺回滚已写出的部分输出。

### 4.3 事件、等待与释放合同

| 事件/状态                            | 严格定义                                                     | 不能据此推导             |
| ------------------------------------ | ------------------------------------------------------------ | ------------------------ |
| am_request_parked                    | 申请已挂起于本 Tile allocator，无 slot/quota/generation 变化 | 已取得任何资源           |
| am_wake                              | lease 已授予并交付给该 context，PC 可越过 ALLOC              | 输入已就绪、计算已开始   |
| am_request_cancelled                 | 挂起请求因 owner fault/reset/rebind 被丢弃，永不唤醒         | 该 owner 的 lease 已回收 |
| step_accepted                        | descriptor 已安全快照                                        | 输入已读、计算已开始     |
| compute_done                         | 所有本步结果已产生并被保护路径接住                           | AM 已提交                |
| step_committed                       | 本步全部 AM 更新完成且可供下一步累加                         | C 已到 L1                |
| input_release                        | 指定层级缓冲的最后读取和全部读者已结束                       | C commit 或 slot_free    |
| output_visible                       | FINAL 结果在约定 L1 目的区域可见                             | 旧访问/事件引用全部退休  |
| slot_free                            | 本代区域可重分配，quota 已归还                               | 旧事件记录可覆盖         |
| chain_done / chain_terminal(SUCCESS) | FINAL 输出可见、旧访问结束、slot_free/quota-- 都已完成       | UCE 已经读取该记录       |
| chain_terminal(ERROR/CANCELLED)      | 失败访问受控结束、slot/配额安全回收后记录失败                | 成功 output_visible      |

硬件 mem_req/mem_resp 继续携带 req_id、owner/generation、解析地址、访问类型、byte mask、data/status；合法请求参与实际 L1 仲裁，不能以接收命令代替数据传输。

WAIT/WAIT_ANY 复用现有事件体系的等价能力，必须原子地检查终态并登记等待者。事件先到时立即满足；已处理的粘性事件从 wait_set 移除，不能形成忙循环。事件身份同时区分 context、entry、generation、seq、kind。

完成事件按 Tile 事件槽实现：每个 `tile.boa.async` 的事件名在程序内唯一，事件槽数在 context 绑定时静态确定并经 admission 检查；身份为 (ctx_id, epoch, event_id)，所以同一 logical_am 的新链与旧链事件互不覆盖。旧链 FINAL 事件在其 slot_free 之后置位并保留到 await；程序先 `ALLOC` 同一 logical_am、后 `await` 旧事件时，新申请在旧链 slot_free 后即被授予，不会自锁（§8.1b Q09）。AM terminal record 是有界诊断环，满则覆盖最旧并计溢出，不参与授予或唤醒。若复用 EventTable 的引用管理，同样需要证明终态保留、有限容量和多代复用。

FINISHING 收尾不依赖已经返回的原 context 再运行并消费事件。必要通知和外部引用完成后可按合同自动退休；有外部等待者时仍需保留可观察终态。context ID/epoch 只有在命令、输入引用、AM 访问和事件引用都安全处理后才能复用，不使用后台 detach。

### 4.4 软件管理 AM 与源 IR 演进

本次源语义与任务见 [BOA_Software_Managed_AM_IR_Plan.md](BOA_Software_Managed_AM_IR_Plan.md)。该计划 §3.4“受限控制流，不增加通用 DAG 执行器”**保留，但本次不实现，也不是软件 gate 的依赖**。下述语法已在 v4 parser/verifier/lowering 中实现（实施报告 §1）。

```mlir
%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"
    alignment = 256 : !tile.l1_buffer<64x256xf32>
%acc = tile.alloc memory_space = "am" logical_slot = 0 output(%out)
    shape = [64, 256] dtype = "f32" alignment = 256
    : !tile.am_buffer<64x256xf32>
```

allocation 表达一个逻辑 AM 申请，只返回 `%acc`。普通 L1 allocation 同样保持一结果，省略 memory_space 仍是 L1；不能把 EVU 的同名 f32 `%acc` 改成 AM。资源不足时 `ALLOC_AM` 挂起并由 slot_free 唤醒（§4.1，已实现）；源程序不增加 SUCCESS/BUSY/ERROR 判断或重试分支。compiler 从线性 AM 生命周期算出静态 bundle（同时存活链数峰值，链在其 FINAL Matmul 之后不再计入），超过 slots_per_context 时编译期拒绝。

Matmul 为 `%done = tile.boa.async "matmul" ins(%a,%b) outs(%out) acc(%acc)`，只返回完成事件：

| 源标志              | AM 动作                | 写 L1 与释放                                |
| ------------------- | ---------------------- | ------------------------------------------- |
| 无标志              | INIT，seq=0            | 不写 L1，链继续                             |
| `accumulate`        | ADD                    | 不写 L1，链继续                             |
| `output`            | INIT + FINAL，one-shot | 最终写 L1，安全后自动释放                   |
| `accumulate output` | ADD + FINAL            | 包含最后一次乘法，最终写 L1，安全后自动释放 |

最终 output 消费 AM 链，不能继续提交或 `tile.free %acc`；L1 `%out` 独立存活，后续 store `%out` 才是 L1→L2。非 output 的完成是 step_committed，output 成功完成是 chain_done；普通 `tile.await` 及输入/输出依赖继续保留。

**本次保持单 block 源 IR，不增加 `tile.try_branch`、`tile.await_any`、`tile.fail`、`cf.br` 或 status/retry 源类型，不放宽 branch/label 白名单，也不实现 CFG verifier。** 具体分支、重试、等待登记和恢复编码留给更低层 IR／执行协议后续设计；共享 AM 的配额、未受理无副作用和安全释放合同仍有效。

旧无 operand Matmul 签名已迁移；真正只有 FLOPs/控制门的用途已改为显式 `tile.boa.estimate.async`。数值 backend 不静默执行 estimate，不保留旧别名或默认空 operand。

本次迁移范围是源语义、typed executable、codec、线性独立 verifier、AM 资源/数值/周期模型、既有 UCE/事件接线和调用方。当前产物与本次实现均为 v4/v4，受影响 artifact 已重生成。64×256 f32 的 64 KiB 单槽需求与新 L1 output 预算分开计算，不承诺 L1 自动减少相同容量。

### 4.5 Rocket / RoCC UCE

复用 [LazyRoCC][up-rocc] 的 Decoupled cmd/resp 等接口；每 Tile 一个裁剪 Rocket hart、默认四个可配置逻辑 context，并包含八 context 验收配置，不将 context 数等同 hart 数。

自定义指令（名称暂定；opcode/funct/位宽**由后续规格冻结**，由一份权威 schema 生成 Scala 常量、C header 和验证 codec，不手写三套编号）：

| 指令         | 语义                                                           | RoCC resp                            |
| ------------ | -------------------------------------------------------------- | ------------------------------------ |
| `AM_REQ`     | 按 §4.1 一次发出；描述符位于 TCM                               | GRANTED(lease) / PARKED(req) / ERROR |
| `BOA_SUBMIT` | 按 §4.2 提交一步                                               | ACCEPTED(token) / FULL / ERROR       |
| `EVT_STATUS` | 读本 hart 各 context 的唤醒/事件就绪位图（电平，非破坏性读）   | bitmap                               |
| `EVT_TAKE`   | 取出某 context 的唤醒记录（lease）或某事件槽的 step/chain 终态 | record                               |
| `REC_REAP`   | 退休已观察的事件槽（可选；context 退休时自动退休），不影响 AM  | ok / ERROR                           |
| `CTX_CANCEL` | fault/reset/rebind：取消挂起请求并回收该 owner 的 lease        | ok                                   |

固件调度：context 状态为 READY/WAIT_EVENT/WAIT_AM/WAIT_ENGINE_QUEUE；调度循环读一次 `EVT_STATUS` 选择可运行 context。WAIT_AM 的 context 只在唤醒位置位后可运行，取出 lease 后 PC 越过 ALLOC，**绝不重发 `AM_REQ`**。没有可运行 context 时，V1 在调度循环中读位图（validator 未实现中断）；RoCC `interrupt` 留作后续优化。读位图只是观察，授予、drain 和 slot_free 都由设备自主完成。

RoCC resp 只返回状态和短 token，完整 descriptor/record 位于受保护 mailbox；resp 在有限周期内返回，busy 接线不得让 core 等整条链结束才处理其他 context。descriptor/mailbox 使用明确的 uncached/TCM 路径和 release/acquire 顺序，保证发布前完整写入；不假定 Rocket D-cache 与 BOA L1 SPM 自动一致。固件保留 Tile Program 执行、wait 和 context switch，不改成通用 DAG 执行器，不引入共享内存 atomic。

## 5. 数据流、控制流和时序路径

### 5.1 独立完成边界

![AM lease、engine 步骤占用与持久事件生命周期](../image/boa_gemmini_lifetimes.svg)

图为逻辑先后，不是周期测量。固定 affinity、AM lease 和 engine 步骤占用是三个不同概念；input_release 不由 C 的状态推导。

- Group `tile.signal input_released` 继续表示该 Task 的 Group 输入不再需要，等待所有逻辑 Task 聚合；不是一个 BOA 内部 load 通知。
- output_visible 仅保证 L1 C 可见；L2 output_ready 还要等 MFE L1→L2 store，HBM 可见还要等 Group DMA store。
- slot_free 归还 AM 和 quota，可以早于事件 reap，并在同一 allocator 串行化点重新评估挂起请求；不能等待 UCE 轮询，也不能先清掉旧完成状态再复用物理槽。
- Task retirement 继续等所有资源/外部引用安全结束；FINISHING 收尾允许自主完成，而缺 FINAL 退出是错误。

### 5.2 两槽软件流水与进展证明（bundle 预留）

> 2026-10-27 起，进展由「一次申请 + 挂起 + slot_free 唤醒 + bundle 预留」保证，已在软件实现并验证；旧稿基于 TRY/BUSY 轮转与 retry snapshot 的伪代码作废。源 IR 仍是单 block、程序序。

每条已建立链只靠它已有的一个 AM slot 完成 `INIT → ADD… → FINAL → DRAIN → RELEASE`，不依赖新申请成功。进展论证如下：

1. **无持有并等待他人**：fresh owner 只在整个 bundle 能放入未预留空槽时获得首个 lease，余下部分为它预留。此后它的续链申请要么使用自己的预留，要么只等自己 FINAL 已受理的链 drain，不等待其他 owner 的 slot。
2. **被等待者不需要 UCE**：续链可能等待的链 FINAL 已被受理，之后的 commit/output/drain/slot_free 都由设备自主完成，不需要已阻塞的 context 再执行指令。
3. **无自锁**：尚未受理 FINAL 的链数达到 bundle 时再申请，是 compiler/verifier 静态拒绝、运行时 ERROR 的合同错误，不会挂起。
4. **fresh 不饿死**：fresh 严格 FIFO、不插队；bundle ≤ slots_per_context ≤ total_slots。持有者依 1–2 最终释放，并在无 lease、无挂起请求时归还 bundle，因此队首最终能整组放入。

例：8 槽、5 个 bundle=2 的 context。前 4 个各获首链并各预留一槽，未预留空槽归零，第 5 个挂起在 FIFO 队首。某个持有者只释放其中一槽时，该槽仍属于它的预留，第 5 个继续等待；该持有者不再有 lease 且没有挂起请求时 bundle 归还，第 5 个才整组获得两槽。不会出现第 5 个只拿一槽、再与他人互等的情况。

进展结论仍是有条件的：有限 K-step、已受理输入最终完成、ready 仲裁与 drain 公平、L1 输出下游最终可用、安全释放不依赖 UCE 轮询。在这些前提下满槽只造成背压；**quota=2 本身不保证无死锁，起作用的是 bundle 预留**。

已有链的 mailbox、供数、输出区域和 drain 通路不能被新申请耗尽；输入 token 只能等待 ready 数据或已受理 producer；不提前占 engine/group 服务资格再等未提交输入。沿用一个驻留 context 一个 memory profile，不额外永久保留空 AM 槽。

软件证据：`test_am_request_parks_once_and_wakes_on_slot_free`、`test_am_bundle_reserves_continuation_and_prevents_hold_and_wait`（含 FIFO 不插队）、`test_am_continuation_waits_only_for_own_draining_final`、`test_am_same_slot_reuse_waits_for_final_release`、`test_am_cancelled_request_is_never_woken`、`test_runtime_alloc_am_blocks_once_and_wakes_on_slot_free`。硬件须以 §8.1b 的 Q 项在 RTL 与 FPGA 上复现，不能只引用软件结果。

### 5.3 功能模型、周期模型与 RTL 联仿

软件已建立共享 AM 的功能与周期合同：owner/quota/affinity 元数据、INIT/ADD/FINAL、挂起/唤醒、bundle、分阶段结果服务和错误回收。但当前是 Tile Group 级 manager、每 Tile 单个 BOA 实例和 binary64 step 数值（§1.3）。RTL 联仿前必须完成 SW-A1–SW-A4，使 validator 的拓扑、engine 选择、数值和终态顺序与硬件同一合同；功能模型不冒充周期模型，所有场景报告验证层级。

RTL 联仿采用进程内 C ABI（C++/Verilator shared library + Python 薄封装），不每条命令启动子进程，不一次性返回矩阵后伪造中间时序：

| 域                         | 权威                                              | 禁止事项                              |
| -------------------------- | ------------------------------------------------- | ------------------------------------- |
| sim cycle                  | pipeline validator                                | RTL 私自跑到完成再回填事件            |
| L1/L2/HBM 字节             | ByteStore / ResolvedMemoryView                    | 第二份不一致的共享内存镜像            |
| 外部内存时序               | 现有 bank/transfer/NoC，加 BOA 请求竞争           | 只调用 read_view 就当作完成 SRAM 时序 |
| AM、结果接收、阵列内部状态 | 当前选中的功能/周期/RTL backend，只有一份状态权威 | Python 和 RTL 同时更新同一 AM         |
| 数值期望                   | 独立 oracle                                       | DUT 结果作为 golden                   |

C ABI 负责版本/能力检查、实例创建/reset/销毁、稳定输入、单周期推进和请求/事件批次；按 beat 批量跨语言，不逐 byte 调 Python。manifest 包含 RTL/source/config hash、精度、array shape、context/group/slot 配置。

cycle c 的顺序约定：交付到期响应并推进一次 RTL edge → 采样请求/完成 → 完成先进入 UCE-local 路径 → UCE 发射形成下一 edge 的输入。新 memory 请求只能在明确的后续服务时刻返回；禁止同一调用内无延迟读写并反向推进 RTL。

沿用现有 `engines.tick → uce.step` 次序；内存服务早于 Tile tick，新增 BOA 请求最早下一轮服务，该边界延迟要纳入对拍。初始时钟 1:1，一次 edge 只推进一次；不承诺多时钟 CDC。真正完成由 backend 的事件产生，不沿用旧 `EngineJob.finish_cycle` 预测。

### 5.4 内存与真实 UCE 接入

runtime 将 logical view 解析成带 allocation/generation/profile 的物理 segments。读字节在实际 read service 时刻取样，写入在服务完成时更新 ByteStore 后回报；不能在 STEP 接受时提前快照 A/B 数据。V1 初始数值 operand 使用 SPM，不兼容 cache/profile 明确拒绝。

L2→L1 的 MFE 流量、L1→阵列的 operand 读取、阵列→AM 结果和 AM 内 RMW 分开计费。取消 RF 管理要求不等于取消 operand 读取或硬件转置开销。过期响应不得写入新 owner，且必须安全结清旧访问引用。

BOA-only 模式由 Python UCE 执行相同协议；Rocket 模式由真实固件执行，**同一个 Task 不能同时由两者解释**。Group admission 经 ControlBackend 等价边界提交已验证 capsule，真实 UCE 负责 L1 view、MFE 请求、`AM_REQ`/`BOA_SUBMIT`、WAIT_AM/等待/错误分支、phase signal 和 return。

非受支持的 Matmul 指令闭包明确拒绝，不能偷偷退回 Python UCE 执行。该闭包不宣称整个 validator ISA 都已成为 Rocket 软件；在 FINISHING 后无需返回旧 context 才能完成硬件释放及通知退休。

两条控制路径比较输出 bytes、所有权和合法事件偏序，不要求总周期相同；Rocket 取指、固件调度与轮询开销必须计入，固件/RTL/hash 和 linker 映射进入 manifest。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 V1 基线与容量

| 参数                                                      | 设计值/规则                                                | 当前软件键/默认                                                              | 性质                                             |
| --------------------------------------------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------ |
| arrays_per_tile（拟新增）                                 | bring-up 1；V1 多阵列基线 4，覆盖 2/4 验证                 | 无；每 Tile 一个 `BOAEngine`                                                 | 一个 engine 对应一份阵列执行资源                 |
| 每 Tile group_count / slots_per_group / 总槽              | 4 / 2 / 8                                                  | `engines.boa.am.group_count`/`slots_per_group` = 4/2，当前按 Tile Group 实例 | SW-A1 后按 Tile 实例；slot 完整落入一个 group    |
| slots_per_context                                         | 2                                                          | `engines.boa.am.slots_per_context` = 2                                       | 硬约束，跨本 Tile engines 计数；V1 不允许改大    |
| bundle_slots                                              | 1 或 2，compiler 计算，loader/verifier 校验                | `_am_peak_live`、`_program_am_bundle`                                        | 程序元数据，不是配置项                           |
| 挂起请求容量                                              | 每 owner ≤1；队列深度 = 每 Tile context 数                 | `SharedAmManager._pending_requests`                                          | 不会溢出，不需要 BUSY                            |
| slot_capacity_bytes                                       | 软件默认 262144（256 KiB）；FPGA 映射见 §7.4.3             | `engines.boa.am.slot_capacity_bytes`                                         | 由 SRAM profile 冻结                             |
| context_count                                             | 默认 4；覆盖 1/2/4/8，受 UCE/memory admission 限制         | `--context-mode`                                                             | 不是 AM 永久预留数，也不是 engine 数             |
| array_rows / array_cols                                   | 默认 16/16；矩形需独立 gate                                | `engines.boa.opa.rows/cols`（旧 OPA 峰值几何）                               | Scala elaboration 与 target fingerprint          |
| tile_m / tile_n / tile_k                                  | 默认 16/16/8；tile_k 也是数值 profile 参数                 | 无                                                                           | 改 tile_k 会改变数值结果                         |
| numeric_profile（拟新增）                                 | `bf16_fp32_mac_rne_v1`                                     | 无；当前为 binary64 step 累加                                                | 进入 target fingerprint                          |
| 结果 FIFO / AM credit / commit / drain / L1 output credit | 软件 step 粒度默认 4 / 8 / 2 / 1 / 1                       | `engines.boa.result_fifo_depth` 等，hardware_config.yaml 104–112 行          | 硬件按微块与 O+R≤Q 冻结，由 PPA exploration 冻结 |
| next_step_mailboxes_per_live_chain                        | 1，加一个当前执行记录                                      | —                                                                            | 授予时保证 continuation 元数据，不建无限步骤队列 |
| 完成事件槽                                                | 每个 typed Matmul 一个，context 绑定时按程序静态事件数分配 | UCE 事件 `ctx{n}:{name}`；`SharedAmManager` terminal 环（有界、覆盖最旧）    | AM 授予从不等待事件退休                          |
| SRAM/bank/adder/drain ports 与延迟                        | 各自独立配置和服务模型                                     | —                                                                            | 由 SRAM profile / PPA exploration 冻结           |
| FPGA 时钟                                                 | 首版 100 MHz 单时钟（Chipyard VCU118 默认）                | `system.clock.core_mhz` = 1000（建模基线）                                   | 板级 fmax 由 PPA exploration 冻结                |
| ISA 位宽、context/事件容量、PE pipeline                   | 不从示例猜编码或硅片频率                                   | —                                                                            | 由后续规格冻结                                   |

slot_capacity 包含 M/N、stride、accType 字节数、padding、alignment 和实际分配粒度。紧密 16×16 FP32 输出的有效数据下界为每槽 1024 bytes；当前 typed fixture 的 64×256 f32 为 64 KiB；软件默认 slot 为 256 KiB，八槽每 Tile 2 MiB。实际宏容量不由这些数字冻结，也不包含结果缓冲/metadata/ECC。

AM 总容量按每 Tile `total_slots × slot_capacity_bytes` 计算，**不按 context_count 永久分区，也不跨 Tile 合并**。配置验证拒绝分组数量不一致、slot 太小、缺少结果接收保障、非法 affinity 或 max_live_slots 大于 2；8 contexts 各一槽必须在其他资源足够的配置下可表达。

一个 context 同时至多驱动本 Tile 的两个 matrix engines；两个槽都在 FINAL_READY/DRAINING 时可以没有任何正在计算的 engine，这不是死锁。四 engine 同时计算至少需要两个提供足够 ready 链的 contexts，并且绑定分布、输入和 AM 带宽满足；这只是必要条件，不是利用率保证。

### 6.2 性能观测与公平比较

| 观测域    | 必须分解的原因/关联状态                                                                                                                                                                            |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 配额/分配 | per-context quota、Tile AM full、logical slot in use、bundle 预留阻塞（区分 fresh FIFO 与 continuation）                                                                                           |
| 申请/唤醒 | request/park/wake/cancel 计数、等待周期、reserved_slots、peak_pending；沿用软件 snapshot/trace 字段名（`request_count`、`parked_count`、`wake_count`、`cancelled_request_count`、`waited_cycles`） |
| 元数据    | mailbox full、terminal 诊断环溢出，与 AM 容量不足分开                                                                                                                                              |
| engine    | 每 engine compute、输入等待、bound-engine 等待、配置切换                                                                                                                                           |
| AM group  | update 数、read/write ports、bank 与 adder 冲突分别计量                                                                                                                                            |
| 结果      | result-credit 阻塞、O/R、高水位、在途微块数                                                                                                                                                        |
| 输出/链   | FINAL_READY、drain 反压、slot-free 延迟、REQ→GRANT、GRANT→FIRST STEP、step 间空洞、FINAL→FREE、总 lease 驻留                                                                                       |
| FPGA 板级 | mcycle、每 engine busy/stall、AM group 端口冲突、trace ring 溢出计数                                                                                                                               |

trace 关联 ctx_key、handle/generation、seq、bound_engine、physical group/region、事件种类；不以一个笼统 engine_busy 隐藏瓶颈。

至少比较：私有与共享 AM 在输出驻留不均时的容量利用；低冲突与热点 group；单 context 两槽与多 context 四 engine；FINAL drain 占 quota 对短任务的影响。私有 AM 只作为明确标注的对照，不是 V1 实现路径。

公平比较保持 MAC 数、AM 容量、输入/输出带宽、时钟和精度相同，不能把新增 SRAM/ports 的收益归给共享调度。输出阶段带宽需求可按 `E×q×b` 分别核算新 partial、ADD 旧值读取和写回，FINAL drain 另计；这不是平均流量或 L1/HBM 流量。

不承诺共享 AM、multi-context 或某种 dataflow 一定更快。性能来自真实请求轨迹；PPA 需统一工艺/器件、时钟约束和工具版本，没有条件只报告结构成本。

## 7. RTL/软件实现建议

### 7.1 Monorepo 组织（拟议，不在本轮建立代码脚手架）

```text
hardware/
  gemmini/
    upstream/                  固定来源的 Gemmini source subtree，保留许可证
    src/main/scala/             阵列适配层；与 BOA 调度代码分离
    src/test/scala/             PE/mesh 数值与形状独立验证
    upstream.lock              上游 SHA、来源、受控修改记录
  boa/
    src/main/scala/             transpose、共享 AM、链表、有限队列、仲裁、epilogue/drain
    src/test/scala/             硬件模块与组级验证
    sim/                       Verilator C ABI、独立 harness
    reference/                 独立数值 oracle 与 golden vectors
  uce/
    src/main/scala/             Rocket/RoCC wrapper、mailbox 和系统连接
    firmware/                  bare-metal 调度程序、自定义指令封装、linker
  toolchain/
    Dockerfile                 统一工具链镜像定义
    toolchain.lock             镜像 digest、所有工具/依赖 hash
    build overlay              指向单份 Gemmini/BOA/UCE 源码的构建配置
  configs/                     单阵列、矩形、多阵列可复现配置
  fpga/
    vcu118/                    Chipyard vcu118 SUB_PROJECT 叠加配置、约束、ILA 与板级脚本
    board-tests/               上板裸机测试、结果与 trace 导出
  scripts/                     构建/运行/复现入口（需在实现时真正提供）
.devcontainer/                 引用同一镜像与工作目录，不建立第二套环境
pipeline_validator/            保持现有 Python 子项目，新增有边界的 backend
examples/generators/            数值 Matmul/Conv 场景生成器
examples/workloads/             生成的 IR fixture
```

source subtree 的选择是本稿建议：本 repo 能直接评审 Gemmini 的必要修改，避免隐藏在另一个浮动仓库。不要同时维护“修改后的源码”和一套会再次应用的重复 patch；从锁定 upstream 到本树的变更应可独立导出和解释。

Chipyard/Rocket/HardFloat 依赖进入容器的锁定 dependency workspace；构建 overlay 只引用 `hardware/gemmini` 的权威源码一次，不同时编译 Chipyard 内另一份 Gemmini。禁止复制三套 Scala/Chisel 版本分别给 BOA、Gemmini、UCE 使用。

### 7.2 组件依赖与交接 guide

保留 C0–C10 系统组件编号，新增 SW-A（FPGA 前软件对齐）与 C11（AMD UltraScale+ 上板）。S0 指独立计划**本次范围内**的任务与软件 gate，已完成（见实施报告）；S0 不含保留的 §3.4 和源级 CFG。C1/C4/C5/C6/C7-A 的软件模型覆盖与 RTL/板上覆盖必须分开报告：

```text
C0-S 软件环境 -> S0 独立AM/IR计划（已完成：一次申请/挂起/唤醒、bundle、分阶段结果服务）

S0 -----------------------> SW-A 软件对齐（A1 每Tile allocator · A2 多engine与affinity · A3 数值profile · A4 终态顺序）
S0 -----------------------> C0-H 硬件工具链构建验证（含 Vivado 只读挂载）
S0 + C0-H ----------------> C2   方形 Gemmini 核心实现
S0 + C2 ------------------> C3   硬件 operand / transpose / L1 接口
S0 + C2 ------------------> C6-H 硬件后处理
C2 + C3 + C6-H + SW-A ----> C7-B RTL 与 validator 联仿
C7-B ---------------------> C8   Rocket / RoCC 真实 UCE
C8 -----------------------> C11-A VCU118 单阵列上板
C2 -----------------------> C9   矩形馈数 / 排空
C8 + C9 ------------------> C10  多阵列共享 AM
C10 + C11-A --------------> C11-B VCU118 多阵列上板（最终验收）
```

`+` 表示全部前置条件；C0-S/H 和 C6 软件/硬件区分环境与实现层级，C7-A/B 区分软件集成与 RTL 联仿，不重复建立 IR/scheduler。环境资料与版本调查可以先做，但不能把已有 RTL 作为 S0 通过的条件。

#### SW-A · P0：FPGA 前软件对齐

validator 必须先与本稿硬件合同一致，联仿和上板的差异才能归因到 RTL。四项可以并行，但都在 C7-B 之前完成：

| 项                          | 现状（§1.3）                                        | 改动                                                                                                                                  | 验收                                                                                     |
| --------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| SW-A1 每 Tile allocator     | `TileGroup` 建一个 manager，由 4 个 Tile 共享       | 每个 `ComputeTile` 持有自己的 `SharedAmManager`（`tile.py` 已有本地构造路径）；报告按 Tile 汇总                                       | Q08；multicontext/split-K 例子的周期与 AM 统计按新拓扑重新记录，不沿用旧数               |
| SW-A2 多 engine 与 affinity | 每 Tile 一个 `BOAEngine`；`bound_engine=slot.group` | 新增 `arrays_per_tile`；按 §4.1 规则独立选择 bound_engine，launch 只进入该 engine；原 engine 忙时等待                                 | T04/T05/T06：单 engine 同时服务超过两个 live handle，强制忙时不迁移                      |
| SW-A3 数值 profile          | binary64 累加整个 step，commit 时舍入               | `boa_numeric` 与测试 oracle 改为 §3.2；`numeric_profile`、`tile_k` 进入 target fingerprint，按严格版本规则升级并重生成受影响 artifact | SoftFloat 交叉校验向量；保留 binary64 与 profile 结果不同的反例；全部 typed fixture 通过 |
| SW-A4 终态顺序              | `chain_done` 记录在 FINAL commit 时写入             | 记录移到 slot_free 之后，或同时记录 output_visible 与 slot_free 时刻                                                                  | 每个成功终态时刻 ≥ slot_free；UCE 事件不早于终态                                         |

guide：这四项会改变 validator 的可观察行为，是有意的合同变更；更新测试期望时要写明依据，不能为了让旧数字通过而保留旧拓扑或旧数值。

#### C0 · P0：环境与来源

- 输入：独立计划的软件环境、§1.4 硬件版本组合与源码。
- 实现：C0-S 先保证现有 Python/xDSL 环境可复现；S0 后 C0-H 再构建锁定的 Scala/Chisel/Rocket/HardFloat/Verilator/RISC-V 工具链，保留许可证和单一 build graph。
- 输出：分层环境清单、镜像 digest/lock 与真实构建入口。
- 验证：软件 parser/CLI 环境不依赖新 RTL；硬件阶段再做干净 elaborate、harness 编译和离线重复，不能仅打印版本。
- guide：宿主不安装依赖；版本不兼容时带证据调整整套 pin，不切 latest。

#### C1 · P0：独立数值 oracle

- 输入：BF16 bits、`bf16_fp32_mac_rne_v1`（§3.2）、tile_k、INIT/ADD/FINAL、已 review 的后处理规则。
- 实现：用精确有理/整数运算求 `a×b+p`，再单次 RNE 到 FP32，然后按块做 AM add；用 Berkeley SoftFloat 3e 的 `f32_mulAdd`/`f32_add` 交叉校验；不调用 DUT 作为唯一 oracle。
- 输出：每个贡献块、AM 提交及最终 C 的精确 golden bytes。
- 验证：非对称输入、tie、signed zero、subnormal、overflow、NaN/Inf、尾块旁路、内部微块 INIT 覆盖一次、bias 一次；与 HardFloat 位级对拍后冻结 NaN 编码。
- guide：Python float 先求和再转 FP32 会双重舍入，不是 FP32 fused 累加；固定每次舍入与分组顺序，不能只比 NumPy 容差。

#### C2 · P0：Gemmini 方形核心

- 输入：规范 A/B beat、shape/有效 mask、FP32 partial 格式、golden vectors。
- 实现：review 后选择单一 dataflow，审计 PE/mesh 中全部 FP32 路径、skew/tag/last/drain 和安全配置切换；Gemmini Float `mac` 已是 HardFloat fused RNE（§1.4 第 7 条），需保证 PE 累加寄存器与 mesh 输出均为 FP32、块起点 +0、块长 tile_k、无效 lane 旁路；保留正确性所需内部状态。
- 输出：方形 adapter、最小不可暂停块的输出量合同、16×16 及另一方形配置。
- 验证：PE→mesh→完整 tile 数值，AM 端延迟时结果不丢；原始 compute_done 不提前完成 AM 步骤。
- guide：只改 accType 或删 square/double-issue 断言不算实现；不建立用户已取消的软件 RF/load 接口。

#### C3 · P0：operand / transpose / L1 访问

- 输入：合法 typed view、generation、已受理 producer 的 input_ready、input_usage 引用和内存背压。
- 实现：真实地址/segment、四种 transpose、有效 lane、请求身份；内部缓冲按 adapter 必需性设计，不强制独立 RF。
- 输出：合法馈数、实际 memory 请求、普通 reader 引用结束；不输出专用内部 load 通知。
- 验证：T15/T18、多个读者、最后读取延迟、释放后覆写、非对称四种转置、旧响应与错误。
- guide：descriptor 快照与 A/B 数据生命周期不同；不能凭 ACCEPTED 或 MAC done 复用 buffer。

#### C4 · P0：共享 AM 功能合同与事件

- 输入：ctx_key、logical_am 0/1、bundle、AM_REQ/STEP、本 Tile 八槽、seq 和 FINAL。
- 实现：动态 lease、两槽 quota、bundle 预留、挂起 FIFO 与唤醒交付、不可变 affinity、INIT/ADD、全成或全败授予、粘性完成和错误收尾。可先用抽象 AM 存储，不伪造周期/RTL。
- 输出：可执行的功能状态机、独立完成边界、owner 映射及事件引用。
- 验证：T01–T09、T14/T17/T19/T21 的资源/功能适用部分及 Q01–Q09，含八 contexts 各一槽和旧事件跨 slot 复用；WAIT_ANY/等待登记竞态随低层方案后续验收。
- guide：复用现有事件框架但补足 ctx epoch/链身份；AM 授予与事件 await/reap 无关，正常和错误 terminal 都能退休。

#### C5 · P0：有限周期资源与进展

- 输入：C4 功能模型、C7-A 两槽程序、每 engine/group/port/adder/结果缓冲的真实配置。
- 实现：current+one mailbox、bound-engine ready 仲裁、O/R credit、route/RMW/writeback、自动 drain 和公平服务；新申请不耗尽 continuation。
- 输出：可审计的 cycle/event/credit trace、按原因分解的阻塞和资源不变量。
- 验证：本次核对 result credit、bank/adder、commit、drain、挂起到唤醒的时延和资源不变量；WAIT_ANY/等待登记的端到端验证留给后续低层方案，不强加源级控制流。
- guide：不使用无限 FIFO、零延迟 AM 或额外永久空槽；不把增加 bank 当作 adder 吞吐；也不把延期的进展验证报告为已通过。

#### C6 · P1：后处理与软件 Conv

- 输入：FINAL AM、bias/scale/clip/quant 参数与 Conv 展平配置。
- 实现：FINAL 后只处理一次，再自动 drain 到已准入 L1；软件 im2col 复用 Matmul，不加硬件 Conv。
- 输出：BF16/INT8 字节与展开流量记录。
- 验证：饱和/tie/zero-point、关闭/开启后处理、小型 Conv padding/stride/dilation 对照独立参考。
- guide：INT8 输出不等于支持 INT8 Matmul；参数非法在输出写入前拒绝。

#### C7 · P0：typed IR、两槽程序和联仿

- 输入：软件管理 AM/IR 独立计划及已准入 views；S0 后的 RTL 层再接 C2/C3/C6-H。
- 实现：C7-A 按独立计划完成单 block AM allocation、四 operand/四模式、已有完成依赖和资源/数值模型（已完成）；不新增源级分支、状态结果或 retry CFG。C7-B 在 SW-A1–SW-A4 完成后接入 RTL；后续低层控制另行验收。
- 输出：v4 typed artifact、compiled-only replay、C 字节和本次资源/事件证据。
- 验证：按独立计划 IR01–IR17 的本次/延期标记执行；重点保留 non-output 不写 L1、accumulate output 自动释放和 f32 partial。T01–T22 是完整系统目标，不等于本次全部闭环。
- guide：拒绝无 ByteStore 的数值路径和旧 operandless Matmul；延期控制流不允许被假成功、无限缓冲或隐式 fallback 掩盖。

#### C8 · P1：Rocket / RoCC UCE

- 输入：已通过单阵列联仿的 top、§4.5 指令 schema、两槽 firmware 和 admitted capsule。
- 实现：真实执行 `AM_REQ`（GRANTED/PARKED）、WAIT_AM 与唤醒位、`BOA_SUBMIT` 的 FULL→WAIT_ENGINE_QUEUE、WAIT/context switch、事件槽退休、`CTX_CANCEL` 和错误收尾；Python UCE 对相同 Task 停用。
- 输出：Rocket 实际驱动 BOA 的系统 top、固件 hash 和指令/事件 trace。
- 验证：每次授予恰好对应一个 `AM_REQ`，挂起后不重发；第三条未完成链申请为 ERROR/fault；`FINAL(slot0)→ALLOC(slot0)→await(旧事件)` 不挂死（Q09）；context 返回后自主 drain/free；完成先于 WAIT、8-context 配置、错误 terminal。
- guide：直接拉 RoCC 信号不算真实 UCE；不能使正常释放依赖已返回 context 重新 poll；读唤醒位图只是观察。

#### C9 · P1：矩形阵列

- 输入：已验证方形 adapter、独立 R/C 和合法 tile 分块。
- 实现：独立推导 row/column skew、transpose、mask、tag/last/drain，不沿用 square block_size 假设。
- 输出：至少 8×16、16×8 与四种 transpose 配置。
- 验证：不同 M/N/K、尾块、第一/最后有效输出、气泡输入以及 complete bytes。
- guide：形状改变不改变 quota、affinity、step_committed 和 FINAL 自动释放合同。

#### C10 · P1：多阵列共享 AM

- 输入：已通过单阵列/UCE/矩形验证的组件、4 engines/4 groups/8 slots，及 2-engine 对照配置。
- 实现：本 Tile 内跨 engine 的共享分配/路由、按 §4.1 规则独立选择的固定链 affinity、独立 group RMW 与公平 drain；只复制必要执行/结果接收资源，不建 private AM，也不跨 Tile 共享。
- 输出：同 context 两链独立推进、多 context 覆盖四 engines、动态八槽利用和完整终态。
- 验证：T01–T22 的适用数值/周期/RTL层；强制原 engine 忙且其他空闲时不迁移；不同 engines 访问同一 group 的不同合法区域；slot 换代后重新分配可改变 engine。
- guide：证明共享的是容量和访问，不要求同一 live handle 跨 engine 更新；四 engines 满载不是单 context 的必达行为。

#### C11 · P1：AMD UltraScale+（VCU118）上板

- 输入：通过 C8（A 阶段）或 C10（B 阶段）的 RTL、锁定 Chipyard `vcu118` harness、C0-H 挂载的 Vivado、validator 生成的测试向量与 manifest。
- 实现：BOA RoCC config 叠加 [`WithVCU118Tweaks`][up-vcu118]；首版 100 MHz 单时钟；§7.4 的 URAM/BRAM/DSP 映射；片上 trace ring 与 PMU 经 UART 导出；L1 output 背压注入寄存器。
- 输出：bitstream、utilization/timing 报告、板上运行 manifest（bitstream/固件/artifact/输入 hash、C 字节、计数器、trace）。
- 验证：§8.1c F01–F08；A 阶段为 1 Tile × 1 阵列，B 阶段为 1 Tile × 4 阵列。
- guide：WNS < 0 的 bitstream 不用于验收；ILA 只辅助调试，不是通过证据；单个 Tile 的 AM/router 不跨 SLR 拆分。

### 7.3 Docker / dev container 可复现契约

容器仍是默认环境。初稿阶段已确认 Docker client/server 可访问；本次文档修订未运行硬件镜像、安装依赖或修改系统文件，不把该历史工具探测当作共享 AM 构建证明。

实现必须提供一个权威入口，下面是**拟交付的命令接口，不是本轮已经存在的脚本**：

```text
hardware/scripts/env build --locked
hardware/scripts/env shell
hardware/scripts/check toolchain
hardware/scripts/check numeric
hardware/scripts/check shared-am-functional
hardware/scripts/check two-slot-progress
hardware/scripts/check shared-am-cycle
hardware/scripts/check single-array
hardware/scripts/check validator-cosim
hardware/scripts/check rocket-uce
hardware/scripts/check rectangular
hardware/scripts/check multi-array
hardware/scripts/check fpga-synth vcu118
hardware/scripts/check fpga-board vcu118
hardware/scripts/reproduce <run-manifest>
```

每个 check 都需要在容器内执行真实程序、输出 manifest/日志/数值结果，不能只是列出下一条手工命令。实施计划中再给出这些入口的精确文件、参数和真实预期输出。

环境约束：

1. 基础镜像按 digest，软件按版本/hash；网络只在显式构建/下载阶段使用。包括 Scala compiler plugin、Maven artifact、Python wheels、RISC-V GCC/binutils、Verilator 及依赖。
2. 不在宿主运行 apt/yum、sudo、全局 pip，不修改 `/etc`、shell profile、Docker daemon 或其他项目文件。依赖获取限容器镜像构建；如需包管理器也只在镜像内部。
3. dev container 引用同一 Dockerfile/digest，映射宿主 UID/GID；源码工作目录固定，构建缓存和结果目录独立，不在工作树散落 root-owned 文件。
4. 不使用 privileged，不挂宿主 Docker socket；simulation 与综合不需要设备直通（上板见第 9 条）。
5. `toolchain.lock` 包含完整版本和来源；当前 environment.yml 只固定 Python 3.11 与 xdsl 0.69.0，不能把它当成已经完整锁定的 Python 环境。
6. 每次运行保存：repo/source hash、dirty diff hash、依赖 pins、镜像 digest、配置、编译器 ABI、输入 seed/hash、固件/RTL/library hash、命令、actual output、golden output、报告、必要波形。
7. 复现必须从 manifest 解析配置和输入；缓存不是正确性前提，第二次离线执行应有相同数值和确定性仿真周期。
8. Vivado 不进入镜像，也不在宿主自动安装：由用户提供已安装路径，只读挂载进容器；版本按所锁 Chipyard/fpga-shells 实测支持的版本选定并写入 `toolchain.lock`（由后续规格冻结）；license 经环境变量传入，不写进仓库。
9. 烧写需要 USB-JTAG。线缆驱动和 udev 规则属于宿主系统修改，只能由用户按 AMD 文档自行安装；容器只在 `fpga-board` 命令中以显式 `--device` 映射所需设备。

### 7.4 FPGA 实现映射（AMD UltraScale+，参考板 VCU118）

本节把 §2–§6 的合同映射到 FPGA。资源与吞吐数字都是**推导值或待综合项**，不是综合结果；所有新增结构遵守前文合同，FPGA 不得以“板上方便”为由放宽 quota、affinity、自动释放或数值 profile。

#### 7.4.1 平台与首版原型范围

| 项目         | 事实/选择                                                                                                   | 来源/性质                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 板卡与器件   | VCU118，XCVU9P                                                                                              | [AMD VCU118 页面][up-vcu118-board]                                              |
| 器件资源     | 1,182,240 LUT、2,364,480 FF、2,160 BRAM（75.9 Mb）、960 URAM（270 Mb）、6,840 DSP、3 个 SLR                 | [DS890][up-ds890] Virtex UltraScale+ Feature Summary 与 SLR 数量表              |
| URAM         | 每块 4K×72 bit，两端口共用一个时钟，各端口可独立读写，可级联                                                | DS890 UltraRAM 节                                                               |
| DSP48E2      | 27×18 补码乘法器 + 48 bit 累加器                                                                            | DS890 DSP 节                                                                    |
| 板级 harness | Chipyard `WithVCU118Tweaks`：UART、SPI SD 卡、DDR backing memory、JTAG，默认 100 MHz                        | [pinned Configs.scala][up-vcu118]；构建流程见 [Chipyard 文档][up-chipyard-fpga] |
| 首版原型     | 1 个 Compute Tile：1 Rocket hart、BOA（1→4 阵列）、Tile 内 8 槽 AM、L1 SPM、Tile DMA；A/B/C 经板载 DDR 交换 | 本稿选择                                                                        |
| 不在首版     | Group L2/MFE 全功能、多 Tile、PCIe/XDMA、HBM 板卡                                                           | 后续规格                                                                        |

[INFERENCE] Group L2/MFE 不属于 BOA 合同；首版用 Rocket 可编程的 Tile DMA 把 DDR 中的 A/B 搬入 L1 SPM，并把 L1 中的 C 写回 DDR。validator 侧以同一 artifact 和输入 hash 运行作对照。

#### 7.4.2 模块划分、时钟与复位

```text
RocketTile --RoCC cmd/resp--> BoaFrontend (AM_REQ / BOA_SUBMIT / EVT_* / CTX_CANCEL)
                                |- TileAmAllocator: quota · bundle · 挂起 FIFO · generation · ChainTable
                                |- StepMailbox x8 + ReadyArbiter(bound_engine)
                                `- EventTracker: wake mailbox · 事件位图 · 程序事件槽
L1 SPM --OperandFeeder/Transpose xE--> GemminiArray xE --ResultCapture xE--> AmRouter(E x 4)
AmRouter --> AmGroup x4 (2 slot URAM + 16-lane FP32 RMW) --> Epilogue/Drain --> L1 SPM
TileDMA (TileLink) <--> DDR (Chipyard backing memory)
```

- V1 所有模块在同一 100 MHz 时钟域，与 §5.3 的 1:1 联仿约定一致。BOA 独立时钟以及 RoCC/内存边界的 async FIFO，等 PPA exploration 后再决定。
- 复位：Tile 同步复位。AM reset 中止所有 lease、清空挂起请求并使每个 slot 的 generation 加一（与软件 `SharedAmManager.reset` 一致），不要求物理清零 URAM。
- TileAmAllocator 是本 Tile 唯一的串行化点：授予判断、入队、slot_free 后重评估都在同一拍序内完成，满足 §4.1 的无丢唤醒要求。

#### 7.4.3 存储映射

| 结构                                 | 映射                                                                         | 推导/约束                                                                              |
| ------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| AM slot（软件默认 256 KiB）          | 8 块 URAM 并排，每块用 64 bit，组成 4096 行 × 16 FP32 lane（512 bit）        | 4096 × 64 B = 256 KiB；每块余 8 bit，可留给 ECC/奇偶                                   |
| AM group（2 slot）                   | 2×8 块 URAM 级联成 8192 行，一套读写端口和一条 16-lane RMW 流水              | 每 Tile 64 块 URAM（占 XCVU9P 6.7%），4 Tile 为 256 块                                 |
| group 端口                           | 端口 A 读（ADD 旧值或 drain），端口 B 写回；ADD 读与 drain 读按年龄/轮转仲裁 | drain 不能被持续 ADD 饿死（§3.5）                                                      |
| 结果 FIFO、mailbox、唤醒记录、事件槽 | LUTRAM/BRAM                                                                  | FIFO/mailbox 随 engines 与微块预留；事件槽按每 context 程序事件上限，由 admission 校验 |
| ChainTable、bundle 表、挂起 FIFO     | 寄存器                                                                       | 8 槽、每 Tile ≤8 context，规模很小                                                     |
| L1 SPM                               | BRAM/URAM bank，按 validator L1 profile 划分                                 | 由 SRAM profile 冻结                                                                   |

同一行 RMW 从读旧值到写回期间保持地址互锁或 forwarding；URAM 读延迟（含所用输出寄存级数）计入互锁窗口。

#### 7.4.4 计算单元与 DSP 预算

每个 MAC 的乘积是 BF16 有效位 8×8 bit 的精确 16 bit 积，然后与 FP32 addend 做一次 RNE 舍入。

| 方案                                    | 做法                                           | 风险/约束                                                                      |
| --------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------ |
| HardFloat `MulAddRecFN`（Gemmini 路径） | BF16 扩到 FP32 后做 24×24 fused MAC            | [INFERENCE] 24 bit 宽于 DSP48E2 的 18 bit 端口，综合若不裁掉补零位约 2 DSP/MAC |
| 定制 BF16×BF16+FP32 fused MAC           | 8×8 乘法（LUT 或 1 DSP）+ FP32 对齐/规格化/RNE | 必须在全部输入上与 HardFloat 路径逐 bit 一致（C1/C2 对拍）                     |

| 配置            | MAC 数 | 2 DSP/MAC           | 1 DSP/MAC      |
| --------------- | ------ | ------------------- | -------------- |
| 1 Tile × 1 阵列 | 256    | 512（7.5%）         | 256（3.7%）    |
| 1 Tile × 4 阵列 | 1,024  | 2,048（29.9%）      | 1,024（15.0%） |
| 4 Tile × 4 阵列 | 4,096  | 8,192（超出 6,840） | 4,096（59.9%） |

AM RMW 每 group 16 个 FP32 adder（每 Tile 64 个），epilogue 另有 16 lane 的 bias/scale/转换；LUT/FF 用量由综合报告冻结。

#### 7.4.5 带宽与吞吐推导

以 16×16 阵列、100 MHz、AM 每 group 每拍读写各 1 行（16 FP32）为例 [INFERENCE，未综合]：

| 项目         | 推导                                                              | 结论                                 |
| ------------ | ----------------------------------------------------------------- | ------------------------------------ |
| 操作数       | 每拍 A、B 各 16 个 BF16 = 64 B；每阵列 6.4 GB/s，4 阵列 25.6 GB/s | L1 SPM bank 宽度需匹配               |
| OS，tile_k=8 | 每块计算 8 拍，排空 16 行 × 16 FP32 需 16 拍                      | 双缓冲重叠后利用率上限仍约 50%       |
| WS，tile_k=8 | 16 行中只有 8 行参与 K                                            | 利用率上限约 50%                     |
| tile_k=16    | 计算与排空/RMW 各 16 拍                                           | 可接近 100%，但数值 profile 随之改变 |
| AmRouter     | E 路 512 bit 结果流到 4 个 group，最多 4×4 交叉                   | 输出打拍；同 group 冲突按 §3.5 仲裁  |

默认 tile_k=8 保持已确认的 16×16×8 功能基线。性能配置是否改为 tile_k=16 或加宽 AM 端口，与 PE 数据流 review 一起决定（§9.1）。

#### 7.4.6 RoCC、内存接口与观测

- RoCC cmd/resp 实现 §4.5 指令，resp 在有限周期内返回；Rocket 不因 BOA 链未完成而停住。
- Tile DMA 与 Rocket 经 TileLink 访问 DDR；BOA 只访问 L1 SPM，不直接访问 DDR。
- 片上 trace ring（BRAM）记录 am_request/park/wake/grant/cancel、step accept/issue/compute_done/commit、output/drain/slot_free 和 terminal，带 mcycle 时间戳与 ctx/lease/generation/engine/group；ring 溢出要计数，溢出的 run 不用于顺序验收。
- PMU 计数与 §6.2 同名，经 UART 导出；输入/输出数据可放 SD 卡。
- ILA 按 Chipyard `debug-bitstream` 流程加入，只用于调试。

#### 7.4.7 时序收敛与布局

- 一个 Tile 的阵列、AmRouter 和 AM group 放在同一 SLR。Tile 内共享 AM 的选择避免了跨 SLR 的宽结果总线。
- Gemmini Mesh 只在 Tile 之间插寄存器，Tile 内 PE 的浮点 MAC 是组合路径；提高频率需要流水化 MAC，会改变 skew 与排空时序，必须重跑 C2 数值和 C7-B 时序对拍。
- URAM 级联与 AmRouter 输出打拍要计入 RMW 互锁窗口和 O/R 预留量。
- 验收只使用 WNS ≥ 0 的实现结果；时序不满足时降频或加流水并更新 manifest，不得忽略违例。

#### 7.4.8 上板对拍流程

```text
validator generator -> artifact + A/B bytes + 独立 oracle 期望 + validator report
     -> manifest（bitstream/固件/artifact/输入 hash）
     -> SD 卡或 JTAG 装入 DDR -> Rocket 固件执行 Tile Program
     -> 导出 C 字节、PMU、trace ring -> 对拍
```

对拍规则：C 字节与 `bf16_fp32_mac_rne_v1` oracle 逐 bit 一致；与时序无关的计数（ALLOC/授予数、commit/final/slot_free 数、终态成功/失败数）与 validator 对同一 artifact 的报告一致；park/wake 次数随时序变化，只检查 park = wake + cancel；每链满足 compute_done ≤ commit ≤ output ≤ slot_free ≤ chain_done；结束时 live/pending/reserved 均为 0。周期数不要求与 validator 相同，但必须记录。

## 8. 验证、bring-up 和验收标准

### 8.1 共享 AM V1 验收矩阵

以下逐项对应输入规格 T01–T22，是完整系统的验收要求；本次文档修订没有执行新测试。F=功能，C=周期，R=RTL，B=FPGA 板上；低层通过不能声称高层通过。

本表是完整 V1 系统验收，不是当前源 IR 软件 gate 的同义词。S0 已完成其中源语义、资源/数值/周期模型和调用方相关的部分（实施报告 §3）；旧稿依赖 TRY/BUSY 重试的条目已按挂起/唤醒协议改写，WAIT_ANY/等待登记竞态的端到端部分仍在后续阶段验收。

| ID  | 场景与必须观察的结果                                                                                                                                                                | 最低验证层/定位            |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| T01 | 同 context 两条未受理 FINAL 的链后再申请第三条：compiler/verifier 静态拒绝，运行时注入则 ERROR/fault 且无副作用；若其中一链 FINAL 已受理，则挂起至其 slot_free；旧链自主 FINAL 释放 | F，quota/自锁              |
| T02 | 多 context 竞争，含八 contexts 各一槽；全局≤8，无四 context 或每 engine 两槽隐限                                                                                                    | F，动态容量                |
| T03 | RESERVED/WAIT_INPUT/FINAL_READY/DRAINING/one-shot 均计数，仅 slot_free 减一次                                                                                                       | F，lease                   |
| T04 | 同 context 两链在不同 engine 独立推进，每条链全部 step 始终绑定原 engine                                                                                                            | F+C，affinity              |
| T05 | bound engine 忙而其他空闲时不迁移，等待原因可见                                                                                                                                     | F+C，仲裁                  |
| T06 | 一个 engine 服务多个 contexts、超过两个 handles，按步公平交替                                                                                                                       | F+C，元数据边界            |
| T07 | 垃圾 AM、单/多段、尾块和初值；INIT 不读残留、内部微块仅首次覆盖、bias 一次                                                                                                          | F 数值+R                   |
| T08 | 重复/跳 seq、重复 INIT、FINAL 后提交、错误 owner 均拒绝且无 AM 副作用                                                                                                               | F，校验                    |
| T09 | 故意延迟 FIFO/RMW/writeback；compute_done 可先到，commit/output 不提前                                                                                                              | F 边界+C/R 时序            |
| T10 | 四 engine burst、多个在途微块；Q=8/O=6/R=2 不再发射，到达只 R→O                                                                                                                     | C/R，真实 credit           |
| T11 | 多 group 并行与热点；吞吐遵守独立 bank/adder 能力，等待公平可观测                                                                                                                   | C/R，服务模型              |
| T12 | 同地址 RMW、写后读、mask、控制位变化不绕过物理地址互锁                                                                                                                              | C/R，hazard                |
| T13 | 全局八槽满且新申请持续到达：新申请挂起于 FIFO、不占 UCE；旧链 mailbox/供数/FINAL/drain 仍推进，slot_free 后按预留/FIFO 授予                                                         | F+C，进展                  |
| T14 | 一链 mailbox 满返回 FULL、不增 seq，其他 context 继续推进；ACCEPTED 后改原 descriptor 不改变行为                                                                                    | F，快照                    |
| T15 | 多输入读者、最后读取延迟、释放后立即覆写；释放安全且结果不变                                                                                                                        | F 数值+C，普通 input_usage |
| T16 | 事件早到、WAIT 边界、多事件乱序；挂起与 slot_free 同周期、取消与授予竞争时无丢唤醒、无重复唤醒、无忙循环                                                                            | F+C/R，等待协议            |
| T17 | slot 复用后旧 generation 的结果/释放/完成不影响新 owner，旧终态仍可观察                                                                                                             | F+C/R，身份                |
| T18 | 不同 context 的 stride/transpose/scale/activation 交错，已接收配置不串扰                                                                                                            | F 数值+R                   |
| T19 | FINAL drain 中退出进入 FINISHING；缺 FINAL/中途故障/部分链失败安全收尾，失败 terminal 不伪装成功                                                                                    | F+C/R，退休                |
| T20 | 单 context 四独立输出最多两个 engine 同时工作；两个 context 各两链在资源充分时可覆盖四 engine                                                                                       | C/R，并发边界              |
| T21 | 授予时 slot/quota/ChainTable/mailbox 一次取得，元数据按 total_slots 静态配齐；完成事件槽在 context 绑定时已分配，授予不依赖事件退休，不存在已占槽却缺元数据的半成功                 | F，原子授予                |
| T22 | L1 drain 反压和持续 ADD；下游恢复且仲裁公平后最终释放，不需要 UCE 轮询                                                                                                              | C/R，自动释放              |

此外保留原项目交付验收：BF16/INT8 全输出 bits 对独立 oracle、方形/矩形四种转置、真实 Rocket 指令执行、compiled-only replay、软件 Conv、容器干净/离线复现。不能用 T 表的协议覆盖替代数值或真实硬件验证。

每个测试结果保存配置、输入 seed/hash、命令/事件 trace、actual/golden bytes 和实际运行层级；不以添加测试文件、打印预期输出或源码字符串断言代替通过。首次失败保存最小复现和波形，不重试到成功。

### 8.1b AM 申请协议补充验收（Q 项）

Q 项把 2026-10-27 已实现的软件协议变成硬件验收；软件证据只证明模型，硬件须在 R（RTL/联仿）和 B（板上）层复现。

| ID  | 场景与必须观察的结果                                                                                                                    | 软件证据                                                                                                                               | 硬件层 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Q01 | 不可授予时只发一次、挂起无 slot/quota/generation 副作用；slot_free 后恰好一次唤醒并交付 lease                                           | `test_am_request_parks_once_and_wakes_on_slot_free`、`test_runtime_alloc_am_blocks_once_and_wakes_on_slot_free`                        | R+B    |
| Q02 | fresh 需整个 bundle 放入未预留空槽；预留保护续链；fresh FIFO 不插队                                                                     | `test_am_bundle_reserves_continuation_and_prevents_hold_and_wait`                                                                      | R      |
| Q03 | 续链只等自身已受理 FINAL 的链 drain，不等其他 owner                                                                                     | `test_am_continuation_waits_only_for_own_draining_final`                                                                               | R      |
| Q04 | 同 logical_am 复用必须等旧链 slot_free，即使预留仍有空槽                                                                                | `test_am_same_slot_reuse_waits_for_final_release`（quota 1/2）                                                                         | R+B    |
| Q05 | fault/reset/rebind 取消的请求永不唤醒，bundle 归还                                                                                      | `test_am_cancelled_request_is_never_woken`                                                                                             | R+B    |
| Q06 | 超出 bundle 的申请是合同错误，不挂起                                                                                                    | `test_ir09_allocation_beyond_bundle_is_rejected_without_side_effect`                                                                   | R      |
| Q07 | FINAL 输出背压时 lease 停在 WAIT_OUTPUT/DRAINING，恢复后自主写回、释放、发事件                                                          | `test_ir14_final_output_backpressure_defers_release_and_event`、`test_ir14_runtime_final_output_backpressure_recovers_without_polling` | R+B    |
| Q08 | 每 Tile 独立 allocator：Tile0 满槽不影响 Tile1，fault 只影响本 Tile                                                                     | 现有身份/隔离测试基于 Group 级 manager；SW-A1 后按 Tile 拓扑更新                                                                       | F+R    |
| Q09 | `FINAL(slot0) → ALLOC(slot0) → await(旧链事件)`，中间不 reap：旧链 slot_free 后新申请被授予，旧事件随后仍可 await，不因事件未退休而挂死 | 实施报告 §4.1 的 runtime smoke（quota=1、输出背压至 3000、5451 cycles 完成）；尚无永久 runtime 回归，随 SW-A1 补齐                     | R+B    |

### 8.1c FPGA 上板验收（F 项，C11）

| ID  | 必须观察的结果                                                                                                                                                                                             |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F01 | 锁定工具链在干净目录生成 bitstream；utilization/timing 报告入 manifest，WNS ≥ 0                                                                                                                            |
| F02 | C 字节与 `bf16_fp32_mac_rne_v1` oracle 逐 bit 一致：f32/BF16 输出、方形/矩形、四种转置、尾块与特殊值向量                                                                                                   |
| F03 | request = grant = 程序执行的 ALLOC 数，park = wake + cancel，final = slot_free；结束时 live/pending/reserved 为 0；与 validator 对同一 artifact 的非时序计数一致（park/wake 次数随时序变化，只记录不比对） |
| F04 | 板级寄存器注入 L1 output 背压：lease 不提前释放；撤除后无需固件动作即完成                                                                                                                                  |
| F05 | 多阵列：同 context 两链绑定不同 engine，trace 中每链所有 step 的 engine 不变；一个 engine 同时服务超过两个 live handle                                                                                     |
| F06 | 每次授予恰好对应一次 `AM_REQ`；WAIT_AM context 只在唤醒位置位后运行                                                                                                                                        |
| F07 | 注入 fault/reset 后 generation 递增、挂起请求不唤醒，再运行不同输入结果正确                                                                                                                                |
| F08 | 多 context、多链、背压的长时间随机负载无泄漏、无挂死；记录 mcycle 与各 stall，不据此宣称 ASIC 性能                                                                                                         |

### 8.2 不变量与故障验收

- 每 context live lease 在 0..2，全局在 0..8；映射一一对应、物理区域不重叠。
- 授予成功加一次，安全 slot_free 减一次；挂起或 ERROR 不改变持有资源；one-shot 无例外。
- 每 owner 至多一个挂起请求；挂起请求不持有 slot/quota/generation；被取消的请求永不唤醒。
- `Σ_owner reserved(owner) ≤ free_slots`；fresh 授予后未预留空槽不为负，bundle 在 owner 无 lease 且无挂起请求时归还。
- AM 授予与唤醒从不等待事件 await/reap；事件槽在 context 绑定时已分配，slot 复用不覆盖旧事件。
- live handle 的 bound_engine 恒定；同 handle 最多一个 issued 且未 committed 步骤，committed 进度不超 submitted。
- 每个实际 FIFO 的 O+R≤Q；到达 R→O，安全出队才释放真实容量，不与 pending_am_updates 混同。
- slot_free 时无待处理步骤、在途 AM 访问或未完成 drain；旧事件仍有独立保留期限。
- 正常 context retire 没有未收尾 AM/输入/事件引用；错误必须先 quiesce，再发布 ERROR/CANCELLED 终态和回收。

在输入访问、阵列执行、AM RMW、输出 drain 注入故障/reset；不提前复用 region/generation，不把部分输出发布为成功。最终回收不依赖失败链的成功 chain_done，也不依赖已返回 context poll。清理后再运行不同输入，确认无残留状态污染；不要求把整个物理 AM 清零。

### 8.3 初稿阶段的既有时序 smoke（历史证据）

以下是初稿阶段执行的**现有 Python 时序模型**实测；本次文档修订没有重新运行它，也不把它当作共享 AM V1 或 Gemmini RTL 验收。命令输出位于 `/tmp`，不修改仓库场景：

```bash
conda run -n elenor-validator python -m pipeline_validator \
  --ir-file examples/workloads/matmul_2048x512x64_boa256x256x32.mlir \
  --hw-override num_dma_channels=2 \
  --hw-override hbm_fixed_latency_cycles=10 \
  --context-mode 4 --device-context-mode 4 \
  --input-binding A=0x100000:262144:r \
  --input-binding B=0x150000:65536:r \
  --input-binding C=0x200000:2097152:w \
  --max-cycles 500000 \
  --compiled-output /tmp/nexus-boa-design-20261026/matmul.compiled.json \
  --json --report /tmp/nexus-boa-design-20261026/matmul.report.json
```

观察结果：

| 项目               | 结果                                                               |
| ------------------ | ------------------------------------------------------------------ |
| 场景               | `matmul_2048x512x64`                                               |
| cycles / completed | `33129` / `true`                                                   |
| report checks      | `task_completed`、`credit_invariant`、`arena_zero_leak` 全部通过   |
| Task leases        | active/routes/retiring 均为 0                                      |
| artifact           | schema `3` / compiler ABI `v3`                                     |
| artifact hash      | `fedd44bfdd2adfbb685187e0f90adb8b5dc95dc4c7b4dca81f15b02953199396` |

该命令没有 input-data 或 Matmul 数值 oracle，当时 BOA 源码也不算矩阵；结果**不证明 BF16、动态 AM quota、affinity、自动 drain 或新事件合同，也不是新阵列性能基线**。当前软件 gate 的实测证据见[实施报告](BOA_Software_Managed_AM_IR_Implementation_Report.md) §4（含 `579 passed, 1 skipped` 全量测试与 AM 挂起/唤醒 smoke）；本版文档修订没有重新运行，也不声称 Scala/RTL/Rocket/FPGA 已实现或通过。

## 9. 风险、取舍和后续细化方向

### 9.1 硬约束与仍需 review 的细化

共享 AM V1 的链级固定 affinity、每 context 两 live slots、正常自动 drain/free，以及 2026-10-27 起的「一次申请 + 挂起 + slot_free 唤醒 + bundle 预留」和 2026-10-28 确认的 Tile 内共享 AM，已经作为本次修订依据；不能在后续实现时重新改为每 engine 配额、永久分区、跨 step 迁移、跨 Tile 共享或可重试 BUSY。

| 待 review 项  | 当前边界                                                                                                       | 冻结前证据                                                           |
| ------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| PE 数据流     | WS/OS 未定；FPGA 推导显示 tile_k=8 时两者利用率上限都约 50%（§7.4.5）                                          | 默认/矩形 shape 下的馈数、FP32 路径、数值顺序与周期                  |
| 数值细则      | profile `bf16_fp32_mac_rne_v1` 已确认；NaN 编码与 INT8 特殊值待 C1 冻结                                        | SoftFloat 交叉校验的 oracle 与 HardFloat/RTL 位级对拍                |
| 物理存储/吞吐 | 每 Tile 4 groups/8 槽；FPGA 首版 URAM 映射见 §7.4.3；bank、adder、结果 credit、drain 宽度未物理定型            | 容量、O/R 不变量、冲突与服务轨迹                                     |
| IR/ABI        | 单结果 AM allocation、四 operand/四模式和 v4；numeric profile、tile_k、arrays_per_tile 进 fingerprint 后需升级 | 线性 codec/独立验证、数值/周期、compiled-only 与调用方迁移           |
| Rocket/环境   | 一 Tile 一 hart、可配置 contexts、锁版本工具链；Vivado 只读挂载                                                | 真实固件执行、构建和 PPA；不把名称当作“微型”证明                     |
| FPGA 平台     | AMD UltraScale+，VCU118 首选（已确认）；首版 1 Tile、100 MHz 单时钟                                            | 综合/实现报告、WNS ≥ 0、§8.1c F 项；U55C 等 HBM 板需另行 review 加入 |

软件 gate 已实现；SW-A1–SW-A4 对齐以及整体硬件/FPGA 的文件级实施计划，仍需 review/授权后推进。

### 9.2 主要风险与停止条件

- **配额与独立性写错**：per-engine quota、context-wide 单链序列化、one-shot 旁路、提前释放 DRAINING 配额都违反 V1。
- **affinity 与占用混淆**：原 engine 忙时迁移、等输入时永久锁 engine、元数据隐含只能两链，都会破坏合同。
- **bundle 与挂起语义写错**：bundle 少报会造成持有并等待；把超出 bundle 的申请挂起会自锁；fresh 插队会饿死大 bundle；把唤醒做成“可重试”而不是交付已授予的 lease，会重新引入丢唤醒竞态。
- **完成边界合并**：mesh last、compute_done、step_committed、output_visible、slot_free、event reap 不能合成一个 done。
- **有界资源遗漏**：FIFO 当前占用不含未来结果，bank 数不代表 adder 数，drain 饥饿会长期占 quota；不得通过无限缓冲或零成本 AM 掩盖。
- **输入取消要求被误解**：取消 RF/load 通知不等于可以提前释放 A/B，普通 reader/producer 安全条件仍需满足。
- **数值和矩形未经验证**：FP32 partial 截断、INIT 内部微块重复覆盖、square assert 删除都不能通过放宽 oracle 掩盖。
- **双重状态和错配工具链**：Python/RTL 只有一个 AM 与控制权威；调研 Gemmini 和 Chipyard 实际 pin 不同，构建证据优先。
- **软件与硬件合同分叉**：SW-A1–SW-A4 未完成就联仿或上板，会把拓扑、affinity、数值或终态顺序差异误判为 RTL 错误；数值 profile 改动需要重算全部 typed fixture 期望。
- **FPGA 资源与时序**：HardFloat FP32 MAC 可能占 2 DSP/MAC，4 Tile × 4 阵列超出 XCVU9P；组合 MAC 限制频率，流水化会改变 skew；URAM 端口在 ADD 与 drain 间冲突。
- **工具与板级环境**：Vivado 授权/版本和 USB-JTAG 驱动属于宿主侧，必须由用户提供；不能为方便而修改宿主系统。

### 9.3 需求追踪与评审门

本表给出**设计覆盖**，不是实现/测试通过状态；R 编号与[输入规格](../../history/nexus_shared_am_v1_requirements_and_design.md)保持一致：

| 需求                                       | proposal 落点                  | 核心验收                      |
| ------------------------------------------ | ------------------------------ | ----------------------------- |
| R01 共享容量/无每 engine 两槽隐限          | §2.1–§2.3、§4.1、C4/C10、SW-A2 | T02/T06/F05                   |
| R02 每 context 两个 live lease             | §2.3、§6.1                     | T01/T03/T20                   |
| R03 链级 bound_engine 不变                 | §2.2、§3.4、§4.1               | T04/T05/F05                   |
| R04 同 context 两链无额外依赖              | §3.4、§4.2、§4.4、§5.2         | T04/T06/T14                   |
| R05 唯一 owner/同 handle 单步更新          | §2.2、§3.3、§3.5               | T08/T09                       |
| R06 授予全成或全败                         | §4.1                           | T21/Q01                       |
| R07 新申请不阻断旧链推进                   | §3.4、§4.1、§5.2               | T01/T13/T22/Q02/Q03           |
| R08 接收/计算/提交/输出/释放分离           | §3.3–§3.6、§4.3、生命周期图    | T03/T09/T22/Q07               |
| R09 不可暂停输出有接收保障                 | §3.5、C5                       | T10                           |
| R10 bank/port/adder 与 RMW 互锁            | §3.5、§6.2、§7.4.3             | T11/T12                       |
| R11 INIT/ADD/mask/精度/最终转换            | §3.1–§3.2、C1/C6、SW-A3        | T07/T08/T12/F02               |
| R12 generation/context epoch 安全          | §2.2、§4.2–§4.3、§8.2          | T08/T17/F07                   |
| R13 持久事件/原子挂起与唤醒                | §4.1、§4.3                     | T16/T17/T21/Q01/Q05/Q09       |
| R14 普通输入最后读取与全读者释放           | §2.4、§4.2–§4.3、C3            | T15；不引入专用内部 load 通知 |
| R15 STEP 接受即固定 descriptor/config      | §4.2                           | T14/T18                       |
| R16 FINAL 自主输出并安全释放               | §3.6、§4.3                     | T03/T13/T22/Q07/F04           |
| R17 slot_free 归还 quota/安全 context 退休 | §3.6、§4.3、§8.2、SW-A4        | T19                           |
| R18 原因统计与关联 trace                   | §6.2、§7.4.6、§8.1             | T05/T11/T20 及全 T 表         |

软件管理 AM、A/B/out/acc、accumulate output 和 f32 直出继续由独立计划追踪；源级 CFG 仍暂缓，旧稿的下层 TRY/BUSY 重试已由挂起/唤醒协议取代。此次 gate 不改变 R01–R18 的系统安全目标。

原需求中仍适用的 Scala/Gemmini、单阵列后多阵列、矩形、BF16/INT8 输出、硬件转置、Rocket 自定义指令、软件 im2col、monorepo/Docker、组件输入输出/guide 和图示，分别落在 §1/§3/§4.5/§7/§8；Tile 内共享 AM、逐 MAC FP32 RNE profile 与 AMD UltraScale+ 平台分别落在 §2.1、§3.2、§7.4。专用 RF 与内部 load 通知要求已取消；同 context 单链和 K-step 迁移设计不再适用。

实现报告必须区分当前覆盖、明确延期、功能/周期/RTL/FPGA/PPA 的实际状态；软件结果不能代替 RTL 或板上的 Q/F 项。

### 9.4 上游证据链接

GitHub 链接固定到本轮核对的 commit，避免 master 漂移；Chipyard/AMD 文档页是当前在线版本，与 pinned 源码冲突时以源码为准。当前仓库事实以 §1.3 所列本地源码为准。

[up-chipyard-hash]: https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/CHIPYARD.hash
[up-parent-gemmini]: https://api.github.com/repos/ucb-bar/chipyard/contents/generators/gemmini?ref=e02074414158c85059c1dcb231ff2f8941254b9a
[up-parent-rocket]: https://api.github.com/repos/ucb-bar/chipyard/contents/generators/rocket-chip?ref=e02074414158c85059c1dcb231ff2f8941254b9a
[up-chipyard-build]: https://github.com/ucb-bar/chipyard/blob/e02074414158c85059c1dcb231ff2f8941254b9a/build.sbt
[up-gemmini-build]: https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/build.sbt
[up-bf16]: https://github.com/ucb-bar/gemmini/blob/25809f78323a729ef76fb68f3cedd8a24da2942b/src/main/scala/gemmini/ConfigsFP.scala#L104-L118
[up-pe]: https://github.com/ucb-bar/gemmini/blob/25809f78323a729ef76fb68f3cedd8a24da2942b/src/main/scala/gemmini/PE.scala#L17-L70
[up-mesh-delay]: https://github.com/ucb-bar/gemmini/blob/25809f78323a729ef76fb68f3cedd8a24da2942b/src/main/scala/gemmini/MeshWithDelays.scala#L34-L75
[up-accumulator]: https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/AccumulatorMem.scala
[up-rocc]: https://github.com/chipsalliance/rocket-chip/blob/c7aabd2823d3d001a7a9e8a982bb65d6cb7c9bdf/src/main/scala/tile/LazyRoCC.scala#L24-L83
[up-arith]: https://github.com/ucb-bar/gemmini/blob/25809f78323a729ef76fb68f3cedd8a24da2942b/src/main/scala/gemmini/Arithmetic.scala#L355-L390
[up-vcu118]: https://github.com/ucb-bar/chipyard/blob/e02074414158c85059c1dcb231ff2f8941254b9a/fpga/src/main/scala/vcu118/Configs.scala#L43-L65
[up-chipyard-fpga]: https://chipyard.readthedocs.io/en/stable/prototyping/general/
[up-ds890]: https://docs.amd.com/v/u/en-US/ds890-ultrascale-overview
[up-vcu118-board]: https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/vcu118.html
