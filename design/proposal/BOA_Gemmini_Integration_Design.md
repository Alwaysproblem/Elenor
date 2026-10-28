# Scala BOA / Gemmini BF16 Matmul 与 pipeline validator 集成设计

- 状态：**总体设计评审稿 v0.4；硬件实现尚未开始；软件前置 gate（源语义/typed executable/AM 资源与数值模型/调用方迁移）已按独立计划实现，见 [实施报告](BOA_Software_Managed_AM_IR_Implementation_Report.md)。源 IR 仍不增加分支/重试控制。**
- 需求优先级：用户最新修改 > [共享 AM V1 需求规格](../../nexus_shared_am_v1_requirements_and_design.md) > [prompt.md](../../prompt.md) 与此前确认中仍适用的内容。
- 阅读顺序：§1 范围 → §4.4 源 IR → [软件管理 AM/IR 独立计划](BOA_Software_Managed_AM_IR_Plan.md) → §3/§5 硬件与系统合同 → §8 验收。
- 本轮同步两份文档，保留软件计划 §3.4 并标记本次不实现；既有图示和共享 AM/自动释放合同不变，不修改实现代码或系统环境。
- 软件计划仍先于硬件实现，但其本次 gate 不含源级 CFG、TRY 状态分支或具体低层重试 codegen；这些留给更低层 IR／执行协议后续设计。整体硬件计划仍需另行 review/授权。

## 1. 定位、目标和 First Silicon cutline

### 1.1 已经由用户确认的选择

| 议题             | 确认结果                                                                                                                                             |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 实现语言         | BOA、Gemmini 适配和 UCE 接口使用 Scala/Chisel，生成 RTL；不是 Scala 数值软件代替硬件                                                                 |
| Gemmini 复用边界 | 复用阵列核心、PE 和浮点运算；BOA 自主管理共享 AM、链表、有限队列和仲裁，不套用完整 Gemmini 控制器                                                    |
| BOA 命名         | 沿用 BOA 历史名称；本设计中指 Gemmini 脉动阵列组，不沿用旧文档对 outer-product primitive 的实现限定                                                  |
| 初始规模         | 先一个阵列；默认物理阵列 16×16，逻辑计算块 M×N×K = 16×16×8                                                                                           |
| 形状演进         | 先方形单阵列，再实现并验证矩形；矩形是必达能力，不是默默删除的需求                                                                                   |
| context          | 默认验证配置仍为 4 个、可配置；每 context 最多两个 live AM lease，跨所有 engine 统计；同 context 两条独立输出链可并行                                |
| engine affinity  | 每条输出链在 TRY_OPEN 成功时固定一个 engine，直到 slot_free 不迁移；不是把整个 context 永久绑定到一个 engine                                         |
| 累加存储         | 多阵列共享一个逻辑 AM 池；V1 多阵列基线为 4 engines、4 AM groups、每组 2 slots，共 8 个动态物理槽                                                    |
| 输入边界         | 取消“必须配置 BOA RF 并向 UCE 报告内部 load 完成”的要求；只保留普通 producer/consumer 输入生命周期                                                   |
| 软件管理 AM      | `tile.alloc memory_space="am"` 返回一个 `!tile.am_buffer`，Matmul 返回一个完成事件；源 IR 不处理 BUSY 分支，低层资源协议后续展开；最终写回后安全归还 |
| 数值             | BF16 A/B 输入、FP32 partial 和 AM、RNE；默认 BF16 输出，支持 f32 原精度直出及可选 INT8 最终转换，不增加 FP32/INT8 输入 Matmul                        |
| 后处理           | 简单 bias、scale、ReLU/clip；另含 INT8 输出量化，带 scale、zero-point、RNE、饱和；不增加 INT8 输入 Matmul                                            |
| 转置             | 支持硬件 A/B transpose；需计入装载延迟、缓冲和资源成本                                                                                               |
| 验证顺序         | 先完成独立计划中的 IR、编译/独立验证、软件数值/周期与调用方迁移 gate，再实现单阵列 BOA RTL、联仿、真实 UCE 和多阵列                                  |
| 真实 UCE         | Rocket + RoCC；保持 Scala/Chisel 生态，不选择 PicoRV32                                                                                               |
| 环境与仓库       | monorepo，Gemmini 与 BOA 分目录；Docker/dev container；不在宿主机执行 apt install 或修改系统配置                                                     |

“微型 UCE”是控制职责目标，**Rocket 并不自动满足微控制器级面积/功耗目标**。首个真实 UCE 用裁剪配置验证功能；最终微型化需有综合证据，不能用名字代替 PPA 结论。

### 1.2 必达边界与非目标

完整交付必须同时具备：软件显式 AM allocation、带 A/B/L1 output/AM acc 的 Matmul、accumulate 与 accumulate output 模式、BF16 输入与 f32/BF16/INT8 最终输出、动态两槽 quota、链级固定 affinity、同 context 两链独立推进、普通输入安全释放、硬件转置、矩形、多阵列共享 AM、Rocket 自定义指令、validator 联仿及容器复现。

**本次范围内的软件/IR gate 仍是硬件实现前置条件。** gate 覆盖 AM/Matmul 源语义、资源和数值/周期模型，不把源级 CFG 或具体分支/重试展开作为前置。之后按「单阵列实现/正确性 → 联仿 → 真实 UCE → 多阵列」推进；更低层控制流与动态恢复另行设计和验收。4-context 仍只是默认配置，不是全局上限。

不纳入本轮：K-step 跨 engine 迁移、同 C 的多 engine split-K、AM 抢占/spill/eviction/搬迁、任意地址 atomic add、驻留 context 动态切换 memory profile、不计配额的旁路 AM、通用 Tile Program DAG 执行器、EVU 长期直接持有 AM 引用，以及 FP32/INT8 输入 Matmul、硬件 Conv/im2col、复杂非线性、物理宏冻结和 PPA 保证。Conv 继续采用**软件 im2col + BF16 Matmul**；已有无关功能不因本规格而删除。

输入仍必须在最后读取及全部读者结束后才允许复用，但不增加专门的 BOA load 进度事件、软件可见 RF slot 或强制独立 RF 模块。必要的馈数/转置寄存器是内部实现细节，不恢复已取消的要求。

### 1.3 当前仓库事实，不等同于拟议硬件

| 事实                                         | 当前源码证据                                                                                                                                  | 设计影响                                                                |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| BOA 只按 ops/峰值计算延迟                    | [engines.py](../../pipeline_validator/engines.py)，`BOAEngine.latency`，202–221 行                                                            | 没有真实矩阵计算，不能换个类名就称为 Gemmini backend                    |
| BOA descriptor 没有 A/B/C operand            | [lowering.py](../../pipeline_validator/compiler/lowering.py)，845–854 行                                                                      | 必须增加 typed operand、dtype、K-chain、AM 与阶段事件契约               |
| `accumulate` 被保留但不驱动 AM 运算          | 同上与 `BOAEngine`                                                                                                                            | 现有两条 accumulate IR 不是持久 AM 的证明                               |
| artifact 严格要求 schema=3 / compiler ABI=v3 | [compiled_program.py](../../pipeline_validator/compiled_program.py)，504–516、893–895、1081–1091 行                                           | 新 typed descriptor 需要协调 codec、verifier、compiler ABI 和产物重生成 |
| UCE 是 Python Tile Program 解释器            | [tile.py](../../pipeline_validator/tile.py)，`TileUCE`、1011–1196、1687–1696 行                                                               | 没有真实 RISC-V 指令执行；RoCC 是新增验证层                             |
| ByteStore 已有真实字节接口                   | [byte_store.py](../../pipeline_validator/memory/byte_store.py)，281–342 行                                                                    | 可作为外部内存字节权威，但它本身不提供 SRAM 时序                        |
| 硬件默认是 8 个 OPA 参数，context 默认 1     | [hardware_config.yaml](../../pipeline_validator/hardware_config.yaml)，81–94 行；[config.py](../../pipeline_validator/config.py)，`SimConfig` | 新场景显式指定阵列数 1、context 4，不能把旧默认当成新硬件               |

旧 [BOA 设计](../elenor_boa/ELENOR_BOA_Design.md) 只参考 ownership、事件身份和故障边界；其 dtype、OPA、reduce tree 等内容不覆盖本需求。

共享 AM 概念到实际仓库对象的映射如下；右侧是**需要新增的设计语义**，不是现有实现声明：

| V1 概念              | 当前真实对象/入口                                                                                                                                                                    | 当前语义与缺口                                                                                                               | 改造边界/兼容性影响                                                                                         |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| tile context / epoch | [tile.py](../../pipeline_validator/tile.py) `_UCEContext`、`TileUCE.bind_context`；[execution_ir.py](../../pipeline_validator/execution_ir.py) `TaskIdentity`                        | 已有 ctx_id 与 TaskIdentity；没有显式 AM context epoch 或 quota                                                              | 为每次 context 绑定建立等价的防复用身份；quota 挂在此 owner，不能用 device slot 代替                        |
| GEMM / K-chain       | [lowering.py](../../pipeline_validator/compiler/lowering.py) `_lower_engine_descriptor(TileBoaOp)`                                                                                   | 只生成 m/n/k/ops/accumulate，无 operand、handle、seq 或 FINAL                                                                | 增加 typed OPEN/STEP 及两逻辑位置的非阻塞控制；同步 compiler/codec/verifier                                 |
| engine 选择          | [tile.py](../../pipeline_validator/tile.py) `_engine_for_queue`；[engines.py](../../pipeline_validator/engines.py) `BOAEngine`                                                       | 指向 Tile 的单个时序 BOA 实例；OPA 数量只影响峰值估算                                                                        | 新共享层维护多个执行资源及不可变 bound_engine，不能仅修改峰值参数                                           |
| AM / 部分结果        | `BOAEngine.latency` 与现有 `ExecEngineDesc`                                                                                                                                          | 没有数值 AM、动态 lease 或 RMW 服务                                                                                          | 新功能模型和周期模型实现所有权、物理区域、INIT/ADD、commit 和 drain                                         |
| event / wait         | [tile.py](../../pipeline_validator/tile.py) `_UCEEventRef`、`events_done`、`_probe_wait_instruction`；[event_table.py](../../pipeline_validator/runtime/event_table.py) `EventTable` | Tile-local runtime id 是 `ctx{ctx_id}:{local_name}`；Group EventTable 已有 generation、consumer_refs、future_uses 和安全回收 | 复用事件管理机制，但补齐链事件身份、持久记录、credit 快照、WAIT_ANY 等价语义；不能宣称 Tile-local 已满足 V1 |
| 输入与输出可见性     | [execution_verifier.py](../../pipeline_validator/execution_verifier.py) launch/WAIT/FREE_L1 检查；[tile_group.py](../../pipeline_validator/tile_group.py) `_on_phase_signal`         | BOA/EVU 目前是 opaque event；Group phase 按全部 Task 聚合                                                                    | typed 输入按真实读者结束，L1 output_visible 与 L2 output_ready 分离，不增专用内部 load 通知                 |
| admission / 退休     | `TileGroup._step_task_admission`、`_step_tiles`；`TileUCE._complete_context`                                                                                                         | 已有 Task lease、memory profile 与 L1 Arena 退休；无 AM FINISHING 收尾                                                       | 保留 memory admission，增加已提交 FINAL 自动收尾和缺 FINAL 错误；Task lease 不是 AM lease                   |

`EventTable.wait` 当前读取终态，不是本稿所有资源域的原子 check-and-register 硬件实现。源码复用和硬件等待协议仍需分别验证。

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

## 2. 职责、非职责和 ownership

### 2.1 硬件划分

![BOA 硬件组件和共享 AM 布局](../image/boa_gemmini_hardware.svg)

本稿的一个 matrix engine 对应一份 Gemmini 阵列执行资源。V1 的 4 engines 经带身份的结果路由访问 4 AM groups；每组两个固定物理区域，由**同一个动态分配与所有权域**管理。group 不对应 engine，也不对应 context。

| 组件                           | 拥有的职责/状态                                                         | 不拥有                                                           |
| ------------------------------ | ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Compiler                       | 逻辑 AM 位置 0/1、有限 K-chain、输入依赖、INIT/ADD/FINAL、两槽软件流水  | 物理 bank、固定 engine 编号、轮询共享 SRAM                       |
| Group / validator runtime      | TaskIdentity、context admission、memory profile、L1/L2 引用、phase 聚合 | 把 Task lease 计作 AM lease，或在 admission 时永久预留两个 AM 槽 |
| UCE / Rocket firmware          | Tile Program PC/寄存器、TRY 提交、WAIT/context switch、已有链续步       | AM bank 仲裁、内部 load 进度、用轮询触发 slot_free               |
| SharedAmAllocator / ChainTable | ctx quota、动态区域、generation、不可变 affinity、链状态                | 可迁移的 engine 选择、每 engine 两槽私有限额                     |
| StepMailbox / ReadyArbiter     | 每链当前记录和一个 next-step mailbox、在 bound_engine 上 ready 仲裁     | 未提交的未来步骤、跨链无依据的程序顺序依赖                       |
| Operand / Transpose Adapter    | 声明 view 的合法读取、四种 A/B transpose、有效 lane 馈数                | 软件可见 RF 分配和专用内部 load 通知                             |
| ResultCapture / AmRouter       | 微块结果 credit、在途身份、合法 group 路由                              | 以 compute_done 冒充 step_committed                              |
| AM group / RMW                 | SRAM 端口、bank、局部 adder、物理地址互锁和 writeback                   | 通用软件 atomic、隐含无限吞吐                                    |
| Epilogue / Drain               | FINAL 后处理、自动输出到已准入 L1、可见性和安全释放                     | 等 UCE 发 FREE、等事件被 poll/reap 才归还 AM                     |
| EventTracker                   | 持久完成/失败记录、等待者、消费引用和安全退休                           | 将事件记录生命周期绑死到物理 AM slot                             |

这些是逻辑职责，可复用或组合已有模块；不要求为名字一致而复制第二套 scheduler/event 系统。

### 2.2 身份分层与链级 affinity

软件语义为 `ctx_key=(tile_context_id, context_epoch)`、`logical_am∈{0,1}`、`acc_handle=opaque(entry_id,generation)`。ctx_key 的作用域为 Physical Tile；跨 Tile 消息还须包含 tile identity。`TaskIdentity` 继续标识原有 Grid/Task，不与 ctx_key、物理 slot 或 engine 混同。

硬件 handle 表至少记录：owner_ctx_key、logical_am、output_chain_id、physical_am_group/region、bound_engine、shape/acc_format/layout、submitted/issued/committed 进度及生命周期。event token 还须区分 context、entry、generation、步骤和事件种类，不能只携带裸 slot 或 generation 数值。

- affinity 在 TRY_OPEN 成功的线性化点固定，直到 slot_free 不变。原 engine 忙时等待它，不能转给空闲 engine。
- affinity 不是 engine 的整链独占预约；某链等输入/下一 descriptor 时，同一 engine 可服务其他绑定到它的链。
- 同 context 两条链可绑定不同 engine，独立 ready；一个 engine 可服务超过两个全局 handle，不能继承隐含的两条本地 AM 元数据上限。
- context switch 不释放 AM、不改变 owner/affinity；只在所有命令和引用安全收尾后复用 context epoch。

ID/寄存器/事件位宽和二进制 ABI **由后续规格冻结**。必须有有限回绕和 quiescence 规则；较大的 generation 不能代替旧访问收尾。

### 2.3 动态两槽配额

令 L(ctx) 为某 context 尚未安全释放的 AM lease 集合：

```text
used_am_slots(ctx) = |L(ctx)|，0 <= used_am_slots(ctx) <= 2
sum_ctx used_am_slots(ctx) = global_live_leases <= 8
```

配额跨该 context 使用的所有 engines 统计。RESERVED、WAIT_STEP/WAIT_INPUT、RUNNING、COMMITTING、FINAL_READY、DRAINING、FAULT_QUIESCING 均计一个槽；one-shot 也使用同一池、计一个短生命周期槽。只有 slot_free 才减一，成功 OPEN 加一，BUSY/ERROR 申请不留下 lease 或半初始化资源。

两个逻辑位置不是永久物理预留。4 contexts 各占两槽，或在 UCE/memory 容量允许时 8 contexts 各占一槽，都可以填满八槽。一个 context 已占两槽，即使全局还有六槽，也必须拒绝其第三次申请；释放后可以获得不同物理区域和新的 engine affinity。

完成记录在 AM 已释放后仍存活，不再计 AM quota。软件未 reap 时可以返回 `event_record_full`/逻辑位置记录未退休，但不能谎报该旧 AM lease 仍存在。

### 2.4 内存与输入生命周期

Tile L1 SPM、Group L2 和阵列组内部共享 AM 是不同存储域。每条链在 OPEN 时校验并关联已准入的 L1 输出区域；不因等 AM 的新任务预取而挤占旧链的输出空间。一个驻留 context 保持一个 memory profile。

本轮**不规定独立 BOA RF、RF 容量/槽位或向 UCE 通知内部装载完成**。仍然必须满足 R14：指定层级的输入缓冲只有在最后实际读取、所有声明读者和在途访问结束后才能复用；STEP 接受仅固定 descriptor，不代表 A/B 已读取。

普通 input_usage/ref tracking 可保守地在已证明没有后续读的步骤完成边界释放，不必追求内部装载后的提前释放，也不能仅按 MAC done 推断。多个读者共享输入时必须聚合全部读者。L1 `tile.free` 使 view 失效，不自动释放整个 Task Arena 的物理预留；Group `input_released` 仍按原 L2 输入和 Task 聚合合同处理。

## 3. 微架构和状态机

### 3.1 固定 tile、阵列数据流和转置

配置仍分开表达 `ArrayRows/ArrayCols` 与 `TileM/TileN/TileK`；默认物理阵列 16×16、逻辑块 16×16×8。先方形，再验证 8×16、16×8 等矩形，合法形状由实际 adapter 和分块规则校验。

**不冻结 WS，也不未经 review 改为 OS。** 新规范固定的是共享 AM 和链级 affinity，不规定 PE 数据流；此前以 K-step 跨阵列迁移支持 WS 的前提已取消。两种内部数据流都必须把本步骤 partial 交给共享 AM，只有实际提交后才能发布 step_committed。不能用 OS 将跨步骤状态一直留在 PE、却把尚未更新 AM 的步骤报告为已提交。

数据流选择需比较默认 shape、馈数、transpose、填充/排空和 FP32 路径；实施前冻结一种已验证映射及其数值顺序，不强制同时实现两种。WS 的 K/N 空间映射约束不能当作所有 dataflow 的通用形状限制。

硬件 A/B transpose 继续在阵列输入适配层实现：A 正常读 `[m,k]`、转置读 `[k,m]`；B 正常读 `[k,n]`、转置读 `[n,k]`。支持四种组合，并验证 view/stride/mask。所需流水寄存器或重排缓冲按实际接口建模，不形成新的 UCE 可见 RF 管理协议。

边界块携带 valid_m/n/k；无效位置不能越界读写，无效 MAC 不得因 `0×Inf` 或额外舍入污染有效结果。零维/零 K 由 compiler 明确 lower 为合法空工作或数学定义的输出路径，不向硬件发射永远等不到 last beat 的工作。

### 3.2 数值契约 v0（保留精度，细则待 review）

保留已确认的 BF16 A/B、FP32 partial 和 AM、RNE，输出支持默认 BF16、f32 原精度直出及可选 INT8。推荐数值顺序仍是每个物理贡献块内按逻辑 k 递增 FP32 FMA，然后共享 AM 内 FP32 add；golden oracle 必须包含物理分块及每次舍入位置。固定 seq 不等于整条 K 连续 FMA 或任意 BLAS 归约树 bitwise 相等。

- seq=0 为 INIT，后续为 ADD；one-shot 为 INIT|FINAL。
- INIT 必须覆盖全部有效输出，不能读上一 owner 的残留值。若一个逻辑 INIT step 内有多个贡献同一输出位置的微块，只有该位置第一次更新是覆盖，后续微块必须 ADD。
- shape/layout/acc_format 在同一链中不变；部分 mask、未初始化位置、同地址写后读及 FINAL read 均进入 hazard 检查。
- bias/初始 C 仅按数学合同应用一次；中间结果不缩位、不激活、不量化；FINAL 提交后才执行 epilogue。
- 具体微块顺序进入 immutable chain/step config 和 oracle，不允许实现模型为了匹配输出擅自改舍入或容差。

特殊值仍是待 review 的建议：不采用 FTZ、RNE、overflow→Inf、NaN 规范化为 quiet NaN，覆盖 signed zero/subnormal/tie/invalid；未批准前不能声称已经冻结实现。

后处理建议顺序为 `FP32 AM → 每列 bias → scalar scale → ReLU/clip → 输出转换`，各阶段可关闭。f32 保持宽度写出，BF16 最终转换一次；INT8 为 `saturate[-128,127](RNE(y * inv_scale) + zero_point)`，inv_scale 有限且为正，zero_point 在 INT8 范围内。NaN→zero_point、±Inf→饱和端点仍待 review；不引入 FP32/INT8 输入 Matmul 或默认 per-channel quant。

软件 Conv 继续先 im2col 再 Matmul；展平布局、padding/stride/dilation 和额外流量必须显式验证，不实例化硬件 LoopConv。

### 3.3 链状态机与三个生命周期

```text
FREE --TRY_OPEN/GRANTED--> RESERVED --> WAIT_STEP
WAIT_STEP --STEP accepted--> WAIT_READY --ready/issue--> RUNNING
RUNNING --全部结果已接住--> COMMITTING
COMMITTING --非FINAL且全部提交--> WAIT_STEP / WAIT_READY
COMMITTING --FINAL且全部提交--> FINAL_READY --> DRAINING
DRAINING --output_visible且旧访问结束--> slot_free --> FREE

任一live状态 --终止性故障--> FAULT_QUIESCING
FAULT_QUIESCING --旧访问安全结束且资源回收--> ERROR/CANCELLED terminal
```

每链一个当前执行记录和一个 next-step mailbox；RUNNING/COMMITTING 时可接收下一步，但同 handle 同时最多一个 issued 且未 committed 的步骤。mailbox 已就绪时，前步 commit 后可以直接转 WAIT_READY，不必人为插空。

必须区分：

1. AM lease 从 OPEN 成功持续到安全 slot_free，包括等待输入和 FINAL drain。
2. engine 只在步骤实际执行和必要交接时占用。V1 保守在 step_committed 且配置切换安全后释放该步骤占用；不等整条链 drain，也不因链等输入而锁住 engine。
3. event record 从 OPEN 预留到 reap 或按协议退休，可以跨越物理 AM 的再次分配。

### 3.4 Ready 仲裁、有限队列和进展路径

STEP 的发射条件为：handle/owner/seq 合法、前步 committed、输入 ready、**bound_engine** 可接收、结果/AM 服务保障已取得、配置切换处于安全边界。调度器只处理已提交工作；同 context 两条链不因共享 context 自动建立依赖。

新 OPEN 的队列与已有链的 continuation mailbox/提交路径隔离或保留不可挤占额度。OPEN 资源不足必须返回 BUSY，不能占住 UCE 或挡住旧链 STEP/FINAL/供数/drain。mailbox 满同样返回 BUSY，submitted 游标不增加。

目标水位只适用于已有合法工作和可用 credit；不复制工作、不以 NOP 填满、不假设 UCE 尚未提交的 FINAL 已存在。各 command/mailbox/result/event 容量独立建模，不用一个 pipeline_depth 替代所有资源。

仲裁在 ready 候选内公平轮转/按年龄选择，优先保障可释放状态的 drain 与已有链；不能空等未 ready 的高优先级候选。多个 group/端口允许的重叠需保留，不以全局串行化掩盖竞争。

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

```text
step_committed =
    end_of_step_seen
    AND all_expected_results_received
    AND pending_am_updates == 0
```

pending_am_updates 覆盖 result capture、route、queue、RMW 和待写回，而非仅 SRAM 内部。它与 FIFO 的 O/R credit 含义不同；在阵列还会产出结果时，计数暂为零不能触发 commit。

group 服务资格可按微块获取，不为等待输入的整条链永久锁 group；多资源同时获取需全成或全败。drain 必须公平取得读端口，不能被持续 ADD 永久饿死。

### 3.6 FINAL 自动释放与 double issue

FINAL 步骤 commit 后，设备自动 epilogue/drain 到 OPEN 时确定的已准入 L1 目的区域。输出可见、没有待处理步骤、所有旧 AM/输出访问 quiescent 后，记录终态、slot_free、quota--；chain_done 的成功条件必须包含这些动作。**正常释放不需要软件 FREE，不等待 UCE poll/ACK/reap。**

V1 每 engine 最多一个执行中的步骤，保留 Gemmini 正确性所需的 propagate/c1/c2/tag pipeline；不能盲目将 n_simultaneous_matmuls 改成 1 或删 PE 状态。多 engine 并行、L1 ping-pong 和已接收下一步的 mailbox 均不等于单 engine double issue。

context 返回时，已提交 FINAL 尚在 drain 则进入 FINISHING；没有 FINAL 的活链是协议错误。终止性错误停止该 context 新提交，对其余已拥有链和访问受控收尾，再报告失败终态并退休；不等待失败链永远不会发生的成功 chain_done，不按 timeout 直接清 quota。若只能受控 reset 恢复，应显式暴露该限制。

## 4. 接口、descriptor、寄存器和协议

### 4.1 TRY_OPEN：非阻塞、全成或全败

以下是共享 AM V1 的**下层协议合同**，不是源 MLIR 的返回值或分支语法；具体更低层 IR 的状态编码、分支与重试展开后续考虑，不纳入本次源级改造。

```text
TRY_OPEN(ctx_key, logical_am, chain_id, output_shape, acc_format,
         layout, output_destination, chain_config)
  -> GRANTED(handle, bound_engine, completion_token)
   | BUSY(reason, retry_event_snapshot)
   | ERROR(reason)
```

成功线性化点必须同时完成：检查 ctx/逻辑位置及 used<2，选择全局空闲 slot，取得 ChainTable、一个 next-step mailbox、最终完成/失败记录和必要元数据，绑定已准入 L1 输出目的地，设置 owner/generation/不可变 bound_engine，置 RESERVED 并 quota++。不能先返回 handle 再等待 AM 或完成记录；不要求在 OPEN 时物理清零，但有效数据 INIT 前禁止读取。

源 IR 的 `tile.alloc memory_space="am"` 只返回一个 AM 结果，logical_slot 为 0/1，显式绑定 L1 output；不返回 TRY 状态或 retry token。下层保证真正 GRANTED 前依赖访问不能发射，不能把入队/BUSY 当成功。如何展开具体重试后续决定，不根据 f32 或变量名隐式改变普通 L1 allocation。

engine 选择可按兼容性和已有绑定工作量、AM group 按空闲和争用估计；可以绑定暂时忙的 engine，这不占用其执行时段。软件不指定物理 bank 或固定 engine 编号。

BUSY 需要区分 context quota、逻辑位置占用、global AM full、metadata/event record 不足；不占 AM、不增 quota、不残留半初始化 handle。传输入队不是 GRANTED；受理有限接口延迟后必须返回，不能因缺 AM 长期挂住 UCE。带 request identity 的等价 exactly-once 机制防止成功 OPEN 被互连重放重复分配。

**失败判断和 retry_event_snapshot 必须在同一资源线性化点获取**，或使用等价的资源可用电平等待；否则“判失败之后释放、读取 snapshot 之前”的变化会被遗漏。credit 唤醒只允许重试，不保证已为请求预留资源。相同规则适用于 mailbox、event record 和 MFE credit。

### 4.2 TRY_SUBMIT_STEP 与 immutable descriptor

```text
TRY_SUBMIT_STEP(handle, seq, A_descriptor, B_descriptor,
                input_ready_tokens, input_usage_handles,
                k_extent, INIT_or_ADD, FINAL)
  -> ACCEPTED(step_token)
   | BUSY(mailbox_credit_event)
   | ERROR(reason)
```

接收前验证 owner/context_epoch/handle generation、bound_engine、shape/布局/范围/mask 和 seq。seq 从 0 连续，拒绝重复、跳号、重复 INIT 和 FINAL 后提交；seq=0 为 INIT，后续 ADD，最后一步带 FINAL；one-shot 为 INIT|FINAL，不能绕过配额。

ACCEPTED 说明 descriptor 和所有影响执行的配置已快照或完整转存到受保护存储，软件可改写原 descriptor 本体；**不说明 A/B 数据可以释放**。FINAL 被接受即关闭此 handle 新步骤入口。submitted、issued、committed 游标分开维护，仅 ACCEPTED 增加提交游标。

允许接受尚未 input-ready 的步骤，但 token 必须对应已经 ready 的输入或已由 MFE/producer 受理的生产操作。不能提交等待任意未来软件信号的步骤，再把其生产者放到可能阻塞的新 OPEN 之后。mailbox 满返回 BUSY，让调用方继续服务另一条链。

源 IR 的 `tile.boa.async "matmul"` 必须显式携带 `ins(A,B)`、`outs(L1 output)`、`acc(AM)` 和 seq。lowering 校验 outs 与 AM allocation 绑定一致后，映射为 INIT/ADD 和 FINAL；硬件协议不必每步重复传完整目的地。`accumulate output` 包含本次乘法，再触发最终写回和自动释放，不是单纯拷贝旧 accumulator。

逻辑 descriptor v0 至少包含：

| 字段组     | 字段与 owner                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------- |
| 版本/身份  | descriptor_version、size、patch_mask、request identity、ctx_key、handle/generation、chain_id、seq |
| 不变链配置 | output shape、acc_format、layout、已准入 output_destination、epilogue 参数；OPEN 固定             |
| 输入       | A/B typed view、allocation/generation、offset/extent/stride、dtype/权限、transpose、有效 mask     |
| 步骤       | k_extent、INIT/ADD、FINAL、input_ready_tokens、input_usage_handles                                |
| 内部状态   | bound_engine、physical group/region、结果 credit；由设备维护，不由 compiler 猜测                  |

运行时 patch 仅允许 patch_mask 声明的字段，且发生在 ACCEPTED 前；成功接受后不能再受软件配置寄存器改写影响。参数非法在可观察 AM 副作用前拒绝；中途故障则走安全收尾，不承诺回滚已写出的部分输出。

### 4.3 事件、等待与释放合同

| 事件/状态                            | 严格定义                                               | 不能据此推导            |
| ------------------------------------ | ------------------------------------------------------ | ----------------------- |
| step_accepted                        | descriptor 已安全快照                                  | 输入已读、计算已开始    |
| compute_done                         | 所有本步结果已产生并被保护路径接住                     | AM 已提交               |
| step_committed                       | 本步全部 AM 更新完成且可供下一步累加                   | C 已到 L1               |
| input_release                        | 指定层级缓冲的最后读取和全部读者已结束                 | C commit 或 slot_free   |
| output_visible                       | FINAL 结果在约定 L1 目的区域可见                       | 旧访问/事件引用全部退休 |
| slot_free                            | 本代区域可重分配，quota 已归还                         | 旧事件记录可覆盖        |
| chain_done / chain_terminal(SUCCESS) | FINAL 输出可见、旧访问结束、slot_free/quota-- 都已完成 | UCE 已经读取该记录      |
| chain_terminal(ERROR/CANCELLED)      | 失败访问受控结束、slot/配额安全回收后记录失败          | 成功 output_visible     |

硬件 mem_req/mem_resp 继续携带 req_id、owner/generation、解析地址、访问类型、byte mask、data/status；合法请求参与实际 L1 仲裁，不能以接收命令代替数据传输。

WAIT/WAIT_ANY 复用现有事件体系的等价能力，必须原子地检查终态并登记等待者。事件先到时立即满足；已处理的粘性事件从 wait_set 移除，不能形成忙循环。事件身份同时区分 context、entry、generation、seq、kind。

建议每个 context 的两个逻辑 AM 位置各有一个 chain-event record：OPEN 成功取得记录，保存身份/进度/终态；AM 可以先释放，记录继续保留至 reap。未 reap 的逻辑位置不能覆盖旧记录，但不得因此推迟物理 slot_free。若复用 EventTable 的引用管理，同样需要证明终态保留、有限容量和多代复用。

FINISHING 收尾不依赖已经返回的原 context 再运行并消费事件。必要通知和外部引用完成后可按合同自动退休；有外部等待者时仍需保留可观察终态。context ID/epoch 只有在命令、输入引用、AM 访问和事件引用都安全处理后才能复用，不使用后台 detach。

### 4.4 软件管理 AM 与源 IR 演进

本次源语义与任务见 [BOA_Software_Managed_AM_IR_Plan.md](BOA_Software_Managed_AM_IR_Plan.md)。该计划 §3.4“受限控制流，不增加通用 DAG 执行器”**保留，但本次不实现，也不是软件 gate 的依赖**。下述为拟实现语法，当前 parser 尚不支持。

```mlir
%out = tile.alloc memory_space = "l1" shape = [64, 256] dtype = "f32"
    alignment = 256 : !tile.l1_buffer<64x256xf32>
%acc = tile.alloc memory_space = "am" logical_slot = 0 output(%out)
    shape = [64, 256] dtype = "f32" alignment = 256
    : !tile.am_buffer<64x256xf32>
```

allocation 表达一个逻辑 AM 申请，只返回 `%acc`。普通 L1 allocation 同样保持一结果，省略 memory_space 仍是 L1；不能把 EVU 的同名 f32 `%acc` 改成 AM。资源不足由下层协议处理，源程序不增加 SUCCESS/BUSY/ERROR 判断或重试分支，也不因此假设申请永远成功。

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

自定义指令语义包括非阻塞 OPEN、非阻塞 STEP、事件检查/WAIT_ANY 等价操作、record reap/status。具体 opcode/funct/位宽**由后续规格冻结**；由一份权威 schema 生成 Scala 常量、C header 和验证 codec，不手写三套编号。

RoCC resp 返回 GRANTED/ACCEPTED/BUSY/ERROR 或短 token；完整 descriptor/record 位于受保护 mailbox。busy 接线不得让 core 等整条链结束才允许处理其他 context。正常 AM drain/free 是设备自主工作，轮询或 reap 只观察/退休事件，绝不触发物理释放。

descriptor/mailbox 使用明确的 uncached/TCM 路径和 release/acquire 顺序，保证发布前完整写入；不假定 Rocket D-cache 与 BOA L1 SPM 自动一致。固件保留 Tile Program 执行、wait 和 context switch，不改成通用 DAG 执行器，不引入共享内存 atomic 或依赖尚未实现的 interrupt。

## 5. 数据流、控制流和时序路径

### 5.1 独立完成边界

![AM lease、engine 步骤占用与持久事件生命周期](../image/boa_gemmini_lifetimes.svg)

图为逻辑先后，不是周期测量。固定 affinity、AM lease 和 engine 步骤占用是三个不同概念；input_release 不由 C 的状态推导。

- Group `tile.signal input_released` 继续表示该 Task 的 Group 输入不再需要，等待所有逻辑 Task 聚合；不是一个 BOA 内部 load 通知。
- output_visible 仅保证 L1 C 可见；L2 output_ready 还要等 MFE L1→L2 store，HBM 可见还要等 Group DMA store。
- slot_free 归还 AM 和 quota，可以早于事件 reap；不能等待 UCE 轮询，也不能先清掉旧完成状态再复用物理槽。
- Task retirement 继续等所有资源/外部引用安全结束；FINISHING 收尾允许自主完成，而缺 FINAL 退出是错误。

### 5.2 两槽软件流水与进展证明

> 本节保留为后续更低层 IR／执行协议的参考，不要求本次源 MLIR 写出下面的循环、条件分支或重试。具体展开和完整动态进展验收不属于当前源级 gate。

每条普通已建立链必须只靠它已有的一个 AM slot 完成 `INIT → ADD… → FINAL → DRAIN → RELEASE`，不能依赖一次新 OPEN 成功。若算法必须同时取得两槽才能让任一链释放，需要单独的成组申请合同或重新分块，quota=2 不自动提供该能力。

以下是两逻辑位置的**下层协议伪代码**，不是本次源 dialect 或新的通用 DAG 执行器；未来可由合适的低层控制实现：

```text
live[0] = empty; live[1] = empty
while 有未开始输出或live非空:
    收集可观察终态
    若任一链ERROR/CANCELLED：停止新工作，安全收尾所有owned工作并结束正常循环
    对chain_done的逻辑位置：reap旧record，随后清空软件live位置

    轮转服务live[0]、live[1]：
        处理普通input_usage释放
        在合法窗口尝试非阻塞MFE供数；受理后才获得ready token
        尝试提交当前seq；只在ACCEPTED时增加submitted游标
        首步INIT，末步FINAL；FINAL accepted后不再提交
        BUSY立即返回服务另一链；同步ERROR进入同一安全收尾路径

    仅对空的逻辑位置尝试新TRY_OPEN：
        GRANTED才取走下一output descriptor
        BUSY保存原子retry snapshot，不循环重试到成功
        ERROR进入安全收尾

    若无进展：WAIT_ANY(尚未解决的链进度事件 + 相关OPEN/credit重试条件)
```

wait_set 必须非空且对应真正能改变当前条件的事件；已处理的完成不能反复唤醒自身。两个软件 live 位置尚未 reap 时，可以保守地保持占用，即使硬件已在 slot_free 减 quota；两种计数不可混用。

必须覆盖自锁反例：同 context open C0/C1 后尝试第三条 C2；即使全局还有六槽，第三条仍 BUSY，调用方必须继续提交 C0/C1 的 FINAL。UCE 切换只能让其他 context 运行，不能生成本 context 尚未提交的 FINAL。

已有链的 mailbox、供数、输出区域和 drain 通路不能被新 OPEN 耗尽。输入 token 只能等待 ready 数据或已受理 producer；不提前占 engine/group 服务资格再等未提交输入。沿用一个驻留 context 一个 memory profile，保护后续输入窗口和已准入 FINAL 输出空间；不额外永久保留一个空 AM 槽。

进展结论是有条件的：有限 K-step、已受理输入最终完成、UCE 持续产生合法续步、ready 仲裁与 drain 公平、输出下游最终可用、安全释放不依赖 UCE 轮询。在这些前提下，满槽只造成背压，至少一条可推进链最终归还槽；**quota=2 本身不保证无死锁**。

### 5.3 功能模型、周期模型与 RTL 联仿

先在现有 validator 路径建立共享 AM 功能合同：真实 owner/quota/affinity、INIT/ADD/FINAL、不同完成边界、descriptor 快照和错误回收；随后加入有限 result credit、bank/adder 服务、RMW/drain latency 和 UCE/mailbox 资源。功能模型不冒充周期模型，所有 T 场景报告验证层级。

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

BOA-only 模式由 Python UCE 执行相同协议；Rocket 模式由真实固件执行，**同一个 Task 不能同时由两者解释**。Group admission 经 ControlBackend 等价边界提交已验证 capsule，真实 UCE 负责 L1 view、MFE 请求、TRY_OPEN/STEP、等待/错误分支、phase signal 和 return。

非受支持的 Matmul 指令闭包明确拒绝，不能偷偷退回 Python UCE 执行。该闭包不宣称整个 validator ISA 都已成为 Rocket 软件；在 FINISHING 后无需返回旧 context 才能完成硬件释放及通知退休。

两条控制路径比较输出 bytes、所有权和合法事件偏序，不要求总周期相同；Rocket 取指、固件调度与轮询开销必须计入，固件/RTL/hash 和 linker 映射进入 manifest。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 V1 基线与容量

| 参数                                          | 设计值/规则                                        | 性质                                                       |
| --------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------- |
| num_engines                                   | bring-up 先 1；V1 多阵列基线 4，覆盖 2/4 验证      | 单 engine 对应一份阵列执行资源                             |
| num_am_groups / slots_per_group / total_slots | 4 / 2 / 8                                          | 一个固定大小 slot 完整落入一个 group；group 与 engine 独立 |
| max_live_slots_per_tile_context               | 2                                                  | 硬约束，跨 engines 计数；V1 不允许改大                     |
| context_count                                 | 默认 4；覆盖 1/2/4/8，受 UCE/memory admission 限制 | 不是 AM 永久预留数，也不是 engine 数                       |
| array_rows / array_cols                       | 默认 16/16；矩形需独立 gate                        | Scala elaboration 与 target fingerprint                    |
| tile_m / tile_n / tile_k                      | 默认 16/16/8，实际映射验证                         | compiler/hardware 不混同物理阵列与 output tile             |
| next_step_mailboxes_per_live_chain            | 1，加一个当前执行记录                              | OPEN 时保证 continuation 元数据，不建无限步骤队列          |
| chain_event_record                            | 每逻辑位置可保留一代完整记录，OPEN 时保证容量      | AM 释放与记录 reap 分离，容量不足返回独立 BUSY             |
| ResultCapture / 最大在途微块                  | 按真实容量、最坏 burst 和 O+R≤Q 冻结               | 不预设无限缓冲或整大逻辑块一次预留                         |
| SRAM/bank/adder/drain ports 与延迟            | 各自独立配置和服务模型                             | 由 SRAM profile / PPA exploration 冻结                     |
| ISA 位宽、context/事件容量、PE pipeline/clock | 不从示例猜编码或硅片频率                           | 由后续规格冻结                                             |

slot_capacity 包含 M/N、stride、accType 字节数、padding、alignment 和实际分配粒度。以已确认的默认紧密 16×16 FP32 输出为例，有效数据下界为每槽 1024 bytes，八槽共 8192 bytes；实际宏容量不由这个下界冻结，也不包含结果缓冲/metadata/ECC。

AM 总容量按 `total_slots × slot_capacity_bytes` 计算，**不再按 context_count 永久分区**。配置验证拒绝分组数量不一致、slot 太小、缺少结果接收保障、非法 affinity 或 max_live_slots 大于 2；8 contexts 各一槽必须在其他资源足够的配置下可表达。

一个 context 同时至多驱动两个 matrix engines；两个槽都在 FINAL_READY/DRAINING 时可没有任何正在计算的 engine，这不是自动的死锁。四 engine 同时计算至少需要两个提供足够 ready 链的 contexts，并且绑定分布、输入和 AM 带宽满足；只是必要条件，不是利用率保证。

### 6.2 性能观测与公平比较

| 观测域    | 必须分解的原因/关联状态                                                                     |
| --------- | ------------------------------------------------------------------------------------------- |
| 配额/分配 | per-context quota、global AM full、logical slot in use                                      |
| 元数据    | mailbox full、event record full，与 AM 容量不足分开                                         |
| engine    | 每 engine compute、输入等待、bound-engine 等待、配置切换                                    |
| AM group  | update 数、read/write ports、bank 与 adder 冲突分别计量                                     |
| 结果      | result-credit 阻塞、O/R、高水位、在途微块数                                                 |
| 输出/链   | FINAL_READY、drain 反压、slot-free 延迟、OPEN→FIRST、step 间空洞、FINAL→FREE、总 lease 驻留 |

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
  scripts/                     构建/运行/复现入口（需在实现时真正提供）
.devcontainer/                 引用同一镜像与工作目录，不建立第二套环境
pipeline_validator/            保持现有 Python 子项目，新增有边界的 backend
examples/generators/            数值 Matmul/Conv 场景生成器
examples/workloads/             生成的 IR fixture
```

source subtree 的选择是本稿建议：本 repo 能直接评审 Gemmini 的必要修改，避免隐藏在另一个浮动仓库。不要同时维护“修改后的源码”和一套会再次应用的重复 patch；从锁定 upstream 到本树的变更应可独立导出和解释。

Chipyard/Rocket/HardFloat 依赖进入容器的锁定 dependency workspace；构建 overlay 只引用 `hardware/gemmini` 的权威源码一次，不同时编译 Chipyard 内另一份 Gemmini。禁止复制三套 Scala/Chisel 版本分别给 BOA、Gemmini、UCE 使用。

### 7.2 组件依赖与交接 guide

保留 C0–C10 系统组件编号。S0 指独立计划**本次范围内**的任务与软件 gate 完成，仍先于新硬件实现；S0 不含保留的 §3.4、源级 CFG 或具体低层分支/重试 codegen。C1/C4/C5/C6/C7-A 的资源/数值模型覆盖与未来低层动态恢复覆盖必须分开报告：

```text
C0-S 软件环境 -> S0 独立AM/IR计划
                  (覆盖本次AM/Matmul语义、模型、C7-A及调用方迁移；不含延期控制流)

S0 -----------------------> C0-H 硬件工具链构建验证
S0 + C0-H ----------------> C2   方形 Gemmini 核心实现
S0 + C2 ------------------> C3   硬件 operand / transpose / L1 接口
S0 + C2 ------------------> C6-H 硬件后处理
C2 + C3 + C6-H + S0 ------> C7-B RTL 与 validator 联仿
C7-B ---------------------> C8   Rocket / RoCC 真实 UCE
C2 -----------------------> C9   矩形馈数 / 排空
C8 + C9 ------------------> C10  多阵列共享 AM 最终验收
```

`+` 表示全部前置条件；C0-S/H 和 C6 软件/硬件区分环境与实现层级，C7-A/B 区分软件集成与 RTL 联仿，不重复建立 IR/scheduler。环境资料与版本调查可以先做，但不能把已有 RTL 作为 S0 通过的条件。

#### C0 · P0：环境与来源

- 输入：独立计划的软件环境、§1.4 硬件版本组合与源码。
- 实现：C0-S 先保证现有 Python/xDSL 环境可复现；S0 后 C0-H 再构建锁定的 Scala/Chisel/Rocket/HardFloat/Verilator/RISC-V 工具链，保留许可证和单一 build graph。
- 输出：分层环境清单、镜像 digest/lock 与真实构建入口。
- 验证：软件 parser/CLI 环境不依赖新 RTL；硬件阶段再做干净 elaborate、harness 编译和离线重复，不能仅打印版本。
- guide：宿主不安装依赖；版本不兼容时带证据调整整套 pin，不切 latest。

#### C1 · P0：独立数值 oracle

- 输入：BF16 bits、物理分块/逻辑 seq、INIT/ADD/FINAL、已 review 的舍入和后处理规则。
- 实现：独立软件 FP32 FMA/add、BF16/INT8 转换；可用锁版本软件浮点库，不调用 DUT 作为唯一 oracle。
- 输出：每个贡献块、AM 提交及最终 C 的精确 golden bytes。
- 验证：非对称输入、tie、signed zero、subnormal、NaN/Inf、尾块、内部微块 INIT 覆盖一次、bias 一次。
- guide：Python float 不是 FP32 fused 累加；固定每次舍入与分组顺序，不能只比 NumPy 容差。

#### C2 · P0：Gemmini 方形核心

- 输入：规范 A/B beat、shape/有效 mask、FP32 partial 格式、golden vectors。
- 实现：review 后选择单一 dataflow，审计 PE/mesh 中全部 FP32 路径、skew/tag/last/drain 和安全配置切换；保留正确性所需内部状态。
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

- 输入：ctx_key、logical_am 0/1、OPEN/STEP、全局八槽、seq 和 FINAL。
- 实现：动态 lease、两槽 quota、不可变 affinity、INIT/ADD、all-or-nothing OPEN、粘性完成和错误收尾。可先用抽象 AM 存储，不伪造周期/RTL。
- 输出：可执行的功能状态机、独立完成边界、owner 映射及事件引用。
- 验证：T01–T09、T14/T17/T19/T21 的资源/功能适用部分，含八 contexts 各一槽和旧事件跨 slot 复用；完整重试/等待登记竞态随低层方案后续验收。
- guide：复用现有事件框架但补足 ctx epoch/链身份；AM 释放与 record reap 分离，正常和错误 terminal 都能退休。

#### C5 · P0：有限周期资源与进展

- 输入：C4 功能模型、C7-A 两槽程序、每 engine/group/port/adder/结果缓冲的真实配置。
- 实现：current+one mailbox、bound-engine ready 仲裁、O/R credit、route/RMW/writeback、自动 drain 和公平服务；OPEN 不耗尽 continuation。
- 输出：可审计的 cycle/event/credit trace、按原因分解的阻塞和资源不变量。
- 验证：本次核对 result credit、bank/adder、commit、drain 和资源不变量；涉及具体重试恢复的满槽/等待登记端到端验证留给后续低层方案，不强加源级控制流。
- guide：不使用无限 FIFO、零延迟 AM 或额外永久空槽；不把增加 bank 当作 adder 吞吐；也不把延期的进展验证报告为已通过。

#### C6 · P1：后处理与软件 Conv

- 输入：FINAL AM、bias/scale/clip/quant 参数与 Conv 展平配置。
- 实现：FINAL 后只处理一次，再自动 drain 到已准入 L1；软件 im2col 复用 Matmul，不加硬件 Conv。
- 输出：BF16/INT8 字节与展开流量记录。
- 验证：饱和/tie/zero-point、关闭/开启后处理、小型 Conv padding/stride/dilation 对照独立参考。
- guide：INT8 输出不等于支持 INT8 Matmul；参数非法在输出写入前拒绝。

#### C7 · P0：typed IR、两槽程序和联仿

- 输入：软件管理 AM/IR 独立计划及已准入 views；S0 后的 RTL 层再接 C2/C3/C6-H。
- 实现：C7-A 按独立计划完成单 block AM allocation、四 operand/四模式、已有完成依赖和资源/数值模型；不新增源级分支、状态结果或 retry CFG。后续低层控制与 C7-B 硬件联仿分别验收。
- 输出：v4 typed artifact、compiled-only replay、C 字节和本次资源/事件证据。
- 验证：按独立计划 IR01–IR17 的本次/延期标记执行；重点保留 non-output 不写 L1、accumulate output 自动释放和 f32 partial。T01–T22 是完整系统目标，不等于本次全部闭环。
- guide：拒绝无 ByteStore 的数值路径和旧 operandless Matmul；延期控制流不允许被假成功、无限缓冲或隐式 fallback 掩盖。

#### C8 · P1：Rocket / RoCC UCE

- 输入：已通过单阵列联仿的 top、统一协议/ISA、两槽 firmware 和 admitted capsule。
- 实现：真实执行非阻塞 OPEN/STEP、WAIT/context switch、record reap 和错误收尾；Python UCE 对相同 Task 停用。
- 输出：Rocket 实际驱动 BOA 的系统 top、固件 hash 和指令/事件 trace。
- 验证：第三次 OPEN BUSY 后原两链仍提交 FINAL；context 返回后自主 drain/free；完成先于 WAIT、8-context 配置、错误 terminal。
- guide：直接拉 RoCC 信号不算真实 UCE；不能使正常释放依赖已返回 context 重新 poll。

#### C9 · P1：矩形阵列

- 输入：已验证方形 adapter、独立 R/C 和合法 tile 分块。
- 实现：独立推导 row/column skew、transpose、mask、tag/last/drain，不沿用 square block_size 假设。
- 输出：至少 8×16、16×8 与四种 transpose 配置。
- 验证：不同 M/N/K、尾块、第一/最后有效输出、气泡输入以及 complete bytes。
- guide：形状改变不改变 quota、affinity、step_committed 和 FINAL 自动释放合同。

#### C10 · P1：多阵列共享 AM

- 输入：已通过单阵列/UCE/矩形验证的组件、4 engines/4 groups/8 slots，及 2-engine 对照配置。
- 实现：跨 engine 的共享分配/路由、固定链 affinity、独立 group RMW 与公平 drain；只复制必要执行/结果接收资源，不建 private AM。
- 输出：同 context 两链独立推进、多 context 覆盖四 engines、动态八槽利用和完整终态。
- 验证：T01–T22 的适用数值/周期/RTL层；强制原 engine 忙且其他空闲时不迁移；不同 engines 访问同一 group 的不同合法区域；slot 换代后重新分配可改变 engine。
- guide：证明共享的是容量和访问，不要求同一 live handle 跨 engine 更新；四 engines 满载不是单 context 的必达行为。

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
hardware/scripts/reproduce <run-manifest>
```

每个 check 都需要在容器内执行真实程序、输出 manifest/日志/数值结果，不能只是列出下一条手工命令。实施计划中再给出这些入口的精确文件、参数和真实预期输出。

环境约束：

1. 基础镜像按 digest，软件按版本/hash；网络只在显式构建/下载阶段使用。包括 Scala compiler plugin、Maven artifact、Python wheels、RISC-V GCC/binutils、Verilator 及依赖。
2. 不在宿主运行 apt/yum、sudo、全局 pip，不修改 `/etc`、shell profile、Docker daemon 或其他项目文件。依赖获取限容器镜像构建；如需包管理器也只在镜像内部。
3. dev container 引用同一 Dockerfile/digest，映射宿主 UID/GID；源码工作目录固定，构建缓存和结果目录独立，不在工作树散落 root-owned 文件。
4. 不使用 privileged，不挂宿主 Docker socket；simulation 默认不需要设备直通。
5. `toolchain.lock` 包含完整版本和来源；当前 environment.yml 只固定 Python 3.11 与 xdsl 0.69.0，不能把它当成已经完整锁定的 Python 环境。
6. 每次运行保存：repo/source hash、dirty diff hash、依赖 pins、镜像 digest、配置、编译器 ABI、输入 seed/hash、固件/RTL/library hash、命令、actual output、golden output、报告、必要波形。
7. 复现必须从 manifest 解析配置和输入；缓存不是正确性前提，第二次离线执行应有相同数值和确定性仿真周期。

## 8. 验证、bring-up 和验收标准

### 8.1 共享 AM V1 验收矩阵

以下逐项对应输入规格 T01–T22，**全部是待实现后运行的验收要求，本次没有执行这些新功能测试**。F=功能，C=周期，R=RTL；F 通过不能声称 C/R 通过。

本表是完整 V1 系统验收，不是当前源 IR 软件 gate 的同义词。S0 只覆盖独立计划标明的本次语义/模型部分；具体低层分支、重试恢复、WAIT_ANY/等待登记相关部分在后续阶段验收，要求保留但不作为本次实现前置。

| ID  | 场景与必须观察的结果                                                                             | 最低验证层/定位            |
| --- | ------------------------------------------------------------------------------------------------ | -------------------------- |
| T01 | 同 context 两槽后第三次申请，即使全局余六槽也 BUSY、无副作用；旧链可 FINAL 释放                  | F，quota/进展              |
| T02 | 多 context 竞争，含八 contexts 各一槽；全局≤8，无四 context 或每 engine 两槽隐限                 | F，动态容量                |
| T03 | RESERVED/WAIT_INPUT/FINAL_READY/DRAINING/one-shot 均计数，仅 slot_free 减一次                    | F，lease                   |
| T04 | 同 context 两链在不同 engine 独立推进，每条链全部 step 始终绑定原 engine                         | F+C，affinity              |
| T05 | bound engine 忙而其他空闲时不迁移，等待原因可见                                                  | F+C，仲裁                  |
| T06 | 一个 engine 服务多个 contexts、超过两个 handles，按步公平交替                                    | F+C，元数据边界            |
| T07 | 垃圾 AM、单/多段、尾块和初值；INIT 不读残留、内部微块仅首次覆盖、bias 一次                       | F 数值+R                   |
| T08 | 重复/跳 seq、重复 INIT、FINAL 后提交、错误 owner 均拒绝且无 AM 副作用                            | F，校验                    |
| T09 | 故意延迟 FIFO/RMW/writeback；compute_done 可先到，commit/output 不提前                           | F 边界+C/R 时序            |
| T10 | 四 engine burst、多个在途微块；Q=8/O=6/R=2 不再发射，到达只 R→O                                  | C/R，真实 credit           |
| T11 | 多 group 并行与热点；吞吐遵守独立 bank/adder 能力，等待公平可观测                                | C/R，服务模型              |
| T12 | 同地址 RMW、写后读、mask、控制位变化不绕过物理地址互锁                                           | C/R，hazard                |
| T13 | 全局八槽满且新 OPEN 持续到达，旧 mailbox/供数/FINAL/drain 仍推进                                 | F+C，进展                  |
| T14 | 一链 mailbox 满而另一链可推进；BUSY 不增 seq，ACCEPTED 后改原 descriptor 不变行为                | F，快照                    |
| T15 | 多输入读者、最后读取延迟、释放后立即覆写；释放安全且结果不变                                     | F 数值+C，普通 input_usage |
| T16 | 事件早到、WAIT 边界、多事件乱序、失败判断与 BUSY 返回间释放资源；无丢唤醒/忙循环                 | F+C/R，等待协议            |
| T17 | slot 复用后旧 generation 的结果/释放/完成不影响新 owner，旧终态仍可观察                          | F+C/R，身份                |
| T18 | 不同 context 的 stride/transpose/scale/activation 交错，已接收配置不串扰                         | F 数值+R                   |
| T19 | FINAL drain 中退出进入 FINISHING；缺 FINAL/中途故障/部分链失败安全收尾，失败 terminal 不伪装成功 | F+C/R，退休                |
| T20 | 单 context 四独立输出最多两个 engine 同时工作；两个 context 各两链在资源充分时可覆盖四 engine    | C/R，并发边界              |
| T21 | OPEN metadata/event 容量不足时全败，无已占槽却缺最终记录的半成功                                 | F，原子分配                |
| T22 | L1 drain 反压和持续 ADD；下游恢复且仲裁公平后最终释放，不需要 UCE 轮询                           | C/R，自动释放              |

此外保留原项目交付验收：BF16/INT8 全输出 bits 对独立 oracle、方形/矩形四种转置、真实 Rocket 指令执行、compiled-only replay、软件 Conv、容器干净/离线复现。不能用 T 表的协议覆盖替代数值或真实硬件验证。

每个测试结果保存配置、输入 seed/hash、命令/事件 trace、actual/golden bytes 和实际运行层级；不以添加测试文件、打印预期输出或源码字符串断言代替通过。首次失败保存最小复现和波形，不重试到成功。

### 8.2 不变量与故障验收

- 每 context live lease 在 0..2，全局在 0..8；映射一一对应、物理区域不重叠。
- OPEN 成功加一次，安全 slot_free 减一次，BUSY 不改变持有资源；one-shot 无例外。
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

该命令没有 input-data 或 Matmul 数值 oracle，BOA 源码也不算矩阵；结果**不证明 BF16、动态 AM quota、affinity、自动 drain 或新事件合同，也不是新阵列性能基线**。本次只验证文档/图片，不声称 T01–T22、Scala/RTL/Rocket 已实现或通过。

## 9. 风险、取舍和后续细化方向

### 9.1 硬约束与仍需 review 的细化

共享 AM V1 的链级固定 affinity、每 context 两 live slots、正常自动 drain/free 等要求已经作为本次修订依据；不能在后续实现时重新改为每 engine 配额、永久分区或跨 step 迁移。

| 待 review 项  | 当前边界                                                                                           | 冻结前证据                                                                  |
| ------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| PE 数据流     | WS/OS 未定；不能用迁移前提选 WS，也不默认为 OS                                                     | 默认/矩形 shape 下的馈数、FP32 路径、数值顺序与周期                         |
| 数值细则      | BF16/FP32/RNE 已确认；特殊值、物理分组和 epilogue 细则按 §3.2 评审                                 | 独立 golden 与 RTL 数值检查                                                 |
| 物理存储/吞吐 | V1 四 groups/八槽；具体 slot alignment、bank、adder、结果 credit、drain 宽度未物理定型             | 容量、O/R 不变量、冲突与服务轨迹                                            |
| IR/ABI        | 单结果 AM allocation、四 operand/四模式和 v4/v4 目标保留；源级分支/状态/重试不做，低层控制后续设计 | 线性 codec/独立验证、数值/周期、compiled-only 与调用方迁移；不要求 CFG gate |
| Rocket/环境   | 一 Tile 一 hart、可配置 contexts、锁版本工具链                                                     | 真实固件执行、构建和 PPA；不把名称当作“微型”证明                            |

用户已要求独立软件/IR 计划，现已另文给出；本轮不实施其代码。整体硬件的文件级计划和实现仍需在软件 gate 及后续 review/授权后推进。

### 9.2 主要风险与停止条件

- **配额与独立性写错**：per-engine quota、context-wide 单链序列化、one-shot 旁路、提前释放 DRAINING 配额都违反 V1。
- **affinity 与占用混淆**：原 engine 忙时迁移、等输入时永久锁 engine、元数据隐含只能两链，都会破坏合同。
- **OPEN 自锁**：只切换其他 context 不能替当前 context 提交 FINAL；BUSY 后必须服务已有链，保护 continuation 和输出路径。
- **完成边界合并**：mesh last、compute_done、step_committed、output_visible、slot_free、event reap 不能合成一个 done。
- **有界资源遗漏**：FIFO 当前占用不含未来结果，bank 数不代表 adder 数，drain 饥饿会长期占 quota；不得通过无限缓冲或零成本 AM 掩盖。
- **输入取消要求被误解**：取消 RF/load 通知不等于可以提前释放 A/B，普通 reader/producer 安全条件仍需满足。
- **数值和矩形未经验证**：FP32 partial 截断、INIT 内部微块重复覆盖、square assert 删除都不能通过放宽 oracle 掩盖。
- **双重状态和错配工具链**：Python/RTL 只有一个 AM 与控制权威；调研 Gemmini 和 Chipyard 实际 pin 不同，构建证据优先。

### 9.3 需求追踪与评审门

本表给出**设计覆盖**，不是实现/测试通过状态；R 编号与[输入规格](../../nexus_shared_am_v1_requirements_and_design.md)保持一致：

| 需求                                       | proposal 落点               | 核心验收                      |
| ------------------------------------------ | --------------------------- | ----------------------------- |
| R01 共享容量/无每 engine 两槽隐限          | §2.1–§2.3、C4/C10           | T02/T06                       |
| R02 每 context 两个 live lease             | §2.3、§6.1                  | T01/T03/T20                   |
| R03 链级 bound_engine 不变                 | §2.2、§3.4                  | T04/T05                       |
| R04 同 context 两链无额外依赖              | §3.4、§4.4、§5.2            | T04/T06/T14                   |
| R05 唯一 owner/同 handle 单步更新          | §2.2、§3.3、§3.5            | T08/T09                       |
| R06 OPEN 全成或全败                        | §4.1                        | T01/T21                       |
| R07 新申请不阻断旧链推进                   | §3.4、§5.2                  | T01/T13/T22                   |
| R08 接收/计算/提交/输出/释放分离           | §3.3–§3.6、§4.3、生命周期图 | T03/T09/T22                   |
| R09 不可暂停输出有接收保障                 | §3.5、C5                    | T10                           |
| R10 bank/port/adder 与 RMW 互锁            | §3.5、§6.2                  | T11/T12                       |
| R11 INIT/ADD/mask/精度/最终转换            | §3.1–§3.2、C1/C6            | T07/T08/T12                   |
| R12 generation/context epoch 安全          | §2.2、§4.2–§4.3、§8.2       | T08/T17                       |
| R13 持久事件/原子等待/重试快照             | §4.1、§4.3                  | T16/T17/T21                   |
| R14 普通输入最后读取与全读者释放           | §2.4、§4.2–§4.3、C3         | T15；不引入专用内部 load 通知 |
| R15 STEP 接受即固定 descriptor/config      | §4.2                        | T14/T18                       |
| R16 FINAL 自主输出并安全释放               | §3.6、§4.3                  | T03/T13/T22                   |
| R17 slot_free 归还 quota/安全 context 退休 | §3.6、§4.3、§8.2            | T19                           |
| R18 原因统计与关联 trace                   | §6.2、§8.1                  | T05/T11/T20 及全 T 表         |

软件管理 AM、A/B/out/acc、accumulate output 和 f32 直出继续由独立计划追踪；源级 CFG 与具体低层分支/重试按最新要求暂缓。此次 gate 不改变 R01–R18 的系统安全目标，也不宣称所有动态恢复场景已经闭环。

原需求中仍适用的 Scala/Gemmini、单阵列后多阵列、矩形、BF16/INT8 输出、硬件转置、Rocket 自定义指令、软件 im2col、monorepo/Docker、组件输入输出/guide 和图示，分别落在 §1/§3/§4.5/§7/§8。专用 RF 与内部 load 通知要求已取消；同 context 单链和 K-step 迁移设计不再适用。

本次软件计划待 review 后执行，当前范围的 gate 仍先于硬件实现；保留的控制流章节及更低层分支/重试后续单列。实现报告必须区分当前覆盖、明确延期、功能/周期/RTL/PPA 的实际状态。

### 9.4 上游证据链接

以下链接均固定到本轮核对的 commit，避免 master 漂移；当前仓库事实以 §1.3 所列本地源码为准。

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
