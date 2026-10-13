# Nexus / pipeline_validator 整体思路的 FPGA + RISC-V 实现可行性分析

> **重评日期：2026-10-10。范围是当前代码描述的完整系统，不再仅评估调度器。** 包含编译与装载接口、CPU/Group/Tile 调度、BOA/EVU/USE、MFE Gather/Scatter、DMA、片上存储与 Cache、Profile 切换、Host 页管理、Stream、事件、故障恢复和观测链路。
>
> **结论（[INFERENCE]）：架构可以用 FPGA + RISC-V 实现；当前代码已经是较具体的控制/数据搬运行为参考，但还不是能够直接转换为完整计算硬件的规格。** 最大缺口已不只是调度协议，而是 BOA/EVU/USE 的真实数值语义与操作数接口、Cache 的物理组织、存储带宽、软硬件 ABI 和物理验证。
>
> **“实现整体思路”“实现完整数值加速器”“达到当前 YAML 默认性能”是三个不同承诺。** 第一项有明确实现路径；第二项必须补齐计算合同并实现真实数据通路；第三项尚无综合、布局布线或板上证据，不能承诺一块 FPGA 原样达到。

本文区分 **源码事实、2026-10-10 运行观察、架构规格意图、[INFERENCE] 工程判断**。所有硬件映射、选型建议和实现步骤均属工程判断；不把 Python 对象当成已存在的寄存器、SRAM 或 RTL。只更新本报告，不修改模型、工作负载或其他规格。

## 1. 本次范围与上一版的实质差别

### 1.1 什么算“完整实现”

完整目标是：软件装载不同程序和 descriptor；RISC-V 处理设备管理与软件策略；FPGA 自主执行任务、搬运真实数据、完成约定算术、管理有限资源，并在错误/复位后安全恢复。不是把 Python 模拟器搬进软核，也不是用倒计时替代计算。

- **Host 软件仍属于系统交付：**xDSL 编译、资源证明、模型/数据准备、装载与结果检查，不需要全部综合成电路。
- **RISC-V 是软件执行载体：**设备程序解释、页分配、资源授权和恢复可移植为固件；当前 `CpuDeviceController` 不是 RISC-V ISA 模拟器，Python handler 也不是现成固件。
- **FPGA 不仅做控制面：**还必须承担 DMA、Cache/MSHR、索引求址、真实数据搬运，以及 BOA/EVU/USE 等约定的计算能力。
- **验证设施与目标硬件分开：**`ByteStore`、虚拟周期、Python callback、JSON trace 属于参考模型/验证环境；硬件实现其对应的数据可见性、状态和 PMU，不逐句翻译这些工具。
- **当前边界是单 Group、四 Tile：**不能将此结论外推为多 Group NoC、分布式一致性或完整数据中心芯片已经可实现/已验证。

后文可以建议降低频率或时分复用，但这只是实现方案选择，**不是把完整目标悄悄缩成单 Tile、仅调度器或只有 memcpy 的演示**。改变存储容量、并行度或算法支持集，必须标注新 target 并重新编译、验证，不能沿用旧性能结论。

### 1.2 旧报告必须纠正的事实

| 旧报告口径                                    | 当前源码事实                                                                                                                | 对硬件判断的影响                                                          |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Device 由模拟器按顺序简单占 slot              | `CpuDeviceController` 有依赖就绪提交、pending、outstanding/completion credit；GroupPort 独立管理 pending 与执行 slot        | CPU 与加速器边界更清楚，不能仍按单一 device PC/slot FSM 估算              |
| UCE 当前 READY context 优先，阻塞才换         | `TileUCE._select_context` 是 eligible-head round-robin、共享单发射；队列接受后才推进 PC                                     | 应实现 RR 资格选择与 exactly-once 接受，不是旧的 current-context 优先策略 |
| UCE 数取两类 context 数最大值                 | `context_count`、`device_context_count`、`group.active_context_capacity` 分别控制每 Tile UCE、CPU outstanding、Group 执行槽 | 三种容量独立，不能互相代替                                                |
| Group 只推进线性 action index                 | 每 root 有注册游标；共享 `GroupScheduler` 有有界 ready-action 窗口；S0 队首受限，S1 可越过 PENDING 依赖，S2 当前同 S1       | 需 action 表、依赖状态、credit、注册 fence，而非一个全局线性 FSM          |
| L2 strict FIFO，仅 final-free 触发重试        | Root/Task 按 SAME、COMPATIBLE 两类 FIFO 队首比较；重试受 pool/event/slot/Profile/gate 等版本变化驱动                        | 同类 FIFO 仍保留，但不等于全局单 FIFO，也不等于只听一次 free 通知         |
| 多 Tile 全部准备后原子 dispatch               | 先登记有界 Grid Route，再逐 Tile 独立准入 Task；一 Tile 受阻不回滚其他 Tile 已提交的 Task                                   | 不应额外加入全 Grid 启动 barrier；部分 Grid 已运行的故障必须 drain        |
| `BARRIER_GROUP` 只计数                        | 注册 fence 与 `barrier_ready` 等待该 Context 的先前 action                                                                  | 是 Context 内 barrier，不是跨 root 全局同步                               |
| Gather 命中由 profile 给出，没有真实 Scatter  | 当前 `tile.gather.global.async` / `tile.scatter.global.async` 读取真实 i32 索引；按地址查 Cache，Scatter 有真实字节覆盖写   | MFE 数据路径已具备可执行参考，必须纳入硬件范围                            |
| 没有需要真正实现的 Cache 数据面               | 当前有 Cache 行、LRU、MSHR 合并、维护、字节有效性和真实下游 clean 事务                                                      | Cache 组织与数据端口成为完整系统主要成本                                  |
| 原控制面 612 KiB/8–15 人月/10–30 W 可代表项目 | 原预算排除了算术、tensor SRAM 和完整数据面                                                                                  | 不能继续作为本次全系统预算                                                |

依据：[S1]–[S9]。仓库 `Limitation.md`、部分 README 仍写“profiled Gather / 没有 Scatter”或 artifact schema 2/ABI v2；本次实际产物为 **schema 3 / compiler ABI v3**，Gather 报告为 `address_resolved_cache`。这些旧文字不能作为当前能力的否定证据；硬件 ABI `v0` 草案又是另一个版本空间。[S1][S6][S15]

## 2. 完整系统能力盘点：哪些可提取，哪些必须新设计

| 子系统                     | 当前实际做到什么                                                                                              | FPGA + RISC-V 落地内容与缺口                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Compiler / Loader          | 显式 compile → immutable artifact → load → run；hash、relocation、binding/权限、资源与依赖证明；Loader 不修图 | Host 侧保留；新增硬件程序/descriptor 编码与能力描述。JSON artifact 不是 FPGA 指令镜像                     |
| Device / Host              | 有界 CPU 解释器、异步 Host 请求；HostSession 执行页分配及经 transfer 计费的读写                               | RISC-V/Host 固件实现协议与例程；不能把 `issue_width` 或 Python handler 的一次调用当真实 CPU 耗时          |
| Group / Tile               | ready-action、逐 Tile Task 准入、UCE RR、有限 engine FIFO、event/phase                                        | RTL 状态表、资格扫描、微序列器与局部 completion；现有行为可作参考                                         |
| Arena / 生命周期           | L2 Root Arena、L1 Task Arena、逐 bank stripe/padding、view/pin/inflight、R lease、只读跨 Context 共享         | 固件管理分配政策，硬件授权/生命周期表；不能 release 一个 view 就返还整个 Arena                            |
| Global / Local DMA         | HBM/L2/L1 多腿传输、outstanding、bank 冲突、generation-aware cancel；ByteStore 可检查字节                     | burst、byte-enable、地址生成、AXI 请求/响应、有限 payload buffer、错误/可见性；stage 延迟公式不是 DMA RTL |
| MFE Gather / Scatter       | 真实索引求址、Cache lookup/refill、MSHR、Gather 重排、重叠 Scatter 有序覆盖                                   | 求址流水、边界检查、有限请求窗/重排队列、写冲突 scoreboard；**不是 ScatterReduce/原子归约**               |
| L1/L2 SRAM / Cache         | Profile 划分 SPM/Cache、模型 LRU、read-only/write-back 维护、精确写范围账本                                   | BRAM/URAM bank、tag/data、端口仲裁、替换与回写；物理 Cache 组织仍需决定                                   |
| Profile / maintenance      | 单一 `ProfileController`、issue/range gate、drain、全成员 Prepare/Commit ACK、generation、recovery            | 可配置 bank 边界/映射与配置广播；不是“修改一个 CSR 后立即换容量”                                          |
| Page pool / PagedAttention | Host 页分配、block table/length 更新、KV Gather/Scatter、scope 静态约束和字节种子                             | 页表/epoch、宿主可见性、MFE 动态访问授权需闭合；PagedAttention 算术仍未实现                               |
| BOA / OPA                  | `m/n/k/ops/accumulate` 与 Roofline/launch 时序                                                                | 矩阵操作数、数据布局、MAC/累加/写回及数值规格；OPA 当前是峰值几何参数，不是 RTL 阵列                      |
| EVU                        | op 名称与 ops/launch 时序；pow 另有 exponent/bytes                                                            | 向量读写、mask/tail、reduce、softmax 等运算的真实语义和数据通路                                           |
| USE                        | 状态引擎的 ops/时钟比延迟模型                                                                                 | state layout、scan/recurrence、状态提交/恢复；不能由延迟模型推导具体算法                                  |
| Collective                 | command/window 与时序 trace                                                                                   | 若交付 reduce/broadcast，必须补参与者、数据格式、归约顺序与真实网络；现模型不提供数值参考                 |
| Stream / event / fault     | credit/leased/popped 状态、EOS/error、generation、fault ring、drain/reset                                     | 有限控制通道、唯一状态 owner/仲裁、迟到响应隔离、CDC；Python 共享对象不等于硬件实现                       |
| PMU / trace / report       | 组件 snapshot、事务与生命周期计数、Perfetto 导出                                                              | 硬件计数/采样环、时间戳、overflow；Host 负责解码/生成 JSON，不把完整 trace 字符串放进 RTL                 |

**可行性提高的原因是合同更具体，不是剩余硬件更少。** 软件/控制/内存模型已有大量可复用语义；真实算术和真实物理接口仍需工程设计。源码中的枚举、类名、形如 `softmax` 的 op 名称，不等于对应算法已经实现。

## 3. 推荐软硬件划分与可编程性

以下为 [INFERENCE] 推荐实现，不代表当前已经有 RTL。

| 部分                                                       | 推荐实现位置                                                | 不应跨越的边界                                                    |
| ---------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| xDSL、lowering、资源/依赖证明、数值参考                    | Host compiler/runtime                                       | FPGA 不解析 Python/xDSL/高层 graph                                |
| `nexus.program` 提交、页管理、资源准备、程序版本、恢复策略 | 管理 RISC-V 固件；Host handler 可保留为 Host 服务或迁入固件 | 必须保持异步请求/完成与可见性合同，不能直接改 live 硬件表         |
| GroupPort、ready-action、Root/Task grant、event/phase      | FPGA 控制器与有界 SRAM/寄存器表                             | 固件分配策略与硬件 grant 状态要有单一权威来源                     |
| Tile UCE                                                   | 每 Tile 小型可编程单发射序列器                              | 执行软件装入的程序；不写死一个 workload，也不替 engine 做内部调度 |
| MFE/DMA、Cache/MSHR、bank 互连                             | FPGA 数据通路                                               | 逐 index、逐 cache line、逐 completion 不绕经管理 CPU             |
| BOA/EVU                                                    | FPGA DSP/LUT/BRAM 数据通路；合适的规则算术可用 HLS/IP       | 保留可编程 descriptor；完成必须来自真正计算/写回                  |
| USE                                                        | 局部 RISC-V/uCtrl 加状态存储，或专用单元                    | 若共享管理核，必须计入并发、延迟和状态隔离代价                    |
| 外部内存、PCIe/AXI、调试                                   | 板卡成熟 IP + 薄适配层                                      | 模型 HBM 通道编号不直接等于物理 HBM stack/AXI port                |

一个管理 RISC-V 加多个局部硬件前端通常比“所有动作都交给一个 CPU”更稳妥。NEORV32 等公开实现证明可综合管理软核和总线接口有成熟路径，但不证明本系统已满足资源/时序要求。[H1]

**同一 bitstream 可变的内容：**任务与依赖、合法 shape、Tile Program、descriptor、placement、buffer 绑定、Profile 选择、页表、支持范围内的调度参数。**通常要改 RTL/bitstream 的内容：**新算术原语、未预留的 queue/context/端口上限、新 Cache 组织、任意 engine 抢占。

UCE 不必是一颗完整 RISC-V；可加载的控制程序已提供软件灵活性。反之，把所有算术交给标量 RISC-V 固件虽然可能得到低速功能实现，却不能冒充 BOA/EVU 并行结构或沿用其性能预测。是否采用这种实现必须作为显式方案，而不是隐藏替代。

## 4. 第一阻塞：计算语义尚不足以驱动完整硬件

### 4.1 不只是“还没有写乘法器”

当前 `TileBoaOp` 主要携带 op 名、`m/n/k/ops/accumulate`，`TileEvuOp` 携带 op 名、`ops`；`_lower_engine_descriptor` 没有为它们生成类似 Gather/Scatter 那样的 source/destination view。`BOAEngine.latency`、`EVUEngine.latency`、`USEEngine.latency` 只算延迟，`Engine.tick` 到时返回完成，不读写 tensor 计算结果。[S6][S11]

因此，即使购买现成 GEMM IP，也还不知道该 descriptor 要从哪些 buffer 读取什么格式、如何累加、向哪里写结果。当前实例中的 QK、PV、softmax、pow、reduce 名称和 ops 数不能补齐这些信息。

### 4.2 完整数值实现前必须冻结

1. **显式操作数合同：**输入/输出 view、offset/stride/layout、mask/tail、原位更新、alias/hazard；每个 op 的读写范围必须进入 compiler proof 和 Loader 校验。
2. **数值合同：**支持 dtype、乘法与累加精度、round/saturate、溢出、NaN/Inf、reduce 顺序及容许误差。默认 YAML 的 BF16 参数和架构文档的 dtype 意图不能代替合同。[A1]
3. **各引擎具体功能：**BOA 的 GEMM/累加/写回；EVU 的算术、归约、softmax 所需 exp/div；USE 的状态算法与 checkpoint；Collective 的实际数据操作。仅保留接口但没有计算不能算完成。
4. **一致的可执行 oracle：**用显式输入产生可对比输出，覆盖尾块、重复累加、极值与多 context。ByteStore 只证明搬运字节，不替代 tensor reference。
5. **完成可见性：**计算输出已写入约定存储且后继可读，才允许输出事件与回收；不是 descriptor 已接受或计算 pipeline 已启动。

**[INFERENCE] 建议先补这一层合同再设计完整计算数据面。** 这不是新增用户没要求的算法，而是将代码里已声称的计算步骤变成可执行硬件所需的最低前提。没有这些输入，无法可靠估算完整 DSP 数、数值误差或开发周期。

### 4.3 控制 ISA 也有残留占位

公开源码 IR 当前以有限单块、展开的控制流为主。内部 `ExecTileOp` 仍包含 MOV/ADD/CMP/LOAD_DESC/STORE_DESC 等枚举，但 UCE 中这些分支只推进 PC；BRP 没有真正谓词求值，PATCH_DESC/FENCE 也不能据当前计时分支认定为完整硬件读写/内存栅栏。[S4]

硬件 ABI 必须标明哪些 opcode 真正支持，并为使用到的指令定义寄存器、谓词、patch 权限和 fence 可见性；不支持的指令必须明确拒绝，不能把模拟器占位 no-op 固化为“已实现”。当前 straight-line 编译输出与未来通用可分支 ISA 应分开验收。

## 5. 第二阻塞：内存系统可以实现，但不能直接翻译 Python 容器

### 5.1 Gather / Scatter 已有可用行为参考

`resolve_indexed_segments` 的求址单位是 remote dtype 的 **元素**：[S7]

```text
remote_element = index[i] * index_scale + offset + task_id * task_stride + j * stride
local_element  = (i * repeat + j) * segment
```

index 从 L1 读取为 little-endian signed i32；负 index、越 view、地址溢出被拒绝。Gather 按 cache line 切请求，执行 L1/L2 lookup、MSHR 合并与回填，再向目的 buffer 交付字节。不能用 HBM miss 数代替 HBM transaction 数：多个 miss 可合并为一次回填。

Scatter 绕过两级 Cache，按段写真实目标。**重叠写保持 ordinal 覆盖顺序，非重叠段可以先完成**；它不是所有段完全串行，更不是 atomic add。编译器的写范围维护与运行期 precise-write ledger 用于后续 Cache 可见性。[S6][S8]

[INFERENCE] RTL 应采用求址流水、有限 index/request window、MSHR waiter 队列、Gather reorder 与 Scatter 区间冲突表。模型中 `slots` 按全部 index 数建 Python 列表，不能把 `window_entries` 误当成所有内部状态已自然有界；最大 index/repeat、段数、payload/reorder bytes 和溢出回压仍要冻结。

### 5.2 Cache：地址真实不等于物理组织真实

`DeterministicLRUCache` 以 allocation/generation/line identity 为 key，使用一个 pool 内的 `OrderedDict` 查找和全局 LRU；它没有普通硬件 set-index/way 冲突模型。**当前命中率是这个组织下的地址驱动模拟结果，不是某个 FPGA Cache 的实测值。**[S8]

默认 line 为 64 B；mode3 时每 Tile L1 Cache 有 8,192 行，L2 Cache 有 65,536 行。将如此大的查找/全局 LRU照搬成宽 CAM 和同拍更新网络，风险显著高于一个小 scoreboard。

[INFERENCE] 有两条真实路线：

- 保留精确模型替换/查找语义，以多周期 tag 搜索和存储化元数据实现，接受更高延迟；不能承诺模型的一拍 L1 lookup。
- 明确选 banked/set-associative Cache、替换算法和端口数，并同步修订 target/model、重新测 hit/miss 和性能。字节功能可保持，但旧命中率与周期预测不可保留不变。

默认 write policy 为 read-only；源码也有 write-back dirty clean/maintenance。它不会让 dirty victim 静默消失，不能将“有 clean 事务”误称为已定义完整的自动 dirty eviction 控制器。若支持 write-back，必须明确 victim 处理、writeback buffer、错误保留与 reset 丢弃规则。

### 5.3 Profile 是共享物理容量的切分，不是额外加一块 Cache

每 bank 的 SPM 与 Cache 来自同一总容量；system reserved SPM 还要从可供 Arena 的部分扣除。**不要把 12 MiB L1/L2 数据容量与 Profile Cache 再相加，重复计算 tensor data。**Tag、MSHR、ECC、有效性和控制 RAM 则另计。[S9][S12]

[INFERENCE] FPGA 可预布置 RAM bank，用地址边界/选择器实现合法模式。切档顺序必须保留：关闭 issue、证明相关引用静止、clean/invalidate、排空下游、Prepare、收齐 ready ACK、Commit、收齐 commit ACK、发布 generation、重新开放。当前成员为 4×16 个 L1 bank 加 16 个 L2 bank；实际 BRAM/URAM 深宽、端口、tag 区域与映射由后续规格冻结。

不必靠 partial reconfiguration 实现每次 Profile 切换；但也不能假设改一组软件容量字段就改变了 FPGA 物理 SRAM。

### 5.4 ByteStore、页所有权与 Host 的剩余边界

ByteStore 是稀疏逐字节 oracle：未初始化读取报错、read-leg 完成捕获数据、write-leg 完成才提交目的。真实硬件需要实际 RAM，而不是 Python page dict。其逐字节 validity 若压成 1 bit/byte，12 MiB 数据就另需 1.5 MiB 位图；覆盖 16 GiB 外部内存则需 2 GiB，尚不含管理开销。**[INFERENCE] 必须区分验证用 validity 与产品承诺的 poison/初始化检查；不能默认硬件免费提供整个 oracle。**[S7]

`PagePoolRegistry` 已有 scope、epoch、allocate/free、lease/root-pin API；HostSession 有 scope 范围检查，编译产物有 scope effect 约束。但当前 MFE launch/子事务路径未调用该 registry 的 `lease/acquire/release`；不能仅凭 `AccessLease` 类和模块注释，就宣称设备侧每个动态索引已做页所有权授权。[S10]

[INFERENCE] 若完整系统依赖页隔离，必须把 scope/epoch 验证接到真实 MFE 访问与终结路径，或冻结并证明等价的静态访问限制。binding 内合法并不自动等于当前 owner 持有该页。Host 页分配和更新 block table/length 的固件耗时、CPU cache clean/fence、DMA 可见性也必须计入；当前同步 Python 页管理不是这些物理代价的证据。

### 5.5 DMA / NoC 从时序路径变成真实事务

`full_memory` 已区分 HBM、Global DMA、NoC VC、L2/L1 bank、Local DMA 的 route 与竞争；NoC 仍是简化单路由器，HBM stage 是固定延迟/带宽模型，不是 DRAM 行、刷新、turnaround 的完整控制器。[S7]

[INFERENCE] 采用成熟 DDR/HBM/AXI IP，补 descriptor fetch、burst/边界拆分、byte-enable、有限在途 tag、payload buffering、响应乱序、总线错误、credit-return 与 backpressure。completion 必须对齐消费者需要的可见性；timeout/reset 不撤销已落内存的写。控制、错误和返还 credit 的路径不得被普通 payload 永久堵死。

## 6. 控制面落地条件：保留当前语义，不恢复旧架构

### 6.1 准入与资源退休

- CPU 接受 root 先占 pending metadata，不等于占 Group execution slot/L2 Arena；`ACTIVE` 与 Group 实际 `active_cycle` 不是同一观察点。
- Root 准入要将 slot、Arena、event/control 预算一起准备好；失败不能泄漏部分资源。
- Grid 是有限登记对象，Task 按 Tile 独立补位。单 Task 提交前可以回滚准备，已提交 Task 只能按真实终结或 fault/drain 退休。
- `input_released`、`output_ready`、`grid_done` 分开聚合；R lease 直到 Task 安全退休才归还。
- `nest.release` / `tile.free` 失效的是 view；owner 安全退休且 pin/inflight/共享引用关闭后才返还 backing/Arena。共享只读 backing 可长于 producer root 生命周期。

这些是当前源码合同，而不是新的全 Grid 原子提交方案。[S2]–[S5]

### 6.2 有界表、同步存储和唯一 owner

当前默认容量见 `config.py`：CPU pending 32/completion 4096；Group active 8/pending 32/action 16/scan 4/event 4096/inflight 32；prefetch/store credit 各 8、dispatch 16；每 Tile UCE 配置 1–8。它们是模型参数，不是已综合上限。[S12]

[INFERENCE] 硬件需为每张表冻结 entry 宽度、深度、端口、更新优先级、回绕与表满行为。Group/Tile 本地 scoreboard 和有限完成 FIFO 比一个全局任意 fan-in CAM 更容易收敛。多源同拍完成应先入队仲裁，不允许覆盖活跃事件。

尤其不能声称模型已经消除了所有共享写：**StreamQueue 是同一个 Python 对象，producer acquire/push 与 consumer pop/release 修改共同 FIFO/credit 状态，安全依赖单线程 step 顺序。**硬件必须由 queue/credit 单元统一更新，或定义等价的分布式 credit 协议；不能让 RISC-V、UCE 和 DMA 直接并发加减同一权威计数。

### 6.3 发射、故障与版本

- UCE ingress 接受、engine 接受、engine 完成是三个边界；event 身份建立与入队提交必须一致，背压期间命令不变且不得重复发射。
- event/phase/task/program/arena/Profile 都要携带对应身份与 generation，不能只比较裸 ID。
- 程序运行期间 text 不可原地改写；新版本装载、验证后供新 launch 使用，旧任务保持原绑定。
- 当前有 drain/reset FSM 和 cancel-confirm 参考；硬件须等待真实 accepted job、MSHR、NoC、DMA 与写响应闭合，再允许资源复用。
- 跨时钟命令/完成用事务级 CDC/异步 FIFO；多 bit 总线逐位同步不保证同一条记录一致。reset 释放、epoch 和迟到 completion 必须一起设计。

上述规则可以综合为硬件，但模型的 Python callback 顺序、`dict.clear()` 或固定 cycle 数不能作为物理正确性的替代。[S4][S5]

## 7. 全系统资源、带宽、功耗与平台判断

### 7.1 当前 target 的规模，而非旧报告的控制器预算

以下为默认 `hardware_config.yaml` 的模型参数与算术推导，不是 FPGA 测量。[S12]

| 项目              | 当前模型配置/推导                                           | 硬件含义                                                             |
| ----------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- |
| 拓扑/时钟         | 1 Group × 4 Tile；core/UCE 1 GHz                            | 1 GHz 是模型基线，不能当 FPGA 已达频率                               |
| 数据 SRAM         | L2 8 MiB + 4×L1 1 MiB = **12 MiB**                          | 尚未计 tag/ECC、控制表、CPU、buffer；Profile 内 SPM/Cache 不重复相加 |
| Bank              | L2 16；每 Tile L1 16，共 80 个成员                          | 容量放得下不代表端口和布线带宽足够                                   |
| Tile Program 容量 | 每 Tile 64 KiB，共 256 KiB                                  | 当前 text size 还是估算，硬件编码后须重新检查驻留预算                |
| mode3 Cache 数据  | L1 合计 2 MiB + L2 4 MiB = 6 MiB                            | 已包含在 12 MiB 内；tag、替换、MSHR 另计                             |
| MSHR              | 每 Tile L1 16；L2 64                                        | entry 深度不决定 waiter/payload 队列容量，后者仍需预算               |
| 外部内存          | 16 GiB、8 模型通道、819.2 GB/s、32 outstanding              | 物理控制器组织/实际带宽须由板卡 target 明确                          |
| 片上带宽          | L2 每 bank 64 GB/s，即合计 1024 GB/s；L1 每 Tile 512 GB/s   | L1 的 512 是每 Tile，不是再乘 16 bank；都不是实测带宽                |
| BOA 并行度        | 每 Tile 8×16×16 = 2048 MAC/cycle；全 Group 8192             | 理想 1 GHz、2 ops/MAC 得 16.384 TOPS；不是数值实现或持续性能         |
| EVU / USE         | 每 Tile EVU 32 lanes；USE 500 MHz、128 KiB state-cache 参数 | 不代表已实现向量算术或实际 state RAM 数据通路                        |

**[INFERENCE] 主要瓶颈会同时来自 RAM 端口、Cache tag/替换、DSP/累加存储和跨 Tile 宽总线。**只统计 BRAM 字节或 LUT，不足以判断是否能放下。

例如假设 FPGA 为 200 MHz，要维持模型 L2 每 bank 64 GB/s，需要每 bank 每拍 320 B，即 2560 bit 数据服务宽度；同频下每 Tile 512 GB/s 需要合计 20,480 bit/cycle。这个推导说明不能仅把 `clock_mhz` 改小，却保留所有 GB/s 和一拍访问假设。

### 7.2 有具体载体，但没有“默认参数原样落板”的证明

以官方公开的 AMD Alveo U55C 为能力参照：[H2]

| 官方项目      | 公布规格                           | 对本设计能说明什么                                                                 |
| ------------- | ---------------------------------- | ---------------------------------------------------------------------------------- |
| HBM2          | 16 GB，460 GB/s                    | 外部容量接近模型量级，但公布带宽低于模型 819.2 GB/s；GB/GiB 精确容量仍需核对       |
| Internal SRAM | 43 MB                              | 原始容量大于 12 MiB 数据 SRAM，但 shell、CPU、tag、复制和端口碎片还要占用          |
| DSP slices    | 9,024                              | 不能按“一 DSP = 一 BF16 MAC”直接认定能放下 8192 个 MAC；需实际算术映射与利用率预算 |
| 板卡功耗      | typical 115 W，maximum total 150 W | 是厂商整卡规格，不是本设计预计或实测功耗                                           |

这是**平台存在性证据，不是推荐采购结论**。高端 FPGA + RISC-V 软核存在承载全系统原型的可能；具体 fit/Fmax 必须拿所选数值格式、Cache 组织、RAM 端口与 shell 做综合/P&R 后判断。带 RISC-V 硬核的 SoC FPGA 也可评估，但“自带 RISC-V”不保证数据面 SRAM/DSP/HBM 足够。

[INFERENCE] 可以时分复用 MAC、降低 lane 数、改用 DDR 或更窄互连来实现功能；但默认 target 性能随之改变。将 L1/L2 外置 DDR 还改变层级竞争，不能只当容量扩展后继续引用原来的 SRAM 延迟。

### 7.3 功耗、布线与成本的新口径

完整范围下，功耗必须计入计算 DSP、SRAM/Cache、DDR/HBM、PHY、时钟、RISC-V、控制与 debug；不能沿用只含控制面的 10–30 W 预留。模拟器 utilization 不能换算瓦数。

[INFERENCE] 物理收敛优先事项：Tile 本地化、bank 级 pipeline、completion 分层聚合、少搬完整 descriptor、多传引用；单独预算跨 die/区域连线、clock/reset fanout 和 trace 汇聚。带 debug 的实现也要检查时序，不只检查去掉 probe 的理想版本。

本次没有本设计 LUT/FF/DSP/BRAM 综合报告，也没有板上电源测量，**不提供新的总瓦数、确定人月或固定板卡报价**。旧的 8–15 人月/4–8 个月仅针对控制面，不能迁移为完整系统工期。成本应按第 10 节工作包，结合数值支持集、IP 复用、板卡/工具授权和验证范围重新估算；初次使用现成板卡可避免另起 PCB/DDR/PDN 项目。

## 8. 软件产物到硬件 ABI：当前已有基础，但仍缺最后一段

`CompiledProgram` 的不可变、严格 JSON、hash/Registry/target、call binding、relocation、resource budget 与独立 Loader 是有价值的软硬件边界。当前源码脱离编译器的执行复放已可运行，不需要 FPGA 上再做 lowering。[S1]

但 `compiler/api.py::_finalize_program_identities` 的 `text_bytes` 仍按指令数和 descriptor 数估算；没有由真实 opcode/operand 编码生成的设备 text 镜像。软件 artifact ABI v3 不能直接作为硬件 ring/指令 ABI。[S11]

[INFERENCE] 需要补齐：

1. 硬件 command/event/descriptor/program image 的版本、大小、对齐、端序、地址宽度与 relocation；所有未定字段由后续规格冻结，不在本报告杜撰编码。
2. 设备 capability 与编译 target 对齐：不支持的 dtype/op/Profile/队列上限在编译或装载时拒绝，不静默降档。
3. CPU 发布 descriptor → 数据 cache maintenance/memory fence → doorbell；设备完成 → Host 读取结果的对应 acquire/可见性边界。
4. resident handle 与 program ID/version/hash/epoch 绑定；真实 fetch/verify/install、running text 保护和错误装载路径。
5. 把 Python Host routine 改成明确的 Host 服务/RISC-V 固件 ABI，覆盖页分配、动态元数据更新、失败与重试的既有语义，不在每个 index 上做远程回调。

`runtime/host_runtime.py`、`kernel_driver.py`、`firmware.py` 中的延迟/计数封装，不是已交付的 OS driver、RISC-V firmware 或 PCIe doorbell 实现。不要因目录和类名齐全而低估这部分工作。

## 9. 本次实际运行证据与证明边界

### 9.1 六次当前场景运行

全部使用 `elenor-validator` 环境、当前 launcher、`fidelity=full_memory` 与 memory trace。前五次导出 trace JSON，PagedAttention 读取 report/PMU，未导出其完整 trace。结果来自本次运行，不沿用旧报告周期。[S13][S14]

| 场景                                    | 模型 cycles | memory transaction 完成/发出 | 本次观察                                                                                         |
| --------------------------------------- | ----------: | ---------------------------: | ------------------------------------------------------------------------------------------------ |
| `gather`                                |       1,210 |                      76 / 76 | 16 个真实 index 读取、16 个 miss 请求、12 次 MSHR merge、256 B Gather；不是 profile 人工指定命中 |
| `indexed-memory`                        |       1,710 |                      24 / 24 | 3 个 Gather 请求、3 个 Scatter 段，各 48 B；重叠写等待 225 cycles                                |
| `profile-reconfiguration`               |         403 |                        0 / 0 | 最终 L1/L2 mode2、generation 各 1；运行中 L1 Prepare/Commit 各 64 ACK，L2 各 16 ACK              |
| `ready-action-branch --group-policy s0` |      16,835 |                        8 / 8 | fast BOA 在 cycle 16,717 开始；slow EVU 在 16,507 结束                                           |
| `ready-action-branch --group-policy s1` |      16,604 |                        8 / 8 | fast BOA 在 cycle 150 开始，slow EVU 到 16,506 才结束；ready-action 允许独立分支补位             |
| `paged-attention-decode`                |     169,588 |            269,995 / 269,995 | 27 次 Host call 全部完成；98,592 个 Gather 请求、96 个 Scatter 段；全流程完成                    |

六次均 `completed=true`、credit 检查通过、Arena 零泄漏检查通过；有 Gather/Scatter 的报告相应守恒和 Gather 零泄漏检查也通过。各次事务 cancelled/faulted 均为零。**这些正常路径没有验证错误注入、reset 故障恢复或硬件 CDC。**

PagedAttention 使用已提交场景：初始长度 `[255,511,767]`、4 steps、16 tokens/page、128 physical pages、placement 15、每 Tile 4 UCE context、CPU outstanding 8、L1 mode3/L2 mode1。其 Gather 分类为 L1 hit 73,728、L2 hit 11,602、HBM miss 13,249、bypass 13，总和等于请求数；这是当前 Cache 组织下的模拟统计。

Profile 场景没有数据事务，证明正常切档/ACK 路径，**不证明 dirty write-back、带流量切档或初始化耗时**。默认导出 trace 不含 pre-run Profile 初始化，不应将表中 ACK 数当成从上电开始的总数。S0/S1 的 engine 区间仍是时序模型，不能理解为真实 BOA/EVU 同拍完成数值计算。

### 9.2 额外字节与 compiled replay 核对

对 `indexed-memory` 的本次持久化 artifact，另用独立 Python 进程调用 `parse_compiled_program → load_program → Simulator.run`，读取保存的 target YAML 和原始输入 bin，**不 import compiler、不重新读取 MLIR**：

- Gather indices 为 `(3,0,3)`，Scatter indices 为 `(1,1,2)`。
- 最终 OUT row1 为 `[10,11,12,13]`，说明后序重叠写覆盖了先序写；row2 为 `[40,41,42,43]`。
- 未写入的 row0/row3 仍触发 uninitialised-byte 错误，没有被隐式补零。
- trace 中 Scatter ordinal 0、2、1 分别在 cycle 1301、1502、1703 提交；done 在 1703，既保留重叠覆盖顺序，也允许不重叠段提前完成。
- replay 的产物版本为 schema 3 / ABI v3，`sys.modules` 未加载 `pipeline_validator.compiler`。

这证明了真实字节搬运/覆盖与软件编译执行边界，**不证明 BOA matmul、EVU softmax/pow 或 attention 输出数值正确**。本次没有运行全套 pytest，也没有 FPGA 综合、RTL 仿真或板上测试。

### 9.3 复现与证据保存

本次报告、前五个 trace、compiled artifact 及 target snapshot、源码 SHA-256/evidence manifest 保存在 `/tmp/nexus-fpga-review-20261010/`。这是本机临时证据，不是仓库永久交付；可用下列命令重新产生。重跑前须确认源码版本，不能将新周期与旧模型参数混算。

```bash
D=/tmp/nexus-fpga-review-20261010

bash examples/run.sh gather --sim-override fidelity=full_memory --memory-trace \
  --trace-json "$D/gather.trace.json" --compiled-output "$D/gather.compiled.json" \
  --report "$D/gather.report.json" --json
bash examples/run.sh indexed-memory --sim-override fidelity=full_memory --memory-trace \
  --trace-json "$D/indexed.trace.json" --compiled-output "$D/indexed.compiled.json" \
  --report "$D/indexed.report.json" --json
bash examples/run.sh profile-reconfiguration --sim-override fidelity=full_memory --memory-trace \
  --trace-json "$D/profile.trace.json" --compiled-output "$D/profile.compiled.json" \
  --report "$D/profile.report.json" --json
for P in s0 s1; do
  bash examples/run.sh ready-action-branch --group-policy "$P" \
    --sim-override fidelity=full_memory --memory-trace \
    --trace-json "$D/$P.trace.json" --compiled-output "$D/$P.compiled.json" \
    --report "$D/$P.report.json" --json
done
bash examples/run.sh paged-attention-decode --sim-override fidelity=full_memory \
  --compiled-output "$D/paged.compiled.json" --report "$D/paged.report.json" --json
```

## 10. 实施工作包与完整交付验收

以下是 [INFERENCE] 工程路径，不是把某一中间阶段冒称最终完成。当前证据支持进入规格收敛与实现探索，不支持直接给出已完成 RTL 的结论。

| 工作包                | 必须交付                                                                           | 退出标准                                                                          |
| --------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| 数值/指令/target 合同 | 各 engine 显式操作数、支持集、数值规则、Cache 组织、能力上限、硬件 ABI             | 编译输出不再依赖计时占位来表达真实计算；软件 oracle 可生成明确期望                |
| RISC-V 与 Host 通路   | 程序装载、命令/完成环、页管理、grant、cache 可见性、恢复                           | 实际软/硬核执行固件；同一 bitstream 装载不同程序，无 Python callback 代执行       |
| 调度与存储数据面      | Root/Task 准入、UCE、event/stream、Arena、Profile、DMA、Cache/MSHR、Gather/Scatter | 随机背压、乱序完成、重复/陈旧事件、部分 Grid 故障、资源守恒与字节 oracle 对比通过 |
| 真实计算与 Collective | BOA/EVU/USE 及约定 Collective 数据路径                                             | 多 shape/dtype/tail/累加/状态场景数值对照；不能只比较 done 或 latency             |
| 全系统集成            | Host 页管理 → KV append → Gather → QK/softmax/PV → 输出与回收                      | PagedAttention 等程序的数据、顺序、生命周期和数值闭环；不以预种子输出代替计算     |
| 物理与板上验收        | 目标器件综合、P&R、CDC/RDC、带 debug 时序、实际接口和功耗                          | 报告 LUT/FF/DSP/RAM、WNS/Fmax、持续带宽、dispatch/wakeup 延迟、板上输入功率与温度 |

控制协议可先用可注入延迟/乱序/错误的 endpoint 验证，算术可分模块对照 oracle；这些是验证夹具，不能作为完整加速器交付。规则算术可以用 HLS/成熟 IP，调度、Cache 一致性、资源生命周期和 CDC 仍需逐项约束与验证。

最后应以**数值/字节结果、因果顺序、资源守恒、隔离与有界前进**判定正确性；用真实硬件延迟更新模拟器。没有必要追求与 Python 的每一个 cycle 相同，更不能冻结虚拟时间来隐藏固件开销。

**最终判断：可以用 FPGA + RISC-V 实现当前整体设计思路；后续改动显著增强了控制、索引访存和资源协议的可落地性。要实现完整的可计算系统，下一优先级是补计算操作数/数值合同、Cache 物理组织与端到端 ABI，而不是继续把所有 engine 当外部黑盒后宣布可行。默认配置的性能、单板 fit、功耗和工期仍必须由真实实现验证。**

## 11. 证据索引

### 11.1 当前源码与运行输入

| 编号 | 文件/入口                                                                                                                                                                                                                                                                                                                                                 | 本报告使用的依据                                                                      |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| S1   | [compiler/api.py](../pipeline_validator/compiler/api.py)、[compiled_program.py](../pipeline_validator/compiled_program.py)、[loader.py](../pipeline_validator/loader.py)、[execution_verifier.py](../pipeline_validator/execution_verifier.py)、[cli.py](../pipeline_validator/cli.py)                                                                    | compile/load 分离、schema/ABI、严格产物、独立校验与复放                               |
| S2   | [device.py](../pipeline_validator/device.py)、[runtime/group_port.py](../pipeline_validator/runtime/group_port.py)、[simulator.py](../pipeline_validator/simulator.py)                                                                                                                                                                                    | CPU pending/dependency、Group slot、Host 与 Group 主循环                              |
| S3   | [group_scheduler.py](../pipeline_validator/group_scheduler.py)、[tile_group_sequencer.py](../pipeline_validator/tile_group_sequencer.py)、[tile_group.py](../pipeline_validator/tile_group.py)                                                                                                                                                            | ready-action、barrier、逐 Tile 准入、Grid/phase/R lease                               |
| S4   | [tile.py](../pipeline_validator/tile.py)：`_select_context`、`_issue_context`、`_enqueue_engine_launch`                                                                                                                                                                                                                                                   | eligible-head RR、控制占位、engine ingress                                            |
| S5   | [runtime/event_table.py](../pipeline_validator/runtime/event_table.py)、[stream_queue.py](../pipeline_validator/stream_queue.py)、[runtime/reset_domain.py](../pipeline_validator/runtime/reset_domain.py)、[runtime/program_table.py](../pipeline_validator/runtime/program_table.py)                                                                    | 事件/credit、drain/reset、program residency                                           |
| S6   | [engines.py](../pipeline_validator/engines.py)：BOA/EVU/USE `latency`、MFE `launch_gather`/`launch_scatter`、Gather/Scatter tick                                                                                                                                                                                                                          | 时序算术与真实索引搬运的区别                                                          |
| S7   | [memory/address_provider.py](../pipeline_validator/memory/address_provider.py)、[memory/byte_store.py](../pipeline_validator/memory/byte_store.py)、[memory/transfer.py](../pipeline_validator/memory/transfer.py)、[memory/noc.py](../pipeline_validator/memory/noc.py)                                                                                  | 求址、oracle、read/write 提交、route 与物理简化                                       |
| S8   | [memory/cache.py](../pipeline_validator/memory/cache.py)、[memory/mshr.py](../pipeline_validator/memory/mshr.py)、[compiler/profile_pass.py](../pipeline_validator/compiler/profile_pass.py)                                                                                                                                                              | LRU/identity、MSHR、维护和写范围合同                                                  |
| S9   | [profiles.py](../pipeline_validator/profiles.py)、[memory/profile_controller.py](../pipeline_validator/memory/profile_controller.py)、[memory/arena.py](../pipeline_validator/memory/arena.py)、[compiler/resources.py](../pipeline_validator/compiler/resources.py)                                                                                      | Profile ACK/generation、bank/Arena 预算与生命周期                                     |
| S10  | [memory/page_pool.py](../pipeline_validator/memory/page_pool.py)、[runtime/host_session.py](../pipeline_validator/runtime/host_session.py)、[paged_attention.py](../pipeline_validator/paged_attention.py)                                                                                                                                                | 页与 scope、Host 事务、PagedAttention handler；lease API 与设备接线的区别             |
| S11  | [dialects/elenor.py](../pipeline_validator/dialects/elenor.py)：`TileBoaOp`/`TileEvuOp`/`TilePowOp`；[compiler/lowering.py](../pipeline_validator/compiler/lowering.py)：`_lower_engine_descriptor`                                                                                                                                                       | 当前计算 descriptor 缺少显式 tensor 输入/输出合同                                     |
| S12  | [hardware_config.yaml](../pipeline_validator/hardware_config.yaml)、[config.py](../pipeline_validator/config.py)                                                                                                                                                                                                                                          | 默认容量/带宽/几何、独立 Device/Group/Tile 容量                                       |
| S13  | [examples/run.sh](../examples/run.sh)、[indexed_gather_scatter.mlir](../examples/workloads/indexed_gather_scatter.mlir)、[gather_indexed.mlir](../examples/workloads/gather_indexed.mlir)、[ready_action_branch.mlir](../examples/scenarios/ready_action_branch.mlir)、[profile_reconfiguration.mlir](../examples/scenarios/profile_reconfiguration.mlir) | 本次 smoke 的 launcher、源 IR 与参数                                                  |
| S14  | [paged_attention_decode_scenario.json](../examples/workloads/paged_attention_decode_scenario.json)、[paged_attention_decode_pipeline.mlir](../examples/workloads/paged_attention_decode_pipeline.mlir)                                                                                                                                                    | 默认 PagedAttention 几何、模式与源 hash                                               |
| S15  | [Limitation.md](../pipeline_validator/Limitation.md)、[README.md](../pipeline_validator/README.md)                                                                                                                                                                                                                                                        | 单 Group/fidelity/初始化 trace 边界；Gather/schema 部分文字滞后，以源码和本次产物为准 |

### 11.2 架构意图与外部能力参照

- [A1] [BOA 设计](../design/elenor_boa/ELENOR_BOA_Design.md)：数值格式、operand fetch、accumulator、writeback 的设计意图，不是当前数值模型实现。
- [A2] [Tile UCE 设计](../design/elenor_tile_uce/ELENOR_Tile_UCE_Design.md)、[Driver/Firmware/Runtime 设计](../design/elenor_driver_firmware/ELENOR_Driver_Firmware_Runtime_Design.md)：可加载程序、ownership、doorbell/patch/residency 草案；与当前模型区分成熟度。
- [H1] [NEORV32 官方项目](https://github.com/stnolting/neorv32)：可综合 RISC-V 管理软核及总线/软件生态；非本设计资源证明。
- [H2] [AMD Alveo U55C 官方规格页](https://www.amd.com/en/products/accelerators/alveo/u55c/a-u55c-p00g-pq-g.html)：本次读取的 HBM、SRAM、DSP 与整卡功耗规格；非本设计实测或 fit 保证。

[S1]: ../pipeline_validator/compiled_program.py
[S2]: ../pipeline_validator/device.py
[S3]: ../pipeline_validator/group_scheduler.py
[S4]: ../pipeline_validator/tile.py
[S5]: ../pipeline_validator/runtime/reset_domain.py
[S6]: ../pipeline_validator/engines.py
[S7]: ../pipeline_validator/memory/address_provider.py
[S8]: ../pipeline_validator/memory/cache.py
[S9]: ../pipeline_validator/memory/profile_controller.py
[S10]: ../pipeline_validator/memory/page_pool.py
[S11]: ../pipeline_validator/compiler/lowering.py
[S12]: ../pipeline_validator/hardware_config.yaml
[S13]: ../examples/run.sh
[S14]: ../examples/workloads/paged_attention_decode_scenario.json
[S15]: ../pipeline_validator/Limitation.md
[A1]: ../design/elenor_boa/ELENOR_BOA_Design.md
[A2]: ../design/elenor_tile_uce/ELENOR_Tile_UCE_Design.md
[H1]: https://github.com/stnolting/neorv32
[H2]: https://www.amd.com/en/products/accelerators/alveo/u55c/a-u55c-p00g-pq-g.html
