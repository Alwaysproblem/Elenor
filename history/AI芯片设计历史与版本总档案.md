# AI 芯片设计历史与版本总档案

整理日期：2026 年 10 月 28 日
已恢复的讨论时间范围：2025 年 8 月至 2026 年 10 月 28 日

这份档案按**历史方案、同一方案的修订、独立探索分支和关键决策**整理你的 AI 芯片设计讨论，不只总结最新 Nexus。被放弃、被收缩、尚未采纳的设计也保留，并说明它们解决什么问题、后来为什么变化。

目前可以恢复的长期主线是：从工业 NPU 的重 runtime 和粗算子调度出发，探索强编译器、显式异步、片上数据驻留、分层控制和可编程计算；经过 Island/Path、BOA/Elenor、独立 CORA 探索，再收敛到 SPMD 与两级多 Context；随后在 NEST/Nexus 中逐步补全资源准入、存储 profile、输入输出生命周期、计算后端和 Shared AM。**这不是一条连续改名链，也没有证据表明历史中每个参数、模块或助手提案都被你采纳。**

## 阅读导航

- [一 档案范围与版本判定](#scope)
- [二 设计与版本总目录](#inventory)
- [三 历史时间线](#timeline)
- [四 逐设计档案](#design-records)
- [五 历史命名与独立方案补录](#named-branches)
- [六 硬件方案与参数沿革](#hardware-history)
- [七 跨版本决策日志](#decision-log)
- [八 编程层次与执行合同](#execution-contracts)
- [九 存储系统与动态调度](#memory-scheduling)
- [十 计算后端与 Shared AM](#compute-am)
- [十一 Irregular 多设备与可观测性](#irregular)
- [十二 编译器与验证路线](#compiler-validation)
- [十三 未定问题与易混淆事项](#open-issues)
- [十四 覆盖范围与原始来源](#sources)

<a id="scope"></a>

## 一 档案范围与版本判定

### 这份档案怎样使用版本一词

1. **原始名称保留。**CORA、HSTI-1、NEST、Nexus、Shared AM V1 等使用对话中实际出现的名称；不另造“V0/V1/V2”充当官方架构编号。
2. **没有正式编号就用日期和描述。**例如“2025 年 8 月单 L2 bytecode 方案”“2026 年 10 月 17 日四个独立 8×8 方案”。日期表示本次找到的讨论时间，不保证是创意首次产生时间。
3. **同名也可能有不同层次。**2026 年 8 月的 Nexus 先是上层提交/编排 IR；后来的 Nexus 项目包含更广架构。不能把它简单写成 NEST 的新名字。
4. **独立分支不强行继承。**CORA 被要求不默认继承 Elenor；7 月末新讨论又明确排除 CORA。RA-X1、HTFA 等助手提案单独归档，不自动成为个人项目的后续版本。
5. **历史源码判断绑定时间。**当时对某个仓库分支的分析，是当时实现的证据，不等于今天代码状态；本次没有重新运行这些历史代码或重新验证 RTL。

### 状态标签

| 标签         | 含义                                                                |
| ------------ | ------------------------------------------------------------------- |
| 用户明确决定 | 你直接提出、确认或纠正的设计要求                                    |
| 用户探索     | 你提出了问题或候选方案，但未找到最终采纳证据                        |
| 助手提案     | 对话中的建议、命名、实现细化或示例；不能自动视为你的决定            |
| 历史实现分析 | 当时对代码或模型的分析，不代表本次重新核验                          |
| 被替代或收缩 | 有后续讨论明确调整了该方案的适用范围                                |
| 待验证       | 有目标、推导、模拟或建议，但还没有足够的数值、协议、RTL 或 PPA 证据 |

本文优先保留“用户原话 → 当时建议 → 后续修正”的顺序。性能、功耗、面积、可发表性及工期分别注明证据等级，不把推测改写成事实。

<a id="inventory"></a>

## 二 设计与版本总目录

以下是架构层目录；硬件内部候选与参数另见第六节。每项均有后文档案或来源，可用原始名称继续检索。

| 讨论时间       | 原始名称或描述                            | 类型与归属                         | 核心内容                                                                        | 状态与后续                                                    |
| -------------- | ----------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 2025-08        | 工业 NPU 两级解释器提案                   | 助手针对用户旧公司架构提出的改造案 | L2/L1 各一层 bytecode，异步 DMA、事件、乒乓缓冲                                 | 很快被用户否定“两份 bytecode”部分                             |
| 2025-08        | 单 L2 bytecode 方案                       | 用户明确修订                       | L2 常驻解释器；L1 固定固件/微内核消费 descriptor                                | 当时旧架构基线；不约束后来 Tile program                       |
| 2025-08-31     | 分页 bytecode                             | 用户明确选择                       | DDR 程序、DCCM page cache、segment/page                                         | 替代滑动窗口取指候选；当时代码草案并非部署证明                |
| 2025-09-04     | 离线 microkernel family                   | 用户明确编译策略                   | 离线挑选/调优，运行编译时 lookup 与 tiling                                      | 不是对后来全部 cost-model 优化的禁令                          |
| 2026-03        | 驻留与数据移动导向的“第四范式”讨论        | 概念探索，名称来自助手表述         | 把局部性、数据驻留和搬运作为调度核心，探索近 SRAM 轻量处理                      | 非已实现芯片，也不是正式版本号                                |
| 2026-03        | Island/Path 与“第五范式”讨论              | 单芯片架构探索                     | Island 作为芯片内 cluster；局部资源与调度，Path 编排跨岛阶段                    | 与 Flat SPMD 比较的收益仍待验证                               |
| 2026-03～04    | BOA 与 reduce network                     | 用户澄清的计算核心及扩展探索       | Block Outer-Product Accumulator；K 维分块累加；向量单元；动态 gather 补强       | 后续 Tile 计算主线的前驱，非完整冻结芯片配置                  |
| 2026-05～07    | Elenor / ELENOR 与可编程 Tile             | 既有个人架构及回顾                 | BOA、EVU、MFE、UCE、L1；Group Scheduler、Region Sequencer、L2；分层 IR          | 已含 Tile-SPMD；不能一概归为纯 MPMD                           |
| 2026-06-11     | Micro-SIMT EVU                            | 用户提出 SIMT 诉求；名称由助手给出 | 不复制 GPU warp scheduler，探索轻量执行前端、mask 与分支                        | 与传统 SIMD、后来自主 RVV 属不同备选                          |
| 2026-06-16     | SuperRegion 与 Pattern Region             | 助手激进案与用户倾向的收缩案       | 任意 tile-dataflow → compute-heavy pattern 模板化流水与粗粒度异步               | 控制跨算子重排与全局 buffer 复杂度                            |
| 2026-06-30     | LTF 放入 MFE                              | 用户明确位置选择                   | memory system 管 transformation，DMA 与 LTF data pipeline 分开                  | 不能与 10 月有限 LTU 候选自动视为同一实现                     |
| 2026-07-14     | B1 + Multi-Context                        | 单 Big Tile 的独立对照案           | 中央多 Context UCE、共享宽引擎、banked SRAM                                     | 不等于 NEST 论文中的 B1 L2-MC 实验标签                        |
| 2026-07-15     | TCP-inspired TCC                          | 助手局部重构提案                   | 保留 Elenor 外层，改造 BOA+EVU 为 contraction/vector chain/local feed           | 不是完整 TCP 替换整机                                         |
| 2026-07-21～26 | CORA                                      | 用户接受命名的独立架构             | Coroutine/context、Matrix/Vector/LSU、计算驱动访存；进一步探索 L2 multi-context | 明确不默认继承 Elenor；MC-PGE 属助手细化                      |
| 2026-07        | RA-X1、HTFA、HFC、EFP 等命名提案          | 助手的独立备选方案                 | 从汽车机器人 SoC、异构路径到通用推理组织的不同尝试                              | 不等于全部采纳，不拼入 Nexus 固定配置                         |
| 2026-07-29～31 | HSTI-1 与 SPMD 收缩                       | 新推理芯片提案；SPMD 是用户决定    | 地址、访存、变换、归约、路由、提交的逻辑抽象；放弃通用 Tile 间不同算子流水      | 保留 Tile 内 async，降低编译器压力                            |
| 2026-07-31～08 | NEST                                      | 论文机制提案，后成为明确讨论基线   | L2 Context + Tile Context；SPMD epoch；分层资源契约                             | 三事件与生命周期成为持续主线；早期单 image epoch 不能无限外推 |
| 2026-08-07     | NEST 简洁 IR 与完整输出生命周期修订       | 用户纠错及接口简化                 | `nest.context` / `nest.tile.program`；输出 store 完成后才释放                   | 不要求把所有 pool/lease/credit 机制暴露到用户 IR              |
| 2026-08-10     | Context-owned L1/ACC slots                | 对通用 FU 间 FIFO 的收缩提案       | 大 tensor 驻留命名 slots；command queue 传 descriptor/token                     | 小 fast path 可选，通用 stream network 不是必要条件           |
| 2026-08-20     | Nexus 上层提交 IR                         | 用户直接给出程序框架               | `nexus.program`、异步提交多个 NEST context、await                               | 当时与 NEST 是上下层关系                                      |
| 2026-08～09    | Cache/SPM 双模式与 memory profile         | 用户逐步明确的存储版本             | 同级 bank 内统一比例；一个 Context 一个 profile；HBM 安全切档                   | FIFO 三模式、活数据在线改档未被确认为当前基线                 |
| 2026-09        | 有限 ready-action 与 memory admission     | 分层调度细化方案                   | Group 提交/执行解耦；Tile eligible heads；completion-driven 补位                | 有限资源与前进性约束仍须联合验证                              |
| 2026-09～10    | Nexus 模型与 real-gather 相关实现快照     | 历史实现分析                       | 生命周期、资源/profile、有限队列、真实 index 搬运等                             | 不把模型当完整数值计算或硬件实测                              |
| 2026-10-01～05 | 有限 Group 控制流与 Context ownership     | 用户明确收窄                       | Group 不做 if/else；增加 for；shared 发布后只读；当时无 atomic                  | 动态分支由 CPU；可写 move 暂无首版收益证明                    |
| 2026-10-17     | Tile Shared 与四个独立 8×8                | 用户明确硬件方向                   | 不复制大块私有输入 SPM；四阵列无互联/广播，当时各自 AM                          | AM 后来转向共享；四阵列不能与旧八阵列参数混合                 |
| 2026-10-19～20 | ring、Reduce Memory、reduce tree          | 用户探索及助手细化                 | 并行 K 分片与部分和汇聚，解耦源 AM 驻留                                         | 后续转向先不用 tree 的顺序 K-step 主线                        |
| 2026-10-20     | 顺序 K-step 与 AM 持久累加                | 用户方向调整                       | FIRST 覆写、后续累加；阵列可穿插不同 Context，旧 C 留 AM                        | 与 PE 内 OS 及 split-K 并行归约严格区分                       |
| 2026-10-21～23 | RISC-V 多 Context、自主 RVV、公共任务管理 | 用户需求与未定微架构备选           | 多程序独立 PC、异步任务继续；设备自主 kernel；共享 Task/Event 管理              | 尚未选定最终核、完整 ABI 或多向量 kernel 驻留能力             |
| 2026-10-26     | Shared AM V1                              | 用户明确首版约束                   | 原 engine affinity；每 Tile Context 跨所有 matrix engines 最多两个 AM slots     | 需求指南阶段，未证实已改仓库或通过 RTL 验收                   |
| 2026-10-28     | 二维 DMA 与间接 AG                        | 后续开放性能问题                   | 区分块间 irregular/块内 regular 与逐元素随机 gather                             | 请求率、outstanding、有效带宽仍待评估                         |

主要来源：[工业起点与单 L2 修订](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=ba2698fc-65fa-437e-85ea-e37497dc58a7) [来源 S04](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=36fc1de6-01c5-4528-9edc-a863d3b4df40)、[早期 Island/BOA](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)、[CORA](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)、[SPMD](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=b628a6f0-c936-44d1-b572-f623c2bb989b)、[NEST](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)、[Nexus IR](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=cee8808a-d227-4db9-8bbf-c63b41cc8dc4)、[Shared AM V1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)。

<a id="timeline"></a>

## 三 历史时间线

### 2025 年 8 月至 12 月 从重 runtime 到程序化调度

- **8 月 20 日**：你描述 ARM Host、自研 L2 controller（DCCM/CSM）、12 个 L1 controller/12 个 NPU slice；每个 op 对应模型算子，runtime 太重。目标是异步、乒乓 buffer 和更细粒度 compiler 优化。[来源 S03](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=ba2698fc-65fa-437e-85ea-e37497dc58a7)
- **8 月后续修订**：助手最初提出 L2/L1 双 bytecode；你明确要求模型只生成 L2 bytecode，由 L2 控制 L1。8 月 30 日综合稿仍沿用单 L2 路线。[来源 S04](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=36fc1de6-01c5-4528-9edc-a863d3b4df40)
- **11～12 月**：讨论 IREE/MLIR 优化复用、Async token/value、CUDA Tile IR 验证与 LLVM IR 提早 load/重排。它们是工具链研究，不意味着整套 runtime 已被采用。[来源 S05](https://chatgpt.com/c/6921c3c8-5d1c-8330-a56c-3a01baf31510?messageId=39a6ef82-005a-4bd7-9ce4-6e9915dbc618)[来源 S06](https://chatgpt.com/c/694b8c19-4548-8327-9fdf-fb16eb74371b?messageId=243bf07b-903d-4aa7-a5c6-28ee3211ff30)[来源 S07](https://chatgpt.com/c/694fec5a-ddb8-832b-aa85-ae7544739e00?messageId=c745f5c4-4d6d-4cf7-b075-643276d38750)

### 2026 年 3 月至 7 月早期 从单芯片组织到独立架构分支

- **3 月 13 日**：区分图分段/fusion、重算子主导 tiling、pipeline 和内存规划；autotuning 是候选选择，不等于 tiling 本身。[来源 S56](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)
- **3 月 21 日**：你纠正 Island 指单芯片内的 cluster，不是多机集群；随后探索 Path 编排、local scheduler、事件与驻留。[来源 S56](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)
- **3 月 25 日**：你仍不确定 Island/Path 相对纯 SPMD 的收益，要求快速实验。建议用同一成本模型比较三类组织，而不是预设胜出。[来源 S56](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)
- **3 月 27 日**：你明确 BOA 是 Block Outer-Product Accumulator，reduce 主要用于 K-tiling 累加；3 nm、单芯片 1 PFLOPS FP16 / 1000 TOPS 是可行性问题而非实测规格。[BOA 原始澄清](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1)
- **4 月 1 日**：讨论 BOA 如何支持 gather，分清 structured window、可分桶 indexed 与真正动态不规则，补充 IMS/AGU/reorder 等候选能力。[gather 原讨论](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=d137bb01-8491-4aab-9c36-ebd65f4f4911)
- **5 月**：Tile 控制 ISA 用 async launch 驱动 MFE/BOA/EVU；程序镜像进入 Tile-local instruction memory。早期 USE 是控制器称谓，不应无日期地与后来的 UCE/USEEngine 混用。[来源 S08](https://chatgpt.com/c/6a19530c-b224-83a3-914d-3ee7d6c44904?messageId=fead0f63-eb08-4c1d-817e-2ac4f3ec6572)
- **7 月 1 日**：你要求两个独立 GEMM 展开 K 循环后跨 tile function 重排。独立任务可以重排；K 累加依赖不能被删除。[来源 S09](https://chatgpt.com/c/6a44b4dc-65bc-83e8-a130-6678dd5fcb6b?messageId=1a9663bb-c0d9-41d1-8120-23da65cdfeda)
- **7 月 8 日**：你拒绝只面向大模型的设计，希望像 GPU 一样通用。后续具体收敛为广泛 AI 推理通用性，而非完整 CUDA/HPC 替代。[用户要求](https://chatgpt.com/c/6a4df94e-c59c-83ee-8aea-006ad1e05265?messageId=43305fca-562d-4939-9399-a29d8a71f2da)

### 2026 年 7 月后半至 8 月 明确 SPMD 并补全生命周期

- **7 月 21 日**：接受 CORA 名称，作为不默认继承 Elenor 的独立架构。[来源 S10](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)
- **7 月 22 日**：反对 Region IR 直接下沉 L2 controller 导致跨 Region reorder/prefetch 空间缩小。[来源 S10](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)
- **7 月 25 日**：明确汽车/机器人与通用 AI 推理；更重视关键资源有效利用和少搬数据，不要求所有单元永远满载。[来源 S01](https://chatgpt.com/c/6a646992-d14c-83ee-b27f-a7bad0d4fb79?messageId=cc79cd85-2847-43be-86be-861ddd73d67a)
- **7 月 26 日**：提出 L2 也增加 multi-context，助手给出 MC-PGE/prefetch context 方案。[来源 S10](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)
- **7 月 29 日**：新讨论明确不考虑 CORA，除非你主动提及；7 月 30 日出现 HSTI-1 提案。[隔离要求](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=083a359b-516b-401b-99bc-ff5009d45852)；[HSTI-1](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=20b35430-8353-474c-87f3-8e0f97fcd9c0)
- **7 月 31 日**：你放弃通用 Tile 间流水，选择 SPMD；同日论文方向收敛为 NEST 的两级 multi-context 与资源合同。[来源 S11](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=b628a6f0-c936-44d1-b572-f623c2bb989b)[来源 S12](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)
- **8 月 3～4 日**：模型评审暴露 Full Contract 串行化、弱 baseline、Tile-only 未真激活等问题；不是已证明架构收益。[来源 S48](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=81f53566-7c09-4b89-8621-81120c8079b2)
- **8 月 7 日**：你把 NEST 设为新背景，并指出生成 IR 缺失输出 buffer 管理；后来要求保留简洁语法，不把复杂资源机制全摊到 IR。[来源 S13](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=b5f8eb2c-e61a-49e8-9d74-0415cf36ec24)
- **8 月 10 日**：质疑通用 direct-forwarding→小 FIFO→L1 FIFO；建议改成 Context-owned named L1/ACC slots 与 token/scoreboard。[用户质疑](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=4f9d1607-01ed-4c61-8330-5d0e888eb27a)
- **8 月 12 日**：明确 Cache/SPM 双模式、FIFO 仍待定；纠正固定批次为完成一个补一个。[来源 S18](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=0bcc2e8e-8787-44d1-a326-d79ef6acc382)[来源 S49](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=c474e393-071a-44d0-94ba-2c4f5fd41010)
- **8 月 16 日**：提出按任务、调度、资源、内存分层的 trace，用等待原因解释收益。[来源 S39](https://chatgpt.com/c/6a81a3e4-5cfc-83e8-9007-2bc85ff78502?messageId=870f28da-c36d-457e-ae67-ec54ced3e31a)
- **8 月 20 日**：你给出 `nexus.program` 与异步提交 NEST context 的上层框架。[来源 S14](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=cee8808a-d227-4db9-8bbf-c63b41cc8dc4)
- **8 月 28 日**：液态网络 LTC/CfC 的 temporal Context、跨 Context micro-batching 成为负载探索，不是新增冻结硬件。[来源 S40](https://chatgpt.com/c/6acc9954-5a68-83ec-ac31-d2aa4a6812a3?messageId=aceb5cde-d2c9-4519-8d9c-cab83b8d2699)

### 2026 年 9 月至 10 月 从执行合同到硬件落点

| 时间            | 修订或新问题                                                          | 历史意义                                                 |
| --------------- | --------------------------------------------------------------------- | -------------------------------------------------------- |
| 9 月 11～12 日  | 有限 action 表、拓扑登记、前置 memory admission                       | 既限制动态性，也避免有限表和输入挤占输出的一类死锁       |
| 9 月 13 日      | 同级所有 bank 内使用相同比例；HBM 边界改 profile                      | 统一容量组织，不等于同时获得两套满带宽                   |
| 9 月 14～15 日  | Group 自定义编排；Tile 用 Triton 表达                                 | 两层靠布局、资源、事件合同衔接                           |
| 9 月 18 日      | 撤回“所有跨 Context await 都必须阻塞 Device”的推论                    | 需要控制准入依赖，不必永久禁止跨 Context 等待            |
| 9 月 23 日      | 一个 Context 一个 memory profile；完成驱动补位                        | 编译器选择合法计划，不预测真实完成顺序                   |
| 9 月 28 日      | 历史源码审查                                                          | 揭示有限扫描、部分队首限制、固定 Tile 映射等具体实现边界 |
| 10 月 1～2 日   | Group 不做 if/else；当时无 atomic；比较 Tile/Group gather             | 控制流和间接访存归属分别收窄，不等于删除 Tile gather     |
| 10 月 5 日      | Context 管理释放；shared 一次设置、发布后只读                         | 可写 move 与远端 L2 copy 留作另案                        |
| 10 月 11～12 日 | 研究目标收窄；调度 RTL/Python 切分；仅借鉴 openTPU sim                | 把合同验证、周期模型、RTL 和 PPA 分开                    |
| 10 月 14～15 日 | 专用 BOA+EVU；Ara 多在途不等于多 PC                                   | 防止复用现成模块时偷换执行模型                           |
| 10 月 17 日     | Tile Shared；4 个独立 8×8；系统 TaskID；Gemmini 后端                  | 从抽象调度进入端口、状态、命令和归属问题                 |
| 10 月 19～20 日 | ring/Reduce Memory → 暂不用 tree，顺序 K-step                         | 保留跨 K 累加，同时降低并行归约复杂度                    |
| 10 月 21～23 日 | RISC-V 多 Context 五路线；自主 RVV；公共 Task Controller；Chisel 联仿 | 仍处于候选选择与渐进验证阶段                             |
| 10 月 26 日     | Shared AM V1 affinity+每 Context 两槽；区分三种 K                     | 最新明确约束是资源合同，而不是完整物理配置               |
| 10 月 28 日     | DMA/间接 AG 与随机 gather 吞吐                                        | 能寻址与能高效执行必须分别证明                           |

表内来源：[调度](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=b34b621f-95e6-4d36-9183-75d2380f9315) [来源 S51](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=d9720292-8f8f-41e1-86ba-1cda1d6e26b7)、[存储](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=0bcc2e8e-8787-44d1-a326-d79ef6acc382) [来源 S20](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7eff8b05-df9c-4435-ad12-03b9e8903f6c) [来源 S21](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=bc75f0ff-84a8-4b4d-be0c-3f3781d61da1) [来源 S52](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=d03c1861-1a68-4ee2-9dd0-fb86be4dea6c)、[编译](https://chatgpt.com/c/6aa7c75a-21bc-83e8-bc55-0a5d342a7b22?messageId=50675cfa-a814-4729-bc1e-dcc27a811935)、[10 月控制与 ownership](https://chatgpt.com/c/6abb7d0e-0bb8-83ee-8042-892cb647deb7?messageId=32ae04b5-99f4-48a3-af3a-5361bef40597) [来源 S53](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=44bc19fb-f07f-4c0a-ab05-e271002e3f1c)、[计算与验证](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786) [来源 S26](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74) [来源 S29](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79) [来源 S30](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda) [来源 S31](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621) [来源 S32](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e) [来源 S36](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=d7fb793b-8c12-4d4e-a287-9bb8df90310b) [来源 S43](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=0f65b652-2cd4-4f2d-96a0-ff552be8f68e) [来源 S57](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=7aa75680-b6b5-4ff0-904a-1864086c52ee)。

<a id="design-records"></a>

## 四 逐设计档案

### 工业 NPU 改造 从双解释器到单 L2 bytecode

**时间与归属：**2025 年 8 月。起点是你描述的旧公司实际组织；解释器设计是当时讨论的改造方案，不应追认成后来个人芯片的第一代量产规格。

- **目标**：消除 op-per-op runtime 的重控制开销，把 tiling、fusion、异步搬运与乒乓缓冲更多交给 compiler。
- **硬件背景**：ARM Host；L2 controller 带 DCCM/CSM；12 个 L1 controller 与 12 个 NPU slice。
- **最初编程方案**：助手提出 L2 bytecode 管 DDR↔CSM、跨 slice 与全局事件，L1 bytecode 管单 slice DMA/compute 微序列。
- **用户修订**：只生成 L2 bytecode；L1 由固定固件或微内核消费 command ring/descriptor。控制流和全局事件留 L2。
- **存储与调度**：异步 DMA/compute、多 buffer、显式依赖；Host 负责装包、参数、doorbell 和等待，而不是逐微操作介入。
- **编译**：模型相关程序与常驻解释器固件分开，不是每个模型都重烧解释器。
- **状态与演进**：单 L2 是这阶段的明确选择，后来 Tile program 是新的设计边界，不能被这条旧约束禁止。长期继承的是薄 runtime、强 compiler 和设备侧异步执行。[来源 S03](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=ba2698fc-65fa-437e-85ea-e37497dc58a7)[来源 S04](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=36fc1de6-01c5-4528-9edc-a863d3b4df40)

### 分页 bytecode 与离线 microkernel 两条早期修订

**2025 年 8 月 31 日，bytecode 分页。**你先讨论 DCCM sliding-window bytecode fetch 与 segmented paging/CALL，随后明确选择分页，并要求给出 C++ 代码供后续讨论。助手骨架包含 L2-only bytecode 的 DDR 程序与 DCCM page cache、L1 固定命令的三条异步 stream、ring/doorbell、MemoryPlan/handles、segment/page、PAGE_HINT/SAFEPOINT，以及最小 relocation/profiling。这是当时的设计与代码草案，不是已部署实现；也不沿用旧骨架中简化的 HAL barrier 来证明硬件内存语义。[滑动窗口问题](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=914de433-644c-43ea-aa7f-f8d25ab20156)；[用户采用分页](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=47004e9d-13a0-475a-9037-a6a716586273)；[当时分页骨架](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=b3289ded-c50f-4499-88ef-905099ba774c)

**2025 年 9 月 4 日，离线最优 microkernel family。**你明确反对为 micro-conv 做全局 cost-model 搜索，希望先离线找到合适 microkernel 家族，再 lookup/tiling。助手细化为有限 tile/dataflow/buffering 候选、离线 profiling（包括多核竞争）、按 shape/dtype/layout/hardware/firmware/compiler revision 索引的 manifest DB，以及无精确命中时的 fallback heuristic。这是早期局部 kernel 选择策略，不构成禁止后来 Nexus 的 schedule/profile 优化。[用户选择](https://chatgpt.com/c/68aff962-3f34-832a-867c-14e54589d5d9?messageId=caea8cdd-64ed-4a4e-8d42-11f14883ae08)；[候选家族与查表提案](https://chatgpt.com/c/68aff962-3f34-832a-867c-14e54589d5d9?messageId=42b30f39-5008-4bc6-a3b7-867febd87235)

### 驻留优先的 Island Path 方案

**时间与归属：**2026 年 3 月。单芯片探索；“第四/第五范式”是对话中的概念命名，不是经过验证的学术分类或官方版本。

- **目标**：围绕数据驻留、搬运距离和阶段间依赖组织计算，比较它是否比平铺 SPMD 更能降低搬运、等待和状态重载。
- **硬件组织**：Island 是芯片内 cluster，含局部计算、SRAM、控制与事件；Path 编排岛间粗粒度阶段。曾探索 SRAM 附近的 reduction/elementwise/accumulation，不代表实现为存内计算。
- **编程模型**：局部阶段在 Island 内执行，跨 Island 通过粗粒度事件/消息连接；不是多机集群的通信方案。
- **存储与事件**：event 是同步状态，doorbell 是通知/唤醒。专用事件网络或现有 NoC 的控制 VC 都曾被提出；不存在“必须物理独立控制网”的最终结论。
- **调度边界**：L2 更像共享资源/事务代理，不必复制完整 cluster scheduler。对 1000+ clusters 的疑问引出分层控制、轻量 local controller、编译期 buffer rotation/顺序与运行时局部不确定性的分工。
- **编译压力**：跨阶段 placement、驻留、buffer lifetime 与通信都需规划；这正是后续对通用 MPMD/空间流水复杂度的顾虑来源之一，但不能据此断言它已被全面否定。
- **验证计划**：Flat SPMD、SPMD+locality-aware grouping、Island+Path 三种同资源模型；负载为 attention-like、Mamba/SSM-like、reduce/merge-heavy；测 cycles、分层搬运/控制 bytes、利用率/空闲、峰值 SRAM/溢出。
- **状态**：你明确询问“到底强多少”，收益未闭合。应保留正反适用区间，不把该方案写成天然优于 SPMD。[来源 S56](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)

### BOA 矩阵核心与不规则访问补强

**时间与归属：**2026 年 3～4 月；BOA 是你直接澄清的计算结构，外围补强包含助手建议。

- **目标**：用块外积计算和 K 维累加构成高效矩阵主路径，配合向量单元和分层架构，而不是照搬 GPU 线程机。
- **计算**：BOA = Block Outer-Product Accumulator。reduce network 在当时主要对应 K-tiling partial sum 的累加，不能把它一开始就泛化为全芯片任意 collective。
- **编程问题**：曾问失去 GPU/IPU 式 threads 会失去什么；后来 task/context 与线程执行模型的取舍由此展开。
- **访存候选**：规则 window gather、可分桶 indexed gather、真正动态 irregular 分开处理。cluster-local IMS 候选含 indexed AGU、gather queue、reorder/pack、valid/mask、scatter write-combine/commit、向量 assist。
- **编译**：尽可能先规则化和分块，让 BOA dense 主路径持续；不能规则化的部分走向量/irregular 路径，而不是给矩阵峰值增加数字就认为 gather 已解决。
- **参数边界**：3 nm、单芯片 1 PFLOPS FP16 / 1000 TOPS 是你提出的可行性目标；32 clusters/256 BOA、约 4 TFLOPS/BOA、640～1060 mm² 等是旧助手估算，未见对应 RTL、综合或工艺验证。
- **状态与演进**：BOA/向量/搬运分工延续；早期 gather 缺口不应被直接说成后来 real-gather 版本仍然缺失。[BOA 澄清与目标](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1)；[历史面积估算](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=c0237d49-5acc-420d-874a-995c11662cf6)；[IMS 候选](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=d137bb01-8491-4aab-9c36-ebd65f4f4911)

### Elenor 与可编程 Tile

**时间与归属：**本次已读资料可恢复到 2026 年上半年及 7 月回顾；最初命名日期未凭回顾消息倒推。

- **目标**：以异构 Tile、显式数据移动和多层 compiler/runtime 分工提高 AI 计算利用率。
- **硬件回顾**：Tile 包含 BOA、EVU、MFE、UCE、L1 SRAM；Tile Group 包含 Group Scheduler、Region Sequencer、Shared L2 SRAM、Group DMA、Collective/Multicast、PMU。
- **编程与编译**：Host IR → Region IR → Device IR；更偏 compiler/region/dataflow 主导。基础已经包含 Tile-SPMD，不能标成“完全 MPMD 的旧版”。
- **Tile 控制细化**：5 月 launch MFE/BOA/EVU，事件连接异步操作，Tile-local instruction memory 存程序，ROM 负责固定启动功能。7 月讨论独立 GEMM 的展开、重排和提前 load。
- **内存与调度**：分层 SRAM、DMA、数据驻留与局部 schedule；若采用跨 Tile producer-consumer pipeline，还须处理 stage placement、route、FIFO sizing、buffer lifetime。
- **状态与演进**：这里的模块表来自历史对照回顾，不是本次代码审计；CORA 后来被明确当作独立探索，不能视为 Elenor 的正式更名。SPMD 的选择是对通用空间流水复杂度的收缩，而非从此才第一次出现 SPMD。[Elenor 回顾](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=1c6dcc67-2a4c-499f-b534-4fc9c55d59c4)[来源 S08](https://chatgpt.com/c/6a19530c-b224-83a3-914d-3ee7d6c44904?messageId=fead0f63-eb08-4c1d-817e-2ac4f3ec6572)[来源 S09](https://chatgpt.com/c/6a44b4dc-65bc-83e8-a130-6678dd5fcb6b?messageId=1a9663bb-c0d9-41d1-8120-23da65cdfeda)

### CORA 及 L2 MC PGE 探索

**时间与归属：**2026 年 7 月 21～26 日。你接受 CORA 名称；英文展开 Coroutine-Oriented Reconfigurable Accelerator 来自助手解释。

- **目标**：访存等待时切换 ready coroutine/context；让 Matrix/Vector/LSU 程序化组合，支持 compute-driven memory 和运行时 index gather。
- **硬件**：Tile 多 context、异构执行后端；后续探索 L2 多 context，而非只在计算侧隐藏延迟。
- **编程**：不是固定 operator pipeline；一个程序 PC 可发多个异步 engine operation，task 与单个 engine operation 不能混为一谈。
- **用户纠错**：反对把 Region IR 原样下沉到资源更小的 L2 controller；跨 Region reorder、提前 prefetch 的 compiler 空间必须保留。
- **MC-PGE 候选**：compiler 拆 Prefetch Context；token、address、resource ready 后发射，可选小窗口；以 L2 sidecar 方式工作，不插入普通 L2 hit 关键路径。
- **存储与调度风险**：多 context 减轻 head-of-line blocking，不会增加 HBM 带宽；L2 容量、重复请求、优先级、输出排空仍须独立处理。token DAG 无环不等于资源无死锁。
- **状态与演进**：这是一条独立探索；7 月 29 日新对话明确排除 CORA。MC-PGE 的名称、窗口和具体模块不能未经证据迁入 Nexus 当前规格。[来源 S10](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)；[命名解释](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=911f35c2-b4b9-4f45-b7dd-78e0929edee1)；[MC-PGE 提案](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=334613f0-c6bc-46b7-b3e9-d501ed44915f)

### HSTI-1 与 SPMD 推理芯片方案

**时间与归属：**2026 年 7 月 29～31 日。HSTI-1 是助手命名提案；采用 SPMD、放弃通用 Tile 间流水是你的明确选择。

- **目标**：通用 AI 推理，尤其汽车机器人、小 batch、irregular 强于传统 NPU、允许略逊于 GPU；功耗面积目标介于专用 NPU 与完整 GPU 之间，但未验证。
- **负载范围**：CNN、ViT、Transformer、Attention/Softmax/LayerNorm、BEV/检测/分割/多模态、Embedding/Gather/Scatter、部分稀疏/图计算、动态 shape/轻量控制流。首版不承诺完整 CUDA/SIMT、递归、数据库或通用科学计算。
- **逻辑抽象**：Address / Acquire / Transform / Reduce / Route / Commit。它描述数据依赖和工作类型，不是六个串行时钟阶段。
- **物理路径**：访存前端、矩阵/向量/SFU、归约与路由/存储等功能域；不能把每个算子机械拆为全芯片统一 load→compute→store。
- **编程模型修订**：同一 kernel/grid，不同参数/program_id；每 Tile 独立 PC 和进度。static grid 与 dynamic work queue 都可保持 SPMD。
- **明确不做什么**：不做任意 Tile-to-Tile 不同算子的 producer-consumer 流水；不强迫 compiler 规划任意 stage placement/FIFO sizing/route/deadlock。
- **保留什么**：Tile 内 async、局部软件流水、时间复用与有限 collective 候选。硬件 context slot 数不能直接当成大 MatMul 可同时驻留数。
- **状态与演进**：SPMD 简化原则成为 NEST 主线；HSTI-1 的具体容量、阵列或吞吐数仍是提案参数，不是全部冻结。[HSTI-1 原案](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=20b35430-8353-474c-87f3-8e0f97fcd9c0)[来源 S11](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=b628a6f0-c936-44d1-b572-f623c2bb989b)

### NEST 两级多 Context 与资源生命周期

**时间与归属：**2026 年 7 月 31 日论文提案；8 月 7 日你明确以最新 NEST 为新背景。

- **目标**：保留 SPMD 编译简单性，用有限动态调度提高关键资源利用率；这是可证伪研究问题，而非已成立的性能结论。
- **核心机制**：Nested Multi-Context；SPMD Epoch；Hierarchical Resource Contract。
- **硬件职责**：L2 Context 隐藏 HBM↔L2、prefetch、grid、drain 等等待；Tile Context 隐藏 L2↔L1、DMA、gather、matrix/vector/store 等局部等待。
- **早期程序约束**：同 epoch 计算使用同一 kernel image，独立 PC/进度；L2 可处于不同 phase。后来 Nexus 是否允许多个 program/image，需要依较新决定，不能永久套用此论文限定。
- **资源合同**：L2 输入/输出、L1、ACC、DMA、MSHR、event、credit 的准入与生命周期；容量与前进性必须一起考虑。
- **三事件**：INPUT_RELEASED、OUTPUT_READY、TASK_DONE 含义不同；任务完成不代表外存写回完成。
- **8 月 7 日修订**：用户指出 output buffer 漏管；输出应经历 reserve/ownership→tile writing→output ready→DMA reading→store_done→free。后来简化 IR，不是否定内部 lease/generation/credit 的安全需求。
- **实现分工备选**：RISC-V 做配置、异常、低频复杂控制；硬件 Context Engine 做 ready/select、token wakeup、频繁切换。它与后来“RISC-V 执行 Context 程序”不矛盾，关键是谁承担频繁状态切换。
- **验证现状**：8 月合成周期模型曾暴露 Full Contract 串行化和 baseline 缺陷；不能把旧 speedup 或功耗估算当作芯片测量。[来源 S12](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)[来源 S13](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=b5f8eb2c-e61a-49e8-9d74-0415cf36ec24)[来源 S48](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=81f53566-7c09-4b89-8621-81120c8079b2)；[8 月硬件与固件分工](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=5c97bfd7-9366-4f1b-9c4c-ec6aa395313a)

### 从通用 FIFO 网络到命名 L1 ACC slots

**时间与归属：**2026 年 8 月 10 日，属于 NEST 附近的 Tile 数据组织候选修订，而非一颗新芯片。

- **原候选**：direct-forwarding、小 FIFO、L1 FIFO 逐层承接 FU 结果，尝试让不同计算单元流式衔接。
- **用户顾虑**：大数据最终还是进入 L1，多 Context 交叉运行使通用 FIFO/转发网络复杂。
- **收缩方案**：Context-owned named L1/ACC result slots；token/scoreboard 表示 ready；command queue 传 descriptor/token，而非承载大 tensor。
- **调度与存储**：计算前预留 output/ACC，多 Context 的驻留槽承担解耦；少数固定 ACC→Vector fast path 可做性能优化，但不能成为正确性必要条件。
- **编译意义**：降低每条 FU-FU 边做 FIFO sizing、任意 stream route 与死锁分析的负担。
- **状态与演进**：用户指出问题，助手给出收缩提案；slot ownership、生命周期和大张量共享存储成为后来 Shared/AM 思路的铺垫。具体 FIFO 深度和容量未冻结。[用户质疑](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=4f9d1607-01ed-4c61-8330-5d0e888eb27a)；[收缩提案](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=633ed92c-3525-477f-b954-47ed6116f51c)

### Nexus 上层编排与后续执行合同

**时间与归属：**2026 年 8 月 20 日已有用户直接给出的 Nexus IR；9～10 月继续扩展执行和实现讨论。下述日期修订不冒充官方 Nexus 版本编号。

- **初始职责**：Host/Nexus 创建 global views、绑定参数、提交 Context、建立外层依赖、await；NEST Context 管 L2 buffer、预取、Tile dispatch、store、release。
- **程序复用**：两个独立实例可重复提交同一个 Context program；两次 async submit 在 await 前，并不保证物理同时运行，但允许资源许可时并发 admission。
- **任务身份**：logical task/grid ID、physical Tile ID、Context slot 不同；后续扩展为系统 TaskID 与 generation，避免复用槽后错收旧完成。
- **8～9 月存储版本**：Cache/SPM 双模式→同级 bank 内统一比例→每 Context 一个 profile→HBM 安全边界改档。Cache budget 与 SPM 硬预留不同。
- **调度版本**：不是固定四任务批次，而是完成驱动补位；Group 有限登记/ready-action，Tile 多 PC/eligible heads；前置 admission 限制并发资源足迹。
- **10 月控制与 ownership 版本**：Group 不做 if/else，动态分支留 CPU，Context 增加 for；Context 是管理和释放基本单位；shared 设置后只读；当时不支持通用 atomic。
- **编译路线**：Group 自定义 MLIR 编排，Tile 从 Triton/TTIR 接入；布局、L1/L2 需求、事件、profile、输入释放和输出可见性都是跨层合同。
- **模型与硬件状态**：Python 可用于合同/调度/存储评估；历史真实 index 搬运不等于完整 BOA/EVU 数值计算。后续倾向 Scala/Chisel、逐组件闭环接入 RTL，不要求一次做完整芯片。
- **关系说明**：Nexus 起初在 NEST 之上，后来名称覆盖更多架构；不能用一条“改名”省略这次层次变化。[来源 S14](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=cee8808a-d227-4db9-8bbf-c63b41cc8dc4)[来源 S18](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=0bcc2e8e-8787-44d1-a326-d79ef6acc382)[来源 S20](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7eff8b05-df9c-4435-ad12-03b9e8903f6c)[来源 S50](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=b34b621f-95e6-4d36-9183-75d2380f9315)[来源 S53](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=44bc19fb-f07f-4c0a-ab05-e271002e3f1c)(https://chatgpt.com/c/6aa7c75a-21bc-83e8-bc55-0a5d342a7b22?messageId=50675cfa-a814-4729-bc1e-dcc27a811935)[来源 S41](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=ac1723fd-2bd3-49b2-971f-c0bf468e0f56)[来源 S43](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=0f65b652-2cd4-4f2d-96a0-ff552be8f68e)[来源 S57](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=7aa75680-b6b5-4ff0-904a-1864086c52ee)

### Shared AM V1 与矩阵后端收敛

**时间与归属：**2026 年 10 月 17～26 日多个方案连续修订；Shared AM V1 是原对话中的明确首版名称。

- **目标**：多个 Context 交叉使用矩阵阵列时，让输出链跨 K-step 持久驻留，减少旧 C 经 Shared 往返；共享 AM 容量，但不引入无边界迁移和并行归约复杂度。
- **前驱方案**：四个独立 8×8、无互联/广播、各自 AM；随后探索 ring/AM-to-AM reduce、专用 Reduce Memory 和 reduce tree；10 月 20 日转向先不用 tree、顺序 K-step。
- **首版用户约束**：默认保留原 engine affinity；每个 Tile Context 在所有矩阵 engine 上总共最多两个 AM slots。
- **助手合同细化**：一条输出链→一个 slot→一个固定 engine；按链动态申请；one-shot 也计数；等待输入时保留槽而不长期独占 engine；安全 drain/回收后归还额度。
- **硬件候选**：多个 AM group 与本地 RMW/adder，engine→AM 互连、slot table、仲裁与结果 FIFO。四组×两槽×16 KiB 只是示例，不是用户两槽限制自然推出的配置。
- **调度关键**：满槽或新链申请拥塞时，存活链 ACCUM/FINAL/drain 必须继续；compute_done、step_committed、output_visible、slot_free 分开。
- **编译关键**：固定 affinity、合法输出 tile、K_total/K_chunk/K_cmd、两槽软件流水与输入双缓冲；两槽不限制一个输出链可有多少 K-step。
- **状态**：对话当时产生需求与指南，明确未修改仓库、未跑模拟器或 RTL 验收。跨 engine migration、split-K 并行归约、无槽 one-shot bypass 是另案或被收紧的旧建议。[来源 S29](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)[来源 S30](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda)[来源 S31](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)[来源 S32](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)[来源 S33](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)[来源 S34](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)[来源 S35](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f)

<a id="named-branches"></a>

## 五 历史命名与独立方案补录

本节专门保留容易被最新基线摘要省略的独立架构、局部重构、论文工作名和实际分支。它们的名称与状态不应互相替代。

### ELENOR Region 主线的内部收缩

#### 既有层级

`Host → Tile Group/Region Sequencer → Tile/UCE → BOA/EVU/MFE/USE`。Tile 有 PC/context/event；Group 有共享 SRAM/DMA/collective、空间分配和粗粒度编排。基础编程已经包含 Tile-SPMD，不是“ELENOR 全是 MPMD”。

#### 6/16：SuperRegion → Pattern Region

助手此前提出 SuperRegion + 任意 tile-dataflow 调度，涉及全图 buffer、跨 op reorder、资源约束排程。随后认可用户的更保守方案：

```text
Graph → Pattern Fusion → Compute-heavy Pattern Regions
      → Region-level Async Dispatch → Tile Group SPMD
```

Region 内固定模板，Region 间隐藏 descriptor/program/weight 预取与前序 store drain。它**不自动等价于依赖 Region 的 compute 并行**，也不能消除所有中间 tensor 落地/屏障。收益与编译器复杂度是主要取舍。

来源：[6/16 修正与两方案完整对比](https://chatgpt.com/c/6a30fc13-f444-83e8-b86c-759c7ad320ce?messageId=480b582e-2857-431a-a79d-e2223036daa3)。

#### 7/8：不能删除 Tile 的执行上下文

HFC 讨论里助手曾简化成没有 PC 的 compute lane。用户追问怎样 overlap IO/compute，并指出修正后和自身 ELENOR 一样。助手最终承认：

- 保留 Chip→Group→Tile；Tile 保留 UCE/PC/context/event。
- Group 降为资源/执行域管理，不重复逐指令发射。
- overlap 来自多 outstanding context + 异步引擎，不是“每一个 ALU 都有 PC”。

来源：[用户追问及纠正](https://chatgpt.com/c/6a4df94e-c59c-83ee-8aea-006ad1e05265?messageId=d254fbcc-6c5a-4013-8c81-bcdbf9f5fe43)、[助手承认与 ELENOR 核心一致](https://chatgpt.com/c/6a4df94e-c59c-83ee-8aea-006ad1e05265?messageId=f30e9bfb-cda3-4bcb-99a1-7348eb52cf89)。

### EFP 与 HFC：两条独立推理架构探索

#### EFP（2026-07-08）

全称 Elastic Flow-Pipeline Processor。flowlet 是可静态调度的小 tile/page/block 子图；算子中间值尽量走 FIFO，依赖用 predicate/token，credit/backpressure 处理延迟。

- 计算：micro-systolic island + vector/reduce/SFU/mask/RF/FIFO。
- 存储：cluster SRAM、DMA/AG，NoC multicast/reduce，KV page DMA；更偏显式 SRAM/FIFO 而非大 cache。
- 执行：static fast path + elastic slow path，编译器做放置、局部时序、FIFO 深度、bank/modulo 排程。
- 目标：LLM、多模态、MoE/attention，非任意 CUDA/C++。
- **结果**：用户要求像 GPU 一样通用，明确指出“你这个只是为了大模型”。不能将 EFP 写成被采纳的主线。

来源：[EFP 提案](https://chatgpt.com/c/6a4df94e-c59c-83ee-8aea-006ad1e05265?messageId=adcedac6-6785-4f8a-9de3-cb477c32babd)、[用户否定适用面](https://chatgpt.com/c/6a4df94e-c59c-83ee-8aea-006ad1e05265?messageId=43305fca-562d-4939-9399-a29d8a71f2da)。

#### HFC（同日）

全称 Heterogeneous Flow Compute，作为回应通用性不足的新提案：

- 层级 `OperationToken → FlowBlock → ComputeCluster → Fabric`。
- FlowBlock 把 load/mma/vector/reduce/store 组织成依赖图，ready 操作驱动执行。
- Cluster 增加标量 RISC 控制 pointer/loop/branch，配 tensor/vector/memory/reduce、SRAM/FIFO。
- Flow ISA 表达 load/compute/branch/reduce/sync/spawn，predicate flow 表达分支合流。
- 助手随后提出 **GPU+HFC Hybrid**：GPU/SIMT 承担通用生态，flow cluster 加速适合负载；不是用户已定案。
- 后续关于删除 Group/Tile 的错误建议被用户纠正，回到 ELENOR 现有层级，差异更接近 dataflow-first vs async-program，而非全新层级。

来源：[HFC / Hybrid 提案](https://chatgpt.com/c/6a4df94e-c59c-83ee-8aea-006ad1e05265?messageId=f9fcdb3e-866c-46bf-81ec-3a15dac517b5)。

> 名称保留：后续文本偶有 EFP/EWF 并写；目前未核实 EWF 有独立定义/版本，不创建虚假的 EWF 架构条目。

### B1 + Multi-Context：一个大 Tile 的资源池化方案

2026-07-14 用户明确要求对比 **B1 + multi context** 与原设计。助手纠正误解：CPU 只入队完整程序，不逐条调度 load/compute/store。

```text
CPU → Central Program FIFO → Central Multi-Context UCE
    → Context-tagged BOA / EVU / MFE queues
    → Unified Banked SRAM → HBM
```

- 原方案：多个小 Tile、各自 UCE/L1/context，空间并行、模块化、隔离与扩展自然，但有 L2→L1 搬运和资源碎片。
- B1：一个逻辑执行域、宽引擎与多个 context，资源池化、少一层 SRAM 搬运、编译器单层 mapping；但大 SRAM/长线/端口、集中调度、故障和 QoS 干扰风险更高。
- 推荐的 B1 变体：**逻辑单 Big Tile，物理 banked/partitionable**；BOA 子阵列可分给不同 context，但没有重新暴露多个 Tile PC/UCE/L1/barrier。
- 大 GEMM 与小矩阵/irregular 的优劣取决于子阵列可分区、outstanding、总 SRAM 带宽、活跃工作集与干扰上界。
- 文中 8 Tile、4K MAC/cycle 等是公平比较/候选配置，不是已实现资源数。

来源：[用户对比要求](https://chatgpt.com/c/6a55fb75-c998-83e8-a69d-f4bcc458c87a?messageId=d6bcea1d-002b-4d77-b508-013ec0230208)、[修正后的 B1 对比](https://chatgpt.com/c/6a55fb75-c998-83e8-a69d-f4bcc458c87a?messageId=f07eab6d-af2e-594f-8502-f32132fe38dd)。

### TCP-inspired TCC：计算侧重构，不是整机换架构

2026-07-15 助手比较三条路线：完整 TCP 替换、原 BOA+EVU 不改、TCP-inspired 重构；推荐第三条，暂名 Tensor Compute Complex（TCC）。

- 保留 Group、UCE 双 context、MFE、USE、L1/L2、event/token、Tile Program。
- 替换/改造局部计算侧：Contraction Engine、Programmable Vector Chain、Local Fetch/Feed、ACC/RF、可选 Transpose-lite/commit。
- 主数据通路 CE→VE 直通，减少 L1 落地；VE→CE 只用小 RF/staging 可选慢路，不做任意全带宽反馈 crossbar。
- MFE 管跨层搬运/动态 gather；TCC local fetch 只管 L1→RF/feed，不能再造一套 HBM master。
- UCE 拥有程序 PC/coroutine；TCC 只有 tagged engine command，不另建独立程序上下文。
- 保留 USE 的 scan/recurrence/persistent state；不因复制 TCP 削弱 Mamba/状态路径。
- “BOA V2”在此是离散 reduction-depth 模式建议，**不是整个 ELENOR V2 发布**。新增面积 10%～15% 是门槛建议，不是综合结果。

来源：[TCC 原提案](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=7c9040fe-a6df-4e01-93e7-b1094223ca82)。

### RA-X1：汽车/机器人完整异构 SoC 平行案

2026-07-25 用户要求先忽略 Elenor/CORA，从汽车与机器人通用芯片重新设计。助手提出 RA-X1：

- CPU：OS、动态控制、planning、复杂数据结构。
- GPU：通用并行、SLAM/点云、图形、未知 kernel 与生态 fallback。
- NPU：tensor/attention，局部静态调度+token/credit，矩阵切片、向量/reduce/SFU、DMA/gather/transpose。
- DSP/ISP：传感器处理；独立 real-time safety island、Guardian AI 提案用于安全/冗余。
- 统一 memory/event/runtime/QoS，计算仍异构，安全隔离不因统一而消失。
- 全局按 deadline/locality/QoS/power 分配，局部编译器静态调度；建议使用既有 GPU IP，不自行开发完整 GPU。
- 4/5nm、45～70W、64GB LPDDR5X、12 NPU cluster、约200 INT8 TOPS 等均为助手目标参数，不是 Nexus 当前规格或实际硅性能。

来源：[用户要求独立设计](https://chatgpt.com/c/6a644e18-cbd4-83ee-90b8-223c15172321?messageId=dcae1260-3590-4b54-8cc0-075383d3640b)、[RA-X1](https://chatgpt.com/c/6a644e18-cbd4-83ee-90b8-223c15172321?messageId=7a9dfea9-828a-47aa-8d9c-dc1586922bf0)。

### HTFA：有限 SIMT 补强 irregular 的三路径案

同日另一讨论将目标明确为通用 AI 推理：irregular 强于传统 NPU、可略弱于 GPU，面积功耗低于 GPU，重点是**当前负载关键资源的有效利用率**，不是所有引擎永不闲。

助手提出 Heterogeneous Task-Flow Accelerator：

1. Tensor path：矩阵切片、local scratch、局部多 context。
2. Vector/Dataflow path：规则向量/reduce/SFU，局部静态 VLIW/dataflow。
3. ISE / Irregular Stream Engine：有限 SIMT、独立 PC/mask/reconvergence、scoreboard，非完整 GPU/CPU OoO。

统一 Tile Task ISA 与 async token；动态调度粒度是 Task/Tile，局部规则计算不必逐指令 warp 调度。gather 用 sector cache/coalescing/outstanding/lane replay；预取不能占掉最后 demand MSHR/pagewalk/atomic/writeback credit。

原提案有 2 compute chiplets + I/O、每 chiplet 8 clusters、HBM NUMA 等示例；**不是用户确认的芯片规格**。它和 RA-X1 的完整 SoC、ELENOR 的程序化 Tile 路线都应分别记录。

来源：[用户完整目标](https://chatgpt.com/c/6a646992-d14c-83ee-b27f-a7bad0d4fb79?messageId=cc79cd85-2847-43be-86be-861ddd73d67a)、[HTFA 提案](https://chatgpt.com/c/6a646992-d14c-83ee-b27f-a7bad0d4fb79?messageId=a00989a6-5245-4446-a0d2-cf9b6c3c7af4)。

### 论文工作名与局部版本名

NEST 论文提案还给过强调资源管理的替代工作名 **HERA**；没有证据表明它是一颗独立芯片或正式新代际。[论文提案](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)

“Elenor-SPMD v2”或“Elenor 第二版简化重构”属于助手比较和建议别名，不是已核对的正式发布记录。[SPMD 与 Elenor 对照](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=1c6dcc67-2a4c-499f-b534-4fc9c55d59c4)

OPA/BOA 路线图、EVU 的 First-Silicon V1A/V1B/V2 评审、Bank Profile V1、Shared AM V1 各指自己的局部对象；不能把相同数字合并成全芯片代际。能恢复的机制分别列在硬件、存储与 Shared AM 章节；未恢复完整原件的局部版本，不补造逐版功能表。[OPA/BOA 路线](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)；[EVU 版本评审](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=b1489649-4764-4561-9cbb-ad1dfbbec4c0)；[Bank Profile 提案](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=35676440-32fd-4ecb-9d8a-a843647be760)；[Shared AM V1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)

### 实际 Nexus 名称 分支与历史提交快照

| 名称／时间                                                                       | 证据与功能边界                                                                                                                                                                                                                                                                                                                                                                                              | 不能推导什么                                                                                               |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `nexus.program` · 2026-08                                                        | 用户给出的上层编排IR；提交NEST context、等待completion。[原文](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=cee8808a-d227-4db9-8bbf-c63b41cc8dc4)                                                                                                                                                                                                                                   | 不能仅因名字Nexus便归为一个正式“V1”发布                                                                    |
| `nexus-v3` @ `1e6e94a65f0a63e611c86ba1d38adedaef0db392` · 2026-09-28             | 旧助手固定commit报告：Device解释Nexus；Group有限action表/扫描；s0队首、s1有限ready；逐Tile task admission而非gang；Tile context-head轮转。未跑仿真。[审查](https://chatgpt.com/c/6aba2e6b-b18c-83e8-8316-ed81630116de?messageId=d5095dae-0ae9-45e7-b9c2-d7b3eb75143c)                                                                                                                                       | 默认8 Group context／16 action／scan4／Tile1等是当时代码参数，不是硬件冻结值；不能证明Transformer80%利用率 |
| `nexus-v3` @ `ef0ab11…` · 2026-10-01                                             | 旧助手称Gather仍是L1_HIT/L2_HIT/HBM_MISS预设访存档位，缺真实AddressProvider／索引求址／Scatter。[来源](https://chatgpt.com/c/6abdec71-c088-83e8-97cb-308af6ee08d0?messageId=2d4acce1-21a0-4be1-b650-ac6159ce0584)                                                                                                                                                                                           | 不能用此断言后来的real-gather分支仍缺真实索引                                                              |
| `nexus-v3-real-gather` @ `eef596c5c2be…` · **2026-10-10 UTC**，后续10/11讨论     | 原始消息时间为 10 月 10 日 12:52 UTC；旧助手称有索引字节读取、地址解析、Gather/Scatter数据搬运及相关测试；BOA/EVU/USE仍主要描述符时延模型；未执行测试／RTL／综合。[来源](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=171b1bc7-44ba-4674-bcbd-2628d09775b8)、[10/11定位](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=ac1723fd-2bd3-49b2-971f-c0bf468e0f56) | real-gather不等于全tensor算术正确、完整cache微架构精确或芯片PPA验证                                        |
| `nexus-v3-real-gather` @ `7ec6f6a64c19a1e640e6fb264f70e22d1ce779e8` · 2026-10-23 | 历史助手确认 `launch(desc, cycle, event_id, transaction=None)`／`tick(cycle)` 引擎边界、ByteStore；建议backend注入及Python＋Verilator联仿。用户明确希望逐组件Scala实现接入Python。[来源](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=98915acb-7c7a-5447-8412-1102a30a6369)                                                                                                         | 接入建议不是已经实现的RTL backend，不是完整RISC-V/RVV程序链已可运行                                        |
| `nexus-v2` · 2026-09-30命中                                                      | 回读只见助手以“例如”列 `main/develop/nexus-v2/nexus-v3/feature/foo` 教迁移Git分支。[原文](https://chatgpt.com/c/69a29061-7fe0-832a-a4ea-981f41a29634?messageId=77e357a0-1f89-4810-a1d8-6eb8bc58b107)                                                                                                                                                                                                        | **不构成真实v2分支或架构规格证据**；本次不补造功能表                                                       |
| Nexus v1／v2正式代际                                                             | 当前可访问精确搜索未找到足够直接证据；V1常是局部设计首版，V2常是其他对象版本／示例                                                                                                                                                                                                                                                                                                                          | 不能由v3倒推出v1/v2功能；也不宣称历史绝无这两版                                                            |

<a id="hardware-history"></a>

## 六 硬件方案与参数沿革

这一节按局部硬件对象记录候选和变化。表中的日期是讨论时间；原文局部 v1/v2 是当时路线图，不是整个芯片的正式代际。正文后面的执行合同章节解释这些选择的共同正确性要求。

### 早期 primitive、OPA/BOA 分层与功能扩展

| 讨论时间与性质            | 具体方案与变化                                                                                                                                                             | 当时理由、代价与状态                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-03-25                | 用户问能否一 cycle 做完整 4×4 MatMul；比较完全组合展开、流水化展开、外积逐 K 更新                                                                                          | 完整 4×4 需要 64 次乘法、48 次加法。组合展开可拉长周期完成；“每拍接纳／产出一块”与“输入后单拍延迟”不同。属于微架构探索，不是已达到单周期 4×4。[来源 H01-1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=f38e5bc6-c9a5-43e3-9fc4-606d6cce2df1) [来源 H01-2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=ef44cbee-6b0f-4c87-a043-2f9a9f6f5f53)                                                                                                                                                                                                                                                                                      |
| 3/25                      | 外积 primitive：A 列、B 行广播 → outer product → reg → add → reg → 本地 C                                                                                                  | 助手讨论双／多 accumulator、乘加解耦、forwarding、部分 K unroll；目的是处理累加反馈依赖，而不是已证明全面超过 Tensor Core。旧回答中“100% 利用”“性能接近”等没有验证支撑，不收入结论。[来源 H01-1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=f38e5bc6-c9a5-43e3-9fc4-606d6cce2df1) [来源 H01-2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=ef44cbee-6b0f-4c87-a043-2f9a9f6f5f53)[来源 H02-1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=763064a4-8d86-452d-86a9-e0c9bff0bff4) [来源 H02-2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=d2d7ede8-9824-4f91-be78-f37adbbf4dd1) |
| 3/25                      | 用户明确要求先说明 shape 选择，不能只盯 4×4                                                                                                                                | 把 `(M,N,K)`、精度、广播扇出、本地累加面积、供数带宽、实际 shape 分布联合看。助手候选为 4×8×8、8×4×8、8×8×8、8×8×16；K=4/8/16 扫描是当时建议，不是后来 K_cmd 上限。[来源 H02-1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=763064a4-8d86-452d-86a9-e0c9bff0bff4) [来源 H02-2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=d2d7ede8-9824-4f91-be78-f37adbbf4dd1)                                                                                                                                                                                                                                                                 |
| 3/27                      | 用户明确 BOA 是 **Block Outer-Product Accumulator**，reduce network 专门处理 K 维分块累加；另有 vector 与层级系统                                                          | “PE + reduce”不应被误读成任意图计算网络。3 nm、单片 1 PFLOPS FP16／1000 TOPS 是可行性目标提问，非产品规格或实测。[来源 H03-1](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1) [来源 H03-2](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=d137bb01-8491-4aab-9c36-ebd65f4f4911)                                                                                                                                                                                                                                                                                                                   |
| 5/19 文档／原型评审       | OPA 保持简单 contraction primitive；BOA 管 K tiling、partial merge、mask、index mapping、reduce/pooling bypass；Cluster 放 Tile Loader、WindowGen、L1                      | 历史助手依据当时附件判断实现主要是 INT8 OPA 原型，完整 BOA tree／window／FP 链尚未闭环；本次未重新读取这些附件或 RTL。不可把“框图支持”写成“实现支持”。[来源 H04-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)                                                                                                                                                                                                                                                                                                                                                                                                      |
| 5/19 助手 OPA 路线图      | v1 INT8×INT8→INT32、external C seed、valid/ready；v2 BF16/FP16→FP32；v3 4–8 份 acc bank/context metadata、tile boundary cooperative switch；后续可选 FP8 scale metadata    | 这是**助手建议的版本序列**。与 10 月“不保存每 context PE 内 OS，而以 AM 持久化”的方案不同。TF32 优先级低也是当时建议，不是禁止支持。[来源 H04-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 5/19 助手 BOA 路线图      | 三路径：GEMM/contraction；Window/Slice→bypass→ADD/MAX/MIN tree→scale；独立 elementwise/post-op                                                                             | 主路径支持 MatMul/BatchMatMul/Conv lowering；旁路支持 reduce/pooling；post-op 支持 bias、scale、clamp、ReLU 等。softmax/norm、复杂 SFU 和动态地址不宜硬塞每个 OPA。版本路线为先完整 GEMM，再 post-op、reduce/pooling、队列/context、indirect loader、dynamic shape/predicate。[来源 H04-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)                                                                                                                                                                                                                                                                              |
| 5/20 FEATHER+/MINISA 借鉴 | 引入 **BOA Fragment**（A/B/C 或 partial fragment）；layout 设置与 compute/mapping 执行解耦；mapping/layout 联合搜索；BOA reduction 加 permute/pack（reorder-in-reduction） | 目标是比线程粗、比整任务 descriptor 细的稳定映射层。助手明确不建议复制完整 FEATHER/NEST/BIRRD 或把 BOA 改成任意 CGRA。论文里的 NEST 是外部组件名，**不是后来用户的 NEST 项目**。[来源 H05-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=115c05d3-07c2-4798-a7bc-6120ba755b75)                                                                                                                                                                                                                                                                                                                                                                            |

早期 `OPA v1/v2/v3`、`BOA v1…v6` 是同一历史回答里的发展路线，不应与 Nexus V1/V3 的软件版本号对应。相同“V1”在不同讨论里代表不同对象。

### BOA 与通用计算如何分工：讨论过的五种不同层级

| 讨论时间与性质       | 方案                                                                                                                                                      | 取舍与后续                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5/19 用户提出        | 半片 BOA＋RISC-V、半片 GPU 架构                                                                                                                           | 用户是“会不会更好”的探索。助手反对机械 50/50，建议 BOA Dense Island＋Flex SIMT/Vector Island＋DMA/AGU/KV Engine，共享 NoC/L2/HBM，统一 runtime/compiler partition；增加第二执行模型、RF、coalescer、reconvergence、同步与布局交接成本。未见用户冻结该双岛布局。[来源 H06-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=c7eef51c-adc9-4a1c-bd9b-71c149883897) [来源 H06-2](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=f9a874e2-8de6-4e1e-8d9b-20d949fb5f28) |
| 5/19 助手 Flex 变体  | 面向 LLM、通用 AI、边缘固定模型分别调整 Dense/Flex/Memory/Control 比例；Flex 可为 4–8 warp/context、16–32 SIMD lanes                                      | 数字只是资源探索；不是同版 Nexus 的固定面积拆分，也不是证明 GPU-like 区域能高效处理所有 irregular。建议显式 ownership/event/fence，而非完整 coherent cache。[来源 H06-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=c7eef51c-adc9-4a1c-bd9b-71c149883897) [来源 H06-2](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=f9a874e2-8de6-4e1e-8d9b-20d949fb5f28)                                                                                                    |
| 7/15 用户提出        | 直接用 TCP 替换 Elenor BOA＋EVU，或保持原案                                                                                                               | 助手比较完整 TCP、原案不改、TCP-inspired TCC；偏好第三种。TCP slice 自带存储、fetch、CE/VE/TE、commit、context，整块放入会与 L1/MFE/UCE 重叠。用户认为成熟并要求评估，不等于已批准替换。[来源 H07-1](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=58f914c0-6d87-48b7-8271-f90d70d93b44) [来源 H07-2](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=7c9040fe-a6df-4e01-93e7-b1094223ca82)                                                                        |
| 7/15 助手 TCC        | 保持 Tile Group/UCE 双 context/MFE/USE/L1/L2/event；局部改为 Contraction Engine＋Programmable Vector Chain＋Local Fetch/Feed＋ACC/RF＋可选 transpose-lite | 高层保留 contraction indexing maps；编译器 lowering 到硬件有限模式，不让硬件直接理解任意 affine map。少数 reduction-depth 档位、固定 CE→VE 直通；VE→CE 经小 RF/staging，只做有限小 tile 慢路径。不做任意 CE↔VE↔TE crossbar。[来源 H07-1](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=58f914c0-6d87-48b7-8271-f90d70d93b44) [来源 H07-2](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=7c9040fe-a6df-4e01-93e7-b1094223ca82)                                    |
| 10/14 用户提出       | 把 4×4 OPA 的 16 个 PE 扩成通用 ALU/FPU/SFU lane，共享一套 load/store；16 个 OPA 共 256 lane                                                              | 用户明确**不是每 PE 一个 CPU**。助手解释为可编程 SIMD/vector 阵列；可全阵列时分复用，也可如 12 OPA MatMul＋4 OPA elementwise 空间分区。共享指令、共享 LSU、共享同份数据不是一回事。[来源 H08-1](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=d11334f7-8ca1-4c6d-8a44-172e6eec96bf) [来源 H08-2](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=c5d6d593-88f0-4814-abe8-bb68232ab3b4)                                                                             |
| 10/14 助手回到原主线 | 专用外积 BOA＋独立 programmable EVU＋解耦 MFE；再加受限矩阵 epilogue                                                                                      | 优势是无依赖 MatMul 与 elementwise 可真正异构并行；把全部 PE 通用化则竞争同一算术池和存储端口。SFU/除法无需每 lane 完整复制。助手随后承认这基本是“保留原架构并完善实现”，不是一项全新架构。[来源 H09-1](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786) [来源 H09-2](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=21dbf1ac-2cd4-40bb-b556-736dc8949e33)                                                                     |

TCC 的“vector chain descriptor，不必独立 Vector PC”属于 7 月提案；6 月 Micro-SIMT 和 10 月自主 RVV 都讨论过独立轻量执行前端。三者不是同一个已冻结 EVU。

### TCP/DPE、大 systolic、小阵列：10 月再次比较并被用户纠正

2026-10-17 当时比较的不是简单“外积比脉动好”，而是三条可挂在同一 Nexus 提交接口后的计算后端：[来源 H10-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=a0433a52-5efe-5d1d-b8e8-f3c0a58174cc)

1. **TCP-style DPE**：把更多计算资源用于 K 方向并行归约，或改为更多输出；需要 operand hold/feed、可配置 reduce、AM/output 与 layout 联动。适合关注 GEMV、短归约、特殊 contraction 的候选；但效益依赖供数，不能仅放一棵树就复制 TCP。
2. **单个 Gemmini-style 阵列**：控制集中、规则 GEMM 复用清楚，适合作为验证基线；一个阵列也可以与其他 context 的 EVU/MFE 并行。默认 16×16 不应被当作 128×128/256×256 超大阵列来批评。
3. **少量小阵列簇**：允许任务分别使用部分矩阵资源；助手先建议 4×8×8＋共享供数广播＋任务 AM＋可选 split-K，命令级选 1/2/4 组，不逐拍任意重组。

随后用户明确纠正为 **4 个独立 8×8，彼此无连接、无广播，各自 AM；编译器 tiling，UCE 提交计算，后续才可能 ring reduce**。[来源 H11-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)

因此应保留两条不同记录：

- **助手协同广播簇提案**，非用户方案；
- **用户独立小阵列版本**，当时明确，后来变更的是 AM 私有／共享组织，不能倒灌成用户已经接受广播。

小阵列的风险是重复 Shared 读取、命令供给、启动／排空与输出管理；好处是小输出块和多独立任务的分配粒度。不能把“8×8 输出阵列”自动解释成每命令 K=8；后来 K_total/K_chunk/K_cmd 已明确拆开。[来源 H10-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=a0433a52-5efe-5d1d-b8e8-f3c0a58174cc)[来源 H11-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)[来源 H15-1](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f) [来源 H15-2](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=ca5a1f4b-da43-56f9-b077-51306105ebf9)

### AM 与归约候选的完整序列

| 讨论时间与性质     | 存储／归约组织                                                                      | 当时动机、状态与限制                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------ | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3–5 月             | PE/OPA 本地 C_reg、外部 C seed、BOA K-partial tree，Cluster SRAM/software reduction | 原始层级累加路线。5 月还提出给每 context 保存 acc_bank/k_count/mode；这是后来 AM 独立持久化之前的备选。[来源 H03-1](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1) [来源 H03-2](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=d137bb01-8491-4aab-9c36-ebd65f4f4911)[来源 H04-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)[来源 H05-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=115c05d3-07c2-4798-a7bc-6120ba755b75)                                    |
| 10/17              | 四独立 8×8，各自 AM                                                                 | 用户当时明确；输入无跨阵列广播，未来可加 ring。AM 与物理 engine 绑定容易理解，但空闲容量不能自动被别的 engine 使用。[来源 H11-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)                                                                                                                                                                                                                                                                                                                                                                                             |
| 10/19              | AM-to-AM／ring reduce                                                               | 对同一矩阵执行实例、同一输出块的互斥 K 分片归约；不是同 context 就可相加。助手建议 ring 初版先传输，owner AM 做累加；需 task/handle/generation、分片 bitmap、数据范围、输出目的地；防重复贡献与迟到写。候选，未成 V1 必需。[来源 H12-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda) [来源 H12-2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=167a29e0-d38d-4ccf-9c0e-0208f6b4e248)                                                                                                                                                              |
| 10/20 07:22 左右   | 专用 Reduce Memory＋共享 reduce tree                                                | partial 可靠转移后源 AM 可提前释放，解耦 OPA 与等待归约；代价是额外 SRAM、读写、树与调度。多任务驻留，ready 任务归约，不让缺片 A 堵齐片 B。被用户随后“先不用 tree”方向收缩。[来源 H12-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda) [来源 H12-2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=167a29e0-d38d-4ccf-9c0e-0208f6b4e248)                                                                                                                                                                                                             |
| 10/20 07:59 左右   | 顺序 K-step：首步 overwrite，后续 accumulate，最后一次输出                          | 用户选择的思路。不是 split-K 并行。PE 内 OS 保存一次执行中的局部状态；AM 保存跨 K-step、跨 context switch 的输出链。X0→Y0→X1→Y1 可复用同阵列，不必把旧 C 经 Shared 往返。[来源 H13-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)                                                                                                                                                                                                                                                                                                                                        |
| 10/20 助手细化     | 多 context 共阵列、不同 AM 区域，按显式逻辑 acc handle 累加                         | 一步完成要求 AM 提交且局部状态可丢弃，不是末次乘法发出。`accumulate` 不能指“用当前 PE 残值”。WS 也能跨步 AM 累加，持久 AM 不是 OS 专属。[来源 H13-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)                                                                                                                                                                                                                                                                                                                                                                         |
| 10/26 用户提出     | 阵列组共享 AM                                                                       | 共享容量可改善私有 AM 碎片与热点，但并不自动允许输出链迁移 engine。后端需服务路由、metadata、result FIFO、drain 与仲裁。[来源 H14-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)                                                                                                                                                                                                                                                                                                                                                                                         |
| 10/26 助手物理建议 | 四组物理 SRAM＋各自累加通路，任意 engine→任意 AM group；全局 slot 表                | `slot` 是逻辑输出链空间，`bank` 是物理端口组织，`group` 是独立累加服务模块；不能把三者等同。单 slot 先落单 group 是降低跨 bank 锁定复杂度的建议，非最终物理实现。[来源 H14-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)                                                                                                                                                                                                                                                                                                                                                |
| 10/26 最新用户 V1  | **保持原 engine affinity；每 tile context 只能使用两个 AM slot**                    | 两槽是跨 matrix engines 总配额，不是每 engine 两槽；一条链一槽一固定 engine。申请后等待输入／未 FIRST／AM 写回／FINAL 后待 drain／安全错误收尾均计数。one-shot 也计入配额。[来源 H16-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e) [来源 H16-2](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4) [来源 H16-3](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=d5a3148e-0038-50f4-b4c3-4abf563328b5)                                                                                      |
| 后续扩展，未冻结   | 链迁移 engine、并行 split-K、多 engine 更新同一输出、条带化 AM                      | 不能因容量共享就说已支持。需处理上一步全提交、PE 残状态、权重重载、输入可达、hazard 与浮点归约顺序；当前 V1 affinity 主动避开这些复杂度。[来源 H14-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)[来源 H16-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e) [来源 H16-2](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4) [来源 H16-3](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=d5a3148e-0038-50f4-b4c3-4abf563328b5) |

### 地址生成、布局、计算的边界

| 讨论时间与性质     | 方案                                                                                                                    | 取舍与状态                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 3–5 月             | Window/Slice Generator 在 OPA 外，Tile Loader／BOA 输入侧；Direct Slice 与 Window 路径分开                              | stride、padding、窗口生成是数据供给，不是每 PE 算术。Gather 是 indexed AGU、队列、coalescing、tag、reorder/pack 问题，不应塞入 OPA。[来源 H04-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)                                                                                                                                                                                                                                                                                                  |
| 4/1                | Cluster-local IMS：indexed AGU、gather queue、reorder/pack、valid/mask、scatter combine/commit、vector irregular assist | 区分 structured/windowed、可 bucket 的 indexed、真正 runtime irregular。是早期缺口评审／建议；不能据此说 10 月仍未有 gather。[来源 H03-1](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1) [来源 H03-2](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=d137bb01-8491-4aab-9c36-ebd65f4f4911)                                                                                                                                                                                 |
| 6/30               | LTF 完全放 BOA 内                                                                                                       | 集成延迟低、可强耦合 layout＋compute；EVU/MFE 无法复用、计算阵列扩展绑住 layout，助手不推荐全部内置。[来源 H17-1](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=b10786f4-5ceb-4a69-ba5c-a49c18103dce) [来源 H17-2](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=ccee2578-fc8c-485b-a100-d002cb03de80) [来源 H17-3](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=8032d359-9e2a-4029-8a89-02fd70082ea0)                                                                                 |
| 6/30               | 完全独立 LTF fabric                                                                                                     | BOA/EVU/MFE 共用，演进独立；增加 hop、仲裁与热点。助手先推荐 shared fabric＋各 engine 轻量 shim 混合方案。[来源 H17-1](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=b10786f4-5ceb-4a69-ba5c-a49c18103dce) [来源 H17-2](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=ccee2578-fc8c-485b-a100-d002cb03de80) [来源 H17-3](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=8032d359-9e2a-4029-8a89-02fd70082ea0)                                                                            |
| 6/30 用户明确      | **把 LTF 放进 MFE**                                                                                                     | 用户选 memory-system owns transformation。助手细化为 MFE command frontend 下 DMA pipeline 与独立 LTF data pipeline，带 bypass、stream transform、按需 reorder；不是仅多一个函数开关。[来源 H17-1](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=b10786f4-5ceb-4a69-ba5c-a49c18103dce) [来源 H17-2](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=ccee2578-fc8c-485b-a100-d002cb03de80) [来源 H17-3](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=8032d359-9e2a-4029-8a89-02fd70082ea0) |
| 6/30 buffer 变体   | 顺序 transform 用 skid/浅 stream buffer；重排用 indexed/banked buffer；避免每类 transform 一份固定大 FIFO               | 1–2 entry skid、2–8 entry stream 只是示例。该旧回答对“哪些操作必需 buffer”有绝对化措辞，本档只保留设计分类，不把所有 transpose/gather 都判为必需固定 reorder SRAM。[来源 H18-1](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=1b9db319-6710-4eae-a283-b664bc976e76)                                                                                                                                                                                                                                                                   |
| 7/15 TCC 局部 feed | MFE 管 tile/global 搬运、page/segment/window；TCC Local Fetch 管 L1→RF/feed/CE/VE、reuse/broadcast                      | 是为了不把完整 TCP TDMA/Fetch 和 Elenor MFE 重复实现；局部 feed 不应独立任意访问 HBM。[来源 H07-1](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=58f914c0-6d87-48b7-8271-f90d70d93b44) [来源 H07-2](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=7c9040fe-a6df-4e01-93e7-b1094223ca82)                                                                                                                                                                                                                        |
| 10/5–6             | 增强 MFE 2D DMA＋消费者布局适配＋有条件 Tile-local LTU                                                                  | 先 metadata/view/fusion 消物化，再连续/二维搬运与多请求在途，再 BOA 右操作数转置、EVU broadcast/shuffle；高频剩余 pack/transpose 才加一条 Tile 共享 LTU。不为每个 context 或每个内存算子复制完整器件。[来源 H19-1](https://chatgpt.com/c/6ac0cc5d-b464-83ee-939a-28b210319244?messageId=e5c632ca-4199-4327-b0ee-3bdf082d8f07)                                                                                                                                                                                                                                |
| 10/6 备选被暂缓    | 完整通用 XLU、独立 Sort／Concat／Expand／NonZero／Scan 单元                                                             | 助手不建议首版直接增加，先在 EVU＋MFE 及分块算法实现；scatter 不等于 scatter-reduce，不能以支持 scatter 推导支持冲突 atomic。[来源 H19-1](https://chatgpt.com/c/6ac0cc5d-b464-83ee-939a-28b210319244?messageId=e5c632ca-4199-4327-b0ee-3bdf082d8f07)                                                                                                                                                                                                                                                                                                         |
| 10/20              | MFE 可 L2→Shared，也可 L2→RF，乃至同份数据 multicast 两端                                                               | RF/Shared 物理合并与搬运能否直达 RF 是两个独立选择。直达需 owner/generation、RF reservation、scoreboard、lane layout 和最后读者；双路径不使 L2/NoC 带宽翻倍。[来源 H21-1](https://chatgpt.com/c/6ad72b86-1d40-83e8-9e9c-805306982abc?messageId=26579bc3-abbf-4008-aaf6-b4f49626714c)                                                                                                                                                                                                                                                                         |

**6 月 LTF 与 10 月 LTU 应分别归档**：前者曾被设想为 MFE 内较广的 data-plane transformation fabric；后者是在 Nexus 当时实现边界下，优先 MFE/消费者适配后才加有限 Tile-local 重排单元。不能仅因缩写相近就断言是同一个已实现模块或一次正式改名。

### RF／L1／Shared：三次不同说法

1. **旧四层思考**：HBM(L3)→Group L2→Tile L1→器件内部 RF/输入 scratchpad。用户认为多了一层大块复制。
2. **2026-10-15 用户合并提案**：取消独立 L1，Tile 本地 SRAM 组织成 RF/VRF，L2 直接写 RF；HBM→L2→Tile RF，保留本地复用和多 context 工作集。助手认可作为候选，但若只是删大 L1 留小 RF，可能把流量推到 L2；少模块名字不等于省面积。[来源 H20-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=4a8dc1bd-ddf5-4d5a-a20a-ff1354cfa5e5) [来源 H20-2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=c9c4d766-a731-4ac8-a54a-5ee56cbf86d2)
3. **2026-10-17 用户重新澄清**：真正目标不是“所有东西叫 RF”，而是**所有计算器件共用主要 Tile 本地张量存储**，不再各自有大输入 scratchpad。允许 UCE GPR、流水寄存器、operand FIFO/latch、矩阵 AM；主要本地层称 Shared Memory。[来源 H22-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=b0c04d5b-6aeb-4a74-8210-0a465fa332eb) [来源 H22-2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=0e71df90-56f0-46a1-84a6-e883870f0dce)
4. **2026-10-20 再问物理统一／直达**：同 bank 池划 RF/Shared 两区域只是容量统一，若还复制就没去掉读写；同数据提供两视图才是真正去重。高频 VRF 端口与 PE feedback AM 仍需单独评估，助手偏首版先统一接口，不强制所有物理 SRAM 合一。[来源 H21-1](https://chatgpt.com/c/6ad72b86-1d40-83e8-9e9c-805306982abc?messageId=26579bc3-abbf-4008-aaf6-b4f49626714c)

RF 可以由 SRAM 实现，但“寄存器名＋VL/operand dependency”与“地址＋长度＋stride＋range ownership”不是同一个访问合同。改 controller 不能自动解决 alias、写后读、in-flight return 路由与多 context 隔离。[来源 H20-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=4a8dc1bd-ddf5-4d5a-a20a-ff1354cfa5e5) [来源 H20-2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=c9c4d766-a731-4ac8-a54a-5ee56cbf86d2)

### EVU 从 descriptor 到 Micro-SIMT 和自主 RVV 的备选序列

| 讨论时间与性质              | 执行模型／配置                                                                                                                     | 状态、选它的理由与代价                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6/11 文档评审               | Descriptor-only predicated vector datapath：op_kind/slot/shape/mask/stride，FSM 执行固定算子，无用户可见 code_addr                 | 当时助手建议 first-silicon V1A，queue depth 1–2；BF16/FP16 输入输出、FP32 reduction，local softmax/norm、尾 mask；scatter/atomic 不放基础闭环，gather 只可选 profile。只是当时文档切线建议，不能据此否定 10 月 gather。[来源 H25-1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=b1489649-4764-4561-9cbb-ad1dfbbec4c0) [来源 H25-2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=031676cf-d07a-4f26-9d46-bb81c8ae35c4)                                                       |
| 6/11                        | Microcoded EVU：外部 descriptor，内部 vload/vadd/vreduce/vexp/vstore 微序列                                                        | 助手 V1.5/V2 折中；复用固定高频算子，外部 ABI 简单，增加 microcode memory、decoder 与验证，但不必完整用户编译器后端。[来源 H25-1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=b1489649-4764-4561-9cbb-ad1dfbbec4c0) [来源 H25-2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=031676cf-d07a-4f26-9d46-bb81c8ae35c4)                                                                                                                                                         |
| 6/11                        | Limited Vector ISA EVU，RVV/SVE-like 的 VL、predicate、vector registers、load/store/reduce/shuffle                                 | 比 descriptor 灵活；需 EVU kernel IR、指令选择、register allocation、code/arg/scratch/event ABI。“like RVV”不是已实现标准 RVV。[来源 H25-1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=b1489649-4764-4561-9cbb-ad1dfbbec4c0) [来源 H25-2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=031676cf-d07a-4f26-9d46-bb81c8ae35c4)                                                                                                                                               |
| 6/11 用户明确偏好           | **没有 warp scheduler 的 SIMT 单元**；助手命名 shared-PC Micro-SIMT／Predicated Microthread Vector Engine                          | 每 lane 有逻辑 thread/lane_id、predicate 和 lane RF；一个 shared PC／single-wave sequencer，无 multi-warp residency、无 per-thread PC，memory stall 主要靠 MFE 与 Tile engine overlap 隐藏。用户说暂不按阶段切分，表达长期方向偏好。[来源 H26-1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=9b765cd6-8e2f-4e54-b272-18ff8596ca98) [来源 H26-2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=8065cf69-e4aa-4956-b624-319ccab2dbdb)                                          |
| 6/11 三种 divergence 子路线 | A if-conversion/select；B structured mask stack；C per-lane PC                                                                     | 助手优先 A；B 长期可选，增加 mask stack 与 structured-control verifier；C 不推荐，会转向 MIMD/threaded-core array。这些是 Micro-SIMT 内部备选，并非三种都实现。[来源 H26-1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=9b765cd6-8e2f-4e54-b272-18ff8596ca98) [来源 H26-2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=8065cf69-e4aa-4956-b624-319ccab2dbdb)                                                                                                               |
| 6/11 双前端提案             | Descriptor frontend＋Micro-SIMT backend，fixed-op 和 code kernel 两模式并存                                                        | 保留常用 fast path，又容纳长尾算子；需要更明确的 ISA、slot/buffer 访问与 fault 语义。没有用户最终冻结 ABI 的证据。[来源 H26-1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=9b765cd6-8e2f-4e54-b272-18ff8596ca98) [来源 H26-2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=8065cf69-e4aa-4956-b624-319ccab2dbdb)                                                                                                                                                            |
| 7/15 TCC                    | CE→VE programmable vector chain，8–16 entry chain descriptor；可不独立 Vector PC                                                   | 用可组合 mul/add/max/exp/convert/reduce 等替代 activation enum；固定 CE→VE route；是局部融合提案，不是完整 multi-kernel EVU。[来源 H07-1](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=58f914c0-6d87-48b7-8271-f90d70d93b44) [来源 H07-2](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=7c9040fe-a6df-4e01-93e7-b1094223ca82)                                                                                                                                                 |
| 10/6                        | 可编程 masked SIMD/vector；Tile UCE 选 task/context，EVU 轻量 micro-PC/sequencer、scoreboard、本地访存前端，一次一个 active kernel | 助手偏好，不另配 GPU warp scheduler。物理 lane 与向量长度不同，SFU/reduction/shuffle 要单独建模。32/64/128 lane 是探索，不冻结。[来源 H27-1](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53) [来源 H27-2](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=9cc8b569-f2d5-4565-af54-87dac3858fbd) [来源 H27-3](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=64cc222d-b067-42a4-a021-5026b3438632)                      |
| 10/6 kernel 分解            | 长等待拆成生成 index→异步 gather→后处理；短本地 gather 可留 kernel                                                                 | UCE 切 context 不等于 busy EVU 能接另一 kernel；若 EVU A 因远端 gather 占住设备，必须拆 kernel 或新增真正多 kernel 驻留机制。[来源 H27-1](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53) [来源 H27-2](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=9cc8b569-f2d5-4565-af54-87dac3858fbd) [来源 H27-3](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=64cc222d-b067-42a4-a021-5026b3438632)                         |
| 10/15–17 Ara 风格后端       | 轻量 multi-context UCE＋可复用 Ara 风格向量后端＋独立 OPA 组                                                                       | 原版 Ara 多条向量指令在途不等于多程序 PC。完整 CVA6 每 Tile 大规模复制的面积／功耗未定；用户质疑过 64 Tile 成本，助手偏不先冻结 full CVA6＋原版 Ara。[来源 H28-1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74) [来源 H28-2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=3e29a423-c782-4d28-a36d-16bd0e9e3186) [来源 H28-3](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=c79258c6-1b93-4220-8017-92b855784278) |
| 10/23 用户                  | 自主 RVV kernel：提交函数入口＋输入输出参数后设备自己跑，不由 UCE 逐条推 RVV 汇编                                                  | 需要本地 PC/取指、标量 RF、地址计算、分支、必要 scalar load/store；Ara 后端本身不构成完整自主 kernel 前端。可轻量 RISC-V＋标准 RVV，也可受限 micro-sequencer。[来源 H30-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=f142eddd-6116-537b-93b0-1a9991927ea8)                                                                                                                                                                                                                                        |
| 10/23 助手首版与扩展        | 首版每向量设备单 active kernel；多 UCE contexts 和 matrix/MFE 并行仍保留。多 kernel RVV 驻留留作另一层扩展                         | “多等候任务”不等于“多 active kernel”。若未来共享程序前端，各 kernel/context 仍需自己的 PC/状态，并共享总发射带宽。[来源 H30-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=f142eddd-6116-537b-93b0-1a9991927ea8)                                                                                                                                                                                                                                                                                    |

### 2026-10-21 五种路线完整保留

用户问如何把 RISC-V 自定义汇编与硬件多 context 融合，助手列以下五条，偏 B、备选 D；**未见用户明确选择最终核或最终流水微架构**。[来源 H29-1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=6f22235b-7231-4426-a6bc-993c71da1653) [来源 H29-2](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=1c2bac32-a0e8-5116-bc7a-1a44a3d374a5)

| 路线                                    | 保留状态／怎样切换                                                                                 | 优点                                                                     | 代价／历史状态                                                                                              |
| --------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| A 软件 coroutine／continuation          | 普通 RISC-V 上软件保存恢复 PC/GPR，运行 ready queue，异步 submit/wait                              | 改硬件少，原型快，容易复用软件工具                                       | 高频切换与唤醒由软件承担；不等价于目标中的低开销硬件驻留。候选／基线                                        |
| B 多份架构 context＋共享顺序流水        | 每 context PC/GPR/必要 CSR；共享 fetch/decode/ALU；wait/yield/未接纳 submit 反压时切 ready context | 与“汇编写做什么／等什么、硬件决定谁运行”契合；已提交异步 engine 任务继续 | 初版建议 2–4 contexts，切换前排尽普通标量流水；新增 owner-tag、event wakeup、配置隔离。助手首选，非用户定案 |
| C IMT／barrel                           | 每周期轮转不同 ready context，共享流水                                                             | 连续隐藏短依赖，公平 interleave                                          | 单 context issue 份额下降，context state 和 pipeline tagging 更强；未必适合主要等待长异步任务。候选         |
| D 普通 RISC-V＋独立多 context sequencer | RISC-V 处理配置／复杂控制；硬件 sequencer 跑受限 async program                                     | 核改动小，保留低开销硬件 ready/select/wakeup                             | 两套执行格式与编译边界；需要证明不是固件调度成为细粒度关键路径。助手备选                                    |
| E 多个小 RISC-V 核共享加速器            | 每核天然独立 PC/GPR，外部共享 matrix/vector/MFE                                                    | 并行来源直观，软件易理解                                                 | 复制 fetch/decode/ALU/cache 等面积功耗，共享引擎仲裁仍在；更多核不创造更多 engine/AM/带宽。候选             |

### 跨时期硬件参数登记

这张表保留历史数字及其身份，**不是把所有数字组装为同一芯片配置**。旧参数即使出现在 YAML 或模拟器中，也不代表对应 RTL/PPA 已成立。

| 时间与方案              | 参数或规模                                                                  | 参数性质                   | 应怎样解读                                                | 来源                                                                                                                                                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2025-08 工业 NPU 背景   | 12 L1 controllers / 12 slices；ARM Host；L2 DCCM/CSM                        | 用户描述的旧公司架构       | 是改造背景，不是当前个人芯片规格                          | [来源 S03](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=ba2698fc-65fa-437e-85ea-e37497dc58a7)                                                                                                                      |
| 2026-03 Island          | 1000+ clusters 的 scheduler 扩展问题                                        | 用户提出的规模问题         | 不是已确定芯片实例，更无控制器面积测量                    | [调度规模问题](https://chatgpt.com/c/69bd6375-49c8-8332-8e2e-9c9cf80abe0b?messageId=d7acf139-5fe6-49e4-8e12-6d915472dd09)                                                                                                                  |
| 2026-03 BOA             | 3 nm、单芯片、1 PFLOPS FP16 / 1000 TOPS                                     | 用户可行性目标             | 不能等同已实现 dense/sparse 性能或确定面积                | [BOA 目标](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1)                                                                                                                      |
| 2026-03 BOA 面积讨论    | 32 clusters / 256 BOA、约 4 TFLOPS/BOA、640～1060 mm²                       | 旧助手提案/估算            | 无工艺库、综合/物理实现校准，不能作为产品 PPA             | [历史估算](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=c0237d49-5acc-420d-874a-995c11662cf6)                                                                                                                      |
| 2026-08 滑动补位        | 一次四个 Context 的例子                                                     | 说明调度语义的例子         | 用户强调完成一个补一个，并未固定硬件为四槽                | [来源 S49](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=c474e393-071a-44d0-94ba-2c4f5fd41010)                                                                                                                      |
| 2026-10-06 EVU 探索     | 32/64/128 lanes；从 64 FP32 元素/周期试起                                   | 助手 sweep/起点建议        | lane 数、vector length、SFU 吞吐分别建模                  | [来源 S24](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53)                                                                                                                      |
| 2026-10-14 旧模型       | 8 个 16×16 OPA；32 EVU lanes；1 GHz；L1 读峰值 512 GB/s                     | 旧分支配置的历史分析       | 不是后来 4×8×8 的同一版本，也不是测得频率/带宽            | [来源 S23](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786)                                                                                                                      |
| 2026-10-14 带宽例子     | 无跨 OPA 复用时矩阵约 512 B/cycle；32-lane BF16 双输入 mul 另需 128 B/cycle | 特定假设下的理论需求估算   | 640 B/cycle 超过旧 512 B/cycle 预算；输出、MFE 等尚需另计 | [来源 S23](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786)                                                                                                                      |
| 2026-10-14 广播候选     | 2×4 协同广播例子约 192 B/cycle 输入                                         | 需要相应广播硬件的理论例子 | 后来用户明确无广播，不能把该流量降低算到独立阵列版本      | [来源 S23](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786)[来源 S29](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79) |
| 2026-10-17 小阵列       | Tile 内四个独立 8×8；无连接/广播；当时各自 AM                               | 用户明确的当时设计         | 后来 AM 改共享；单条命令 K 不必等于 8                     | [来源 S29](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)                                                                                                                      |
| 2026-10 RISC-V 原型     | 初版 2～4 scalar Context；另有两 Context Ara 验证建议                       | 助手原型规模建议           | 不是最终 Context 数，也不是完整 RVV Context 的成本        | [来源 S27](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=6f22235b-7231-4426-a6bc-993c71da1653)(https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74)           |
| 2026-10-20 输出容量例子 | 256×256 FP32 输出占 256 KiB                                                 | 算术示例                   | 说明输出 tile 必须匹配 AM 容量，不是冻结 AM=256 KiB       | [来源 S31](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)                                                                                                                      |
| 2026-10-23 最小联仿     | 一个 Tile、两个 Context、一个整数向量加 RTL EVU                             | 助手验证起点               | 为缩小闭环，不是最终芯片规模                              | [来源 S43](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=0f65b652-2cd4-4f2d-96a0-ff552be8f68e)                                                                                                                      |
| 2026-10-26 Shared AM V1 | 每 Tile Context 跨全部矩阵 engine 最多两个 AM slots；固定 engine affinity   | 用户明确软件/资源合同      | 不能推出总槽数、每槽容量、Context 驻留上限                | [来源 S32](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)[来源 S33](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4) |
| 2026-10-26 AM 物理示例  | 4 groups×2 slots；64×64×32 bit=16 KiB/slot；合计 128 KiB                    | 助手示例配置               | slot、bank、group 不等价；还需服务带宽/adder/端口         | [来源 S34](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)                                                                                                                      |
| 2026-10-26 K 例子       | 16×16 阵列处理 16×16×256；可 1 个长命令或 16 个 K=16 命令                   | 计算分解示例               | 同一输出链复用同一 AM，命令数与槽数无直接对应             | [来源 S35](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f)                                                                                                                      |
| 2026-10-28 irregular    | 逐元素随机 4 B 请求到每 index 连续 KB 级块                                  | 建议评估区间               | 小请求前端与大块 DMA 不能用同一个峰值带宽假设             | [来源 S36](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=d7fb793b-8c12-4d4e-a287-9bb8df90310b)                                                                                                                      |

### 其他历史计算与带宽例子

以下补充的是不同候选下的估算口径，不是性能实测。表中未写工艺实现的 GHz 数值都只是计算假设。

| 讨论时间与方案    | 数字及计算口径                                                                                            | 证据级别与边界                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 3/25 完整 4×4     | 64 mul＋48 add；16 棵四输入求和；完全展开与流水化是不同 latency/throughput 方案                           | 推导／探索；不是 4×4 外积每拍 16 mul 就完成完整 K=4 GEMM。[来源 H01-1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=f38e5bc6-c9a5-43e3-9fc4-606d6cce2df1) [来源 H01-2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=ef44cbee-6b0f-4c87-a043-2f9a9f6f5f53)                                                                                                                                                                  |
| 3/25 shape 扫描   | 4×8×8、8×4×8、8×8×8、8×8×16；K 候选 4/8/16                                                                | 助手候选，不是最终阵列尺寸／ISA K 限制。[来源 H02-1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=763064a4-8d86-452d-86a9-e0c9bff0bff4) [来源 H02-2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=d2d7ede8-9824-4f91-be78-f37adbbf4dd1)                                                                                                                                                                                    |
| 5/19 OPA context  | 4–8 contexts，各 acc bank/k_count/mode                                                                    | 助手发展路线，不是后来 UCE 配额或 AM 两槽规则。[来源 H04-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)                                                                                                                                                                                                                                                                                                     |
| 5/19 BOA＋Flex    | 版本 A：60/20/17/3%；B：50/30/17/3%；C：75/10/12/3%，依次 Dense/Flex/Memory/Control                       | 面向不同负载的助手示例面积预算；未综合，也不应把各类宽区间独立相加成超 100% 的规格。[来源 H06-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=c7eef51c-adc9-4a1c-bd9b-71c149883897) [来源 H06-2](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=f9a874e2-8de6-4e1e-8d9b-20d949fb5f28)                                                                                                                                        |
| 5/19 Flex core    | 4–8 warps/contexts、16–32 SIMD lanes                                                                      | 双岛提案，与 6/11“无 multi-warp EVU”是不同候选。[来源 H06-1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=c7eef51c-adc9-4a1c-bd9b-71c149883897) [来源 H06-2](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=f9a874e2-8de6-4e1e-8d9b-20d949fb5f28)                                                                                                                                                                            |
| 6/11 Micro-SIMT   | 32 lane 示例；每 lane 16×32-bit 寄存器，另 scalar_reg[16]、pred_reg[8]、shared PC/mask                    | 仅 state struct 例子，lane RF 共 2 KiB 不含其他状态；不是当时完整实现容量。[来源 H26-1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=9b765cd6-8e2f-4e54-b272-18ff8596ca98) [来源 H26-2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=8065cf69-e4aa-4956-b624-319ccab2dbdb)                                                                                                                                                 |
| 6/30 LTF          | skid 1–2 entry；ordered stream FIFO 2–8；reorder buffer 按 transform 需要                                 | 深度举例，未定每个接口／burst/outstanding 的最终 credit 合同。[来源 H18-1](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=1b9db319-6710-4eae-a283-b664bc976e76)                                                                                                                                                                                                                                                                                      |
| 7/15 TCC          | reduce_depth {0,1,2,3}；output_width wide/medium/narrow；8–16 vector micro-op；计算侧面积增量门槛 10–15%  | 全是助手候选／设计门槛，绝非已测增加幅度。CE/VE/commit 应分别建模。[来源 H07-1](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=58f914c0-6d87-48b7-8271-f90d70d93b44) [来源 H07-2](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=7c9040fe-a6df-4e01-93e7-b1094223ca82)                                                                                                                                                         |
| 10/6 EVU 带宽     | 32 lane FP32 `C=A+B`，EVU 读256＋写128=384 GB/s；若一次 L2 搬入/搬出，MFE 另384，L1 总768（读384＋写384） | 单层实际访问必须相加，不是 HBM 768；BF16 对应总384。中间留 VRF 的融合 kernel 会降低平均 SRAM 流量。[来源 H27-1](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53) [来源 H27-2](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=9cc8b569-f2d5-4565-af54-87dac3858fbd) [来源 H27-3](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=64cc222d-b067-42a4-a021-5026b3438632) |
| 10/14 通用 OPA    | 16 个4×4=256 lanes，BF16 @1GHz：独立外积输入256 GB/s；独立 binary elementwise 读1024、写512 GB/s          | 假设每 lane 每拍有效运算。共享 LSU 不会消除有效字节；如果有4×4协同广播，源端独特输入可降64 GB/s，但 OPA 入口接收总量仍256。[来源 H08-1](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=d11334f7-8ca1-4c6d-8a44-172e6eec96bf) [来源 H08-2](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=c5d6d593-88f0-4814-abe8-bb68232ab3b4)                                                                                                 |
| 10/17 等 MAC 比较 | 1×16×16、4×8×8、16×4×4 均256乘法器；OS 满载输入期 BF16边界输入分别64、128、256 B/cycle                    | 不含startup/drain/writeback；不是通用 WS 公式，也不是实际性能结果。广播可降源端重复读取，但用户后来明确无广播。[来源 H10-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=a0433a52-5efe-5d1d-b8e8-f3c0a58174cc)[来源 H11-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)                                                                                                              |
| 10/17 轻量 DPE    | 8×32 BF16 operand hold=512B，双缓冲1KiB                                                                   | 用于说明 DPE 不一定需要大独立 scratchpad；不代表已采用8×32 DPE。[来源 H10-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=a0433a52-5efe-5d1d-b8e8-f3c0a58174cc)                                                                                                                                                                                                                                                                                    |
| 10/17 split-K例子 | [8×1024]×[1024×8]，四阵列各 K=256；4份FP32 partial读取1024B、最终写256B、192 add                          | 归约算账例子。8最终元素/拍的四输入树需128B/拍 AM输入；树不创造 AM 带宽。[来源 H10-1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=a0433a52-5efe-5d1d-b8e8-f3c0a58174cc)                                                                                                                                                                                                                                                                            |

容量的两个基本关系可用于检查历史例子，不代表新规格：

- 输出累加存储：`S_AM = M_tile × N_tile × bytes_per_accumulator`。
- 双缓冲 A/B 输入：`2 × K_chunk × (M_tile × bytes_A + N_tile × bytes_B) ≤ S_AB`。

同一输出形状下，增加 K 不增加 AM 元素数，但增加输入流量、执行时间、槽寿命和数值误差/溢出风险。1 GHz、64 个 FP32 FMA lanes 的纯理论峰值是 128 GFLOPS，不是 1 TFLOPS；这也不包含访存、SFU 和 reduction 的实际限制。[来源 S24](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53)[来源 S35](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f)

<a id="decision-log"></a>

## 七 跨版本决策日志

这张表记录“改变了什么”，不把最新结果覆盖到旧方案。没有明确否定的候选写作“未采纳证据”而非“已永久放弃”。

| 时间            | 旧方案或问题                                 | 后来的决定或修订                                      | 保留的洞见与边界                                                      | 来源                                                                                                                                                                                                                                                                                                                                                            |
| --------------- | -------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025-08         | L2/L1 各自生成 bytecode                      | 用户要求只生成 L2 bytecode；L1 固定命令执行           | 薄 runtime 与设备异步保留；后来 Tile program 不受旧案限制             | [来源 S04](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=36fc1de6-01c5-4528-9edc-a863d3b4df40)                                                                                                                                                                                                                                           |
| 2025-08-31      | DCCM 滑动窗口取 bytecode 与分段/分页候选     | 用户明确采用分页实现                                  | DDR 程序与 DCCM page cache、segment/page、hint/safepoint 属该历史方案 | [分页选择](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=47004e9d-13a0-475a-9037-a6a716586273)                                                                                                                                                                                                                                           |
| 2025-09-04      | 全局 cost model 搜索 micro-conv 组合         | 用户选择离线最优 microkernel family，再查表与 tiling  | 不等于禁止后来 Nexus 的合法 schedule/profile 优化                     | [离线路线](https://chatgpt.com/c/68aff962-3f34-832a-867c-14e54589d5d9?messageId=caea8cdd-64ed-4a4e-8d42-11f14883ae08)                                                                                                                                                                                                                                           |
| 2026-03-21      | 把 Island 扩大解释成多芯片集群               | 用户纠正为单芯片内 cluster                            | 不能把旧助手的数据中心方案算作用户芯片设计                            | [来源 S56](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)                                                                                                                                                                                                                                           |
| 2026-03-27      | PE+reduce 含义不清                           | 用户明确 BOA，reduce 用于 K-tiling 累加               | 并非天然拥有任意全局归约/原子能力                                     | [来源 S56](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)                                                                                                                                                                                                                                           |
| 2026-07         | 偏 LLM 的专用方案                            | 用户要求更广的 AI 推理、汽车机器人                    | 不等于承诺完整 GPU/HPC 兼容                                           | [来源 S01](https://chatgpt.com/c/6a646992-d14c-83ee-b27f-a7bad0d4fb79?messageId=cc79cd85-2847-43be-86be-861ddd73d67a)                                                                                                                                                                                                                                           |
| 2026-07-21      | 默认沿用 Elenor 约束                         | CORA 独立讨论                                         | 不强造 CORA 是 Elenor 正式下一代的关系                                | [来源 S10](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)                                                                                                                                                                                                                                           |
| 2026-07-22～26  | Region IR 下沉 L2 缩小重排空间               | 保留跨 Region compiler 优化；探索 L2 multi-context    | MC-PGE 是候选，不是后来全部架构的必选模块                             | [来源 S10](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)                                                                                                                                                                                                                                           |
| 2026-07-29      | 新案混入 CORA                                | 用户要求除非提及否则不考虑 CORA                       | 版本隔离是用户明确要求                                                | [来源 S11](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=b628a6f0-c936-44d1-b572-f623c2bb989b)                                                                                                                                                                                                                                           |
| 2026-07-31      | 任意 Tile 间 producer-consumer 流水/MPMD     | 用户选择 SPMD，降低 compiler 压力                     | Tile 内 async、独立 PC、局部流水仍保留                                | [来源 S11](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=b628a6f0-c936-44d1-b572-f623c2bb989b)                                                                                                                                                                                                                                           |
| 2026-08-07      | output buffer 生命周期遗漏                   | store_done 后才释放输出                               | 三事件不能替代所有写回、引用和 safe retirement                        | [来源 S13](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=b5f8eb2c-e61a-49e8-9d74-0415cf36ec24)                                                                                                                                                                                                                                           |
| 2026-08-07 后续 | 把 pool、lease、credit 全暴露在 IR           | 用户要求简洁 `nest.context` / `nest.tile.program`     | 简化表述不等于删除内部安全机制                                        | [来源 S13](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=b5f8eb2c-e61a-49e8-9d74-0415cf36ec24)                                                                                                                                                                                                                                           |
| 2026-08-10      | 通用 FU 间 forwarding/FIFO 层层连接          | 倾向命名 L1/ACC slots 与 token/scoreboard             | 固定 fast path 可选，大张量不必走通用 FIFO                            | [收缩案](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=633ed92c-3525-477f-b954-47ed6116f51c)                                                                                                                                                                                                                                             |
| 2026-08-12      | 固定 mix-window 批次都结束才补位             | 用户要求一个完成即可补下一个                          | “四个”只是例子，不是冻结 Context 数                                   | [来源 S49](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=c474e393-071a-44d0-94ba-2c4f5fd41010)                                                                                                                                                                                                                                           |
| 2026-08～09     | SPM/Cache/FIFO 模式边界与 bank 划分不明      | 确认 Cache/SPM；同级 bank 内采用相同比例              | FIFO 仍未定；容量统一不保证双满带宽                                   | [来源 S18](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=0bcc2e8e-8787-44d1-a326-d79ef6acc382)                                                                                                                                                                                                                                           |
| 2026-09         | 只要 DAG 无环或内存够就无死锁                | 加有限表拓扑登记、memory admission、完成/排空资源分析 | 每个机制只排除特定死锁，不是全系统证明                                | [来源 S50](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=b34b621f-95e6-4d36-9183-75d2380f9315)[来源 S51](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=d9720292-8f8f-41e1-86ba-1cda1d6e26b7)                                                                                                                      |
| 2026-09-18      | 所有跨 Context 等待都必须阻塞 Device         | 助手撤回过度推论                                      | 依赖可在准入前表达，避免等待者占生产者资源                            | [来源 S52](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=d03c1861-1a68-4ee2-9dd0-fb86be4dea6c)                                                                                                                                                                                                                                           |
| 2026-09-23      | profile 可能随执行动态混用                   | 用户明确一个 Context 一个 profile                     | 程序实现版本、执行 assignment、物理域 profile 要分开                  | [来源 S20](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7eff8b05-df9c-4435-ad12-03b9e8903f6c)                                                                                                                                                                                                                                           |
| 2026-10-01      | Group 承担复杂动态控制流                     | 用户要求不做 if/else，动态分支由 CPU；增加 for        | “有界计数且次数预知”是助手细化，非完整用户规范                        | [来源 S15](https://chatgpt.com/c/6abb7d0e-0bb8-83ee-8042-892cb647deb7?messageId=32ae04b5-99f4-48a3-af3a-5361bef40597)                                                                                                                                                                                                                                           |
| 2026-10-02      | 另一模型建议 Group gather 取代 Tile gather   | 用户澄清原方案只有 Tile gather，比较尚未决定          | Group 页表预取可增强，不必删除计算驱动 Tile gather                    | [来源 S54](https://chatgpt.com/c/6abf1244-7a3c-83ee-9527-ff150a4e701a?messageId=4666263b-e276-4a5c-ab5d-8cbe2f95254d)                                                                                                                                                                                                                                           |
| 2026-10-05      | 跨 Context 任意 copy/move 被当成通用优化     | Context ownership、shared 发布后只读；move 收益需证明 | Context 内 in-place/reuse 不因此被禁止                                | [来源 S53](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=44bc19fb-f07f-4c0a-ab05-e271002e3f1c)(https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=83bd4615-28c7-4a8c-9a00-61ab3a52d5c9)                                                                                                                                |
| 2026-10-11～23  | 一次完成全芯片/购买 FPGA 才能验证            | 收窄为个人可承担的 Python+局部 RTL 闭环               | 逻辑/协议/周期与 PPA/硅片证据分开                                     | [来源 S02](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=ac5d71a8-fff0-49ea-881c-99c3a7f55687)[来源 S42](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=379dcbda-2e9c-48de-82a3-b64d44ba045f)[来源 S43](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=0f65b652-2cd4-4f2d-96a0-ff552be8f68e) |
| 2026-10-15      | Ara 多在途向量操作等同多程序 Context         | 澄清还需多 PC、状态隔离、owner completion、唤醒       | 外部队列或增加两个 PC 都不足                                          | [来源 S26](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74)                                                                                                                                                                                                                                           |
| 2026-10-17      | 每设备私有大输入 scratchpad、先从 L1 复制    | 用户要求主要张量直接来自 Tile Shared                  | AM、GPR、pipeline register、小 FIFO 允许独立                          | [来源 S25](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=b0c04d5b-6aeb-4a74-8210-0a465fa332eb)                                                                                                                                                                                                                                           |
| 2026-10-17      | 误解为协同广播阵列簇                         | 用户纠正四个独立 8×8，无连接/广播                     | 当时各自 AM，后来共享 AM；不混入旧 8×16×16 配置                       | [来源 S29](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)                                                                                                                                                                                                                                           |
| 2026-10-19～20  | ring/reduce tree/Reduce Memory               | 用户转向先不用 tree，顺序 K-step 累加                 | 并行 split-K 与顺序 K 链是不同算法组织                                | [来源 S30](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda)[来源 S31](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)                                                                                                                      |
| 2026-10-20      | 穿插 Context 要保存 PE 多份 OS 或往返 Shared | AM 持久存旧 C，PE 只保存当前 step 状态                | 通过显式 acc handle 与 step_committed 保障正确性                      | [来源 S31](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)                                                                                                                                                                                                                                           |
| 2026-10-23      | UCE 逐条参与向量汇编                         | 用户希望向量设备收函数入口和参数后自主执行            | 设备仍需自己的执行前端/状态，不是后端凭空自治                         | [来源 S28](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=f142eddd-6116-537b-93b0-1a9991927ea8)                                                                                                                                                                                                                                           |
| 2026-10-26      | AM 共享被理解为自由迁移/无限并发             | V1 固定 engine affinity，每 Context 总共两槽          | 共享容量不自动提供迁移；affinity 不等于 engine 独占                   | [来源 S32](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)[来源 S33](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)                                                                                                                      |
| 2026-10-26      | 较早 one-shot bypass 不占持久 AM 的例外      | 后续 V1 指南将 one-shot 也计入两槽                    | 不能把旧旁路例外混回最新限制                                          | [来源 S33](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)                                                                                                                                                                                                                                           |
| 2026-10-26      | 两槽限制被误解为 K 只能短或两步              | 区分 K_total/K_chunk/K_cmd                            | AM 随输出面积而非 K 增长；长 K 增加驻留时间和数值风险                 | [来源 S35](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f)                                                                                                                                                                                                                                           |
| 2026-10-28      | 二维 DMA、动态 stride 被等同高效 gather      | 区分规则地址、index-dependent AG 与请求吞吐           | 能表示地址不等于能高效服务逐元素随机请求                              | [来源 S36](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=d7fb793b-8c12-4d4e-a287-9bb8df90310b)                                                                                                                                                                                                                                           |

<a id="execution-contracts"></a>

## 八 编程层次与执行合同

### 控制与 IR 的其他历史版本

| 时间       | 当时方案                                                | 差异与边界                                                                                 | 来源和状态                                                                                                                                                                       |
| ---------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-05-31 | Host Runtime／Device Runtime／Tile Runtime 三层         | 当时用于划分软件责任的草案；不证明最终存在三套软件 runtime                                 | 助手草案。[来源](https://chatgpt.com/c/6a1b8c44-a1bc-83a5-913d-f7cfd7d793ea?messageId=b4a86e78-3c34-41d2-bd41-8bf05bf42ba9)                                                      |
| 2026-07-22 | CORA：compiler 全局排程＋CPU runtime 保存绑定提交命令流 | L2/HBM 回到数据面；异步引擎队列执行；保留 compiler reorder、prefetch、lifetime 能力        | 助手接受纠正后的分支方案，**不是取消后来 NEST Group context**。[修正](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=630036f1-2536-41e1-b034-f704e3c28b08) |
| 2026-09-11 | Device／Group／Tile 三级明确化                          | Device CPU IR；`nest.context` Group；tile program Tile。不是 Group 内分前后端后多了一层 IR | 用户明确。[原文](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=b34b621f-95e6-4d36-9183-75d2380f9315)                                                      |

### Host Nexus Group Tile 的职责边界

Host/Nexus 负责绑定参数和 global views、提交 Context、建立外层依赖并等待完成。partition 若只是建立互不重叠的视图，属于 metadata 操作，不等于搬运数据。相同逻辑的两个实例可复用同一 NEST context program；两次 async submit 放在 await 前，在无依赖且资源许可时可并发准入。提交 FIFO 不意味着执行或完成 FIFO。[来源 S14](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=cee8808a-d227-4db9-8bbf-c63b41cc8dc4)

Group/L2 Context 负责 L2 输入输出、HBM/L2 预取、SPMD 任务分发、输出写回及资源回收。Tile program 负责局部地址、L1 数据准备、矩阵或向量计算、局部异步与完成信号。Tile task ID 是逻辑工作身份，不是固定物理 Tile ID，也不是硬件 context slot。

在当前有限控制流方向中，Group 不承担数据相关 if/else。你提出只增加 for；助手建议进一步限定为进入循环前次数已知的有界计数循环，把 EOS、while 和动态分支留给 CPU。这里“有界且次数预知”是建议细化，不能冒充你已经冻结的完整控制流规范。[来源 S15](https://chatgpt.com/c/6abb7d0e-0bb8-83ee-8042-892cb647deb7?messageId=32ae04b5-99f4-48a3-af3a-5361bef40597)

### SPMD 与多 Context 分别解决什么

SPMD 表示同一个逻辑 kernel/grid 用不同参数或 program_id 执行，不意味着所有 Tile 锁步。每个 Tile 可以有独立 PC、Context 和进度；static grid 或 dynamic work queue 都可保持 SPMD。Tile 内异步、软件流水与时间复用仍然成立。[来源 S11](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=b628a6f0-c936-44d1-b572-f623c2bb989b)

多 Context 在一个实例等待数据、事件或资源时推进别的 ready 实例。它提高的是利用率和延迟隐藏能力，不增加物理峰值。一个 Context 也可以先发出多个独立异步操作再 await；不能把所有 overlap 都归功于 multi-context。队列更深也不等于引擎数量更多。

逻辑 Context、物理 slot 和长期 task/event ID 必须分离。Context slot 被复用后，旧异步完成仍应路由到提交时的 owner/generation，而不是当前占槽者。驻留数还受到 L1、L2、AM 和事件资源约束，不等于硬件支持的 PC 数。

10月17日，你进一步明确任务身份必须覆盖整个系统，包括 Group，而不只是 Tile 本地槽；希望有系统可识别的任务编号，或由 L2 管理复用。助手提出 HomeGroupID、TaskSlot、Generation 组合及本地 engine tag 到系统 TaskID 的映射；这是候选编码，不是要求编号全局永不重复。Group 只有在旧引用和在途事务排空后才能复用，有限 generation 不能替代安全回收。[来源 S57](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=7aa75680-b6b5-4ff0-904a-1864086c52ee)

### 输入输出与资源的完整生命周期

你给定的 Group 链条可概括为：预取输入，异步分发；相关 Tile 全部 input_released 后释放输入；output_ready 后启动 store；store_done 后释放输出；最后等待 Context 收尾。Tile 链条是：异步 load、等待输入、计算、按最后一次读取释放输入、异步 store、等待写入可见、发出 output_ready。[来源 S13](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=b5f8eb2c-e61a-49e8-9d74-0415cf36ec24)

| 事件或状态             | 可以据此做什么                             | 不能据此推断什么                   |
| ---------------------- | ------------------------------------------ | ---------------------------------- |
| command accepted       | 命令已被可靠接收，后续不能重复提交         | 计算或搬运已完成                   |
| input_released         | 指定层输入已无剩余读取，可按所有权规则回收 | 其他层副本或其他消费者也已释放     |
| output_ready           | 输出在约定位置对后继可见                   | 最终外存写回已完成                 |
| task_done 或 grid_done | 相应任务计算或定义的任务范围结束           | 所有异步搬运已排空、所有资源可复用 |
| store_done             | Group 最终输出写回已完成                   | 无须检查其他 pin、引用或错误收尾   |
| safe retirement        | 在途操作和所有权均安全结束                 | 可跳过正常错误与取消协议           |

input_released 不是某一次 load_done。K0 搬完后，任务可能仍需 K1 至 K3；当下 outstanding 为零也可能只是下一次读尚未发出。任务级释放至少需要读取集合已经关闭、pending loads 为零且没有重试或未来读取。Group 还要汇总所有当前与未来消费者，检查在途读写和仍存活引用。[来源 S16](https://chatgpt.com/c/6ad72b86-1d40-83e8-9e9c-805306982abc?messageId=26579bc3-abbf-4008-aaf6-b4f49626714c)

输出同理：source read 完成不一定代表目标写入已提交。Gemmini 等后端的局部 completed 信号要先核对语义，不能直接当作 L2 output_ready。每级副本按自己的最后一次使用回收，计算推进也不必等待 Group 完成 free。

### 两级编译接口不能只传一个 kernel 名字

Group/Tile 合同至少包含参数和布局、stride 与 task 坐标、访问范围、L1/L2 工作集、缓冲数、事件需求、memory profile、输入释放与输出可见性。Tile 版本改变 L1 需求，可能迫使 Group 调整 tiling 或并发；Group 改 L2 layout，也可能需要重新生成 Tile 程序。[来源 S17](https://chatgpt.com/c/6aa7c75a-21bc-83e8-bc55-0a5d342a7b22?messageId=50675cfa-a814-4729-bc1e-dcc27a811935)

这种合同是编译器与动态硬件之间的边界：编译器产出合法计划及需求，硬件在真实依赖、队列、配额和背压下推进。不能要求编译器预测唯一完成顺序，也不能让硬件通过无界队列掩盖不合法的资源需求。

<a id="memory-scheduling"></a>

## 九 存储系统与动态调度

### SRAM 分区的阶段性收敛

8月12日，你明确每个 bank 至少支持 SPM/Cache 两种模式，FIFO 仍在考虑。9月13日进一步确认同一级所有 bank 采用一致比例的 bank 内 Region 分区，而不是固定一半 bank 做 SPM、另一半做 Cache；L1 和 L2 可以分别选择比例。[来源 S18](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=0bcc2e8e-8787-44d1-a326-d79ef6acc382)

同级统一比例不意味着同周期所有 bank 只能访问同一种 Region，也不保证两条访问路径各得全带宽。容量划分与带宽份额是两件事，最终仍受端口、bank conflict、tag/MSHR 和仲裁限制。不能继续沿用9月13日早晨“bank 内同时分区尚未定”的旧摘要，也不能把 FIFO 三模式写成冻结规格。

规则 RHS 的历史路径是 HBM→L2 SPM→L1 SPM；Tile gather 面向全局地址，通过 L1 Cache→L2 Cache→HBM 服务，不是只读一个预先构造的“L2 gather cache 对象”。Cache 复用与显式 SPM 驻留必须分开计账。[来源 S19](https://chatgpt.com/c/6a685920-7d7c-83ee-b25a-5b43878fd86d?messageId=20ab2e80-d38e-4809-ab2d-104edca09975)

### 存储策略中未被单一 profile 结论覆盖的备选

| 时间       | 历史变体                                                          | 分界与取舍                                                                                                                                         | 来源和状态                                                                                                                                                                                                                                                                                                                      |
| ---------- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-07 | 单缓冲多 Context vs 少 Context 双缓冲                             | workspace 必须单独预算；不默认每个 buffer 都 ping-pong。双缓冲加强单Context流水，但可能减少驻留数                                                  | 用户纠正 L1/L2责任和workspace；助手明确对照，未冻结唯一slots数。[原问](https://chatgpt.com/c/6a685920-7d7c-83ee-b25a-5b43878fd86d?messageId=20ab2e80-d38e-4809-ab2d-104edca09975)、[修正](https://chatgpt.com/c/6a685920-7d7c-83ee-b25a-5b43878fd86d?messageId=56ac7f38-4a68-464e-a699-7ce689a98f47)                            |
| 2026-09-14 | Runtime自由选择allowed_profiles vs 固定阶段Profile＋动态admission | 前者即时选档/版本，可能增加延迟波动；后者compiler选择Profile、tiling、阶段、HBM边界，runtime仅按依赖/资源补位。`allowed`是合法集，不强迫在线自由换 | 用户探索，助手推荐后一条V1；不把两条强贴未证实的A/B名字。[用户](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=6edf115a-cf88-4ea9-889a-1e07196703e2)、[对照](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=34b12a99-5333-4f17-8bbf-571ed97b0c46)                                   |
| 2026-09-14 | 每Context私有Cache／共享Cache硬分区／普通共享Cache                | 助手推荐物理共享、不按Context硬容量分区；SPM硬预留，cache_target为性能hint，不可相加当独占容量保证                                                 | 助手比较和V1建议，不是用户每项确认。[来源](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7345c2e7-917e-4890-b3fe-6679bffebf81)                                                                                                                                                                           |
| 2026-09-15 | Bank Profile V1 调度/Runtime/IR proposal v0.1                     | 固定阶段Profile＋阶段内动态admission＋HBM安全切档；seal只封新根成员，旧任务和子task仍推进                                                          | 用户要求先不做compiler优化；同日出现两份相近摘要/不同文件名，不能合并为同一已核对附件。[第一份](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=35676440-32fd-4ecb-9d8a-a843647be760)、[第二份](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=eebe66ad-116e-433d-9b8d-fa9b0715f4a9) |

### 一个 Context 一个 memory profile

9月23日，你明确一个 Context 只能有一个 memory profile，同时坚持 completion-driven 补位。固定 profile 并不等于固定并发批次、固定执行时长或固定完成顺序。[来源 S20](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7eff8b05-df9c-4435-ad12-03b9e8903f6c)

助手建议把程序版本可用的 allowed_profiles、执行计划选定的 assigned_profile、物理资源域当前 profile 分开。程序因更小 SPM 需要重新 tiling/buffering/layout，就是另一个版本，不能只改属性。较大 SPM 档位也不必然兼容：Cache 容量可能减少，地址映射也可能变化。

SPM 是可兑现的硬预留；共享 Cache budget 若无隔离机制，通常只能作性能提示，不能宣称独占。已预留但未填数据的区域也不能随意改成 Cache。L2 合同覆盖 Group 的预取、输出和临时驻留；L1 合同描述 Tile 本地工作区，由 Tile 做实际准入与分配，不能由 L2 越权分配。[来源 S19](https://chatgpt.com/c/6a685920-7d7c-83ee-b25a-5b43878fd86d?messageId=20ab2e80-d38e-4809-ab2d-104edca09975)[来源 S21](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=bc75f0ff-84a8-4b4d-be0c-3f3781d61da1)

9月28日旧源码分析曾描述 Context 内 dispatch 间切换 L1 profile。这与9月23日约束需要核对 Context/profile 的作用域及版本：可能是 Group Context 与 Tile/L1 合同不同，也可能尚未收敛。现有材料不足以直接认定代码违反决定，亦不能把二者无说明合并成现状。[来源 S22](https://chatgpt.com/c/6aba2e6b-b18c-83e8-8316-ed81630116de?messageId=d5095dae-0ae9-45e7-b9c2-d7b3eb75143c)

### 调度候选与实验对照目录

这些是同一架构中的策略、窗口语义或消融实验，不是新的芯片代际。特别是 **B1 + Multi-Context 大 Tile** 与论文里的 **B1 L2-MC Only** 只是不相关讨论中重用了字母编号。

| 时间           | 策略或实验版                                                | 关键区别                                                                                                                                             | 来源和状态                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------- | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-06-30     | lookahead/promote 浅窗口                                    | `window=2`：P0 running/retiring，P1 candidate；lookahead 看见，promote 只在 alias／safe point／credit／slot 合法时提前准备，launch 另有边界          | 旧 ELENOR 探索；助手建议降为 metadata/hint，不做强语义 ISA／复杂动态 promote。[来源](https://chatgpt.com/c/6a439320-bc00-83e8-a55d-a30af22a6121?messageId=a0c4375d-0d66-4d9c-8527-40bb8a33f66e)                                                                                                                                                                                                     |
| 2026-06-30     | 编译器 Tile Group DAG grouping/reorder/fusion＋轻量 promote | compiler 用依赖、liveness、reuse、engine balance、slot pressure 全局排程序；UCE 做局部安全机会；不是只按 tensor 内存大小拼组                         | 用户想法由助手展开；区分 operator fusion、region grouping、horizontal batching。[来源](https://chatgpt.com/c/6a439320-bc00-83e8-a55d-a30af22a6121?messageId=332ca78d-f355-44f5-9575-aadcc8e988cb)                                                                                                                                                                                                   |
| 2026-07-01～02 | 小物理 issue window＋逻辑 pending graph／token table        | 操作被引擎接受即释放 issue 位置，完成仍由 inflight token 跟踪；非 ready 动作留 pending；A/B/C 要按 tile 暴露依赖，而非黑盒串行程序                   | 助手提案，未确认采用。[前置方案](https://chatgpt.com/c/6a4519ea-6bb8-83e8-9205-81fa67cd57e8?messageId=1f9e5781-3c6c-4d42-8c75-fa88cdcd3aa7)                                                                                                                                                                                                                                                         |
| 2026-07-02     | sliding-window dataflow scheduler                           | streaming decode frontier＋dependency scoreboard＋ready set＋issue selection；这里 `2` 被另定义为每周期发射选择数量，不是总在途容量                  | 用户要求连续滑窗，助手提案。原答“必须全 DAG 实体化／C永远可进”过强；后来的有限表与背压更严谨，不能照搬。[用户](https://chatgpt.com/c/6a4519ea-6bb8-83e8-9205-81fa67cd57e8?messageId=92c54257-8904-49eb-96c4-a38ef4505d76)、[提案](https://chatgpt.com/c/6a4519ea-6bb8-83e8-9205-81fa67cd57e8?messageId=3993f6fd-3443-4944-9051-fe2a9b76312f)                                                        |
| 2026-07-31     | B0 Strong Static Baseline                                   | L2=1、Tile=1；仍必须有 async DMA、ping-pong、静态预取、SRAM规划、fusion；非故意弱化的全串行 baseline                                                 | NEST 论文对照，不是用户新架构代际。[实验表](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)                                                                                                                                                                                                                                              |
| 同上           | B1 L2-MC Only                                               | L2多 context、Tile单 context；隔离 HBM↔L2/prefetch/grid/drain 隐藏收益                                                                               | 同一论文消融设计，非已验证收益。[同源](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)                                                                                                                                                                                                                                                   |
| 同上           | B2 Tile-MC Only                                             | L2单 context、Tile多 context；隔离局部 DMA/gather/compute/store 等待隐藏                                                                             | 同一论文消融设计。[同源](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)                                                                                                                                                                                                                                                                 |
| 同上           | B3 Nested-MC 简单配额 vs Proposed 完整 Contract             | 两级均多 context；比较简单配额和完整分层资源承诺；8月模型评审发现 Full Contract 曾反而串行化并发，后续与 safe quota 无差别                           | 助手研究方案＋历史合成模型评审，不是实测。[方案](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)、[首轮](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=81f53566-7c09-4b89-8621-81120c8079b2)、[第二轮](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=3b4dedc2-71f9-422c-a6a8-b57536a02309) |
| 同上           | MPMD comparator／论文称 Upper Bound                         | 放松单 kernel epoch；需要空间 placement、routing、FIFO 等代价；没有证明时只能叫对照，不能真宣称上界                                                  | 助手论文候选；用户此前选SPMD以降 compiler 压力。[来源](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)                                                                                                                                                                                                                                   |
| 2026-09-11     | 纯 context-head／纯 ready-action／两者融合                  | 三路线被比较；推荐保留 context 的状态与归属、提交 PC，由 Group 有限 ready-action 后端扩大已登记可见范围；Tile继续独立PC队首切换                      | 用户要求比较，助手建议融合；非三层都 OoO。[融合原答](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=aaca6c71-da0e-4fa5-86b2-6bf012442e12)、[三级落位](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=be820549-c964-4f12-a769-bd2400ef80f2)                                                                                                              |
| 2026-09-11     | S0／S1／S2 调度比较                                         | S0 优化 NEST 队首；S1 小 ready-action region；S2 更大 ready-action 范围。记录 resource_blocked 与 ordering_blocked，比较可达频率后的时间，不只cycles | 助手实验设计；9/28旧代码报告 s0/s1，未证实S2实现。[来源](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=aaca6c71-da0e-4fa5-86b2-6bf012442e12)                                                                                                                                                                                                                                 |

### 补位执行与有限 ready action

8月12日你反驳固定 mix-window：不是一批四个全部结束才换下一批，而是任何一个完成后，后继可运行 Context 补位；“四个”只是示例。[来源 S49](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=c474e393-071a-44d0-94ba-2c4f5fd41010)

9月的助手方案是 Group 多 Context 提交前端加有限 ready-action 后端，Tile 保留独立 PC 并在各 Context 当前可发射指令头中选择。Group 登记成功即可推进 submission PC，不必等该 action 完成；后端只能选择已登记、依赖和资源满足的工作，不是任意全图动态重排。[来源 S50](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=b34b621f-95e6-4d36-9183-75d2380f9315)

一个 Context 因配额或参数未备无法登记，切换别的 Context 可能有用；整张 action 表满时切换无用，应停止前端而保持后端、DMA、完成回收继续。action 表存的是控制 descriptor，扩大可见窗口不应自动扩大张量预取驻留。循环参数、event 实例和 buffer handle 要在登记时固定，不能继续引用变化中的循环变量。

ready-action 也救不了尚未提交的工作。过早 await 阻止独立 B 登记时，后端无法猜出 B；Tile Context 内顺序发射也不意味着必须等上一条完成才发下一条。登记到程序末尾更不等于 Context 已安全结束。

9月28日旧源码快照还显示有限扫描、部分队列只看队首、任务到 Tile 固定映射，以及逐 Tile 准入而非 gang scheduling。它说明 action 层减少头阻塞不等于全系统无头阻塞；这些是当时未跑仿真的源码判断，不是现时硬件规格。[来源 S22](https://chatgpt.com/c/6aba2e6b-b18c-83e8-8316-ed81630116de?messageId=d5095dae-0ae9-45e7-b9c2-d7b3eb75143c)

### 9 月提案与实现快照尚未对齐之处

9 月 15 日的 Bank Profile v0.1 交付摘要曾提多 Tile 原子 Dispatch；9 月 28 日固定提交的历史源码审查却描述逐 Tile task admission，而非 gang scheduling。这是提案与实现快照的差异，不能静默当成同一合同。9 月 15 日两个摘要还分别提到 18 条不变量/32 项测试与 31 项验收用例；本次没有逐一读取附件，更没有证据表明这些验收已实际执行。[第一份摘要](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=35676440-32fd-4ecb-9d8a-a843647be760)；[第二份摘要](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=eebe66ad-116e-433d-9b8d-fa9b0715f4a9)[来源 S22](https://chatgpt.com/c/6aba2e6b-b18c-83e8-8316-ed81630116de?messageId=d5095dae-0ae9-45e7-b9c2-d7b3eb75143c)

### Memory admission 与前进性

你提出前置 memory admission，排除输入占满 SRAM、输出无处安放的容量死锁。预留应覆盖输入、输出、workspace 和允许的预取/重排峰值；不能只证明某一条理想静态时间线放得下。也可保留 buffer reuse 依赖，限制合法重排范围。[来源 S51](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=d9720292-8f8f-41e1-86ba-1cda1d6e26b7)

reservation 已包含实际使用量，不能重复相加；未写入的预留输出不能再借给别的任务。L1/L2、多 pool 和逐 bank 分别核算，总 bytes 足够不代表地址或 bank 组织合法。workspace 是否双缓冲、需要几个 slot，是编译器选择，不是 workspace 的天然属性。

完整拓扑登记可以排除“消费者占满有限 action 表，生产者尚未登记”的一类死锁；前置内存预留可以排除一类容量死锁，但两者都不是全系统无死锁证明。command/event/completion credits、MSHR/refill/writeback、返回路由、profile 切换和 AM 续算/排空仍有独立资源依赖。

因此 admission 失败必须允许旧任务 store、completion、release 前进；新链排队不能阻塞旧链 ACCUM/FINAL。数据 DAG 无环不代表资源等待无环，切 Context 也不一定能解救只有原 Context 才会提交 FINAL 的链。

### Profile 切换与 HBM 安全边界

你接受 V1 在改档时切开执行，通过 HBM 安全边界保简洁：跨阶段需要的数据先完整写入 HBM，再由后续阶段消费，不要求带活跃 SRAM 数据在线迁移。[来源 S21](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=bc75f0ff-84a8-4b4d-be0c-3f3781d61da1)

共同协议是关闭旧工作集合，允许已准入/登记旧任务继续推进，完成必要写回与消费者可见性，排空 DMA、cache refill、fabric 和引用，再配置新 profile、等待受影响 bank 确认后重新准入。seal 不能阻断旧任务收尾必需的内部动作，否则会自锁；一个 Context 完成也不自动证明整个配置域可改档。

跨 Context await 本身并未被永久禁止。助手9月18日撤回了“Context 只能等自己的 event、其他等待都必须让 Device 阻塞”的过度推论。关键是避免等待者先占住生产者必要资源；在准入前表达依赖可保留独立任务重叠。[来源 S52](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=d03c1861-1a68-4ee2-9dd0-fb86be4dea6c)

Profile 优化应比较完整计划：每个 Context 的 profile、阶段划分与合法登记顺序，在实际 runtime 准入/释放/FIFO 规则下评估 makespan。不能给 P100/P75 配一个与调度无关的固定成本，也不能把更多 MatMul 同驻或更少切档直接等同于更快。若 V1 不 backfill，离线优化器也不能偷偷假设可跳过队首。[来源 S20](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7eff8b05-df9c-4435-ad12-03b9e8903f6c)

### Context 所有权与 readonly shared

10月5日，你明确 Context 是管理和释放的基本单元，shared buffer 只有一次设置机会、发布后只读。跨 Context 任意拿私有 buffer 做 copy，会遇到源提前回收、复用和完成顺序不确定的问题。[来源 S53](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=44bc19fb-f07f-4c0a-ab05-e271002e3f1c)

Context 主导所有权并不必然要求所有 backing 一直保留到 Context 末尾；旧实现曾区分 view、backing、arena、pin 与 transaction，用最后引用决定物理回收。这里应区分设计层所有权与历史回收实现，而不是用后者推翻前者。in-place、同 Context reuse 和 readonly share 的收益，也不应被重复计作跨 Context move。

### Tile gather Group gather 与 PagedAttention

10月2日，你说明原方案只有 Tile 级 gather/scatter，L1/L2 支持 Cache，Group 不支持 gather；让 Group 增加 gather 并删除 Tile gather 是另一个模型提出的待比较方案，不是你的决定。[来源 S54](https://chatgpt.com/c/6abf1244-7a3c-83ee-9527-ff150a4e701a?messageId=4666263b-e276-4a5c-ab5d-8cbe2f95254d)

Tile 自主间接访存支持 MatMul→生成 index→Gather→后续计算的局部链；Group gather 适合已知页表的批量分页预取，可作为增强，不必删除 Tile 能力。物理 gather 引擎集中在 Group，与只允许 Group 程序发请求是两件事。将 Tile index 上交 Group 会增加发布、通知、回传与潜在同步开销。

PagedAttention 是块间页表间接、块内规则访问，不要求先将整个 KV 物化成连续大 tensor。Group staging 要进入 L2 预算，计入 Cache 重复副本与实际读写流量；两边比较应允许相同块级请求、异步重叠和 SRAM/带宽条件。其收益更多来自 KV 空间管理、碎片、共享和 batch，而非 gather 天生快于连续访问。

10月1日你确认当时不支持 atomic。助手提出的可行路径是：KV 历史多读，append 由 runtime 分配 slot 且唯一写者；跨 Tile 各写私有 partial，再由指定 owner 或分阶段归约；页表、分配、回收和共享前缀 refcount 放 CPU/runtime。Softmax 需要合并最大值、归一化量和加权输出状态，不能直接把局部归一化结果相加。[来源 S55](https://chatgpt.com/c/6abdec71-c088-83e8-97cb-308af6ee08d0?messageId=c2ee61d8-2c2f-40cd-bf81-66629f20d1b6)

真正冲突 scatter-reduce、动态队列等仍需分组、目标独占、串行或 fallback。这些是无通用 atomic 的可行方案，不是已经实现性能的证明；10月1日模型的 profiled gather 缺口，也不能拿来断言后来 real-gather 分支仍未改进。

<a id="compute-am"></a>

## 十 计算后端与 Shared AM

### 矩阵向量搬运的分工

历史建议的主线是共享编程协议、地址空间和存储管理，而非强制所有运算共用同一批乘法器。BOA/OPA 适合矩阵 contraction 与局部累加；EVU 是可编程非矩阵主路径，不只是 bias/ReLU 附件；MFE 负责搬运与布局处理。受限 epilogue 可以承担简单 bias、激活、舍入和类型转换，复杂 SFU、mask、shuffle 与向量链仍交给 EVU。[来源 S23](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786)

这种分工允许独立 MatMul 与 element-wise 在不同引擎重叠。若统一算术池，需要显式竞争或分区，不能免费获得相同并行度。小 OPA 可以是物理构建粒度，但不必把每个微小阵列操作都变成独立 Group action。

2026年10月14日对旧分支的源码分析认为：UCE 每拍从 eligible heads 选择最多一条操作，BOA/EVU/MFE 有各自入口队列；当时 BOA/EVU 仍是各自单执行资源。这里的队列与吞吐判断只对应旧版本，不是 10月28日当前代码审计。

EVU 的 lane 数、向量长度和指令吞吐要分开。SFU、reduction、shuffle 与本地读写需独立建模，不能只以 ops/峰值估时间。若一个单 active-kernel EVU 被长 gather 占住，UCE 切 Context 并不能让另一个向量 kernel 自动使用该后端；可探索把长等待拆为生成索引、异步 gather、后处理。[来源 S24](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53)

### Tile Shared 与私有状态的边界

10月17日，你明确不希望每个计算器件都有一块大输入 scratchpad，并在计算前从 L1 再复制一遍；希望主要张量直接来自 Tile L1/Shared。你允许 OPA/Tensor Core 专用 AM，UCE 通用寄存器也单独保留。[来源 S25](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=b0c04d5b-6aeb-4a74-8210-0a465fa332eb)

因此合理表述是 HBM → Group L2 → Tile Shared → 计算器件。大块输入、输出和中间张量尽量共享主要本地层；少量 operand FIFO/latch、流水寄存器、控制寄存器与 AM 可以独立。这不等于“所有 RF 与 SPM 已决定物理合一”。

10月20日对 RF/shared 统一的讨论进一步区分：同一 bank 池划两个区域只是容量统一，数据复制仍有读写；同一份数据同时提供 shared 与 operand 访问视图才真正消除整块复制。EVU 高频 VRF 与 PE 反馈累加的端口、时序需求仍须单独评估。助手建议先统一接口，而不是首版强行统一所有物理存储。[来源 S16](https://chatgpt.com/c/6ad72b86-1d40-83e8-9e9c-805306982abc?messageId=26579bc3-abbf-4008-aaf6-b4f49626714c)

MFE 直达 RF 也可以独立于物理合一。接口需要 owner/context 与 generation、slot reservation、scoreboard、最后读取、目标可见性和 lane layout 转换。L2→Shared 与 L2→RF 两条路径不会让 L2/NoC 带宽翻倍；同份数据 multicast 到两端还要承担双端写入和反压。

### RISC V 控制与硬件多 Context

你希望多个程序同 Tile、独立 PC，A 等访存时 B 能推进，A 已发出的异步操作继续运行。Ara 的多条在途向量操作与 dispatcher/sequencer 并不原生等于多个驻留程序 Context；增加两个 PC 或外部任务队列也不足以完成状态隔离、唤醒和完成归属。[来源 S26](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74)

10月21日助手给出五条路线：软件协程；多份 PC/GPR/CSR 共享顺序流水并在等待时切换；逐周期交错 IMT/barrel；普通 RISC-V 加独立多 Context sequencer；多个小核共享加速器。助手偏好第二条、备选第四条，但尚无明确最终核或微架构选择。[来源 S27](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=6f22235b-7231-4426-a6bc-993c71da1653)

在这些方案中，汇编表达做什么与等什么，硬件选择哪个 ready Context 运行。submit 成功是命令接纳，未接纳可以无副作用重试；已接纳不能因切换重发。wait 需要持久事件与无丢唤醒协议；共享配置应在 submit 快照或按 Context 隔离，避免 A 配 stride、B 改 stride、A 用错配置。

拟议的 submit/token/wait/yield/exit 不是标准 RISC-V 现成指令，标准 WFI 也不是 wait(task token)。完整 RVV Context 还有向量寄存器、vl、vtype、vstart 等状态，不能按标量 Context 成本估算。异步 fault、release/acquire 与非一致缓存维护也应属于接口合同，而非后补细节。

### 自主向量 kernel 与公共任务管理

10月23日，你希望向量设备接收函数入口和输入输出参数后自主运行，不让 UCE 逐条参与 RVV 汇编。自主执行端因此需要自己的 PC、取指、标量寄存器、地址计算、分支及必要 load/store；可以采用轻量 RISC-V+RVV，或能力更受限的微程序 sequencer，Ara 数据后端本身不能替代这些前端功能。[来源 S28](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=f142eddd-6116-537b-93b0-1a9991927ea8)

助手建议共用 Task Table、提交、依赖/admission、每设备 ready list 和 Event Table/完成路由，各设备保留本地 sequencer、scoreboard 与执行依赖。共享管理不等于全部设备排进一个 strict FIFO，否则一个被堵住的矩阵任务会阻塞向量和搬运。多个设备同拍完成也需要可靠仲裁、缓冲和背压。

V1 单个向量设备一次一个 active kernel 是建议简化；若要求向量 kernel A 等待时 B 进入同一个 RVV 后端，则是另一层多 kernel 驻留能力。多 UCE Context、多个等待任务和异构并发，并不自动包含这项能力。

### 小矩阵阵列与 K 累加链

10月17日，你纠正为四个独立 8×8 单元，彼此无连接、无广播，各自 AM，由编译器 tiling，未来才可能 ring reduce。不能把此前协同广播簇或旧 8×16×16 OPA 配置混到这个版本中。独立小阵列改善小输出块和多任务粒度，但可能增加命令供给压力与 Shared 重复读取。[来源 S29](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)

10月19日至20日曾探索 ring/AM-to-AM reduce 和专用 Reduce Memory。前者必须匹配同一执行实例、矩阵任务、输出块和互不重叠 K 分片；后者用额外存储解耦部分和等待。两者都需可靠 owner/tag、输出空间、排空和防死锁机制。你随后提出先不用 reduce tree，K-step 0 overwrite，后续 accumulate，当前主线转向顺序 K 链。[来源 S30](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda)

PE 内 output-stationary 状态与 AM 持久累加是两种机制。一个 K-step 内可以 OS；在安全边界把新 partial 写入或累加到该输出链的 AM，再让其他 Context 使用阵列。X0→Y0→X1→Y1 不必将旧 C 经 Shared 写回再读，也不需要 PE 保存多 Context 的 OS 状态。accumulate 应指向显式逻辑 acc handle，而不是盲用当前 PE 残值。[来源 S31](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)

### Shared AM V1 的最新明确规则

10月26日，你明确规定：默认保持原 engine affinity；软件上每个 Tile Context 只能使用两个 AM slots。这是最新用户约束，应优先于更早的私有 AM、迁移或旁路设想。[来源 S32](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)

据此形成的指南解释如下。两槽是一个 Context 跨全部矩阵引擎的总配额，不是每个 engine 两槽；从成功申请直到安全回收都计数，包括等输入、FIRST 前、计算、待 AM 更新提交、FINAL 后 drain 与错误收尾。one-shot GEMM 也计入，不能沿用早先“有输出旁路即可不占槽”的例外。[来源 S33](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)

槽按输出链动态分配，不是 Context 创建时永久预留两槽。因此假设全局八槽，也不能推导出只能驻留四个 Context。一条输出链绑定一个 slot 和一个固定 engine，直到释放；同一 Context 的两条链可以绑定不同 engine。等待输入时保留 AM，但不独占 engine，engine 可穿插其他输出链。

在一条链同一时刻只由其固定 engine 推进的前提下，单 Context 同时使用的矩阵 engine 数最多为二。若四个独立 engine 都要工作，至少需要两个有足够 ready 链的 Context，并同时满足输入与 AM 服务带宽；这不限制 UCE、MFE 和向量的重叠。

助手建议 TRY_OPEN 区分 SUCCESS、BUSY、ERROR：成功必须真正取得物理槽、Context 配额、元数据和完成记录，并固定 affinity；BUSY 未受理且无持久副作用。软件第三次申请即使全局尚有空槽也不能成功，更不能等待第三次成功才推进旧两条链。优先收完成、推进已有链供数和续步，再申请新链。[来源 S33](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)

| AM 事件        | 应表达的边界                                 |
| -------------- | -------------------------------------------- |
| compute_done   | 本步结果已经产生并被后端接收                 |
| step_committed | 该步 AM 更新全部提交，下一 K-step 可安全使用 |
| input_release  | 指定输入已无剩余读者                         |
| output_visible | 最终输出在约定 L1/Shared 位置可见            |
| slot_free      | 无在途引用，可复用物理槽并归还 Context 配额  |

最终自动 drain 和安全释放是指南建议，不应依赖 UCE 轮询触发。失败链也要有明确的错误终止与资源排空状态。10月26日对话说明当时产出的是需求和改造指南，尚未修改 Nexus 仓库、运行模拟器或 RTL 验收；这些规则不能表述为已经实现。[来源 S33](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)

### AM 物理实现与带宽风险

共享的是容量和服务入口，不意味着多个 engine 可以无约束同时写同一输出。slot 是逻辑输出链空间，bank 是物理端口单元，group 是累加服务模块。四个 group、每组两槽、每槽 16 KiB 等属于助手示例，不是两槽规则自动确定的硬件规格。[来源 S34](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)

建议在 AM 侧完成 old C 读、加法、写回；同 handle 一次推进一个 K-step，不同 handle 可独立。即使不提供软件通用 atomic，内部 RMW、同地址 hazard 和 forwarding 仍不可省略。多 bank 不等于多 adder；1R1W 和单端口 1RW 的服务能力也不同。

旧 Gemmini 提交的历史分析指出 mesh 输出 Valid 无 ready。因此不可在输出端加仲裁器后假定阵列自动停住；发射不可暂停块之前，要为所有不可停止在途结果预留接收 credit。step_committed 应覆盖 FIFO、互连、AM 读、加法和写队列，而不是只看 mesh 最后一行完成。[来源 S34](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)

新链申请入口不能堵住存活链 ACCUM、FINAL 与 drain；槽满时必须仍能推进旧链并释放。V1 固定 affinity，跨 engine 迁移和 split-K 并行归约属于后续能力，不能由“共享 AM”四字自动推导出来。

### K 维度与容量的三个尺度

K_total 是整个归约长度，K_chunk 是输入分块长度，K_cmd 是单条底层命令处理长度。固定输出形状和累加位宽时，AM 容量取决于输出元素数，不随 K 增长；K 增加的是输入量、时间和槽的占用寿命。[来源 S35](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f)

例如 16×16 阵列处理 16×16×256，既可能是单命令流式处理 K=256，也可能是十六个 K=16 命令累加同一槽。两槽限制同时存活的输出链数，不限制一条链的 K-step 数量。输入双缓冲则仍受 A/B 分块容量限制。

K 也不能笼统称为无限：INT32 累加有溢出边界，分块后仍累加到同一 INT32 不会解决溢出；浮点有舍入误差和顺序差异。固定归约顺序可增加可复现性，不等于与另一顺序逐位一致。

<a id="irregular"></a>

## 十一 Irregular 多设备与可观测性

### 二维 DMA 与间接地址生成

10月28日讨论的关键是区分规则多维地址和数据依赖地址。base/stride/count 能表达二维 DMA；运行时修改 stride/count 不等于消费运行时 index vector。dynamic gather 需要根据 index[i] 生成地址；间接 AG 加一维 burst 已能做整行 gather，二维的价值是在选中块内部仍有 stride。[来源 S36](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=d7fb793b-8c12-4d4e-a287-9bb8df90310b)

当前 DMA 路线最适合“块间 irregular、块内 regular”，例如按 index 取一行、一个 tile 或 KV page。逐元素随机 4 B gather 则要求另一档请求前端：index 供给、每周期提交率、outstanding、返回匹配与重排、请求合并、bank/channel 冲突和缓存复用都可能成为瓶颈。

能寻址不等于高效。DMA 通道数不等于在途请求数；HBM 峰值不等于有效字节吞吐。GPU 也不能把随机地址变成连续访问，其优势更多来自成熟 lane 级 LSU、小请求控制、多 warp 隐藏延迟、缓存和原子通路。对大块 gather、片上驻留或与矩阵流水紧密融合的负载，也不应预设 GPU 一定更快。

Scatter assignment、重复覆盖的顺序语义、scatter reduction 必须分开。无冲突写不必 atomic；有冲突 reduce 需要 atomic、分组归约、目标独占或受控串行等机制。建议评估每个 index 的连续字节数、局部/全局随机与热点程度、冲突率，并同时测 request/s、有效带宽、队列占用和端到端耗时。[来源 S36](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=d7fb793b-8c12-4d4e-a287-9bb8df90310b)

### Layout Copy 与跨 Context ownership

Context 内合法 in-place、buffer reuse 和局部异步，与跨 Context 可写 ownership transfer 是不同能力。readonly shared 已能消除一部分复制，不能再把同一收益重复算给 move。历史助手后来承认，没有目标模型证据证明跨 Context move 值得首版承担准入、取消与回收复杂度。[来源 S37](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=83bd4615-28c7-4a8c-9a00-61ab3a52d5c9)

转交 ownership 也不会自动消除物化 transpose、真正 concat 或完整 expand 写出。应分别判断 metadata view、consumer 是否支持该 layout、以及是否必须实际重排。DMA、二维 copy 或 LTU 都不能绕过源存活、目标独占和完成依赖。

### 多 Tile 与多设备不是同一层扩展

10月5日，你担心结果经 HBM/DDR 后由 CPU dispatch 太慢，询问 group copy 是否有利于多设备。讨论首先区分 CPU 提交控制、设备 HBM 暂存、主机 DDR staging：CPU 提交不代表 CPU 复制张量，经过设备 HBM 也不代表经过主机内存。[来源 S38](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=fb6fc45f-fec3-4015-8847-86a600771afd)

独立请求并行与粗粒度流水较容易延续 Context 模型；细粒度 tensor parallel 还需要 send/recv、all-reduce、all-gather 和相应生命周期，copy 不能替代 reduce。助手提出先用 CPU 粗粒度提交、设备异步 P2P 和外层通信 buffer 管理，避免每步 CPU 阻塞等待。

本地 Group L2 copy 不会自动提供远端寻址、接收准入、权限或 remote completion。直接 L2→远端 L2 需要新合同；即使少了一次 HBM 流量，也可能因长期占住 L2 降低并发。该能力应由真实瓶颈决定，尚非已实现多设备系统。

### Trace 应解释等待而非只展示忙碌

8月16日的 trace 建议分 Task/Context、Scheduler、Resource、Memory/Counter 四类视图，用 flow 连接 Host→Group→Tile→DMA/Matrix/Vector。逻辑任务与硬件 slot 分开，A/B prefetch 分轨，input_released、output_ready、store、complete 分事件。[来源 S39](https://chatgpt.com/c/6a81a3e4-5cfc-83e8-9007-2bc85ff78502?messageId=870f28da-c36d-457e-ae67-ec54ced3e31a)

需要记录等待预取、Group/Tile admission、load/store、matrix/vector、事件、L1/L2 容量、MSHR 和输出 buffer 等原因；资源侧同时记录 active/ready/blocked contexts、队列深度、SPM/cache 占用、outstanding 与有效带宽。只有这样才能区分谁慢、为什么没有运行、硬件是否真空闲。

这套 trace 应服务 Static、L2-MC、Tile-MC、Nested-MC 同条件对比，不能只展示更密集的时间轴便认定性能提高。当时是可观测性设计建议，并不表示完整工具已经实现。

### 其他算子与负载探索

还讨论过 grid_sample 与 mul 的 fusion/tiling，以及 LTC/CfC 液态网络。前者可确认你提出过问题，不宜把未恢复的结论写成已定案 lowering；后者涉及小矩阵、稀疏连接、递归依赖和实时延迟，persistent temporal Context 与跨 Context micro-batching 属于探索，不是当前新增硬件要求。[来源 S40](https://chatgpt.com/c/6acc9954-5a68-83ec-ac31-d2aa4a6812a3?messageId=aceb5cde-d2c9-4519-8d9c-cab83b8d2699)

<a id="compiler-validation"></a>

## 十二 编译器与验证路线

### 编译器关注点与接入选择

IREE/MLIR、LLVM、CUDA Tile 等讨论的共同目标是复用现有优化与验证方法，不等于已经采用各自完整 runtime 或原生 ISA。MLIR Async 的 token/value 描述依赖，实际线程和并发由 lowering/backend/runtime 实现；IR 标了 async 不会自动产生硬件重叠。[来源 S05](https://chatgpt.com/c/6921c3c8-5d1c-8330-a56c-3a01baf31510?messageId=39a6ef82-005a-4bd7-9ce4-6e9915dbc618)[来源 S06](https://chatgpt.com/c/694b8c19-4548-8327-9fdf-fb16eb74371b?messageId=243bf07b-903d-4aa7-a5c6-28ee3211ff30)

你对 LLVM IR 更关心提前 load、隐藏访存与提高 pipeline 效率。重排必须保持数据和内存依赖、alias、dominance 与 speculation 安全，同时考虑活跃区间和寄存器压力。合法重排也可能因工作集扩大而变慢。[来源 S07](https://chatgpt.com/c/694fec5a-ddb8-832b-aa85-ae7544739e00?messageId=c745f5c4-4d6d-4cf7-b075-643276d38750)

9月最具体的路线是 Group 自定义 MLIR dialect/pass，负责子图到 Context、L2 规划、事件与 admission；Tile 用 Triton 描述块计算，从 TTIR 分叉做目标合法性、地址模式、矩阵/向量映射、L1 布局与事件 lowering。你确认的是总体分工理解，完整后端与 JIT 并未因此成为已完成事实。[来源 S17](https://chatgpt.com/c/6aa7c75a-21bc-83e8-bc55-0a5d342a7b22?messageId=50675cfa-a814-4729-bc1e-dcc27a811935)

不能直接移植 NVIDIA TTGIR 的 warp/layout 假设，num_warps 也不是 Context 数。tl.load 并非天然 DMA：规则访问可转预取/SPM，动态间接地址需要 gather/cache；如果 index 来自前一计算，不能无条件提前 Group prefetch。一个 Triton kernel 也不必固定等于一个 Context。

助手建议先固定 Context 外壳和受限 Tile backend，再自动生成资源契约与 Group 外壳，最后扩展完整 JIT/driver；不支持的 atomic 等必须明确拒绝或合法 fallback。这样更符合有限资源下快速验证的目标。

### Python 模拟器的价值与缺口

2026年10月10日至11日对当时版本的分析把模拟器定位为“执行合同验证器和调度/存储系统评估器”。它可以研究 IR/lowering、准入、事件、生命周期、profile、有限队列，比较登记顺序、await 位置、预取、双缓冲、Context 大小和存储规划，并记录等待原因。[来源 S41](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=ac1723fd-2bd3-49b2-971f-c0bf468e0f56)

当时 Gather/Scatter 可由真实 index 驱动字节搬运，但 BOA/EVU/USE 尚未执行完整张量算术。ByteStore 有字节不等于独立数值 oracle；Gather 读真 index 也不证明先前 MatMul 真算出了正确 index。描述符级延迟尚不能充分刻画尾块利用率、寄存器压力、SFU 或逐拍共享 SRAM 端口竞争。

full_memory 不等于微架构精确，LRU/容量模型不等于真实 set conflict 和 tag throughput。compile_program 当时主要做验证、lowering、资源/profile 与产物处理，不是全图 tiling/fusion/搜索已经自动完成。release/free 使 view 失效，也不必然立刻归还父 Arena 容量。

同一个模拟器中的 A/B 不会自动抵消偏差。固定硬件只改计划是编译优化；改变 Context 数、DMA 通道、SRAM 是架构探索，应分别报告。需要独立数值参考、参数敏感性和局部 RTL 校准，而不是把模拟器 speedup 直接当芯片结论。

### 两条互补的局部 RTL 路线

10月11日的调度器先行方案，让 RTL 负责 registration PC/fence、action table、依赖查询、仲裁、inflight、事件、credit 和退休；Python 保留有限内存、引擎与外部响应。GroupScheduler 不等于整个调度器，root sequencer、Tile UCE 与准入/生命周期逻辑也要明确边界。[来源 S42](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=379dcbda-2e9c-48de-82a3-b64d44ba045f)

你不想把张量 SRAM 全部写成 RTL，并不等于不验证调度状态表。状态可以先用定长数组或有限端口模型；如果 Python 预先算好 ready 向量，只让 RTL 选一个任务，便没有验证依赖查询硬件成本。probe 应无副作用，握手后才 commit；组合 eval 不能反复分配和提交。

10月23日的另一方案是可替换 engine backend：保留 Python 系统主循环，将一个 BOA/EVU 换成 RTL，真实 ready/backpressure/done 决定后续调度，退出人工 finish_cycle 决定完成的路径。未接收的 launch 可重试，已接收的任务只能执行并完成一次。[来源 S43](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=0f65b652-2cd4-4f2d-96a0-ff552be8f68e)

建议技术链路为 Scala/Chisel → SystemVerilog → Verilator C++ → wrapper/pybind11 → Python。不是每周期与 Scala/JVM 通信。cocotb 适合组件独立驱动与检查，pybind11 更贴近保留 Python 总控；两者都要明确唯一时间推进者。

一个 Tile、两个 Context、一个整数向量加 RTL EVU、有限队列与内存，是建议的最小闭环，不是最终硬件规模。需要检查未接收重试不丢、等待时其他 Context 真前进、真实完成改变补位、结果一致、只完成一次、资源不提前归还。

### 联仿必须保留真实约束

一个组件只能有一个状态拥有者。Python 可以当环境和独立参考，但不能替 RTL 预选任务，也不能用无界队列吸收背压。ByteStore 可以继续保存 SRAM/HBM 字节，但访问要经过 bank、端口、延迟、outstanding、仲裁和响应反压模型。[来源 S42](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=379dcbda-2e9c-48de-82a3-b64d44ba045f)[来源 S43](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=0f65b652-2cd4-4f2d-96a0-ff552be8f68e)

周期可见性应统一为驱动输入、组合稳定、采握手、时钟沿、采状态。eval 不自动等于一拍，launch 与 tick 也不能各推进一拍。Python 调用先后不应意外创造零周期唤醒。功能和任务归属要一致；只有采用相同周期合同，才要求逐拍一致。

没有 FPGA 不妨碍逻辑、协议和局部周期验证。但频率、面积、功耗需要综合与目标工艺/物理实现；联仿不能证明硅片 IO、良率或完整系统性能。每一层结论都应保留对应证据边界。

### 开源底座的复用边界

Gemmini 是矩阵后端候选，不自动替代 Nexus 的调度、准入与生命周期。其内部 load/store/execute、ROB 和队列提供设备内部重叠，不是 Group/Tile 多 PC。你希望一条汇编提交 MatMul，可以探索 custom 指令前端快照参数并内部展开 preload/compute；这是 RISC-V custom/RoCC 路线，不是标准 RVV，也不意味着矩阵单周期完成。[来源 S44](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=c577d408-4616-574b-b696-7405b5fb9a05)

Ara/CVA6 可参考向量数据通路和控制分工；Rocket/Shuttle+Saturn、T1 是 Scala/RVV 方向候选；Vortex/Virgo、SNAX、Spatz/Snitch、MemPool 等是不同粒度参考。它们不天然带有 Nexus TaskID、owner/generation、admission、共享可见性与多 Context 合同，也没有被确认全部集成。[来源 S26](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74)[来源 S45](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=feb5de91-6135-5bac-8083-011b375687be)

你对 openTPU 明确限定只借鉴 sim 方法，不参考架构。历史分析认为可借鉴同一产物与初态的双后端差分、独立数学参考、合法随机程序、语义 trace 和严格报告 skip。独立 RTL executable 不等于在线逐周期嵌入 Python 系统；固定 halt 后额外等待若干周期也不能替代真正排空证明。[来源 S46](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=338db66a-e872-4d88-b9d4-007e7b31c0e2)

chisel-npu 与 Gemmini 的比较是源码学习，不是项目迁移决定。历史分析关注独立程序引擎、多位宽 RF 和 chaining，但提醒不同执行路径不能合并宣称“全部上板”；小寄存器数组不证明大 SRAM 端口问题已经解决，一拍发射不等于系统持续满吞吐。没有公平吞吐/PPA 证据支持全面优于 Gemmini。[来源 S47](https://chatgpt.com/c/6ad9d043-bd18-83ee-8667-526a9c4878dc?messageId=a6ba43e1-a539-48a1-a1fd-b75cbae57c72)

### 论文与性能评估留下的教训

NEST 的可证伪问题是：两级 Context 能否保持 SPMD 编译简单，同时提高关键资源利用率；资源契约能否安全支持并发。multi-context、软件流水和分层调度本身不能直接宣称原创。[来源 S12](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)

8月的模型评审曾发现 Full Contract 几乎串行化 nested 并发；后续静态双缓冲和 pipeline 消融未明显生效、Tile-only 未真正激活，收益可能主要来自 DMA 利用率提升或多任务吞吐，而非单请求延迟。用户提供的数据本就标注为确定性合成周期模型，而非硬件测量。[来源 S48](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=81f53566-7c09-4b89-8621-81120c8079b2)

公平对比至少保持总容量、端口、带宽、算力和 workload 一致，提供异步 DMA、静态预取、双缓冲、SRAM 规划与 fusion 的强 baseline，再比较 Static、L2-only、Tile-only、Nested 和资源契约。报告 makespan、吞吐、延迟、搬运量、资源占用和 stall reason，不能只摘最佳 speedup。旧功耗百分比和工期估计不应进入当前规格。

<a id="open-issues"></a>

## 十三 未定问题与易混淆事项

### 仍应保留为开放设计问题

| 领域            | 已有约束                                                      | 尚未闭合的问题                                                                 |
| --------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 项目版本        | 可按真实分支、提交与日期区分                                  | 历史命名、文档稿号、代码分支是否一一对应，没有完整发布记录                     |
| Group/Tile 控制 | SPMD、两级 Context、Group 不做 if/else、Context 增加 for      | loop 精确语义、并存 program/image 的范围、错误取消 ABI                         |
| Context 微架构  | 多程序独立 PC、已发异步任务可继续                             | 最终 RISC-V 核、等待切换/IMT/独立 sequencer 的选择与成本                       |
| 向量执行        | EVU 不只是 epilogue；自主 kernel 是方向                       | 单 active kernel 还是多 kernel 驻留；向量状态隔离与细粒度访存延迟隐藏          |
| 公共任务管理    | Task/Event 归属与安全复用必须明确                             | 具体 TaskDesc/ISA、完成带宽、异常路由、generation 溢出与 drain 规则            |
| 存储 profile    | 同级 bank 内统一比例；一个 Context 一个 profile；HBM 安全边界 | Group/Tile profile 作用域与旧实现的切换描述需对齐；具体档位未统一冻结          |
| Shared/RF/AM    | 主要张量来自 Shared；AM/GPR/小流水缓冲可独立                  | 高频 VRF 是否物理共享、端口与 bank 组织、MFE 直达路径与真实吞吐                |
| Shared AM V1    | 固定 affinity、每 Context 总两槽                              | 物理总容量、group/bank/adder 数、非停顿结果 credit、错误链 drain 的 RTL 证明   |
| irregular       | 保留 Tile 动态 gather 需求，Group staging 是另案              | request/s、MSHR/outstanding、reorder、热点与冲突性能，是否需通用 atomic        |
| 跨 Context/设备 | readonly share、外层依赖、异步 P2P 可探索                     | 可写 move、远端 L2 直通、细粒度 collective 的收益是否抵消复杂度                |
| 编译器          | Group 自定义编排与 Tile Triton 接入                           | 完整 lowering、资源合同自动生成、profile/schedule 搜索、fallback 与 JIT/driver |
| 验证            | Python 合同模型加局部 RTL 是可行研究路线                      | 独立数值 oracle、真实端口/反压、综合频率/面积/功耗与端到端强 baseline          |

这些问题分别见前文原始来源；“待闭合”不表示后来必然没有实现，只表示本次恢复材料不足以确认完成。

### 十八条不能混写的边界

1. **CORA→Elenor→NEST→Nexus 不是一条改名链。**Elenor 已先存在，CORA 被明确独立，新案又排除 CORA。
2. **助手命名不等于用户正式项目。**RA-X1、HTFA、HFC、EFP、Micro-SIMT 等需保留来源身份；未确认名称不补成官方版本。
3. **SPMD 不等于锁步 SIMD。**不同 Tile/Context 可以独立 PC、进度与参数。
4. **放弃通用 Tile 间流水不等于放弃所有 pipeline。**Tile 内 async、预取/计算/写回重叠仍是主线。
5. **一个 Context 的异步 overlap 不等于多 Context 的收益。**两者必须分别消融。
6. **Context slot、logical task、engine command 与 system TaskID 不是一个编号。**旧异步完成必须回到原 owner。
7. **command accepted 不等于 done。**已接纳不能切换后重发；未接纳重试需无副作用。
8. **load_done、input_released、output_ready、task_done、store_done、safe retirement 不等价。**每个事件只允许它合同中规定的回收或消费。
9. **token DAG 无环不等于资源无死锁。**SRAM、AM、表项、返回路径和 completion credit 都可能形成等待环。
10. **Cache budget 不等于 SPM reservation。**总 bytes 足够不等于逐 bank 和端口可服务。
11. **completion-driven 不等于自由乱序。**有限 action 表只能执行已登记且依赖/资源满足的工作。
12. **Shared 容量统一不等于所有 RF 物理合一。**去除大块复制，不意味着去除 AM、控制寄存器或流水状态。
13. **共享 AM 不等于 engine 迁移。**V1 明确 affinity；等待输入时也不等于独占 engine。
14. **顺序 K-step 不等于 split-K 并行归约。**ring/tree 的数据分解与数值顺序是另一选择。
15. **两槽不等于两步 K，也不等于每 engine 两槽。**它限制 Context 同时存活的输出链总数。
16. **RVV、自定义 MatMul 和 token wait 不是同一标准 ISA。**custom/RoCC 不是标准 RVV，WFI 不是 task-token wait。
17. **真实 index 字节搬运不等于完整张量数值执行。**功能、协议、周期、RTL、PPA 与芯片测量分别举证。
18. **有理论峰值不等于有系统吞吐。**旧模型频率/带宽、助手面积功耗估算和综合周期模型的 speedup 都不能直接写成产品规格。

### 后续按什么顺序补档更有价值

- 找回早期完整设计原件、架构图、正式 proposal 与原始参数表，给每项补实际修改日期与状态，不把本文日期误当正式发布日。
- 把真实代码分支/commit 与对应 IR、memory policy、模型参数一一关联；特别核对 profile 的 Context 作用域与历史实现差异。
- 把 Shared AM V1 的需求、实现 commit 与验收结果链接起来，确认两槽、one-shot、affinity、满槽前进性和安全回收是否全部成立。
- 用同一数值 oracle、资源预算和强 baseline 追踪各方案；明确收益来自计算、数据移动、延迟隐藏、准入还是并发请求，而不是只累积最好看的 speedup。

<a id="sources"></a>

## 十四 覆盖范围与原始来源

### 覆盖范围与缺口

这份总档案尽量保留本次能够恢复的历史设计、修订和重点，但**不宣称已读取账户全部历史聊天或所有旧附件**。

- 原始对话中部分长消息只能恢复到可见段落；结论仅使用已经读到的用户要求、方案和纠正，不补写不可见内容。
- 找到过《NEST 对话 Markdown 文档》的记录，但对应会话/消息本次不可读取；不能用其标题或摘要推定全文已恢复，也不能声称已取回旧附件。[该历史会话](https://chatgpt.com/c/6a798898-34f8-83ee-88e1-c9c98c7eaa8d?messageId=13e4cb91-c8f0-424c-9971-92bfccadddf9)
- 10 月 26 日 Shared AM V1 对话说明曾生成需求指南；本次使用可读的对话总结，未逐条核对旧文件所称全部 18 条要求与 22 个验收场景。[对话总结](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)；[历史指南入口](https://chatgpt.com/library?file_id=libfile_8195cbfd61c48191a01557e8a9979c78&account_id=personal)
- 部分早期架构图内容未恢复；4 月 BOA gather 讨论不能视为已重新审阅原图。[原问题](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=1fd19ed5-35f8-44b3-89d2-ae973b878c08)
- `grid_sample + mul` 已确认讨论过，但未恢复完整回答，因此没有编造当时已经定案的 lowering。[原问题](https://chatgpt.com/c/6acc9954-5a68-83ec-ac31-d2aa4a6812a3?messageId=aceb5cde-d2c9-4519-8d9c-cab83b8d2699)
- ELENOR BOA/UCE/EVU/First Silicon 等旧附件未全部恢复；也没有完整 Elenor V1/V2/V3 发布文件集。字面 policy A/B 的原始定义尚未恢复，不补造其语义。
- 本文整理历史讨论，不是重新审计 GitHub、运行 Python 模拟器、修改 RTL 或验证旧外部项目的最新状态。

### 按主题回到原始对话

下列索引保留完整原始消息链接。正文中的同号引用指向该主题的代表消息；若一项包含用户要求、后续澄清与助手提案，可用这里的多条链接比对完整演进。

**S01 2026年7月 设计宗旨与目标** [原文1](https://chatgpt.com/c/6a646992-d14c-83ee-b27f-a7bad0d4fb79?messageId=cc79cd85-2847-43be-86be-861ddd73d67a)

**S02 2026年10月11日 个人研究目标** [原文1](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=ac5d71a8-fff0-49ea-881c-99c3a7f55687)

**S03 2025年8月20日 工业 NPU 原始需求** [原文1](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=ba2698fc-65fa-437e-85ea-e37497dc58a7)

**S04 2025年8月 单 L2 字节码修正** [原文1](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=36fc1de6-01c5-4528-9edc-a863d3b4df40)　[原文2](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=3572962e-1f88-47e8-963a-2dabc9bbb731)　[原文3](https://chatgpt.com/c/68a5bbdb-9b60-8324-9060-57434a696d7b?messageId=d199c712-07d0-4f69-8e05-e6ff6ccc76d0)

**S05 2025年11月 IREE 与 MLIR 优化复用** [原文1](https://chatgpt.com/c/6921c3c8-5d1c-8330-a56c-3a01baf31510?messageId=39a6ef82-005a-4bd7-9ce4-6e9915dbc618)　[原文2](https://chatgpt.com/c/6921c3c8-5d1c-8330-a56c-3a01baf31510?messageId=c725c5b6-458d-48fd-ab3e-f888693177bc)

**S06 2025年12月 MLIR Async 与 CUDA Tile IR** [原文1](https://chatgpt.com/c/694b8c19-4548-8327-9fdf-fb16eb74371b?messageId=243bf07b-903d-4aa7-a5c6-28ee3211ff30)　[原文2](https://chatgpt.com/c/694e707a-79c8-832b-afc2-f4e3d079cac1?messageId=e3473184-8661-4145-abfb-bde71c389218)

**S07 2025年12月 LLVM IR 层重排** [原文1](https://chatgpt.com/c/694fec5a-ddb8-832b-aa85-ae7544739e00?messageId=c745f5c4-4d6d-4cf7-b075-643276d38750)

**S08 2026年5月 Tile 控制 ISA 与 launch** [原文1](https://chatgpt.com/c/6a19530c-b224-83a3-914d-3ee7d6c44904?messageId=fead0f63-eb08-4c1d-817e-2ac4f3ec6572)

**S09 2026年7月1日 跨 tile function 重排** [原文1](https://chatgpt.com/c/6a44b4dc-65bc-83e8-a130-6678dd5fcb6b?messageId=1a9663bb-c0d9-41d1-8120-23da65cdfeda)

**S10 2026年7月 CORA 命名与控制边界** [原文1](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=89f9397d-745f-40a6-909a-50b33e9e7f39)　[原文2](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=416f80df-3122-44f8-a0d6-1d87fe32b8ff)　[原文3](https://chatgpt.com/c/6a5f63e8-24e8-83ee-bd57-b7ec12877cf1?messageId=eeb138c4-b5cc-4cd9-955e-c9a41317d843)

**S11 2026年7月 隔离 CORA 与选择 SPMD** [原文1](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=b628a6f0-c936-44d1-b572-f623c2bb989b)　[原文2](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=083a359b-516b-401b-99bc-ff5009d45852)　[原文3](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=f29bc173-6b90-4ac4-9db7-30919b37fafc)

**S12 2026年7月31日 NEST 论文机制与评估计划** [原文1](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=83a46c45-1143-433c-ac77-037f1e080f72)

**S13 2026年8月7日 NEST IR 与输出生命周期** [原文1](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=b5f8eb2c-e61a-49e8-9d74-0415cf36ec24)　[原文2](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=99fa4dbd-84f9-487b-991b-48d4558398f7)　[原文3](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=ae88dea3-33ab-40b7-a2cf-2428ba5e8586)

**S14 2026年8月20日 Nexus 上层提交 IR** [原文1](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=cee8808a-d227-4db9-8bbf-c63b41cc8dc4)　[原文2](https://chatgpt.com/c/6a7441dc-106c-83ee-be18-e369fbe11110?messageId=9d965648-d905-4384-b464-8f6236716f36)

**S15 2026年10月1日 Group 不支持 if else** [原文1](https://chatgpt.com/c/6abb7d0e-0bb8-83ee-8042-892cb647deb7?messageId=32ae04b5-99f4-48a3-af3a-5361bef40597)　[原文2](https://chatgpt.com/c/6abb7d0e-0bb8-83ee-8042-892cb647deb7?messageId=f8f973a4-f8f8-46db-b91b-53828931c548)

**S16 2026年10月20日 Shared RF MFE 与释放协议** [原文1](https://chatgpt.com/c/6ad72b86-1d40-83e8-9e9c-805306982abc?messageId=26579bc3-abbf-4008-aaf6-b4f49626714c)　[原文2](https://chatgpt.com/c/6ad72b86-1d40-83e8-9e9c-805306982abc?messageId=48d31e3c-991c-4b13-b102-a3f84d5a3390)

**S17 2026年9月 Triton 与 Group Tile 编译合同** [原文1](https://chatgpt.com/c/6aa7c75a-21bc-83e8-bc55-0a5d342a7b22?messageId=50675cfa-a814-4729-bc1e-dcc27a811935)　[原文2](https://chatgpt.com/c/6aa7c75a-21bc-83e8-bc55-0a5d342a7b22?messageId=7b2f4d03-503f-4f2a-bac3-6d2d9fe2e1f7)　[原文3](https://chatgpt.com/c/6aa7c75a-21bc-83e8-bc55-0a5d342a7b22?messageId=09e01a9b-7aa0-4955-8eba-efd23c637bf2)

**S18 2026年8至9月 Bank 双模式与同级统一分区** [原文1](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=0bcc2e8e-8787-44d1-a326-d79ef6acc382)　[原文2](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=608e1b05-6fab-4a4f-9c1b-7ba897ed8347)　[原文3](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=e1359226-a92a-4041-9216-9e02a33bea0a)　[原文4](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=a6cc2e3e-7fe0-412c-a228-c643d4b90ca5)

**S19 2026年8月7日 L1 L2 分配与 gather 路径** [原文1](https://chatgpt.com/c/6a685920-7d7c-83ee-b25a-5b43878fd86d?messageId=20ab2e80-d38e-4809-ab2d-104edca09975)　[原文2](https://chatgpt.com/c/6a685920-7d7c-83ee-b25a-5b43878fd86d?messageId=56ac7f38-4a68-464e-a699-7ce689a98f47)

**S20 2026年9月23日 单 Context 单 profile** [原文1](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7eff8b05-df9c-4435-ad12-03b9e8903f6c)　[原文2](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=47ed5cbc-e0dd-4f2f-9bb6-424ad559be46)

**S21 2026年9月13日 HBM 边界与资源合同** [原文1](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=bc75f0ff-84a8-4b4d-be0c-3f3781d61da1)　[原文2](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=dae07ccf-b952-481e-86e6-ebe82062b487)

**S22 2026年9月28日 历史源码审查** [原文1](https://chatgpt.com/c/6aba2e6b-b18c-83e8-8316-ed81630116de?messageId=d5095dae-0ae9-45e7-b9c2-d7b3eb75143c)

**S23 2026年10月14日 BOA EVU 分工与旧模型** [原文1](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786)

**S24 2026年10月6日 EVU 程序化与性能建模** [原文1](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53)

**S25 2026年10月17日 Tile Shared 用户澄清** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=b0c04d5b-6aeb-4a74-8210-0a465fa332eb)　[原文2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=0e71df90-56f0-46a1-84a6-e883870f0dce)

**S26 2026年10月15日 Ara 多 Context 边界** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74)

**S27 2026年10月21日 RISC V 多 Context 五条路线** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=6f22235b-7231-4426-a6bc-993c71da1653)　[原文2](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=1c2bac32-a0e8-5116-bc7a-1a44a3d374a5)

**S28 2026年10月23日 自主 RVV 与公共任务管理** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=f142eddd-6116-537b-93b0-1a9991927ea8)

**S29 2026年10月17日 四个独立 8×8 用户纠正** [原文1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)

**S30 2026年10月19至20日 ring 与 Reduce Memory 候选** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda)　[原文2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=167a29e0-d38d-4ccf-9c0e-0208f6b4e248)

**S31 2026年10月20日 K step 与 AM 持久累加** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)

**S32 2026年10月26日 Shared AM V1 用户规定** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)

**S33 2026年10月26日 Shared AM V1 指南摘要** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)

**S34 2026年10月26日 共享 AM 硬件与带宽建议** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)

**S35 2026年10月26日 K_total K_chunk K_cmd** [原文1](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f)　[原文2](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=ca5a1f4b-da43-56f9-b077-51306105ebf9)

**S36 2026年10月28日 DMA AG 与 irregular 性能** [原文1](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=d7fb793b-8c12-4d4e-a287-9bb8df90310b)　[原文2](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=e8c70d7b-882c-5288-9634-94ce88194f77)　[原文3](https://chatgpt.com/c/6ae15ddf-0670-83e8-ab0d-b99a9dad965e?messageId=f16f876d-14c6-54e2-a882-6f1b5d225886)

**S37 2026年10月5日 跨 Context move 收益纠偏** [原文1](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=83bd4615-28c7-4a8c-9a00-61ab3a52d5c9)

**S38 2026年10月5日 多设备与 P2P** [原文1](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=fb6fc45f-fec3-4015-8847-86a600771afd)　[原文2](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=56a06188-8fbf-4816-8d37-fdd9c5db7524)

**S39 2026年8月16日 Trace hierarchy** [原文1](https://chatgpt.com/c/6a81a3e4-5cfc-83e8-9007-2bc85ff78502?messageId=870f28da-c36d-457e-ae67-ec54ced3e31a)

**S40 2026年8至10月 算子与液态网络探索** [原文1](https://chatgpt.com/c/6acc9954-5a68-83ec-ac31-d2aa4a6812a3?messageId=aceb5cde-d2c9-4519-8d9c-cab83b8d2699)　[原文2](https://chatgpt.com/c/6a919b73-40b4-83ee-8548-8a918705114d?messageId=f8b49436-f0c6-4f75-8a06-9acdf7a73e9e)

**S41 2026年10月11日 Python validator 能力边界** [原文1](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=ac1723fd-2bd3-49b2-971f-c0bf468e0f56)　[原文2](https://chatgpt.com/c/6aca30a7-ca88-83ec-a606-67114c0fbfa1?messageId=c907076f-993b-44b7-9af1-5697915fa3a1)

**S42 2026年10月11日 调度 RTL 与 Python 环境切分** [原文1](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=379dcbda-2e9c-48de-82a3-b64d44ba045f)　[原文2](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=8960adcb-cbc1-451e-afbd-766057ba5f7e)

**S43 2026年10月23日 Scala 倾向与逐组件联仿** [原文1](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=0f65b652-2cd4-4f2d-96a0-ff552be8f68e)　[原文2](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=c141414a-cb4b-4a1c-beeb-df6185723a67)　[原文3](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=98915acb-7c7a-5447-8412-1102a30a6369)

**S44 2026年10月17日 Gemmini 后端与 custom MatMul** [原文1](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=c577d408-4616-574b-b696-7405b5fb9a05)　[原文2](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=2535c482-cc70-5c9a-9e2c-fbd95adcba0e)

**S45 2026年10月 开源向量与系统候选** [原文1](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=feb5de91-6135-5bac-8083-011b375687be)　[原文2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=c79258c6-1b93-4220-8017-92b855784278)

**S46 2026年10月12日 openTPU 仅借鉴 sim** [原文1](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=338db66a-e872-4d88-b9d4-007e7b31c0e2)　[原文2](https://chatgpt.com/c/6acb3c76-dc04-83ec-9d87-4c5a4d3dfbf6?messageId=788b39b2-046f-4481-95c3-366d6afe7474)

**S47 2026年10月22日 chisel npu 与 Gemmini 比较** [原文1](https://chatgpt.com/c/6ad9d043-bd18-83ee-8667-526a9c4878dc?messageId=a6ba43e1-a539-48a1-a1fd-b75cbae57c72)　[原文2](https://chatgpt.com/c/6ad9d043-bd18-83ee-8667-526a9c4878dc?messageId=9c226ff2-2670-5533-a1ee-1c6fc7a8963c)

**S48 2026年8月 模型实验评审与证据等级** [原文1](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=81f53566-7c09-4b89-8621-81120c8079b2)　[原文2](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=3b4dedc2-71f9-422c-a6a8-b57536a02309)　[原文3](https://chatgpt.com/c/6a71de1f-0810-83e8-910a-4750fa44e196?messageId=edacd9fe-a74a-4065-aa95-c11eaf4a01c5)

**S49 2026年8月12日 completion driven 滑动补位** [原文1](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=c474e393-071a-44d0-94ba-2c4f5fd41010)

**S50 2026年9月11日 分层调度与有限 action 表** [原文1](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=b34b621f-95e6-4d36-9183-75d2380f9315)　[原文2](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=be820549-c964-4f12-a769-bd2400ef80f2)　[原文3](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=18919c34-04dd-4614-8780-ddf83eec2bd0)

**S51 2026年9月11至12日 拓扑登记与 memory admission** [原文1](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=d9720292-8f8f-41e1-86ba-1cda1d6e26b7)　[原文2](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=1a08bf8b-403e-4db0-8cf1-51787a6d0b54)　[原文3](https://chatgpt.com/c/6aa3ee92-6bfc-83e8-b43c-15c34a255742?messageId=bfead4ad-6dab-47f3-85fd-e379af234733)

**S52 2026年9月18日 跨 Context 等待纠偏** [原文1](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=d03c1861-1a68-4ee2-9dd0-fb86be4dea6c)

**S53 2026年10月5日 Context ownership 与 readonly share** [原文1](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=44bc19fb-f07f-4c0a-ab05-e271002e3f1c)　[原文2](https://chatgpt.com/c/6ac46ddf-b068-83e8-8373-8968d2cd955d?messageId=f7df95ff-cae5-478d-a722-0b1c06cb3eec)

**S54 2026年10月2日 Tile 与 Group gather 比较** [原文1](https://chatgpt.com/c/6abf1244-7a3c-83ee-9527-ff150a4e701a?messageId=4666263b-e276-4a5c-ab5d-8cbe2f95254d)　[原文2](https://chatgpt.com/c/6abf1244-7a3c-83ee-9527-ff150a4e701a?messageId=4d1a49fd-cd2c-4635-bfc5-6a80d4674bd8)

**S55 2026年10月1日 无 atomic 的 PagedAttention** [原文1](https://chatgpt.com/c/6abdec71-c088-83e8-97cb-308af6ee08d0?messageId=c2ee61d8-2c2f-40cd-bf81-66629f20d1b6)　[原文2](https://chatgpt.com/c/6abdec71-c088-83e8-97cb-308af6ee08d0?messageId=a0ab9884-3318-452a-b51c-47da777295cc)

**S56 2026年3至4月 Tiling Island Path BOA 前驱讨论** [原文1](https://chatgpt.com/c/69b3c416-ada4-832a-ac9f-87f9dd8d7a6d?messageId=a0ec1f97-4247-460c-b60c-04a4149e0e84)　[原文2](https://chatgpt.com/c/69bd6375-49c8-8332-8e2e-9c9cf80abe0b?messageId=e37d32ca-94c3-4610-9534-993d8333e556)　[原文3](https://chatgpt.com/c/69bd6375-49c8-8332-8e2e-9c9cf80abe0b?messageId=61cbb7d2-8dad-4049-84f0-99e97c0ecb83)　[原文4](https://chatgpt.com/c/69bd6375-49c8-8332-8e2e-9c9cf80abe0b?messageId=a0219e3e-a52c-48b9-994a-9e52f7355e36)　[原文5](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1)　[原文6](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=d137bb01-8491-4aab-9c36-ebd65f4f4911)

**S57 2026年10月17日 系统级 TaskID 与安全复用** [原文1](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=7aa75680-b6b5-4ff0-904a-1864086c52ee)　[原文2](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=5afaae2a-309c-5bda-8ade-c5b7e174825e)

### 硬件历史的补充原始来源

**H01 2026-03-25，4×4一cycle与早期primitive** [原文1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=f38e5bc6-c9a5-43e3-9fc4-606d6cce2df1)　[原文2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=ef44cbee-6b0f-4c87-a043-2f9a9f6f5f53)

**H02 2026-03-25，shape选择** [原文1](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=763064a4-8d86-452d-86a9-e0c9bff0bff4)　[原文2](https://chatgpt.com/c/69c3981c-f910-8333-af31-baa4f968ffef?messageId=d2d7ede8-9824-4f91-be78-f37adbbf4dd1)

**H03 2026-03-27与04-01，BOA定义／目标／gather** [原文1](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=47b5a8f7-f5a5-4d82-865a-68227f2c23d1)　[原文2](https://chatgpt.com/c/69c4fbb8-9aac-832a-a3b0-1a175a47957d?messageId=d137bb01-8491-4aab-9c36-ebd65f4f4911)

**H04 2026-05-19 06:38 UTC，OPA/BOA评审及版本路线** [原文1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=bad29109-1ca2-4453-a9c9-065f2d4d5d79)

**H05 2026-05-20 02:16 UTC，FEATHER+/MINISA借鉴** [原文1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=115c05d3-07c2-4798-a7bc-6120ba755b75)

**H06 2026-05-19 09:16 UTC，BOA＋Flex双岛** [原文1](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=c7eef51c-adc9-4a1c-bd9b-71c149883897)　[原文2](https://chatgpt.com/c/6a0c051f-c648-83a3-adae-6ccd810965f5?messageId=f9a874e2-8de6-4e1e-8d9b-20d949fb5f28)

**H07 2026-07-15 02:03–02:05 UTC，TCP替换三方案** [原文1](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=58f914c0-6d87-48b7-8271-f90d70d93b44)　[原文2](https://chatgpt.com/c/6a562cbb-4328-83ee-b2ac-dae5b7ad3b94?messageId=7c9040fe-a6df-4e01-93e7-b1094223ca82)

**H08 2026-10-14 08:28–08:35 UTC，OPA通用PE与共享LSU** [原文1](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=d11334f7-8ca1-4c6d-8a44-172e6eec96bf)　[原文2](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=c5d6d593-88f0-4814-abe8-bb68232ab3b4)

**H09 2026-10-14，专用BOA＋EVU与旧配置** [原文1](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=17b1cf20-73ab-487b-b771-aa9b23d70786)　[原文2](https://chatgpt.com/c/6acdcc25-13b4-83e8-99cf-a2c96449e6ad?messageId=21dbf1ac-2cd4-40bb-b556-736dc8949e33)

**H10 2026-10-17 13:59 UTC，DPE／单阵列／小阵列簇比较** [原文1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=a0433a52-5efe-5d1d-b8e8-f3c0a58174cc)

**H11 2026-10-17，用户四独立8×8纠正** [原文1](https://chatgpt.com/c/6ad377b4-8338-83e8-8c6e-1e8edd966207?messageId=af77c502-9589-4e7b-86e3-840f05d89f79)

**H12 2026-10-19至20，ring与Reduce Memory** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=ab30b99b-b4f6-419e-b91d-87df54849bda)　[原文2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=167a29e0-d38d-4ccf-9c0e-0208f6b4e248)

**H13 2026-10-20，K-step与持久AM** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=6f64f011-11d6-4c4b-be8a-7d6e1fca3621)

**H14 2026-10-26，共享AM物理建议** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=fd703440-acc3-5fef-bde2-7bce63546bb1)

**H15 2026-10-26，K_total/K_chunk/K_cmd** [原文1](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=5c33af74-9582-59ac-a2ae-66b14339c88f)　[原文2](https://chatgpt.com/c/6adf1f1b-f6b8-83ee-8b14-f1611a1345d2?messageId=ca5a1f4b-da43-56f9-b077-51306105ebf9)

**H16 2026-10-26，Shared AM V1规则** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c6e53288-a6ff-4cf6-b98d-2b17541a879e)　[原文2](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=c43305ce-d2d3-5252-87e0-065284c1f0c4)　[原文3](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=d5a3148e-0038-50f4-b4c3-4abf563328b5)

**H17 2026-06-30 02:39–02:43 UTC，LTF位置三路及用户选择** [原文1](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=b10786f4-5ceb-4a69-ba5c-a49c18103dce)　[原文2](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=ccee2578-fc8c-485b-a100-d002cb03de80)　[原文3](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=8032d359-9e2a-4029-8a89-02fd70082ea0)

**H18 2026-06-30 02:45 UTC，LTF buffers** [原文1](https://chatgpt.com/c/6a422d12-d7c0-83e8-918c-7bf6fb90cc50?messageId=1b9db319-6710-4eae-a283-b664bc976e76)

**H19 2026-10-06 00:44 UTC，MFE优先与有限LTU** [原文1](https://chatgpt.com/c/6ac0cc5d-b464-83ee-939a-28b210319244?messageId=e5c632ca-4199-4327-b0ee-3bdf082d8f07)

**H20 2026-10-15 14:05–14:10 UTC，L1/RF合并候选** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=4a8dc1bd-ddf5-4d5a-a20a-ff1354cfa5e5)　[原文2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=c9c4d766-a731-4ac8-a54a-5ee56cbf86d2)

**H21 2026-10-20，RF/Shared与MFE直达** [原文1](https://chatgpt.com/c/6ad72b86-1d40-83e8-9e9c-805306982abc?messageId=26579bc3-abbf-4008-aaf6-b4f49626714c)

**H22 2026-10-17，Shared而非“全RF”** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=b0c04d5b-6aeb-4a74-8210-0a465fa332eb)　[原文2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=0e71df90-56f0-46a1-84a6-e883870f0dce)

**H23 2026-08至10月，SRAM/profile/gather** [原文1](https://chatgpt.com/c/6a7ad9ed-9d60-83ee-9bc5-c9c6ce4b9288?messageId=0bcc2e8e-8787-44d1-a326-d79ef6acc382)　[原文2](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=608e1b05-6fab-4a4f-9c1b-7ba897ed8347)　[原文3](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=e1359226-a92a-4041-9216-9e02a33bea0a)　[原文4](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=bc75f0ff-84a8-4b4d-be0c-3f3781d61da1)　[原文5](https://chatgpt.com/c/6aa6880f-8324-83e8-b947-2c23650c1d8d?messageId=7eff8b05-df9c-4435-ad12-03b9e8903f6c)　[原文6](https://chatgpt.com/c/6abf1244-7a3c-83ee-9527-ff150a4e701a?messageId=4666263b-e276-4a5c-ab5d-8cbe2f95254d)

**H24 2026-08-10，FIFO收缩** [原文1](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=4f9d1607-01ed-4c61-8330-5d0e888eb27a)　[原文2](https://chatgpt.com/c/6a6b5824-1a44-83ee-bb57-0c9b54cf8ee9?messageId=633ed92c-3525-477f-b954-47ed6116f51c)

**H25 2026-06-11 07:43–07:54 UTC，EVU descriptor/microcode/vector ISA** [原文1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=b1489649-4764-4561-9cbb-ad1dfbbec4c0)　[原文2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=031676cf-d07a-4f26-9d46-bb81c8ae35c4)

**H26 2026-06-11 08:01–08:02 UTC，Micro-SIMT** [原文1](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=9b765cd6-8e2f-4e54-b272-18ff8596ca98)　[原文2](https://chatgpt.com/c/6a2a671c-f534-83a4-9f09-f406bc0181ac?messageId=8065cf69-e4aa-4956-b624-319ccab2dbdb)

**H27 2026-10-06，EVU与带宽** [原文1](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=5ebc07a3-a9c1-4494-b4b4-9fa47f469f53)　[原文2](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=9cc8b569-f2d5-4565-af54-87dac3858fbd)　[原文3](https://chatgpt.com/c/6ac475e9-5494-83ee-8408-2f03c7bc9486?messageId=64cc222d-b067-42a4-a021-5026b3438632)

**H28 2026-10-15至17，Ara边界与开源候选** [原文1](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=54ac9c1e-11b0-4bf4-91b6-75b20f2d3d74)　[原文2](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=3e29a423-c782-4d28-a36d-16bd0e9e3186)　[原文3](https://chatgpt.com/c/6acf4b26-b844-83ec-ad22-2c607b9a61b8?messageId=c79258c6-1b93-4220-8017-92b855784278)

**H29 2026-10-21，RISC-V A–E** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=6f22235b-7231-4426-a6bc-993c71da1653)　[原文2](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=1c2bac32-a0e8-5116-bc7a-1a44a3d374a5)

**H30 2026-10-23，自主RVV与公共任务管理** [原文1](https://chatgpt.com/c/6ad8305d-3ed8-83ee-a2f7-c91c96c12857?messageId=f142eddd-6116-537b-93b0-1a9991927ea8)

**H31 2026-10-23，Scala/Chisel向量底座** [原文1](https://chatgpt.com/c/6ad9d1a8-ebd8-83e8-aeac-5fff7d2f3bf6?messageId=feb5de91-6135-5bac-8083-011b375687be)

**H32 2026-10-17，Gemmini与单条custom MatMul** [原文1](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=c577d408-4616-574b-b696-7405b5fb9a05)　[原文2](https://chatgpt.com/c/6ad309e6-3c78-83ee-b743-02bb8c6cebaf?messageId=2535c482-cc70-5c9a-9e2c-fbd95adcba0e)
