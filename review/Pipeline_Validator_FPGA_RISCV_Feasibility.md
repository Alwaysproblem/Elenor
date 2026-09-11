# ELENOR / Nexus 可编程调度逻辑的 FPGA 固化可行性分析

> **一句话总结与结论（[INFERENCE]）：ELENOR / Nexus 的调度逻辑可以采用“FPGA 固化事件、队列、背压与资源准入等实时机制，RISC-V 和可加载 Tile Program 保留软件灵活性”的方式实现，主要工程代价在软硬件协议、时序布线和一致性验证，而非完整计算阵列。**
>
> **本报告评估的是：参考 `pipeline_validator` 中的调度思路，实现真正驱动 engine / DMA 的 FPGA 调度控制器，并保留软件可编程性。不是移植 Python 模拟器，也不是在 FPGA 上复现模拟器的虚拟时钟与延迟模型。**
>
> **[INFERENCE] 结论：可行，而且适合采用“软件定义任务与策略、可编程序列器执行控制流、硬件维护实时协议”的分层方案。** 应固化的是依赖检查、资源准入、event/credit、背压和错误隔离等机制；不应将某个 matmul/pow 工作负载的步骤、shape、地址、依赖关系和放置方案写死在 RTL 中。
>
> **推荐形态：Host compiler/runtime + 一个管理级 RISC-V/uCtrl + FPGA 中的 Group Sequencer、Tile UCE、event/queue/resource 单元。** RISC-V 执行管理固件，不运行模拟器；UCE 执行软件生成并装入的 Tile Program，不是某个固定 workload 的专用 FSM。

## 1. 评估目标：把调度器做出来，而不是把模拟器做出来

### 1.1 目标系统需要具备的能力

1. 软件提交不同的 TileGroupTask、Tile Program、descriptor 和资源绑定，不重新生成 FPGA bitstream 就能改变受支持的计算流程。
2. FPGA 根据真实的 `ready/valid`、完成事件、credit 和资源状态决定何时派发工作，而不是等待一个根据 `ops/bytes` 算出来的倒计时。
3. 一个 context 阻塞时，能够在资源和策略允许的范围内推进其他工作；不要求 CPU 介入每次 engine completion。
4. 软件仍控制任务组织、内存分配方案、shape 版本、资源分区和提交策略；硬件阻止越界、重复发射、过早释放、陈旧完成和资源冲突。

**这里的“固化”不是“完全不可编程”。** 可以将小指令集、队列协议和状态机固定成硬件电路，同时把它们执行的程序、描述符和支持范围内的调度参数放在可更新存储中。

### 1.2 范围边界

- **范围内：**设备级 command/slot 管理、Group Task 推进、Tile UCE、多 context 等待与切换、engine ingress、event/phase、stream credit、资源准入与回收、PMU、fault/reset 协同。
- **作为接口对象：**BOA、EVU、MFE、USE、DMA、memory system。分析它们必须给调度器提供什么协议，但不要求本项目同时实现其完整数值/存储数据通路。
- **不属于本次目标：**CPython/xDSL 的 RISC-V 移植、FPGA 仿真加速器、全功能 AI 芯片、完整 HBM/NoC/cache 数据通路。

本文区分 **源码事实、运行观察、架构文档意图** 与 **[INFERENCE] 工程建议/预算**。后者不是综合、布局布线、功耗实测或已冻结规格。本文替换上一版偏向“模拟器迁移”的分析口径，不改变现有源码与架构规格。

## 2. 从 Python 实现中应提取什么调度思路

### 2.1 已存在的机制及其硬件含义

| 调度层/机制          | 当前 Python 实现                                                                                               | 应提取的硬件机制                                             | 不应照搬的部分                                                                       |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Device submit / slot | `Simulator._run_model` 按提交顺序推进，支持 slot pinning 和 first-free 分配；无 slot 则等待；完成后回收        | 可用 slot 表、命令接受/等待、完成退役                        | Python 的 device PC 循环不是完整多队列优先级调度器，也不应在硬件中直接解释高层 graph |
| Group Task           | `TileGroupSequencer` 解释 action list，推进 action index，执行 DMA、dispatch、wait、release 等                 | 固定 action op 集合 + 可加载 action 数据 + wait/issue 状态机 | Group Task 不是第二套通用 program/ISA；不要引入无必要的 group 级 CPU                 |
| Tile UCE context     | 当前 READY context 优先继续执行；当前不可运行时按循环顺序找下一个 READY context；另有 held-launch 优先重试路径 | 多份 context 状态 + 共享单 issue 前端 + 阻塞感知选择器       | **不是每拍 round-robin 时间片调度**，也不是 CPU 抢占式线程调度                       |
| held-launch          | ingress FIFO 满则进入 `WAIT_ENGINE_QUEUE`；保留指令，成功入队才推进 PC；重试不重复发射 `uce_issue`             | 稳定的待发命令、ready/valid、exactly-once 接受语义           | trace 中的 issue attempt 不能当成 engine 已接受或已执行                              |
| Event wait           | 等待指定完成事件；完成通知解除对应等待                                                                         | event scoreboard、wait 状态、局部唤醒                        | Python 字符串/set/dict 需变成有界、带身份与版本的硬件记录                            |
| Phase aggregation    | 按 grid、launch generation、logical task 聚合 `input_released` / `output_ready`，防重复/陈旧信号               | 带 generation 的 task 位图/计数和完成聚合                    | 不能只按 physical tile mask 或 UCE context ID 聚合                                   |
| L2 admission         | 合法但暂时放不下的 bundle 进入等待；只在 final-free 引起容量变化时重试，严格 FIFO                              | 事件驱动准入、资源授权、等待队列与去重通知                   | 不必把 Python first-fit/free-extent 搜索器做成全组合逻辑                             |
| Role dispatch        | 多 Tile 资源规划成功后提交、绑定；失败回滚                                                                     | 预留—确认—提交协议，防止部分启动                             | 软件单线程下的“原子操作”不是 FPGA 中一个时钟能完成的组合逻辑                         |
| Stream Queue         | credit、leased credit、已 pop 未 release、EOS/error                                                            | 有限 FIFO、credit 单元、控制 token 和错误通道                | EOS 不消耗数据 credit，不代表实际控制 token 不需要存储                               |
| Fault / drain        | 停止新工作，清理 outstanding、event、credit 和 residency                                                       | 硬件快停、事务跟踪、隔离与固件恢复策略                       | Python 清对象不能等价为取消已发到总线的真实事务                                      |

代码依据见 [S1]、[S2]、[S3]、[S4]。这些机制具有较小、明确的状态空间，适合硬件实现；复杂的任务选择和分配政策仍可由软件决定。

### 2.2 不能把当前模拟器当成已完成的控制器 ISA

当前代码还存在明显的行为级简化：

- `tile.py` 的 `MOV/ADD/CMP/LOAD_DESC/STORE_DESC` 等分支只推进 PC，`BRP` 直接跳转，没有完整模拟真实寄存器/谓词计算。不能因此声称动态控制指令已经具备可直接转成 RTL 的完整语义。[S2]
- `BARRIER_GROUP` 目前只是计数并推进 action；真实硬件需要参与者到达、generation、错误与超时协议。[S3]
- Engine 完成由延迟模型驱动，Gather 命中由 profile 提供；这些不是调度控制器应实现的运行机制。[S5]
- 单 Group、logical task 到 Tile 的当前映射，以及没有真实多 Group 竞争，都限制了已有证据的外推范围。[S6]

**本项目应保留调度的因果关系和安全性，不要求保留 Python 的每一个模型周期。** FPGA 增加取指、仲裁、跨域和握手延迟是正常的；必须把真实延迟反馈给模拟器，而不是用虚拟时钟隐藏它们。

### 2.3 必须明确区分的三个概念

- **Device slot：**设备侧一个 context/task 实例占用的执行管理位置。
- **Tile UCE context：**某个 Tile 内的一份 PC、等待状态、程序/descriptor 绑定和事件归属。
- **Engine job：**已经进入某个 engine 的独立任务；UCE 等待或换 context 时，它可能仍在运行。

因此，增加 device slot 不等于增加 Tile；切换 UCE context 不等于暂停或抢占已经启动的 BOA job；一个 group context 完成，也不等于所有其他 context 都要停下来。

当前模拟器允许两类 context 数各为 1–8，有效 UCE 数取二者最大值。架构 UCE 文档则将 First Silicon 定为单 context，将最多两个 context 的协程式切换列为扩展。**这些是不同成熟度的配置，不应把模拟器的 8-context 探索能力误称为已冻结硬件规格。**[S1]、[A1]

## 3. 推荐架构：软件策略 + 可编程序列器 + 硬件协议

### 3.1 结构与 ownership

```text
Host compiler / runtime
  ├─ 模型 lowering、任务拆分、Tile Program、descriptor template
  ├─ 数据布局、依赖、shape 版本、全局资源与提交方案
  └─ command / package / binding
                 │
                 ▼
设备管理 RISC-V / uCtrl
  ├─ command 合法性与上下文管理、shape 路径选择
  ├─ 内存分配策略、资源授权准备、故障恢复
  └─ 配置/发布已准备任务，不逐次轮询 engine 完成
                 │
                 ▼
FPGA 调度控制面
  ├─ Global Scheduler：ready command、依赖、slot、准入与退役
  ├─ Tile Group Sequencer：action index、role dispatch、group wait
  ├─ Program Residency Manager：miss 处理、验证、install-ready
  ├─ Resource / Event / Stream 单元：资源协议与真实状态
  └─ Tile UCE × 4
       ├─ resident Tile Program fetch/decode/控制执行
       ├─ context PC / wait / outstanding scoreboard
       └─ engine command ingress + completion adapter
                 │ ready/valid、completion、error
                 ▼
       BOA / EVU / MFE / USE / DMA
       各自拥有数据通路与内部资源仲裁
```

**[INFERENCE] 首选一个管理级 RISC-V 加每 Tile 一个小型可编程 UCE，不是每个 engine 配一个 CPU，也不是用一个 CPU 同步处理所有 Tile 的每次完成事件。** RISC-V 可以是 FPGA 软核、SoC FPGA 硬核或板外管理处理器；选择取决于接口与部署，而非调度语义。

| 对象/决策                                 | 推荐 owner                        | 必须遵守的边界                                             |
| ----------------------------------------- | --------------------------------- | ---------------------------------------------------------- |
| Graph lowering、kernel 组合、task 拆分    | Host compiler/runtime             | FPGA 不识别 ONNX/高层 graph                                |
| 跨任务提交策略、资源分区、shape 版本选择  | Runtime/管理固件                  | 通过命令与受支持参数影响调度，不直接改正在运行的硬件状态   |
| 已准备命令的依赖检查、slot grant、退役    | Global Scheduler                  | 固件不与它并发修改 slot/event 的同一权威记录               |
| Group action 推进、共驻留 candidate 检查  | Tile Group Sequencer / Dispatcher | UCE 不另设一个 group-level 调度器                          |
| Tile Program PC、wait、局部 runnable pick | Tile UCE                          | 不负责全局资源政策或 engine 内部调度                       |
| Stream credit / FIFO 状态                 | Stream Queue 单元                 | CPU/UCE 通过协议请求操作，不直接加减 credit                |
| L1/L2 实际 bank 仲裁、DMA 事务            | Memory/DMA 单元                   | 调度器消费其资源与完成信号，不重复实现第二套 bank 仲裁     |
| MFE 的 cache/MSHR、Gather/地址生成        | MFE / memory subsystem            | 不放入通用任务调度器；调度器只见 launch、ready、完成和错误 |
| USE 状态计算、checkpoint 内容             | USE                               | UCE/RISC-V 调度组件不因此承担全部状态数据通路              |

这与现有架构的分层总体一致。[A1]、[A2]、[A3] 需要特别澄清 event ownership：Global Scheduler 文档由 Event Fabric/Scheduler 维护 event，Driver/Firmware 文档又列 firmware 写 event 状态。[A3]、[A4] **[INFERENCE] 推荐硬件 scoreboard 为实时状态的唯一写入者；固件如需 signal/cancel，通过经过校验和仲裁的请求更新它，而不是与 completion 电路同时直接写同一表项。**

### 3.2 为什么不是“全部 RTL 写死”或“全部 RISC-V 固件”

| 方案                                          | 优点                                 | 主要代价                                                                  | 本项目判断                                                     |
| --------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 为单个 workload 写固定 RTL FSM                | 少量固定步骤时简单、时序确定         | 改依赖、步骤或 shape 路径可能重做 RTL/bitstream                           | 不符合软件灵活性的要求                                         |
| 单个 RISC-V 用固件解释所有 Tile 控制与完成    | 政策容易修改、调试直观               | 每条 launch/wait 都有指令与总线开销，多个 Tile 完成易形成串行瓶颈         | 可作为低发射率实现，但必须测量，不能假设能承载任意细粒度流水线 |
| 可编程序列器 + 硬件 event/queue + 管理 RISC-V | 控制步骤可更新，快路径自治，职责明确 | 需要小型控制指令语义、程序装载、固件/RTL 接口与验证                       | **推荐**                                                       |
| 每 Tile RISC-V + 自定义 launch/wait 辅助单元  | C/编译器工具链成熟，局部软件可扩展   | 核、指令存储、debug 和运行时开销复制到每个 Tile；仍需硬件 scoreboard/FIFO | 若未来 Tile 控制算法明显超出小指令集，再评估                   |

以上为 [INFERENCE] 工程比较。软件灵活性并不一定要求每个 Tile 都运行 C 程序；**可加载 Tile Program 本身已经提供了软件可编程性。**

## 4. 软件究竟还能改变哪些东西

### 4.1 不重做 bitstream 的灵活性

下表是推荐方案应支持的能力边界，不表示当前已有对应 RTL。

| 修改需求                                          | 软件改变什么                                              | 是否需要重做 FPGA                                      |
| ------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------ |
| matmul 后接 pow，改成接另一已支持 EVU 操作        | Tile Program、descriptor 与 event 依赖                    | 不需要，前提是 engine 已支持该操作                     |
| 改 tile size、迭代数、shape 版本                  | 编译出的程序/descriptor、合法的 shape 参数                | 在 engine/存储/指令能力范围内不需要                    |
| 调整 load/compute/store 重叠、ping-pong buffer 数 | Tile Program 中 launch/wait 顺序、slot/frame 绑定         | 在容量和 hazard 规则范围内不需要；硬件仍检查冲突       |
| 改参与 Tile、任务分块、数据地址                   | role binding、tile mask、task range、descriptor           | 在已实现 topology 和 placement 能力范围内不需要        |
| 根据运行 shape 选择另一执行路径                   | 管理固件选已编译版本，或使用已实现的 UCE 控制指令         | 不需要；不能依赖模拟器中尚未实现完整语义的谓词运算     |
| 调整跨任务优先顺序和资源份额                      | 软件提交顺序、资源授权；若已实现则配置 queue policy/quota | 不需要，但不能超出硬件已有策略模式                     |
| 改 allocator 策略，例如 first-fit 改为预留池      | 管理软件的分配策略与授权结果                              | 接口与 owner/lifetime 契约不变时，不需要修改调度快路径 |
| 增加硬件中不存在的调度 opcode、端口或 event 格式  | 新控制原语/接口                                           | 需要 RTL/bitstream 与 ABI 演进                         |
| 将非抢占 engine 改为可抢占                        | 新增 checkpoint/停止/恢复协议                             | 不能靠修改固件开关完成                                 |
| 超出已综合的最大 context/queue 数                 | 改变物理容量                                              | 通常需要重新综合；运行时只能在已实现上限以内启用/停用  |

**“软件可配置”不是“任意新算法都能通过 CSR 实现”。** FPGA 能执行哪些原语、具有多少状态存储和端口，仍然由硬件决定。初版不需要任意调度脚本引擎、复杂乱序或 partial reconfiguration 才能灵活。

### 4.2 初版建议保持的策略与可变项

**[INFERENCE] 推荐基线：**

- Group action 保持有限 op、线性 action index；具体 action list 由软件生成，不增加 group-level 分支 ISA。
- Tile UCE 保持单 issue、程序内顺序执行，支持规范化后的控制/launch/wait/stream/patch 原语。
- 保留阻塞感知 context 切换、held-launch 和独立 engine FIFO；不引入 engine 抢占。
- L2 admission 默认保持当前 strict FIFO 的可解释行为。软件可在入队前调整提交顺序；若要对已等待队列做 priority/bypass，需要单独定义公平性、饥饿和顺序契约，不能默默改变现有规则。
- 队列数量、context 上限、等待 fan-in、超时粒度等是能力参数；具体编码与数值由后续规格冻结。

这样固化的是“如何安全等待、发射、完成和回收”，而不是“哪个模型一定先执行哪一层”。

### 4.3 更新程序和策略时的安全边界

[A1]、[A4] 已要求 running program 不可 patch、descriptor 只允许修改已声明字段。推荐沿用：

1. Compiler 生成程序/descriptor，runtime 注册 `program_id/version/hash` 并发布任务。
2. Group 侧 Residency Manager 保证程序已经装入并可执行；具体加载/校验可由硬件与管理固件协同完成。UCE 只接收 resident handle，compiler 不增加显式 program-load 指令。
3. 新程序使用新版本或 inactive 存储区域，验证完成后供新 launch 使用；旧 context 继续执行旧版本。
4. 更新 live descriptor、queue policy 或 frame 时，只能在约定的提交边界或 drain 后生效；不能令在途命令一半使用旧配置、一半使用新配置。
5. CPU 发布描述符后再发布 ring entry/doorbell；使用平台要求的 memory ordering 与 cache maintenance。doorbell 不是对 CPU cache 可见性的自动保证。[A4]

## 5. 从模拟器调度走向真实硬件，必须补齐的协议

### 5.1 发射与完成：把倒计时替换成真实因果关系

```text
UCE 准备命令与 event 身份
  -> ingress 接受：命令成为硬件持有的待执行工作
  -> engine 接受：开始其内部执行/排队
  -> engine 返回 done / error 与原 event 身份
  -> Event Unit 更新 scoreboard
  -> 依赖满足的 UCE / Group action 被唤醒
```

**[INFERENCE]** ingress 和 engine 之间可有 FIFO，因此两次接受可能不是同一拍。FIFO 未接受前，待发命令与关联 PC/身份保持稳定；接受后不能重复发射。event/outstanding 的分配与接收也必须一致，防止零延迟/早完成响应先于 scoreboard 建立。

完成事件必须代表约定的**可见性边界**：例如 store 的完成不能只是“写请求进入队列”，而必须满足消费者读取与 buffer 回收所需的可见性。超时、ERROR、RESET 不能冒充 DONE。

这保留 held-launch 思路，但不照搬 Python 固定的 engine 延迟。将来测到的 dispatch、completion-to-wakeup、credit-return 延迟应回填到模拟器。

### 5.2 Event 与 phase：固定机制、动态依赖

- 软件指定谁依赖谁；硬件检查这些已声明依赖是否满足，不进行任意图搜索。
- Event 记录需要上下文归属、sequence/generation、状态和 producer 身份；不能只用一个裸 event ID。
- `input_released`、`output_ready`、`grid_done` 必须分开。前者可允许输入 consumer pin 解除；输出可见不自动代表输出 buffer 可回收；grid 返回也不自动代表所有外部 store 已完成。
- Phase aggregation 依据 logical task 身份和 launch/grid generation；同一 task 重复 signal 不得重复计数。
- 使用有限表深度与明确回压；表满不能覆盖仍在等待的事件。reset 后迟到 completion 必须被隔离。

**[INFERENCE]** 4 Tile、有限 context 下，这可以用 BRAM/寄存器、位图、比较器与分级仲裁实现。真正影响时序的是多源同拍更新、表端口和 fanout，不是事件概念本身。

### 5.3 内存分配留软件，资源安全留硬件

**不建议把 `BankedFreeExtentAllocator` 的完整动态搜索硬化。** 当前 Python 的 first-fit、克隆 free map、跨 bank 搜索与回滚，是分配策略的参考实现，不是必须保留的调度电路。[S7]

推荐分工：

| 软件/管理固件                                   | FPGA 资源协议单元                                         |
| ----------------------------------------------- | --------------------------------------------------------- |
| 选择 static/ping-pong/pool/free-extent 分配方式 | 检查授权对象的 owner、generation、边界与权限              |
| 为 context 准备 L1/L2 frame/buffer 方案         | 维护 live/pin/in-flight、hazard、reserved/committed 状态  |
| 维护 free map 的单一权威副本                    | 只在依赖和在途访问均已满足时确认可以回收                  |
| 收到真实回收通知后提交 free-map 更新            | 发出 release 完成通知，不把普通 phase signal 当成最终释放 |
| 为待准入任务提供合法 grant                      | grant 到齐后再使能 dispatch，禁止部分资源成功就启动       |

这里需要一个**单写入者协议**，而不是软件和硬件各维护一份可随意修改的 free map。固件完成最终回收提交后，发布带版本的容量变化；等待队列据此重试。只收到 `input_released` 或一次 unpin，不能假设 allocator 已经 final-free。

这也符合 UCE 架构文档“compiler/runtime 预留 slot，UCE 做 owner/lifetime 检查、不引入 V1 硬件动态 allocator”的方向。[A1]

**与上一版口径的关键差别：**固件分配需要多少真实时间，就消耗多少真实时间；等待 grant 的任务停在准入状态，其他合法任务继续运行。不存在冻结整个芯片的“virtual cycle”来掩盖固件延迟。软件分配延迟可通过提前准备和批量授权摊薄，但必须纳入性能预算。

### 5.4 多 Tile 原子 dispatch

**[INFERENCE]** 推荐采用小型预留—提交协议：

1. Dispatcher 根据软件 role binding 确认目标 Tile、resident program、context 和资源授权。
2. 对全部参与 Tile 预留资源，但暂不允许任何 engine 启动。
3. 所有参与方准备成功后发布统一的 dispatch generation/commit。
4. 提交前失败可撤销预留；一旦有工作实际启动，后续失败按 fault/drain 处理，不能假装没有执行过。

“原子”指不能部分通过资源检查就产生不可回退副作用，不要求四个 Tile 在同一个物理时钟沿同时开始第一条指令。集中式小 Group reservation owner 可以避免分布式抢占竞争，不需要引入复杂全局一致性协议。

### 5.5 背压和故障路径不能被数据流堵死

- Stream data credit 不足时，EOS/error 仍必须有受控的存储/传播方案；不能靠无限队列保证。
- Command、completion、fault、credit-return 不能形成互相等待的环：尤其不能让返回 credit 的 completion 被等待该 credit 的数据占满通道。
- 停止新 issue、捕获故障身份和保持在途事务计数由硬件快速执行；选择重试、销毁 context、重建资源由管理软件处理。
- 对真实 DMA 写，reset 不等于撤销已发生的内存副作用。必须 drain、处理迟到响应，并在确认安全后回收资源。

上述协议是“软件灵活但硬件安全”的基础，不能因软件可信就省略。

## 6. 调度器的性能和资源代价

### 6.1 应关注的性能指标

真正需要约束的是：

- 每 Tile / 每 Group 每秒能接受多少条 command。
- completion 到依赖任务重新可发射的延迟。
- FIFO credit 返回到下一条命令接受的延迟。
- 空闲 slot 回收、资源 grant、role dispatch 的延迟。
- 多个 Tile 同时完成时的峰值 event 接收与排空能力。
- 管理固件处理新任务、分配和故障的持续速率及最坏等待时间。

**[INFERENCE] 示例：**若局部 UCE 为 150 MHz，平均每条控制指令消耗 3 clocks，则每 Tile 理想上限约 5,000 万条控制指令/秒；一个任务若需要 8 条控制指令，未计阻塞前约为 625 万任务/秒，而不是 5,000 万个 engine job/秒。程序 fetch、patch、wait、event 更新都占成本，必须用实际程序测量。

再看固件预算：`固件占用率 ≈ 每秒管理决策数 × 每次决策CPU周期 / CPU频率`。假设 200 MHz CPU、每次管理决策 2,000 周期、每秒 1 万次决策，占用率是 10%；若每秒 10 万次，则理论上已占满一个核，尚未计缓存 miss、总线等待和错误处理。

**因此管理 CPU 应以已准备任务/资源批次为粒度，不能每个 Tile launch 都走一次重固件路径。** 以上是参数推导，不是实际频率或性能承诺。

### 6.2 存储主要属于控制状态，不是 tensor 容量

把调度器做成硬件，不要求将模型里的 16 GiB HBM、8 MiB Group SRAM、4 × 1 MiB L1 都计入“调度器资源”。它们属于被调度的数据系统。Gather cache/MSHR 全体、BOA DSP 阵列也不属于这个控制器的必要组成。

**[INFERENCE] 一个用于估算方法的有界例子：**4 Tile、每 Tile 最大 8 UCE context；下列记录宽度/表深度只是预算假设，不冻结 ABI，也不表示必须将当前 depth=1 的队列扩大到 16。

| 控制存储          | 假设                                                      |    原始容量 |
| ----------------- | --------------------------------------------------------- | ----------: |
| Context 状态      | 4 × 8 × 256 B                                             |       8 KiB |
| Event 表          | 1,024 × 32 B                                              |      32 KiB |
| Engine ingress    | 4 Tile × 6 类队列 × 16 entry × 32 B                       |      12 KiB |
| Tile Program 存储 | 4 × 64 KiB                                                |     256 KiB |
| Descriptor window | 4 × 8 KiB                                                 |      32 KiB |
| 资源/授权目录     | 512 × 32 B                                                |      16 KiB |
| 可选 trace buffer | 256 KiB                                                   |     256 KiB |
| **以上示例合计**  | 不含 RISC-V 固件存储、其他 group/ring/stream 表和实现冗余 | **612 KiB** |

这说明调度器更接近一个带若干表和队列的小型控制子系统，而不是多 MiB 宽端口 tensor SRAM 加矩阵阵列。

**资源风险仍然存在：**BRAM 宽度碎片、读写端口、表复制、多个 completion 同拍写入、softcore/debug、程序/descriptor 窗口、stream 数量都可能扩大占用。当前没有 RTL，不能给出可靠 LUT/FF 数，也不能保证某块小板一定放得下。

### 6.3 微序列器与 RISC-V 的额外代价

- 可加载 UCE 比写死 FSM 多出取指存储、PC、decode、小型寄存器/谓词状态与程序版本检查，换来软件改变控制流的能力。
- 每 Tile 复制 RISC-V 则还要复制 CPU 核、指令/数据存储、debug、可能的 cache 和总线接口；因此不把它作为默认起点。
- 共享一个管理 RISC-V 可以摊薄成本；若后续管理速率不足，先测瓶颈，再决定批处理、增加局部硬件原语或分组管理，而不是一开始加入大量核。

[INFERENCE] 对当前 4 Tile 调度原型，首先应做事件与队列吞吐预算，不应优先采购最多 DSP 或带 HBM 的器件。

## 7. 功耗、布线和实现难度

### 7.1 功耗：控制器的主要消耗在哪里

**[INFERENCE] 相对于完整 AI 数据通路，调度控制面的主要功耗来源是：**时钟网络、context/event/queue 表访问、状态翻转、跨模块广播、RISC-V 固件运行和调试输出，而不是数千个 MAC。

不能从模拟器的 utilization 或 `uce_active` 直接换算瓦数：它没有真实逻辑映射、布线电容、器件漏电、温度与翻转率。硬件调度也不保证一定比固件更省总能量；小负载时 FPGA 静态功耗可能占很大比例。

**可以用于立项讨论、但不能用于电源选型的 [INFERENCE] 预留：**若使用中档 FPGA/SoC FPGA、单管理核、4 Tile 控制面，目标约 100–200 MHz，且不计真实高吞吐 engine/HBM，整板先按约 **10–30 W 量级**讨论电源与散热。选大型 FPGA、重度 DDR/高速接口或大量 debug 时可能明显超出；该范围不是“调度 RTL 本身消耗 10–30 W”，也不是任何具体板卡实测。

更可靠的判断必须经过：资源预算 → 综合 → post-route activity-based 功耗估算 → 板上电源轨/整板输入与温度测量。[H3] 应分别报告管理 CPU、FPGA 静态、调度逻辑、外设与数据通路功耗，不能混在一起。

减少功耗的有效方向：

- 等待 context 用事件/位图触发或有限仲裁，不让管理 CPU 无意义忙轮询。
- 表与队列使用 clock enable / RAM enable；时钟 gating 使用厂商支持资源。
- completion 优先在 Tile 本地消费，只有 group 级事件向上聚合。
- PMU 本地计数，trace 按需启用并批量导出。
- 不为“软件灵活”引入不必要的通用 CPU 核和全局全连接总线。

### 7.2 布线：这里更怕 fanout 和表端口，而不是算术深度

| 物理风险                     | 原因                                          | 推荐处理及代价                                                                      |
| ---------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------- |
| 全局 event / reset 高 fanout | 多 context、多个 Tile 同时消费状态            | 本地 scoreboard、分层聚合与复位；增加少量状态复制与管线                             |
| 大规模 wakeup CAM            | 每个完成 event 都和所有等待项并行比较         | 有界 event table、waiter 位图/列表、分拍或分 bank 更新；牺牲部分唤醒延迟换面积/时序 |
| ready/grant 组合长路径       | dependency、resource、engine-ready 串成一条链 | 分级寄存化，定义清楚每一级的接受与预留；不要靠 false path 掩盖                      |
| completion 突发              | 多 engine 同拍写同一 event/PMU 表             | 每源小 FIFO、本地汇聚、多 bank 或可证明足够的串行排空能力                           |
| 跨 Tile 宽 command 总线      | 中央调度器搬运完整 descriptor                 | 全局只发送 task/descriptor 引用，Tile 本地取 descriptor；数据通路不穿过调度器       |
| debug 拖累时序               | trace 汇聚和大量 probes                       | 独立预算采集存储/链路；检查带 debug 的实现，不只检查无 probe 版本                   |

**[INFERENCE] 不建议将全部 engine command 和每拍 credit 都集中经过一个中央 RISC-V/总线。** 例如 4 Tile 各每秒 100 万条、每条 32 B 命令，单向记录量已达 128 MB/s，尚未计 completion/descriptor 和总线开销。局部控制与引用式命令能减少这种集中流量。

### 7.3 时钟、CDC、reset

- 100–200 MHz 可作为初始实现探索目标，不是已达成的 Fmax；不应拿模拟器 1 GHz 参数当 FPGA 时钟要求。
- 调度控制核心可先单时钟；管理核、DDR/PCIe 和 engine 若属于其他域，通过握手/异步 FIFO连接。
- 多 bit 命令不能每个 bit 各加两级同步器后就当作一致数据；需要事务级 CDC 协议。
- reset 释放需按域同步；跨域 outstanding、epoch 和迟到响应要一起处理。
- 增加流水线会改变物理延迟，但只要满足协议与规定顺序，就没有必要追求与 Python 逐周期相同。

### 7.4 PCB 和平台选择

**[INFERENCE] 首个原型采用现成开发板，不自研 PCB，也不以 HBM 卡作为默认选择。** 普通 FPGA + 成熟软核，或带 RISC-V 硬核的 SoC FPGA，都是可行载体。

公开平台可作能力参照：Microchip PolarFire SoC/Icicle 集成应用 RISC-V 核与 FPGA fabric；NEORV32 提供可综合 MCU 级 RISC-V 软核及自定义接口。[H1]、[H2] 它们证明实现路径和工具生态存在，不证明本设计已放置布线成功。

开发板将 BGA、供电和 DDR PCB 设计成本转移给板卡厂商，但使用者仍需满足供电、散热和接口限制。若自研板，才需要额外承担阻抗控制、差分/DDR 时序匹配、参考平面、PDN、供电时序、EMI 和热设计。**这些板级成本不是“调度逻辑固化”本身必需的第一步。**

## 8. 工程工作量与落地条件

### 8.1 工作量应按“调度控制面”估算

以下为 **[INFERENCE] 预算范围**：已有 FPGA/固件经验，复用成熟板卡、CPU 核与接口 IP；单 Group、4 Tile，完整覆盖约定的多 context、event、queue、admission、资源生命周期、fault/reset 和软件装载路径。没有偷偷把它缩减成只会发一个固定命令的演示。

| 工作包                         | [INFERENCE] 工作量 | 真正的难点                                                          |
| ------------------------------ | ------------------ | ------------------------------------------------------------------- |
| 调度契约与软件产物接口         | 1–2 人月           | UCE 指令语义、event 身份、原子准入、软件/硬件 owner、能力上限       |
| FPGA 调度控制面                | 3–5 人月           | 序列器、scoreboard、queue/credit、多 context、预留提交、fault/reset |
| 管理固件与装载/配置接口        | 1–2 人月           | task/资源授权、程序版本、ring 可见性、恢复与诊断                    |
| 验证、集成、板上调试与时序收敛 | 3–6 人月           | completion 乱序/突发、背压、陈旧事件、回收竞态、CDC、实际接口       |
| **合计**                       | **约 8–15 人月**   | 不含新 BOA/EVU/MFE 数据通路、自研 PCB、量产认证或任意硬件抢占       |

两到三名有经验工程师承担时，日历时间可先按 **4–8 个月量级**排期，但接口冻结、板上问题和工具收敛会形成串行阻塞。若要同时新做 DMA、复杂 IOMMU 或 engine 协议适配，必须额外计入，不能藏在该范围中。

与上一版“重写整个模拟器模型”的预算相比，本项目不需要硬化 profiled cache LRU、全内存时序模型或 trace 的 JSON 生成；工作重点转为真实调度协议和可编程控制器。

### 8.2 现金成本和维护成本

`项目成本 = 人月 × 团队全成本单价 + 开发板/工具/IP + 仪器/备件 + 集成与维护`

例如按每人月 5 万元的**假设财务单价**，8–15 人月对应 40–75 万元人力成本；这不是薪资报价或项目正式报价。板卡与授权需按现有设备、器件和 IP 询价，本次没有市场报价依据。

长期成本来自软件/RTL 的配套演进：新增一个调度原语，可能影响 compiler lowering、程序/descriptor ABI、固件、RTL、trace decoder 与验证。**将灵活性集中在程序和 descriptor、将稳定机制集中在硬件，可以显著减少必须重新综合的修改类型，但不能消除接口维护。**

### 8.3 进入实施前必须决定的事项

1. **语义与 cutline：**实现几个最大 context、哪些 UCE 控制指令、哪些 switch 条件；明确模拟器探索能力和 First Silicon 规范的差异。
2. **唯一 owner：**event、slot、frame、free map、credit、program residency 各由谁维护，谁只能发请求。
3. **界面和容量：**command/event/descriptor 的版本、字段、宽度、表深度、回绕与超限行为，均由后续规格冻结。
4. **服务速率：**控制指令 issue、completion burst、唤醒延迟、固件准入速率与真实软件提交开销。
5. **故障安全：**何时可以取消、何时必须 drain、什么时候允许资源复用；completion 和 timeout 同拍如何裁决。

这些问题不需要先有完整 BOA 才能解决，但在它们未明确前，也不能承诺确定的 LUT、功耗或板卡型号。

## 9. 验证证据：调度思路确实具有可提取的行为

### 9.1 本次重新运行的调度场景

运行的是现有 Python 模型，用于核对调度思路；不是 FPGA 功能或性能实测。全部开启 memory trace，检查模型完成、credit 守恒与 memory transaction 发出/完成一致性。

| 场景                        | 报告 cycles | transaction 完成/发出 | 从 trace 核对的调度行为                                                                                       |
| --------------------------- | ----------: | --------------------: | ------------------------------------------------------------------------------------------------------------- |
| `matmul-pow-free-slot`      |     226,921 |             128 / 128 | `mm_m0n0` 在 106468 完成，`pow_y0` 在 106469 复用 slot 0；无需等到其他 matmul context 最晚的 162834           |
| `matmul-pow-data-dep`       |     365,735 |             128 / 128 | `pow_np_c0` 在 116729 提交，等待两个所需 producer 的 106468/116728 完成，但不等无关 producer 的 155143/162834 |
| `matmul17-pow-tail-overlap` |     367,484 |             145 / 145 | 两个 pow 在 118478/164584 提交，早于 tail context 的 220946 完成；首个 pow EVU 服务在 Tile 1 的 187657 开始   |
| `l2-admission-wait`         |      29,052 |               16 / 16 | `ctx_b` 先占 device slot 等待容量；在 6982 收到容量变化后仅重试一次并 admitted，早于 `ctx_a` 的 29051 完成    |

这四个场景都完成且 credit 检查通过。提交、准入与实际 engine 服务是不同观察点，不能互换。tail 场景证明其他任务可在 tail context 仍活跃时执行，**不证明抢占 tail engine，也不证明任意两个 BOA/EVU 数据通路都在同拍计算。**

复现命令：

```bash
bash examples/run.sh matmul-pow-free-slot --memory-trace --trace-json /tmp/sched-free.json --json
bash examples/run.sh matmul-pow-data-dep --memory-trace --trace-json /tmp/sched-dep.json --json
bash examples/run.sh matmul17-pow-tail-overlap --memory-trace --trace-json /tmp/sched-tail.json --json
bash examples/run.sh l2-admission-wait --memory-trace --trace-json /tmp/sched-admission.json --json
```

这些例子使用 launcher 自带的 context、容量与延迟 override。[S8] 模型中的 “done 后下一 cycle 提交” 和 “release 同 cycle admitted” 是当前实现的时序，不是要求未来真实 FPGA 无条件达到一拍唤醒。外部 IR 的普通报告检查主要是完成/credit；本次额外从 trace 核对了表中的因果关系，没有运行全套 pytest。

### 9.2 FPGA 验收应该比较什么

**[INFERENCE] 推荐以协议和因果约束为主要验收依据：**

- 同一已接受 command 恰好执行一次；背压期间不能丢命令、改身份或重复分配 event。
- 依赖未满足不启动；与依赖无关的任务不被人为全局串行化。
- 有限资源不超配；phase、store visibility 与最终回收关系正确。
- completion 次序和延迟变化时仍正确；旧 generation、重复完成、reset 后迟到响应不污染新任务。
- 软件能在同一 bitstream 上装载至少两种不同调度流程，以及不同合法 shape/role binding，证明不是将一个 workload 写死。
- 记录实际 dispatch/wakeup/credit-return 延迟与资源利用率，再将硬件特征反馈给模型。

初期可以用**可控制延迟、乱序和错误的协议测试端点**验证调度器；它只是验证夹具，不是交付的真实 engine，也不能据此声称完成芯片。之后应接入已有真实 DMA/engine 接口验证命令接受、数据可见性和故障边界。无需为了分析/验证调度器，先实现全部 AI 算术。

最终物理证据仍需目标器件的综合、post-route、CDC/RDC、板上交互和功耗测量；Python 场景无法代替这些证据。

## 10. 结论与证据索引

### 10.1 最终建议

**[INFERENCE] 可以将这套初步调度思路实现到 FPGA，且不必牺牲软件灵活性。** 推荐将系统设计成：

- **软件决定任务与资源方案：**做什么、依赖谁、用哪个程序和数据布局、何时提交。
- **可编程 UCE/Group 控制器执行软件给出的控制对象：**具体的 load/compute/store 编排通过程序与 action 数据改变。
- **硬件强制实时安全机制：**资源检查、event/credit、握手、背压、exactly-once 接受、隔离和快速故障停止。
- **管理 RISC-V 处理变化快而不宜全硬化的工作：**分配策略、shape 版本、配置与恢复；不成为每条 engine 命令的串行中转站。

**成本的重点是控制协议、状态表端口、跨域/布线、验证和软硬件接口，而不是 HBM 容量或矩阵 DSP 数量。** 软件灵活性的重点是可加载程序/descriptor 和清晰的政策接口，而不是让软件随时修改任意正在运行的硬件状态。

### 10.2 当前源码依据

| 编号 | 文件/符号                                                                                                                                         | 本报告使用的事实                                                |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| S1   | [simulator.py][S1]：`Simulator.__init__`、`_run_model`；[config.py](../pipeline_validator/config.py)：`MAX_CONTEXT_COUNT`                         | 共享 Group、slot 分配/退役、context 数、device submit/await     |
| S2   | [tile.py][S2]：`step`、`_select_context`、`_retry_held_launch_issue`、`_issue_context`、`_enqueue_engine_launch`，约 243–308、532–671、785–877 行 | 当前 context 优先、held-launch、issue/accept 区别、控制指令简化 |
| S3   | [tile_group_sequencer.py][S3]：`step`、各 action 分支，约 129–309 行                                                                              | Group action index、wait/dispatch/release、barrier 简化         |
| S4   | [tile_group.py][S4]：准入约 603–733、`dispatch_role`、phase 约 1220–1293 行；[stream_queue.py](../pipeline_validator/stream_queue.py)             | final-free/FIFO 准入、原子 dispatch、phase 身份、credit/EOS     |
| S5   | [engines.py][S5]；[memory/cache.py](../pipeline_validator/memory/cache.py)                                                                        | engine latency/profile 不属于真实调度器应复制的机制             |
| S6   | [Limitation.md][S6]                                                                                                                               | 单 Group、无 tensor 数值、profiled Gather 和映射边界            |
| S7   | [memory/allocator.py][S7]：`plan_bundle`、`_plan_one`、`commit`                                                                                   | 软件分配器策略与资源生命周期                                    |
| S8   | [examples/run.sh][S8]；[examples/workloads](../examples/workloads)；[examples/scenarios](../examples/scenarios)                                   | 本次运行场景、各自参数和依赖设计                                |

### 10.3 架构文档依据

- [A1] **Tile UCE**：§1–2、§3.2.1、§3.3–3.5、§4.6。可加载 Tile Program、单 issue、context 边界、无 V1 硬件动态 allocator、engine 握手。
- [A2] **Tile Group Sequencer**：§1–3。Group Task 是 descriptor/action list，不是 group-level program；group admission 与 UCE 的分工。
- [A3] **Global Scheduler**：§2–3。命令仲裁、依赖/资源检查、event scoreboard、pipeline 寄存化。
- [A4] **Driver/Firmware/Runtime**：§2.5、§4.5–4.6、§5.2。event ownership 需对齐；doorbell 可见性、受限 patch、程序驻留与 running text 不可改。

### 10.4 外部能力与方法依据

- [H1] Microchip，**PolarFire SoC Product Overview，DS60001656K**：RISC-V + FPGA fabric 的集成平台及 Icicle 开发板。仅作平台能力参照，不作本设计资源或功耗证明。
- [H2] **NEORV32 官方项目**：可综合 MCU-class RISC-V 软核、接口与软件生态。仅作管理核选项，不要求使用该核。
- [H3] Xilinx，**UG907 v2018.3，Power Analysis and Optimization**：静态/动态功耗、活动率、post-route 分析方法。实际实施应使用所选器件与工具版本的对应指南。

[S1]: ../pipeline_validator/simulator.py
[S2]: ../pipeline_validator/tile.py
[S3]: ../pipeline_validator/tile_group_sequencer.py
[S4]: ../pipeline_validator/tile_group.py
[S5]: ../pipeline_validator/engines.py
[S6]: ../pipeline_validator/Limitation.md
[S7]: ../pipeline_validator/memory/allocator.py
[S8]: ../examples/run.sh
[A1]: ../design/elenor_tile_uce/ELENOR_Tile_UCE_Design.md
[A2]: ../design/elenor_tile_group_sequencer/ELENOR_Tile_Group_Sequencer_Design.md
[A3]: ../design/elenor_global_scheduler/ELENOR_Global_Scheduler_Design.md
[A4]: ../design/elenor_driver_firmware/ELENOR_Driver_Firmware_Runtime_Design.md
[H1]: https://ww1.microchip.com/downloads/aemDocuments/documents/FPGA/ProductDocuments/ProductBrief/PolarFire-SoC-Product-Overview-60001656.pdf
[H2]: https://github.com/stnolting/neorv32
[H3]: https://docs.amd.com/api/khub/documents/qILQe_LXLgAWMu~mbALIoA/content
