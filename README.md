# Nexus — ELENOR Pipeline Validator

> 本仓库由人类与大模型（LLM）共同打造：架构设想、代码实现、测试与设计文档均在人机协作中迭代完成。

## 项目简介

`pipeline_validator/` 是一个 cycle-accurate 的 **runtime / memory 契约验证器**，以一个
FPGA slice 的规模（1 个 Tile Group × 4 个 Compute Tile）模拟 ELENOR 风格加速器的
`Graph → Context → Grid → Task → Tile Program → Engine` 完整控制路径。

它不是"给什么跑什么"的通用时序模拟器：程序必须先通过编译期证明，runtime 只执行已被
证明合法的不可变产物，且另一个独立验证器会重新推导编译器的每一个决定。执行路径是
显式的、单向的：

```text
source xDSL ModuleOp
  └─ compiler.compile_program(...)            # 证明 + lowering
       └─ CompiledProgram（不可变 artifact，schema_version=2 / compiler_abi=v2）
            └─ loader.load_program(...)       # 只读加载，不导入 compiler
                 └─ LoadedProgram
                      └─ Simulator.run(LoadedProgram)   # 唯一接受的输入类型
```

`Simulator.run` 拒绝源 `ModuleOp`、私有 DTO 和未加载的 `CompiledProgram`；不存在
runtime 源 IR lowering 回退路径。

## 核心架构思想

### 1. 编译期证明，运行期零补救（prove, don't repair）

这是整个项目的第一性原则，贯穿每一层：

- **编译器是证明器**。`compile_program` 在产出可执行体的同时，对 _每一个_ advertised
  Profile mode、_每一个_ 真实出现的 L1/L2 mode 组合、逐 bank 的 striped 布局与 padding、
  Frame Slot、事件 live frontier、Grid/控制资源上限、`R × 子 Arena` 包络做静态证明。
  证明不过就拒绝编译——compiler 从不悄悄丢弃非法 mode，从不自动调低
  `requested_contexts_per_tile`。
- **产物携带证明**。`CompiledProgram` 是深度不可变对象，内嵌 canonical source、
  Profile Registry、dependency proofs、static effects、resource budgets、binding
  guards、source map、重定位表和四个 SHA-256（source / registry / target / artifact）。
  持久化 codec 用显式 allowlist 解析：未知字段、重复 JSON key、非有限数值、版本
  不符、hash 不符一律拒绝。
- **验证器不信任编译器**。`execution_verifier.py`（全仓库最大的模块）不 import
  compiler、不修复图，而是独立验证静态 executable 的 ordering / dependency contract
  与资源、冒险不变量：transfer/view/descriptor 语义、跨 root 数据冒险、maintenance
  覆盖与前序、事件 live frontier、L2 padded span 不重叠且逐 bank 守恒。它校验的是
  静态契约——动态 issue 顺序由 runtime 的 `GroupScheduler` 按 scan / credit /
  backpressure 决定，不在静态验证范围内。Loader 只做验证与绑定检查——不修复
  依赖、不补插等待、不换 Profile、不重建源 IR。
- **Runtime 不发明任何同步**。普通 await 就是普通 await，不会被隐式提升为全局
  barrier；maintenance 只能来自编译器生成的描述符。非法程序在任何 cycle 运行之前
  就已经失败。

### 2. Compute / Control / Data Movement 分层进入 IR 与硬件边界

`Compute != Control != Data Movement` 不是口号，而是 IR 的所有权类型系统：

| 前缀      | 所有者                 | 能表达什么                                              | 不能表达什么                 |
| --------- | ---------------------- | ------------------------------------------------------- | ---------------------------- |
| `tile.*`  | 一个 Task/Tile Program | L1 alloc、load/store、engine launch、local await/signal | 跨 Task 调度、L2 分配        |
| `nest.*`  | 一个 root Context      | L2 alloc、prefetch/store、dispatch、release、组内 await | Tile 内 L1 布局、Host 控制流 |
| `nexus.*` | Host/CPU 模型          | submit_context、跨 root 依赖、await、return             | 任何片上资源细节             |

硬件边界同样强制：`device.py` 的 CPU 控制器只依赖执行 DTO 与消息协议，**永远观察不到**
TileGroup slot、sequencer、Tile context 或 per-task PC；`runtime/group_port.py` 是有界
消息适配器，root 请求先只占用 pending 元数据，Group slot / L2 Arena / 事件预算 / 执行
所有权在真正准入时才**原子提交**。两阶段准入让"排队"与"占用硬件"成为两个可分别观测
的状态。

### 3. 一切资源有限，且每类等待都可归因

模型里没有无界队列。Group 调度器就是一张有限表：`action_capacity=16`、
`scan_width=4`、每 Context quota=8、`event_capacity=4096`、inflight=32、
prefetch/store/dispatch 各自有 credit；Event Table 使用**编译器证明的 live frontier**
（而非全部历史事件）。Task 准入的失败被精确分类为：

`WAIT_CAPACITY` / `WAIT_FRAGMENTATION` / `WAIT_SLOT` / `WAIT_CONTROL_RESOURCE` /
`WAIT_CONTEXT_LIMIT`

纯 plan 无副作用；永久不可能的请求（空池也放不下的 per-bank 需求）直接 fault，而不是
进入等待队列假装有机会。总空闲字节不能掩盖 stripe 碎片化失败。所有等待类都进 PMU——
报告由真实快照生成，`report.py` 明确**不**从 trace 名字反推硬件事件。

### 4. 分层 SRAM Profile：可重构片上内存的一等公民

L1/L2 各自独立地在 SPM / Cache 之间划分（schema-2 YAML 声明每层 mode 表），而"切换
分区"是被完整建模的显式操作，而不是配置魔法：

- 每次真实切换都是编译产物中的 `ProfileReconfigDesc`，携带 expected/target mode、
  Registry hash、generation 绑定的 await 证明、受影响域与成员清单，执行固定十二步
  序列 `ACQUIRE → CHECK_FRONTIER → CLOSE_ISSUE → DRAIN_REFERENCES → CLEAN_INVALIDATE
→ DRAIN_DOWNSTREAM → PREPARE → WAIT_READY_ACK → COMMIT → WAIT_COMMIT_ACK →
OPEN_ISSUE → RELEASE`。
- `ProfileController` 是唯一 L1/L2 writer，Prepare/Commit ACK 走有界成员总线（每
  Tick 一对请求/响应），按成员身份与 generation 校验。**generation 单调递增**：迟到
  的返回写不进已被复用的新 generation——ABA 问题在模型层面被封死。
- Profile 身份是**准入维度**：root 与 Task 都只对活跃 Profile 比较 SAME / COMPATIBLE
  两类 FIFO 头；Profile 不兼容的工作根本不是 runtime 候选，切换必须由编译器先行
  安排。
- L1-only 重构不触碰活父 L2 Arena；L2-only 重构必须先关闭一切可能触及 L2 的工作。

## 调度设计

三层调度，各自有界、各自可观测：

```text
CPU:  CpuDeviceController（device.py）
        有限 pending/completion credit，解释 nexus.program，只发消息
  └─► Group: GroupScheduler（group_scheduler.py）—— ready-action 窗口
        每 cycle 一次 REGISTER + 一次 ISSUE，有限 action 表 + 有界扫描
        └─► Tile: tile.py —— 每 Tile 每 Tick 至多原子提交一个 Task
              eligible-head round-robin，UCE context 是精确物理资源
```

- **Ready-action（S0/S1 策略）**：S0 每 Context 只许队首 action 参与发射（in-order），
  资源就绪的非队首被挡住时计入 `group_ordering_stall` PMU；S1（默认）允许扫描窗口
  越过 PENDING 依赖乱序发射，依赖未就绪就跳过继续扫。S2 是保留的探索值，当前与
  S1 共用发射路径。普通 action 的 `depends_on` 以 `action.dependencies` 直接携带，
  不再被额外串行化；`nest.await` 按 operand 显式 lower 为一一对应的 `WAIT_EVENT`
  action（前端提交栅栏），`nest.barrier` lower 为 `BARRIER_GROUP`（Context 局部
  全前缀完成栅栏）——两者都不会被改写为全局 barrier。
- **完成与注册解耦**：`note_completion` 独立于 action 表回收 credit 与事件——表满
  永远阻塞不了完成回收，从结构上排除"表满死锁"。
- **SAME/COMPATIBLE 双 FIFO 头**：root 准入对活跃 L2 Profile、Task 准入对每 Tile 活跃
  L1 Profile 各自维护两类队首；SAME 优先，SAME 队首无法完整提交时 COMPATIBLE 队首
  可补位，但同类内不许越头。这是"Profile 感知的准入"，不是通用乱序调度。
- **原子提交，无回滚**：一个 Task 的提交原子覆盖 L1 Arena、UCE pin/Slot、Frame/
  控制状态、父 L2 pin 和 R lease；一个 Tile 受阻不回滚其他 Tile 已提交的 Task。
  Grid 在任何选中 Tile 仍 pending/active/retiring 时不能完成。
- **R lease 共驻约束**：`requested_contexts_per_tile` 以 `(parent binding, launch
generation, tile)` 为键约束同 Tile 共驻 Task 数，只在 Task 安全退役或确认取消时
  归还——`tile.free`、`input_released`、`output_ready`、cancel 请求都不会提前归还。

## 内存与生命周期模型

- **Arena 所有权**：一次 root 调用持有一个 L2 Arena（`RootInvocation`），一个已提交
  Task 持有一个 L1 Arena（`TaskIdentity`）。`nest.release` / `tile.free` 只使命名
  view 失效，不归还父 Arena；Arena 只在 owner 安全退役时归还。零字节 owner 也有显式
  Arena/控制/lease 记录。
- **L2 backing/claim 分离**：readonly 共享的多个 view 别名同一个物理 backing，不增加
  物理占用；claim 走 `DECLARED → BOUND → RELEASED` 状态机；`nest.publish` 在生产者
  最后一次访问后封存只读导出，原 Arena 可先于读者退役，快照单列 origin-retired
  backing，协议级字节总量按 backing 身份去重。
- **编译期不复用已释放 L2 区域**：每个本地 L2 分配有独立不重叠的 stripe-rounded
  padded span，验证器拒绝 span 重叠或逐 bank 不守恒的布局。
- **逐腿传输状态机**：`memory/transfer.py` 把每次搬运建模为多腿 route
  （HBM → Global DMA → NoC → L2 bank → L1 bank → Local DMA），每腿完成后才发射下一腿，
  按 stage 报告等待原因做 PMU 归因；cancel 由 generation 隔离。

## Fidelity 阶梯与字节证明

| Fidelity      | 分配/寻址                           | 传输时序                        |
| ------------- | ----------------------------------- | ------------------------------- |
| `timing_only` | 仅逻辑契约                          | 单腿折叠                        |
| `runtime`     | 真实 HBM binding + L1/L2 Arena/view | 单腿折叠                        |
| `full_memory` | 真实 binding/Arena + bank segment   | HBM/DMA/NoC/L2/L1/LocalDMA 全腿 |

三级 fidelity 执行**完全相同**的资源、Profile、maintenance、generation 与 gate 契约；
只有 `full_memory` + 注入 `ByteStore` 才证明字节搬运。`ByteStore` 是稀疏 oracle：拒绝
未初始化读（无隐式 zero-fill）、校验 host seed 覆盖、读腿完成时捕获源字节、写腿完成
时才提交目标字节。没有 `ByteStore` 时，Gather 的命中序列是源码 authored 的确定性
profile，报告中如实标注 `deterministic_profiled_not_address_or_value_accurate`——
cache 容量从不预测命中率，BOA/EVU/USE 不计算张量数值。

## 设计定位与组合创新

本项目不与通用模拟器竞争：gem5、GPGPU-Sim 等成熟工具提供丰富的可配置调度策略、
cache 层级与互连模型，面向通用微架构探索；RTL 仿真提供实现级精确验证。本项目的
关注点不同——把 **runtime / memory 契约本身**当作验证对象：程序合法性在运行前
证明，资源行为在运行中按契约检查。以下四项特性各自在已有工作中均有先例，本项目
的主张仅是它们的**组合**在一个可执行模型中闭环：

1. **Sealed artifact**：编译产物深度不可变、携带证明与 hash 链（source / registry /
   target / artifact 四个 SHA-256），严格 codec 拒绝未知字段与重复 key；编译一次
   即可在独立进程只读重放（hermetic replay），replay 路径不导入 compiler。
2. **独立 verifier**：`execution_verifier.py` 不信任 compiler 输出，独立验证静态
   ordering / dependency contract、资源预算覆盖、hazard 与事件 live frontier 等
   不变量；动态 issue 顺序由 runtime 调度器决定，不在其内。
3. **Profile / generation 准入**：SRAM 分区切换是编译描述符 + 成员 ACK 协议 +
   单调 generation 隔离；root / Task 准入按活跃 Profile 分 SAME / COMPATIBLE 两类
   FIFO 头。
4. **Typed stalls**：等待在模型内带类型（五类 `WAIT_*`、credit / epoch / ordering
   stall），PMU 由真实快照累积，报告不从事后 trace 名字反推。

关注点差异（仅陈述定位，不声称对方缺失能力）：

| 维度       | 通用 cycle 模拟器（gem5、GPGPU-Sim 等）                 | 本项目                                              |
| ---------- | ------------------------------------------------------- | --------------------------------------------------- |
| 定位       | 通用微架构研究：可配置调度器、cache、互连的设计空间扫描 | 面向一种目标契约（ELENOR runtime/memory）的定点验证 |
| 输入       | 用户或工具链直接提供的程序                              | 仅接受编译期证明通过并封印的 artifact               |
| 合法性保障 | 按配置执行并给出统计                                    | 任何 cycle 之前完成静态证明，verifier 独立重推导    |
| 片上 SRAM  | 可配置 cache / scratchpad 层级                          | 显式 Profile 分区 + 描述符化切换 + generation 隔离  |
| stall 观测 | 统计计数与流水线级可视化                                | 模型内带类型的等待 + 快照累积 PMU                   |

有意保留的边界（不是疏漏）：单 Tile Group、固定 4-Tile 拓扑、无 task stealing、FIFO
队首约束、静态 Arena 布局、Profile 物理值 `由后续规格冻结`。完整清单见
[`pipeline_validator/Limitation.md`](./pipeline_validator/Limitation.md)。

## 代码结构

```text
pipeline_validator/
├── compiler/            # 证明型编译器：api(流水线) / lowering / resources / profile_pass
├── dialects/elenor.py   # xDSL 自定义汇编方言（tile.* / nest.* / nexus.*）
├── workload_ir.py       # 源 IR parse / print / verify
├── execution_ir.py      # 冻结的 compiler/loader/runtime DTO
├── compiled_program.py  # 不可变 package + 严格 codec
├── execution_verifier.py# 独立只读语义验证器（不信任 compiler）
├── loader.py            # 只读加载：验证 + binding 检查 → LoadedProgram
├── profiles.py          # Profile Registry、契约、布局、控制描述符
├── config.py            # HardwareConfig / SimConfig / schema-2 YAML
├── device.py            # CPU 侧控制器（DevicePort 消息协议）
├── runtime/             # group_port(消息适配) / event_table / program_table / reset / fault
├── group_scheduler.py   # 有限 ready-action 调度器（S0/S1）
├── tile_group.py        # root/Grid 准入与退役
├── tile.py              # 每 Tile Task 准入、UCE context、eligible-head RR
├── tile_group_sequencer.py
├── engines.py           # BOA/EVU/USE Roofline 单 job 时序 + MFE 多 lane
├── memory/              # arena / profile_controller / transfer / cache / allocator /
│                        # noc / mshr / byte_store / l1_slot_frame / hbm_region / payload
├── stream_queue.py      # credit/backpressure/EOS 生产者-消费者契约
├── simulator.py         # 只消费 LoadedProgram 的 cycle driver
├── pmu.py / trace.py / report.py   # 观测性：PMU、Perfetto trace、快照报告
└── cli.py               # 源编译与 compiled replay 入口
```

## 快速开始

```bash
# Python 3.11 conda 环境（依赖：xdsl、pyyaml、pytest）
conda env create -f pipeline_validator/environment.yml
conda activate elenor-validator

# 全部测试
python -m pytest pipeline_validator/tests/ -v

# 可运行示例目录
bash examples/run.sh list

# 运行一个 workload（compile → persist → load → run 全链路）
bash examples/run.sh gather-matmul --json

# 追加 Perfetto trace 与内存细节
bash examples/run.sh matmul-2048x512-boa256 \
  --trace-json /tmp/matmul.json --memory-trace
```

CLI 常用路径：

```bash
# 只编译，产出 review.json + .exec.txt + .compiled.mlir.txt + .target.yaml
python -m pipeline_validator --ir-file path/to/workload.mlir \
  --input-binding Y=0x100000:131072:rw \
  --compile-only --compiled-output artifacts/review.json

# 独立进程只读重放 artifact（不读源文件、不导入 compiler）
python -m pipeline_validator --compiled-file artifacts/review.json \
  --hw-config artifacts/review.target.yaml \
  --input-binding Y=0x100000:131072:rw
```

常用开关：`--group-policy s0|s1|s2`、`--context-mode N`、`--device-context-mode N`、
`--sim-override KEY=VALUE`（含 `fidelity=full_memory`）、`--max-cycles N`、
`--json` / `--report`、`--detailed`、`--trace-json` / `--trace-html` /
`--memory-trace`。完整语义见
[`pipeline_validator/README.md`](./pipeline_validator/README.md)。

## 质量门槛

- **测试**：`python -m pytest pipeline_validator/tests/ -v` —— 452 个 collected
  用例（390 个测试函数，含 parametrize 展开），全部通过方可提交
- **Lint / 类型**：`conda run -n elenor-validator pre-commit run -a`（ruff + mypy 零
  错误 + prettier）
- **风格**：行宽 108、2 空格缩进、双引号、LF；所有 Python 模块均使用
  `from __future__ import annotations`

## 文档导航（validator）

- [`pipeline_validator/IR_SPEC.md`](./pipeline_validator/IR_SPEC.md) — 源与可执行 IR 规范
  （契约、调度、准入、Arena 生命周期的权威定义）
- [`pipeline_validator/README.md`](./pipeline_validator/README.md) — 详细使用与语义手册
- [`pipeline_validator/Limitation.md`](./pipeline_validator/Limitation.md) — 有意保留的建模边界
- [`examples/README.md`](./examples/README.md) — 示例索引与 MLIR 修改约束

## 下一步大概想法：ELENOR AI 加速器

`design/` 下的 28 份架构规格是下一步的大概想法——把 validator 已建模的 runtime /
memory 行为落地为一颗真实 AI 加速器。四类引擎覆盖未来工作负载空间：

| 子系统  | 职责              | 典型工作负载                                                        |
| ------- | ----------------- | ------------------------------------------------------------------- |
| **BOA** | Dense Compute     | GEMM、Conv、QK、AV、Expert MLP                                      |
| **EVU** | Irregular Compute | Softmax、Norm、RoPE、Activation、Gather/Scatter、Tail 处理          |
| **MFE** | Memory Flow       | Page Stream、Segment Stream、Sparse / 布局变换相关数据流            |
| **USE** | State / Control   | Scan、Recurrence、Dynamic Shape Assist、Token Routing、Event Assist |

目标配置从边缘到数据中心统一复用：Edge（8–16 Tile，LPDDR）、Balanced（64 Tile，
HBM/DDR）、High End（128 Tile，HBM）。

![ELENOR overview](./image/Elenor_v0.png)

设计文档入口：[`design/ELENOR_Architecture_Design_v1.md`](./design/ELENOR_Architecture_Design_v1.md)
（总体架构），模块规格位于 `design/elenor_<module>/`，覆盖芯片顶层、四大引擎、
片上组织、软件栈与验证计划；架构评审记录见 `review/`。PDF 导出：

```bash
bash scripts/generate_pdf.sh -f design/ELENOR_Architecture_Design_v1.md
```

## License

See [`LICENSE`](./LICENSE).
