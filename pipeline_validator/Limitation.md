# 局限性

## 模型覆盖范围

当前实现已经覆盖以下合同与运行路径：

- 源 `ModuleOp` 必须显式经过
  `compiler.compile_program → CompiledProgram → loader.load_program →
LoadedProgram → Simulator.run`；Simulator 不包含源码 lowering fallback。
- 编译产物深不可变、可严格 JSON 持久化，并绑定 source/Registry/target/
  artifact hash、调用点、relocation、source map、依赖证明、Profile 入口/
  出口与静态资源预算；Loader 只读验证，不重编译或修图。
- CPU Device interpreter、消息端口、Group S0/S1/S2 ready-action 窗口、
  Tile UCE eligible-head RR 与独立完成。
- Root pending metadata 与 Group execution Slot/L2 Arena/event/control 资源
  分离；待准入 Root 不预占这些硬件资源。
- Grid Route 有界登记、每 Tile 独立 Task 补位；某 Tile 暂时阻塞不会回滚
  其他 Tile 已提交的 Task。
- L2 Root Arena 与 L1 Task Arena 的逐 Bank stripe 预留、padding、owner、
  allocation/Profile generation、view/pin/inflight 生命周期检查。
- `nest.release`/`tile.free` 只失效局部 view；仅 owner 安全退休才返还整个
  Arena。R lease 仅在 Task 安全退休或取消隔离确认后归还。
- L1/L2 独立 Profile Registry、编译期 SAME/COMPATIBLE 绑定、显式普通
  await、完整 reconfiguration/maintenance 描述符、全成员
  Prepare/Commit ACK、唯一 `ProfileController` writer 与分层 generation。
- HBM→Cache 冲突的编译期范围维护、write-back clean 的真实下游事务、
  issue/range gate，以及 fault 后显式 recovery。
- HBM binding、Global DMA、NoC、L2/L1 Bank、Local DMA、MSHR/Cache 和
  generation-aware cancel-confirm 路径。
- PMU、Perfetto trace、Profile/member ACK、Arena/view/R lease 与逐 Bank
  守恒计数；报告直接读取组件 snapshot，不从事件名称反推状态。

以上是 simulator 的可执行合同，不代表 RTL、PPA 或真实芯片规格已经冻结。

## 有意保留的架构与调度限制

- **单 Tile Group**：不模拟多 Group 之间的 NoC、全局内存或 Collective
  竞争。Collective 只有 1-cycle command/window 与 trace，reduce/
  broadcast datapath、带宽和数值未建模。
- **固定拓扑与映射**：当前目标是一个四 Tile slice。非空
  `nest.task.range` 数量必须等于 placement popcount，logical Task 仍按
  1:1 映射到选中 Tile；没有 task stealing、动态负载均衡或
  oversubscription。
- **有限且简单的源码控制流**：公开 Context/Tile Program 是有限单块、
  直线控制流；没有通用 CFG、循环、动态递归或跨 Context L2 SSA。
- **FIFO 头约束**：Root 和每 Tile 都只比较 SAME/COMPATIBLE 两类的 FIFO
  队首；SAME 队首受阻时可由 COMPATIBLE 队首补位，但不会越过同类队首
  做小对象装箱。没有用户优先级、aging、抢占或迁移已提交 Task。
- **静态 Arena 布局**：只在编译器能证明 view 生命周期不重叠时复用
  Slot/offset；L2 复用还需要 release 后有支配它的 Context barrier。
  Runtime 不压缩碎片、不搬迁 Arena，也不动态借用另一个 owner 的预留。
- **统一同层模式**：一个 Profile 覆盖该层的全部成员；不支持 per-bank、
  per-Context 私有 Cache 分区或运行时试档。L1 与 L2 可独立切换，但实际
  组合必须在编译产物中预先验证。
- **共享 Cache hint 不是配额**：资源合同的 `target_bytes` 与 Gather 的
  `cache_target_bytes` 不保留私有容量，也不相加扣 Arena；Cache 容量只由
  active Profile 决定。
- **Profile 物理值未冻结**：schema 2 默认 mode、reserved bytes、
  mapping/cache organization、timeout 是 `simulator_experiment` 建模选择，
  `由后续规格冻结`。模型不声称 SRAM macro、Tag/ECC 面积、编码或命令总线
  对应真实器件。
- **显式入口/恢复**：warm run 当前模式必须等于 compiled entry Profile。
  Simulator 不自动 reset、切回 mode 或重新规划；调用者必须在静止且已
  隔离后执行显式 recovery。
- **有限控制资源是合同的一部分**：Event Table 使用编译证明的 live
  frontier，不再按所有历史事件计数，但仍受静态容量限制。Grid Route、
  pending Root、action/inflight/DMA/ACK/Frame Slot 等也都是有限资源；
  过大的单产物会被编译器或 Loader 拒绝。

## 时序、引擎与互连简化

- `full_memory` 才按独立 stage 模拟 HBM outstanding/channel、Global DMA、
  NoC VC credit、L2/L1 Bank segment 与 Local DMA；同 Bank 串行、不同 Bank
  可以重叠。
- `timing_only`/`runtime` 把普通 transfer 折叠为单腿带宽+launch 延迟，
  因而不提供逐 Bank 时序结论；但二者仍执行资源合同、Profile/维护命令、
  gate、generation 和静态预算检查。
- BOA/EVU/USE 是 Roofline/launch 时序模型，单 job 非流水；MFE 有可配置
  load/store lane，每 lane 内串行，未建模更多真实端口/仲裁细节。
- Stream Queue 模拟 credit/backpressure/EOS，但不是 RTL FIFO/CDC 对拍。
- `clock_mhz` 与 CPU `issue_width` 都是模拟参数，不是实测 Fmax 或 CPU IPC。
  真实 CPU/FPGA 异步时钟、PCIe/AXI/doorbell 物理链路、CDC、综合、
  P&R、功耗与热设计均不在模型内。
- Program residency/cold-warm 只建模 program ID/version/hash/epoch 与
  fetch/install metadata。可选 `same_program` epoch 只限制 dispatch，
  不冻结无关 DMA。

## Trace 的初始化可观测性限制

- 默认通过 `Simulator.run` 导出的 workload trace 不包含 pre-run Profile
  初始化过程，不是从设备初始化开始的完整生命周期时间轴。初始化在
  workload cycle 0 之前完成，其耗时也不计入 report 的 workload cycles。
- 底层 Tracer 仍会生成初始化事件，但 `Simulator.run` 在初始化完成、
  workload 启动前调用 `discard_profile_initialization()`，移除
  `profile_initialize`、`profile_initialized`，以及
  `stage == "INITIALIZE"` 的 member request/ACK。因此正常导出的
  JSON/HTML trace 默认看不到这些初始化事件。
- 该处理是**省略初始化阶段，而不是修正初始化事件的时间戳**。过滤条件
  是事件类型和 stage，不判断事件是否 overlap；运行中的
  `profile_command`、`profile_step` 和 `PREPARE`/`COMMIT` member
  事件不会因 overlap 被删除。
- Cycle-0 容量基线、复用 Tracer 时的历史 workload 记录及持久 HBM
  binding/counter 均保留；不清空整个 Tracer，也不平移 workload 时间戳，
  以保持 trace 时间与 report、`accepted_cycle`/`completion_cycle` 对应。
- 这一范围适合分析 workload 执行，但**不能从当前导出的 trace 中检查
  初始 Profile 设置的耗时及其与第一笔 DMA 的完整先后关系**。初始化
  overlap 消失只说明初始化事件已被排除，不是完整启动时间轴已正确
  呈现的证明；实际先初始化、再启动 workload 的顺序由执行代码保证。
- 当前没有独立 startup trace、保留初始化事件的导出开关或独立 epoch
  导出机制。若需完整生命周期可观测性，必须显式补充这些能力，并明确
  初始化时钟与 workload cycle 的对应关系，不能仅靠隐藏事件代替。

## 数值与 Cache 证明边界

- 默认模拟不执行 BOA/EVU/USE tensor 算术；这些 op 的 bytes/ops 只驱动
  时序与资源模型。
- Gather 仍由源码 `tile.profiled.access` 显式给出
  `L1_HIT`/`L2_HIT`/`HBM_MISS`。Cache 容量不会推导命中率，也不会把 hit
  改写成 miss；`line`/`merge` 在非 oracle 模式是 opaque profile identity。
- 没有真实 index tensor 求址、通用 `AddressProvider`、Scatter、
  ScatterReduce 或 FIFO delivery；因此默认 Gather 统计不是实测
  address-accurate/value-accurate hit rate。
- 可选 `ByteStore` 只在 `full_memory` 下提供稀疏、逐字节有效性 oracle：
  host seed 必须落在实际 binding，未初始化读取报错，read leg 完成时捕获
  bytes，write leg 完成时才提交目的。它能证明真实 copy 路径和显式绑定
  source range 的 Gather/Cache 维护结果，但不会把 BOA/EVU/USE 升级成
  tensor 数值模拟。
- 非 oracle Gather 的 Cache provenance 是 allocation-qualified 的保守
  whole-view 范围；oracle Gather 必须逐调用
  `bind_profiled_source(binding_id, request_id, source_offset)`，否则拒绝字节
  检查。Cache test seed 必须满足 line 对齐、active capacity 与 write
  policy。

## Fidelity 边界

- `timing_only`：无物理 L1/L2/HBM allocation handle，普通 transfer 单腿；
  仍验证源码、artifact、binding、Profile 与控制合同。
- `runtime`：真实 HBM binding、L1/L2 Arena/view、owner/generation/pin/
  capacity/lifetime；普通 transfer 仍是折叠单腿。
- `full_memory`：在 `runtime` 上增加逐腿 route、Bank/NoC/DMA/HBM 竞争；
  只有注入 `ByteStore` 后才能声明真实 byte copy 可见性。
- 三种 fidelity 都不是 RTL cycle equivalence。跨 fidelity 的周期差异包含
  抽象层级差异，不能解释成硬件性能误差。

## 源 IR 与地址表达限制

- `nest.subview` / `tile.subview` 的 `strides` 必须全 1；不支持 inline
  `%Y[...]` 下标糖或 view chain。
- `nest.context` 形参仅允许零个或多个 `!nest.global_memref`，后接零个或
  多个只读 `!nest.l2_buffer` 导入；submit actual 必须保持同样顺序，且
  每项 shape 和 dtype 都必须精确匹配。
- `nexus.program` 外部输入仍只能是 HBM global memref；导入的 L2 形参只
  能通过 context submit 传给 tile program。
- dispatch 的 L2 actual 必须是完整的本地 `nest.alloc` 或只读导入形参；
  subview 仍要求 unit strides，且只支持一个 `task_dim`。
- transfer 字节数从 shape/dtype/view 推导，不接受独立 `bytes=N` 覆盖。
- 物理 transfer 只接受连续 row-major view；非连续切片在验证阶段拒绝。
- 输入 binding 按名字匹配，没有按位置 fallback。不同外部 binding 的
  IOVA 区间不能重叠；实际地址可合法 relocation，但不能违反编译时范围、
  权限和 alias guard。
- 公开 phase 聚合只支持 `#nest.aggregate<all_tasks>`，没有 quorum/subset。
  普通 `nexus.await`/`nest.await` 保持局部事件语义，不是隐式全局静默。

- 同一 `nest.context` 执行实例内的多个 Tile task 可通过显式声明的
  `sharing="context-local"` allocation 读写共享 L2 backing；仍由所属 context
  显式 release，不提供锁、原子性或重叠写入的线程安全保证，也不能跨 context
  导出。跨 context 的 L2 共享只能用 `sharing="readonly"`：完整初始化后
  恰好执行一次 producer `nest.publish`，共享引用锚定 producer submit
  实例与 export slot；reader 必须在同一 Group/L2 Profile epoch。
  发布后不可变，consumer 形参只读，每个导入恰好 release 一次。
  不支持动态 Gather Cache 重设计、跨 context 可写 alias 或跨 Profile 共享 backing。

## 持久化与复放限制

- 编译 artifact schema 当前为 2、compiler ABI 为 `v2`；旧 schema 1 / ABI
  `v0`、`v1` 产物被拒绝，必须从 source 重编译。硬件 YAML schema 为 2，
  二者不是同一版本号。
- `.target.yaml` 只保存完整 `HardwareConfig`，不保存 `SimConfig`。独立
  `--compiled-file` 复放必须另行提供与编译时相同的
  `context_count`/`device_context_count`、Device pending/completion 以及
  Group active/pending/action/quota/scan/event/inflight/prefetch/store/
  dispatch 静态容量。
- fidelity、trace、max-cycles、seed、时序 knob 与 scheduler policy 不进入
  static target hash；更改它们可能改变观测周期，但不能改变已编译资源/
  Profile 语义。
- trace counter 是 change-only，不是每 cycle 采样；collapsed-leg fidelity
  的 flow 只有 `s`+`f`。报告中的 Profile/Arena/lease 结论来自 bounded
  snapshot；workload 的编译控制序列与执行历史应结合 executable dump
  和 Perfetto trace 查看，但不包含上述被省略的 pre-run Profile 初始化事件。
