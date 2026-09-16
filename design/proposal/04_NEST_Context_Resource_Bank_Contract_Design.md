# NEST SRAM Profile、资源合同与实施计划

> 状态：待评审设计，**尚未实现**。本文只规定目标行为、编译器与 validator 的职责及后续实施要求，不代表现有 parser、Runtime 或硬件已支持新增功能。
> 依据：[NEST_SRAM_Bank_Runtime_IR_Proposal_v0.1.md](../../NEST_SRAM_Bank_Runtime_IR_Proposal_v0.1.md)、当前 `pipeline_validator` 源码及用户 review。**L2 mode 由 `nest.context` 指定，L1 mode 由 Task 启动操作指定；await 保持现有事件等待写法。编译器生成显式等待和配置执行序列，validator 只验证并执行，不现场补依赖或配置步骤。**
> 软件接口为 contract v0；物理编码、目标维护能力、粒度及控制时序由后续规格冻结。

## 1. 定位、目标和 First Silicon cutline

### 1.1 当前设计

| 议题 | 决策 |
|---|---|
| 模式定义 | L1/L2 各自有 N 个字节配置 mode；每项明确 SPM/Cache bytes，不使用百分比输入。 |
| L2 选择 | `nest.context.resource_contract` 声明 `l2_mode`、本层 `allowed_profiles` 和 L2 Arena 需求。一个根 Context 的 L2 mode 在其生命周期内固定。 |
| L1 选择 | Task 启动操作 `nest.dispatch.tasks.async` 显式携带 `l1_mode`；被调用 `tile.program` 的资源合同声明 L1 `allowed_profiles` 和每 Task 资源需求。 |
| 等待写法 | 保持 `nexus.await %context_event`、`nest.await %grid_event`、`tile.await %engine_event`，不要求用户在 await 上填写配置域、完成等级或切换授权。 |
| Context 内切换 | `nest.context` 内可以用 `nest.await` 等待旧 Grid，为后续 Task 的 **L1-only** 切换准备；配置未变的父 L2 Arena 可继续存活。 |
| L2 切换 | 在 Device 控制流等待相关根 Context 退休后进行。仍持有旧 L2 Arena 的根不能在自己体内改变 L2 mode。 |
| 完整切换等待 | 普通 await 等指定事件；编译器另输出有序的 Profile 配置事务，执行全域排空、Cache 维护及 Commit ACK。事件完成与可安全改档不是同一条件。 |
| 配置范围 | 同层 Domain 的所有 Bank 始终使用同一 mode；所有 Tile 的 L1 同属配置域。Task 的 mode 不是私有硬件分区。 |
| 补位 | 安全、依赖、资源和基础优先级先行；同等条件优先 SAME，其次 COMPATIBLE，同类 FIFO。Context 比较 L2，Task 比较 L1。 |
| 软件边界 | 编译器产生完整、可检查的执行产物；validator 不隐式运行依赖推导、Profile 规划或程序身份生成。保留只读验证、机械解码和真实硬件调度。 |

**本计划采用保留 await 原写法的方案。** 不将普通 await 偷偷升级为全设备 Barrier，也不把它当作只要存在就能切换的标记。完整安全性由“编译器插入的事件等待 + 编译器生成的完整配置事务 + Runtime 的真实计数/ACK检查”共同保证。

### 1.2 范围与约束

- 保留 ready-action、Tile eligible-head、SPMD task 身份、placement、显式 UCE pin、`bindings/ins/outs` 和正常异步事件模型。
- 不增加自动 Profile/Tiling 优化、live SRAM migration、抢占恢复或跨 Context 私有 Cache 配额。
- 不因源码里出现不同 mode，就由 validator 自动加一条等待、补一条数据依赖或现场生成配置事务。
- `nest.task.range` 只定义逻辑任务索引，不是启动操作；本文的 **Task op 指 `nest.dispatch.tasks.async`**。
- 同一层统一配置不等于统一访问类型：各 Bank 可同时服务其 SPM/Cache Region 中的不同请求。不同 Tile 的 L1 容量不能合并给某一个 Tile 使用。
- SPM 使用 Context/Task-lifetime Arena 硬预留；局部 Buffer release 不向其他 Context 提前返还 Arena 配额。

## 2. 职责、非职责和 ownership

### 2.1 对象与配置责任

| 对象 | 拥有的内容 | 不拥有的内容 |
|---|---|---|
| L2 Profile Domain | 目标定义的全部相关 L2 Bank 的统一 mode/generation | 各 Context 的独立 mode 寄存器 |
| L1 Profile Domain | 全部 Tile L1 Bank 的统一 mode/generation | 每个 Task 单独切自己所在 Tile 的配置 |
| Allocation Pool | 每 Group 的 L2 池、每 Tile 的 L1 池及独立资源账本 | 跨 Tile 借用容量 |
| `nest.context` | L2 mode、L2 允许集合、L2 Arena、子任务并发上限 | 固定整个 Context 的 L1 mode |
| `nest.dispatch.tasks.async` | 本次 Task/Grid 的 L1 基准请求 `l1_mode`、程序引用和依赖 | L2 mode 覆盖、配置寄存器写权限 |
| `tile.program` | 固定代码/布局、L1 允许集合、L1 Arena/控制预算、跨层访问能力要求 | 第二个独立的默认 L1 mode 选择入口 |

L1/L2 选择独立，但**一个具体调用的跨层组合仍需验证**。例如 Task 要求 L2 Cache 路径，而父 Context 的已绑定 L2 mode 无 Cache 且无合法 Bypass，则该调用非法。Task 不通过携带另一个 `l2_mode` 来覆盖父合同。

### 2.2 编译器、Loader 与 Runtime

| 组件 | 必须负责 | 禁止承担 |
|---|---|---|
| Registry builder | 字节模式、目标几何/系统预留、组织/能力、ABI/hash | 执行中改变模式表 |
| 编译器 | 资源布局、依赖与别名分析、分层 Profile 绑定、await 插入、配置序列、代码/descriptor 生成及身份 | 运行时物理 Slot/地址选择 |
| validator Loader | 解析已编译产物，检查版本、容量、绑定、事件来源、控制流与配置序列完整性 | 修补非法输入、补依赖、重排程序或重新编译 |
| Device 控制器 | 执行 submit/await/return 和已编译 Device 配置命令 | 根据尚未显式等待的未来 Context 偷改 L2 |
| Group 控制器 | 执行 Context 内普通 await、已编译 L1 配置命令及 ready-action | 将本地事件等待解释成跨 owner 的隐式全局等待 |
| Group/Tile Admission | 当前容量、Slot、lease、pin、原子 plan/commit/rollback、亲和性补位 | 从程序体重新求静态资源需求 |
| Profile controller / Backend | 唯一配置写入权、目标固定的全域排空/维护/ACK、代数与故障隔离 | 发现 mode 不同后自行插入程序步骤 |

普通 await 不持有配置锁。**配置事务**在其执行点按固定 Domain 顺序获取控制权；等待期间不能持有阻止旧请求返回或资源释放的 allocator 锁。两个控制层提交到同一配置写入者，不各建一套寄存器所有权。

## 3. 微架构和状态机

### 3.1 从源 IR 到执行产物

```text
Source MLIR: Context(L2) -> Task(L1), ordinary await/events
                         |
                 Explicit compiler invocation
                         |
  Dependency analysis / layout / mode binding / await insertion
                         |
  CompiledProgram: Device + Group + Tile instructions / descriptors
                         |
           validator load + read-only verification
                         |
 Device PC -> Group ready-action -> Tile admission / execution
                         |
    Profile controller / memory / completion / fault handling
```

编译产物必须可 dump：能看到每条 WAIT、配置事务和后继 dispatch，以及它们对应的源 IR 位置。配置事务不能只存在于注释或 validator 的运行时推导分支里。

### 3.2 普通 await 与配置事务的关系

```text
source nest.await %g0
          |
compiled WAIT_EVENT g0          # 普通事件等待，原语义
          |
compiled PROFILE_RECONFIG L1    # 仅编译器认定确有切换时生成
          |
complete old L1 work / close L1 issue / maintain / drain
          |
Prepare -> Ready ACK -> Commit -> Commit ACK
          |
compiled DISPATCH task_b        # ACK 前不得注册/准入到新配置
```

- `await` 只等待它列出的成功事件，并阻止本控制流越过该等待点；其他旧工作与完成路径继续推进。
- `PROFILE_RECONFIG` 是**编译器产生的低层控制操作**。它执行目标已经定义的完整配置协议，并在完成前保持后继发射门关闭；Runtime 不因一个 mode 字段不同而创建这条操作。
- `WAIT_EVENT` 完成不等于硬件已静默。旧 Cache miss/refill、写缓冲、在途 DMA/NoC 返回和维护 ACK 仍由配置事务检查、等待。
- 普通等待后没有 Profile 变化时，不生成配置事务，不清 Cache、不增 generation；数据可见性需要的维护操作另由编译器显式生成。
- 缺少前置 await、漏等旧工作或缺配置事务的可执行产物直接拒绝。不能以“当前恰好空闲”让非法产物通过。

### 3.3 两级队列与补位

```text
REGISTERED -> WAIT_DEPENDENCY -> READY_ADMISSION
                       |
       WAIT_CAPACITY / WAIT_FRAGMENTATION / WAIT_SLOT /
             WAIT_CONTROL_RESOURCE / WAIT_CONTEXT_LIMIT
                       |
              PENDING_ACTIVATION -> ACTIVE -> RETIRING -> DONE
```

未 Commit 的候选只占有界登记/Route 元数据；成功 Ticket 的 Arena、Slot 和控制资源必须同时计账。配置控制命令是注册与发射屏障；完成处理、释放和错误收敛不能被候选表满或未来提交堵塞。

亲和性按层比较，且先通过全部依赖、资源、配置入口及权限检查：

| 层 | 基准 | 合法允许集合 | SAME / COMPATIBLE |
|---|---|---|---|
| Group Context 补位 | Context 的 `l2_mode` | Context 的 L2 `allowed_profiles`，并满足该调用的子任务计划 | 当前 L2 等于基准为 SAME；不同但已验证可运行为 COMPATIBLE。 |
| Tile Task 补位 | 启动 op 的 `l1_mode` | 被调用 Tile Program 的 L1 `allowed_profiles`，结合该调用的跨层检查 | 当前 L1 等于请求为 SAME；不同但可在当前 L1 执行为 COMPATIBLE。 |

同等基础优先级下 SAME 优于 COMPATIBLE，同类按 ready 序号 FIFO。只对可比较桶队首做无副作用 plan；SAME 队首不能完整准入而 COMPATIBLE 可以时，允许后者补位。同桶不越过资源阻塞队首挑更小请求。Profile 优先级不越过数据依赖、R 上限、配置命令或 await。

集合外请求不能作为普通容量等待无限挂起：编译器必须给出先等待再切换的路径。Runtime 不改变已经选择的程序、Tiling、布局和允许集合。无限流式提交的公平性需要独立的 aging/配额设计；当前只保证有限合法工作在既定公平服务前提下可结束。

## 4. 接口、descriptor、寄存器和协议

新增资源属性与执行产物接口均为待实现草案。**本节 await 的写法沿用当前语法，不增加属性或新操作数种类。**

### 4.1 显式字节模式

每个字段以**单 Bank**为单位。L1/L2 模式表独立；ID 是本层索引，不可按大小比较容量或跨层混用。

```yaml
memory:
  target:
    # 示例值，不是当前代码默认值或已冻结硬件规格。
    l1:
      system_reserved_spm_per_bank: 2048
    l2:
      system_reserved_spm_per_bank: 4096
  profile_source:
    kind: simulator_experiment
    partition_granule_bytes: 64
    l1:
      modes:
        0: { spm_bytes_per_bank: 65536, cache_bytes_per_bank: 0 }
        1: { spm_bytes_per_bank: 57344, cache_bytes_per_bank: 8192 }
        2: { spm_bytes_per_bank: 49152, cache_bytes_per_bank: 16384 }
    l2:
      modes:
        0: { spm_bytes_per_bank: 524288, cache_bytes_per_bank: 0 }
        1: { spm_bytes_per_bank: 458752, cache_bytes_per_bank: 65536 }
        2: { spm_bytes_per_bank: 393216, cache_bytes_per_bank: 131072 }
```

- 每层 N 由该层表长决定。两个 bytes 字段必须显式提供，非负、符合组织/粒度，并满足 `SPM + Cache = bank_bytes`；不自动补余量或取整。
- `system_reserved_spm_per_bank` 来自该层不可变目标描述，缺失不能默认为 0；SPM 字段包含它，用户可用量另扣除。目标 geometry、预留、映射/Cache 组织和能力都纳入 Registry hash。
- 配置的百分比只作为由 bytes 计算的只读展示值，不是输入。同 ID 改 bytes、目标预留或能力后，必须重新编译/验证产物，不能提交时重新查可变表改变语义。
- 同层按完整有效配置检测别名：不同名称/ID但 bytes、geometry、映射/组织/能力相同的重复项拒绝；不同层不做跨层去重。
- simulator 实验目标可以验证显式字节档位；真实设备只接受其 Registry 已实现的合法组织。字节守恒不等于硬件可实现。
- Pool 容量由 Bank 数乘出；L1 是单 Tile 容量，不能乘 Tile 数给一个 Tile 准入。Cache 数据、Tag/ECC/MSHR/控制存储的核算范围必须明确。

### 4.2 L2 Context 合同

```mlir
resource_contract = #nest.context_resources<
  l2_mode = 0,
  allowed_profiles = [0, 2],
  logical_tasks = 3,
  l2_spm_bytes = 4096,
  requested_contexts_per_tile = 1>
```

- `l2_mode` 是该根 Context 的 L2 基准请求，必须在本合同的 `allowed_profiles` 中；列表中的 ID **全部属于 L2**。
- L2 允许集合必须显式、非空，每个条目都满足 Context 的 L2 布局/访问能力及相关编译调用计划。Runtime 不补基准、不删非法项、不扩集合。
- Context 不声明 L1 mode。其 Task 可以在同一个父 Context 内按显式等待顺序使用不同 L1 mode；父 Context 的 L2 地址、数据和 generation 在 L1-only 切换中保持有效。
- `logical_tasks` 与所有 dispatch 的任务数对照；`l2_spm_bytes` 是显式布局预留量，包含 padding/在途存储，不是共享 Cache 配额。
- `requested_contexts_per_tile = R` 是此父调用在单 Tile 上全部 Grid 合计的已提交 Task lease 上限；不是 L1 模式选择。

### 4.3 L1 Tile Program 合同与 Task op

```mlir
// 挂在 tile.program @task_a 上的静态资源合同：
resource_contract = #tile.resources<
  allowed_profiles = [0, 2],
  tile_l1_spm_bytes_per_context = 4096>

// 本次启动的基准 L1 请求由 Task op 明确携带：
%g, %r, %w = nest.dispatch.tasks.async @task_a
  l1_mode = 0
  tasks(%tasks) globals() bindings() ins() outs()
  signal_policy {}
  : (!nest.event<"g">, !nest.event<"">, !nest.event<"">)
```

`nest.dispatch.tasks.async` 是当前真实任务启动 op；`nest.task.range` 只产生任务范围，不携带配置选择。`l1_mode` 与现有 `context = N` 独立：前者请求 L1 Profile，后者仅 pin 一个 UCE context Slot。

Tile Program 的 `allowed_profiles` 中 ID **全部属于 L1**；Task 的基准 `l1_mode` 必须在该列表中。静态程序合同给出固定代码/布局/资源需求，具体调用给出基准请求，不在程序头再提供另一个默认选择。允许集合是本层可执行能力，不是要求按列表顺序试档。

编译器可以在已验证允许范围内复用当前 L1 mode，并把该调用的实际绑定写入执行产物。例如 Task 基准为 0、程序允许 `[0,2]`、当前 L1 为 2，则可按 COMPATIBLE 执行；只允许 `[2]` 的 Task 在当前 L1 为 0 时必须先等待并切换。Trace 记录基准与实际 mode。

Task 不携带 L2 selector。程序若有 L2 Cache、Bypass、mapping 或一致性要求，以**访问能力约束**声明，并在父 Context 的实际 L2 绑定下验证。两层允许集合不自动意味着所有交叉组合都合法；编译器验证实际调用及允许复用的组合，并将结果冻结到调用绑定。Runtime 不从两个列表临时推导跨层兼容性。

### 4.4 资源与生命周期约束

`arena_v0` 分层硬预留：根 Context 持有 L2 Arena，Task 持有本 Tile 的 L1 Arena。V1 使用有明确 layout/alignment 的 striped Arena，逐 Bank 长度包含 padding；不允许运行中增长。

设某父调用的 Task 并发上限为 R，许可子程序在对应 L1 Profile 下的最大 Arena 为 E：

```text
R <= physical Tile context slot count
R × E <= this Tile's user-usable L1 SPM
for every bank b:
    R × max(child Arena bytes[b]) <= user-usable L1 SPM[b]
```

还要检查 Slot、控制预算、连续跨度和 alignment。R=4却只有一个 Task 能放下的合同在 prepare 拒绝；不由 Runtime 偷降 R。L2 也逐 Bank 检查空池可行性，不要求所有根同时驻留。

Runtime 使用 per-`(parent_invocation_id, launch_generation, tile_id)` 的 Task lease 计数；覆盖全部 Grid 的 pending-activation、active、retiring Task。Commit 扣除，安全 retire/cancel 确认才归还；失败事务整体回滚，`input_released`、`tile.free` 或取消请求本身不归还名额。

`tile.free/nest.release` 失效逻辑 Buffer view，可在同 Arena 的已验证布局内复用；Arena 配额在其 owner 退休后才返还 Pool。完整调用闭包、真实最后访问、pin、generation 和 SlotFrame 检查均保留。

L1-only 切换要求旧 L1 用户 Arena 全部退休，但**不要求父 L2 Arena 或不涉及旧 L1 访问的 L2 生命周期 pin 归零**。禁止后续 Task 继续使用旧 L1 handle；L2 数据交接必须使用稳定的 L2 view。改变 L2 时则必须先让相关根退出，不能保留旧 L2 Arena 后原地改边界。

### 4.5 普通 await：保留什么、不能保证什么

保留原写法和普通事件等待语义：

```mlir
nexus.await %context_done
nest.await %grid_done
nest.await %grid_a, %grid_b
tile.await %load_done
```

| 写法 | 等待内容 | 单靠它不能推出 |
|---|---|---|
| `nexus.await` | 指定根 Context 的成功完成/退休，阻塞后续 Device 提交 | 所有无关根均已结束、Cache/Fabric 已完成全域维护 |
| `nest.await` | 指定本 Context 内事件；保留注册 fence，阻塞本 Context 后续动作登记 | 其他根的 Grid 已结束、全层 L1 已静默 |
| `tile.await` | 当前 UCE context 的指定引擎事件 | 本 Task 已退休、其他 UCE context 已退出 |

**采用原写法的理由**：程序已经通过 Context/Task 声明目标层和 mode，编译器已掌握布局、依赖和 target topology，无需让每条 await 再携带一份维护合同。常规算子等待也不应因引入 Profile 而全部变成全域同步。

**成立条件**：不能将 `await %g` 完成等同于可直接写配置寄存器。编译器必须另外生成完整配置执行序列，Runtime 必须执行其真实排空/维护/ACK协议。少了这些步骤的产物拒绝，不靠加强每个 await 的隐式含义兜底。

用于 L1 切换准备的事件至少覆盖旧 `grid_done`；不能只等 `input_released/output_ready`，因为 Task 可能仍计算或持有 L1。其他尚未结束、会触碰目标 Domain 的 DMA/引擎动作也必须纳入编译器的等待或配置收敛分析。

### 4.6 Context 内 L1 切换与跨 Context 安全

在同一 `nest.context` 内，合法顺序是：

```text
Task A (L1 mode 0)
  -> nest.await A.grid_done
  -> compiler-emitted L1 configuration transaction
  -> Task B (L1 mode 2)
```

父 Context 的 L2 mode 不变，L2 Arena 不需要为了 L1 切换而释放，因此不会出现“等待自己持有的 L2 Arena 归零”的死锁。L2-only DMA 若不再可能向旧 L1 发请求，可以按已编译依赖继续推进；由旧 L1 维护产生的 L2/HBM 事务仍需等待。

但同层所有 Tile 的 L1 统一配置，**只等自己的 Grid 不覆盖其他根 Context 的 L1 工作**。当前 Group 事件有 owner/generation 隔离，不能在某个根的 `nest.await` 里直接等待另一个根的事件。

V1 采取保守、可验证的编译规则：执行含真实 L1 改档点的根时，Device 控制流先用 `nexus.await` 排空其他能产生 L1 请求的根，并在该根结束前不提交新的 L1 生产者。纯 L2 工作只有在编译器证明不访问该 L1 Domain时才可重叠。执行产物记录此排他要求，Runtime 检查而不自行寻找“差不多空闲”的时机。

这不要求所有 Context 永远串行：不改变 L1 mode、或全部 Task 可在当前配置下兼容运行的多个根仍可重叠并补位。更精细的跨根同步需另行证明；本计划不增加跨 owner 的隐式 Group Barrier。

若父 Context 后半段还需要不同 L2 mode，编译器必须将工作拆成不同根、将仍需保存的数据通过合法后备路径交接，并在 Device 层插入 `nexus.await`。Context 内的 L1 准备等待不授权修改仍被自身占用的 L2。

### 4.7 Review 用 MLIR 示例

下面是**目标语法草案**：新增分层资源属性和 Task 的 `l1_mode` 尚未实现，await 写法本身不变。例子只验证资源租约和配置顺序，不代表算术或 Cache 数值结果。

目标初始配置已经由启动协议验证为 L2.mode0、L1.mode0。每个 Arena 为16 Bank×256 B=4096 B。`ctx_tasks` 的 L2 Arena 在其三次 Task 启动及内部 L1 切换期间保持存活；它是此时唯一可以投递 L1 工作的根。

```mlir
builtin.module {
  tile.program @task_a (%task : !nest.task)
    resource_contract = #tile.resources<
      allowed_profiles = [0, 2],
      tile_l1_spm_bytes_per_context = 4096>
  {
    %tmp = tile.alloc shape = [4096] dtype = "i8" alignment = 256
      : !tile.l1_buffer<4096xi8>
    tile.free %tmp
    tile.return
  }

  tile.program @task_b (%task : !nest.task)
    resource_contract = #tile.resources<
      allowed_profiles = [2],
      tile_l1_spm_bytes_per_context = 4096>
  {
    %tmp = tile.alloc shape = [4096] dtype = "i8" alignment = 256
      : !tile.l1_buffer<4096xi8>
    tile.free %tmp
    tile.return
  }

  nest.context @ctx_tasks placement = 1
    resource_contract = #nest.context_resources<
      l2_mode = 0,
      allowed_profiles = [0, 2],
      logical_tasks = 3,
      l2_spm_bytes = 4096,
      requested_contexts_per_tile = 1>
  {
    %keep = nest.alloc slot = "keep" role = "in"
      shape = [4096] dtype = "i8" alignment = 256
      : !nest.l2_buffer<4096xi8>
    %tasks = nest.task.range from = 0 to = 1 : !nest.task_range

    %g0, %r0, %w0 = nest.dispatch.tasks.async @task_a
      l1_mode = 0
      tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
      : (!nest.event<"g0">, !nest.event<"">, !nest.event<"">)

    // 编译器插入的普通等待，等待旧 Grid 完成；父 L2 Arena 保留。
    nest.await %g0

    // 编译器在执行产物的此处生成 L1 配置事务，完整排空并提交 mode2。
    %g1, %r1, %w1 = nest.dispatch.tasks.async @task_b
      l1_mode = 2
      tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
      : (!nest.event<"g1">, !nest.event<"">, !nest.event<"">)

    // task_a 请求基准0，但允许当前L1.mode2，故兼容运行，不切回0。
    %g2, %r2, %w2 = nest.dispatch.tasks.async @task_a
      l1_mode = 0
      tasks(%tasks) globals() bindings() ins() outs() signal_policy {}
      : (!nest.event<"g2">, !nest.event<"">, !nest.event<"">)

    nest.await %g1, %g2
    nest.release %keep
    nest.return
  }

  nest.context @ctx_l2_2 placement = 1
    resource_contract = #nest.context_resources<
      l2_mode = 2,
      allowed_profiles = [2],
      logical_tasks = 0,
      l2_spm_bytes = 4096,
      requested_contexts_per_tile = 1>
  {
    %tmp = nest.alloc slot = "tmp" role = "in"
      shape = [4096] dtype = "i8" alignment = 256
      : !nest.l2_buffer<4096xi8>
    nest.release %tmp
    nest.return
  }

  nexus.program @review {
    %c0 = nexus.submit_context.async @ctx_tasks : !nexus.event<"c0">

    // Device层等待整个根退休，才可准备L2切换。
    nexus.await %c0

    // 编译器在执行产物的此处生成L2配置事务；L1保持mode2。
    %c1 = nexus.submit_context.async @ctx_l2_2 : !nexus.event<"c1">
    nexus.await %c1
    nexus.return
  }
}
```

程序没有 L2 读写 Task，故 dispatch 的 `bindings/ins/outs` 为空，只有 `grid_done` 使用有效事件名；两个空标签结果不参与任何 await。L2 Buffer只是保留的只读槽位，不读未初始化数据，也不声明没有实际writer/Store的输出角色。`logical_tasks = 3` 对应三次各一个逻辑 Task。R=1 使同一父调用同 Tile 的已提交 Task最多一个；`g2` 可登记，但必须等 `g1` 真正退休归还名额后准入，不需要为了兼容补位新增配置等待。

源例子中的配置注释对应下面**必须真实存在于编译产物**的执行序列，不授权 validator 现场生成命令：

```text
Group ctx_tasks:
  WAIT_EVENT(g0)
  PROFILE_RECONFIG(L1, target=mode2, source_task=g1)
    close L1 issue -> drain old L1 references -> cache maintenance
    -> downstream completion -> all-member Prepare/Commit/ACK
  DISPATCH(task_b, requested_l1=2, resolved_l1=2)
  DISPATCH(task_a, requested_l1=0, resolved_l1=2)   # no reconfiguration

Device review:
  WAIT_CONTEXT(c0)
  PROFILE_RECONFIG(L2, target=mode2, source_context=c1)
    drain affected upstream/downstream work -> maintenance -> Commit/ACK
  SUBMIT(ctx_l2_2, resolved_l2=2)
```

编译器 dump 必须给出完整序列、关联等待指令和源位置；运行时只消费该序列。`PROFILE_RECONFIG` 内部实现目标层的完整协议，不把额外属性附到用户 await 上。首次执行要求启动状态与编译入口假设一致；不能构造无事件的假 await。没有可验证 Reset/初始静止证明的入口拒绝，或由明确的系统初始化/恢复协议建立该状态，不能借初始化绕过运行中的等待要求。

### 4.8 接口与错误边界

- 编译器入口接收源 MLIR、目标 Registry、合法运行参数/alias 假设，输出不可变 `CompiledProgram` 与可读执行 IR dump。
- validator 的正式运行入口消费 `CompiledProgram`；Loader 不自动调用源码编译流程。开发工具可依次显式调用 compile 和 run，但必须产生可检查中间产物，不能把两步藏在 `Simulator.run` 中。
- 程序/合同/调用绑定包含源哈希、代码版本、Registry ABI/hash、布局、静态依赖、基准与实际 mode、配置入口条件。动态 launch ID、Slot 与物理地址仍在运行期实例化。
- 列表外 mode、跨层能力不符、入口配置不符、缺必需 await、缺配置事务、跨根 L1 冲突、旧代请求均是明确错误，不作为普通容量等待。
- 字节或允许集合改变后，必须重编译 Profile绑定、等待与配置序列。不能只换配置表继续执行旧产物。

## 5. 数据流、控制流和时序路径

### 5.1 Context 内 L1-only 切换

1. 编译器确认父 L2 绑定与数据生命周期稳定，等待旧 Grid 及其它会访问 L1 的已发请求；插入普通 `nest.await`。
2. 该 await 是本 Context 的注册 fence；旧任务/完成路径继续推进。正常事件完成后进入编译好的 L1 配置命令，而不是直接准入新 Task。
3. 配置命令确认设备级排他约束有效、所有旧 L1 Task/Route/Frame/Arena 已退休，不存在仍可能产生旧 L1 请求的其他根；否则报告计划/运行状态不一致，不自动重排其他根。
4. 关闭整个 L1 Domain 的新用户请求；收敛全部旧 L1 miss/refill/写缓冲/引擎/DMA/NoC 引用。L1 维护若产生 L2/HBM事务，必须等待其真实完成。
5. 全成员 Prepare/Commit/ACK 后更新 **L1** generation和用户容量，重建本层 free map/Cache视图。L2 mode、L2 generation与稳定父 Arena不变。
6. 才允许新 Task 的 L1 Admission/Frame bind/发射；分配和 pin 失败按既有原子事务回滚。

“完整”针对实际重配域及它的访问闭包，不等于要求无关层的全部用户存储都归零。旧 L1 handle不能逃逸到后继Task；不能只看 `tile.free` 或某一个 UCE Slot空闲就提交新配置。

### 5.2 Device 层 L2 切换

编译器将后继需要的数据输出到 HBM，并在 Device 侧用普通 `nexus.await` 等待全部相关根 `context_done`。所有受影响 L2 Arena 必须退休，所有可能向该 L2 发请求的 Tile/L1工作也纳入闭包。

随后执行已编译的 L2 配置事务：获取唯一控制权、关闭新用户请求、必要上游/下游维护、DMA/NoC/refill归零、全成员ACK，成功后只增加实际改变配置的 Domain generation。L1数值可以不变，但其旧请求和依赖不能遗漏。

在仍持有旧 L2 Arena 的 Context 内，仅等自己的 Grid不能授权 L2重配。需要这种变化的程序由编译器切成多个根，并在 Device IR明确交接和等待。

### 5.3 完成事件与资源退休

| 完成对象 | 保证 | 不保证 |
|---|---|---|
| `input_released` | 指定输入的消费者读取结束 | Task计算结束、L1 Arena已退、整个Grid完成 |
| `output_ready` | 指定层级输出可用 | 所有Task退休或Host/HBM可见 |
| `grid_done` | 全部预期Task正常结束，相关L1 frame/资源按协议退休 | 其他根的Grid完成、L2父Arena释放或全域Cache维护 |
| `context_done` | 根的子任务、必需输出与资源退休完成 | 整个共享Domain已经完成配置维护 |
| 配置命令完成 | 目标Domain访问闭包静默、维护及所有成员Commit ACK完成 | 后续另一次切换可省略await |

若代码只尝试释放而仍有 pin/release-pending，不能据事件完成位忽略残留账本；配置命令必须再次核对真实资源/在途状态。错误事件、超时或取消不能转成成功等待记录。

源读取结束不等于目的可见。HBM导出、范围失效/一致性要求由编译器明确生成；不能认为普通 DMA completion 自动清理所有共享 Cache旧副本。只有全域静默维护能力时，相应交接需要编译器排空相关工作并生成维护序列；无必要能力则拒绝，而不是靠一个 await名词兜底。

### 5.4 配置事务、锁与故障

每条配置命令是有完成边界的控制操作，不能被后继 dispatch越过；它不是向普通队列随意追加的异步写寄存器。必要控制存储和完成通道独立于用户SPM，不能等待自身即将回收的资源。

事件await和配置命令之间的寄存器/账本版本必须可检查。新用户活动若破坏编译前提，命令拒绝或fault；不能使用过去一个已完成await作为永久授权。相同Profile不重复Commit，但显式数据依赖等待照常执行。

提交配置中途失败保持门关闭，受控隔离/reset，不直接写回旧mode就假装原数据恢复。取消仅是请求：已发配置命令、DMA及迟到返回必须收敛/隔离确认后才能释放资源或开放新工作；不得因删除队列项而漏掉已开始的配置事务。

### 5.5 有界执行与进展

保留 CPU-before-Group、完成后收集、依赖次Tick可见等现有协同原则；普通 await 阻塞后继登记，不停止旧工作所需的引擎与完成推进。

等待图不能形成“父持L2等子、子需要另一个未准入根才能结束”或“await前生产者依赖await后提交”的环。编译器必须证明切换点可到达及相关工作有限；validator验证这些前提和实际资源账本，不尝试在线修复。

Group等待不能引用另一个根owner的事件。跨根排空使用Device事件和编译好的提交顺序；控制队列满、正常资源等待、取消和Completion均需保留推进路径。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 容量公式

```text
C = target per-pool configurable data capacity
B = bank count per pool
c = C / B
s = selected mode.spm_bytes_per_bank
k = selected mode.cache_bytes_per_bank
r = memory.target[level].system_reserved_spm_per_bank
q = lcm(partition_granule_bytes, cache_line_bytes)

C > 0, B > 0, C % B == 0
s >= 0, k >= 0, s + k == c
s % q == 0, k % q == 0, 0 <= r <= s

SPM data = B * s
shared Cache data = B * k
user SPM = B * (s - r)
```

所有算术检查溢出；合法bytes还需满足mapping/Cache组织。以下采用 §4.1 的示例，L1是单Tile、L2是单Group，单位B：

| 层级 / mode | 每 Bank SPM | 每 Bank Cache | Bank 数 | Pool SPM | Pool shared Cache | Pool 用户可用 SPM |
|---|---:|---:|---:|---:|---:|---:|
| L1.mode0 | 65536 | 0 | 16 | 1048576 | 0 | 1015808 |
| L1.mode1 | 57344 | 8192 | 16 | 917504 | 131072 | 884736 |
| L1.mode2 | 49152 | 16384 | 16 | 786432 | 262144 | 753664 |
| L2.mode0 | 524288 | 0 | 16 | 8388608 | 0 | 8323072 |
| L2.mode1 | 458752 | 65536 | 16 | 7340032 | 1048576 | 7274496 |
| L2.mode2 | 393216 | 131072 | 16 | 6291456 | 2097152 | 6225920 |

L1.mode1改为53248/12288 B后，单Tile用户SPM为 `16×(53248−2048)=819200 B`。49152+8192 B不守恒，65535+1 B不满足64 B粒度，均拒绝。模式比较不按ID大小，也不按占比近似猜合法。

在参考提案的测试geometry（每Tile 4×32 KiB、每Bank用户SPM22 KiB）中，R=4、每Task6 KiB/Bank需要24 KiB/Bank，空池仍不可能，加载即拒绝。一个Bank需要23 KiB、其余只需1 KiB，总量虽小也非法。

### 6.2 Cache 与性能边界

每层 Cache 默认 pool-shared，`cache.required/access/bypass` 是能力要求，`cache.target_bytes` 是工作集hint，不按Context相加扣容量。普通Context退出不全池清Cache；需要正确的地址空间/对象版本/line身份和失效协议。

当前 metadata Cache 的命中结果来自 IR profile。修改容量不能自动预测新的hit/miss或多Context干扰；有数值/命中率结论时需真实访问驱动模型或与配置匹配的测量profile。零Cache必须走目标支持的真实bypass路径，不能保留不可能的hit。

SPM/cache容量比例不等于端口带宽比例。L1-only切换保留L2存储可减少HBM搬运，但全层L1排空、Cache维护和编译器为跨根安全插入的序列化仍有开销；不能只测寄存器提交时间。

### 6.3 可观察输出

- 分别记录 Context 请求/实际L2 mode、Task 请求/实际L1 mode、各层Registry/hash/generation，不用一个混合Profile字段掩盖归属。
- 记录 Await指令及事件、配置命令开始/结束、等待旧工作/维护/Fabric/ACK的时间、与源op的映射。
- 同时报告Arena预留、live Buffer、系统预留、共享Cache和R lease；L1改档前后L2 handle/generation连续性应可核对。
- 编译产物哈希和指令/依赖清单应随报告保留，以证明哪些依赖、等待或配置来自编译器，而不是validator临时生成。

## 7. RTL/软件实现建议

### 7.1 编译器必须承担的工作（独立清单）

| ID | 编译器工作 | 必须输出/证明 |
|---|---|---|
| C01 | 解析目标Registry与分层合同 | Context的L2允许集合、Task/程序的L1允许集合、模式/组织/版本合法性 |
| C02 | 资源分析与布局 | 含padding/双缓冲/在途存储的逐Bank Arena、slot/event/Grid预算、R×L1空池检查；显式layout与静态bytes |
| C03 | 数据效应与alias分析 | L2/global RAW/WAR/WAW、输入输出里程碑、调用点alias假设和完整依赖；不把可安全并行的动作人为全序化 |
| C04 | Profile控制流分析 | 每次根调用的L2绑定、每次Task的L1绑定、L1入口/退出配置效应、跨层访问合法性；多次调用可有独立编译绑定 |
| C05 | await插入与跨根调度 | Context内L1准备等待、Device层L2/跨根等待、完整旧工作集合、切换点可达性；不能只检查相邻一条Task |
| C06 | 配置与可见性代码生成 | 必要HBM导出/维护、完整PROFILE_RECONFIG序列、阻止后继越过的控制顺序、目标能力约束；普通await语法不扩展 |
| C07 | 低层执行对象生成 | Role/dispatch索引、release消费者/生产者索引、engine/DMA descriptors、静态view表达式、显式等待/返回闭包 |
| C08 | 程序身份与不可变产物 | program_id/version/hash、合同/layout/Registry指纹、重定位与调用绑定；不留0值让模拟器补 |
| C09 | 输出与诊断 | 源IR、编译后可读执行IR、依赖/配置命令来源、编译错误及必要运行前guard；供独立validator消费 |

这些工作在独立编译入口执行，建议组织为单独的 `compiler/` 功能模块；可复用现有轻量IR/schema与验证函数，不为目录分层而无关搬动全仓文件。**编译器验证不等于validator可以再次编译**：同一个检查器可复用，但正式运行路径不能调用会修改程序含义的pass。

### 7.2 当前 validator 隐式 lowering 的具体边界

以下为本轮读取源码和运行现有fixture lowering得到的事实；这里只制定迁移，不修改源码。

| 当前位置 | 观察到的行为 | 计划归属与处理 |
|---|---|---|
| `ir_lowering.py::_build_action_dependencies`（328–422） | 从L2读写、global重叠区间及Gather访问生成RAW/WAR/WAW；为release、barrier/return补依赖闭包 | **移到编译器**。执行产物必须已有完整符号依赖；validator验证缺边/非法边，不补边。 |
| `ir_lowering.py::lower_model_ir`（103–111） | 对实参alias的submit重新运行依赖生成，原地修改共享task.actions | **移到编译器的调用点绑定**。当前生成器从已有依赖集合起步，重复分析会累积边；不能让一次调用的alias关系污染所有调用模板。 |
| `ir_lowering.py::_lower_context`（157–169、288–303）、`_register_role` | 根据dispatch访问关系索引读写者，为release生成ordinal列表，并建立role表 | **由编译器输出**。validator保留owner/生命周期检查及动态pin操作，不再扫描程序生成静态消费者集合。 |
| `_lower_program`、`_lower_engine_descriptor`、view/bytes构造 | 生成Tile指令、engine参数、Buffer/view描述和静态字节数 | 静态代码生成归编译器；机械解码、类型/shape一致性及溢出校验可以保留，不称为优化pass。 |
| `simulator.py::_assign_program_ids/_program_hash`（375–406） | 在运行入口给未设置身份的程序分配ID、计算hash | **移到编译/打包**。Loader验证产物身份而非补默认值；程序驻留/cold-warm状态仍属Runtime。 |
| `tile_group_sequencer.py::dispatch_program_key`（167–176） | program_id/hash缺失时用程序名CRC回退 | 产物必须显式提供身份；切换后缺失即拒绝，避免名称代替编译版本。 |
| `tile_group_sequencer.py::dependencies_for`（124–132） | 在登记时把WAIT参数和release事件再次并入依赖 | 编译产物规范化一次；若保留解码适配器，只验证两种表示一致，不用它修复缺失依赖。 |
| `NestAwaitOp/NexusAwaitOp`及普通WAIT/WAITALL翻译 | await操作数直接映射为等待指令 | **保留语义和机械解码**。`tile.await`多操作数映射WAITALL，不是自动插入新的终结等待。 |
| `workload_ir.py::verify_*` | 校验绑定、访问集、事件来源、view及释放/设备依赖合法性 | **保留只读验证**，共享检查工具可以重用；发现错误就拒绝，不改图。 |
| `tile_group.py::_prepare_context_launch` | 按launch/slot克隆实例、重定位事件/queue身份、绑定实参 | **保留Runtime实例化**。动态launch identity不是编译器静态资源分析，必须区分。 |
| Group/Tile Admission、scheduler、barrier_ready、事件表 | 实际容量竞争、Slot选择、registration fence、完成/退休、背压与故障 | **保留硬件行为**。不能为了移出lowering删掉这些执行语义。 |

具体例证：对当前 `examples/fixtures/pow_single_context.mlir` 执行现有 lowering，Store 的依赖包含 `ev_dma_pow_in0` 和 `ev_outready_pow0`；源Store只显式写了后者。无操作数的 `nest.return` 生成含5个前驱事件的完成依赖。Tile的三个单事件await仅生成三个WAIT；lower后program_id/hash仍为0，之后由Simulator填充。由此不能笼统地说所有lowering都是简单读取，也不能把WAITALL解码误删为隐式同步。

Host地址/别名是晚绑定事实时，编译器输出可检查假设或保守依赖；Loader用实际bindings验证。若不满足，拒绝或要求显式重新编译，不能在正式运行入口按新alias重写共享图。允许按调用点生成不同产物，但必须从不可变源模板生成并有独立身份。

### 7.3 正式运行入口与迁移清单

当前调用链是 `cli.load_workload_ir → Simulator.run/_run_model → lower_workload_ir/lower_model_ir → 动态程序身份填充 → Group加载`。因此仅移动一个helper的文件位置不足以完成职责拆分。

目标流程：

```text
compiler.compile(source_module, target, binding_assumptions)
    -> CompiledProgram + readable executable IR dump

validator.load(compiled_program, actual_bindings)
    -> read-only checks + launch relocation

Simulator.run(loaded_program)
    -> execute exactly the supplied action/instruction/configuration sequence
```

- 从 `Simulator.run/_run_model` 的正式路径移除高层编译pass；不保留默认“给源码就悄悄编译”的回退入口。
- 迁移CLI、workload builders、examples runner及直接调用lower函数的测试：构建/编译是显式前置步骤，Runtime输入统一为已编译产物。
- 可以保留用于读已完成MLIR的1:1解码器，但它不能发现hazard后加边、选择mode、插await、生成缺失配置事务或补程序ID。
- 公共 await 仍是普通事件操作；可执行产物的配置指令是真实、可dump的执行项，不是藏在await附加属性中的另一个调度器。
- 编译后产物按调用绑定冻结，源/module/contract/layout/Registry有稳定hash；Runtime实例只重定位明确允许的字段，不修改共享模板。
- 在提取现有静态编译工作时，先证明同输入的指令、依赖和结果等价；新增分层Profile、Arena与逐Tile准入的行为变化另建基线，不能混在一次“行为不变重构”声明中。

### 7.4 实施步骤与验收

| 步骤 | 主要范围 | 完成条件 |
|---|---|---|
| A：分层Schema | modes/target、Context L2合同、Tile L1合同、Task `l1_mode`、CompiledProgram schema | L2/L1字段各归其层；允许集合包含基准请求；跨层非法调用拒绝。 |
| B：提取编译流程 | 依赖/alias、role/release索引、静态descriptor和程序身份、显式编译入口 | 同输入编译输出可复核；Simulator正式路径不再合成依赖或填身份。 |
| C：Profile与等待编译 | 模式绑定、普通await插入、跨根L1排他分析、配置序列和HBM发布 | `nest.context`内可见L1准备await；所有实际改档在执行产物中有完整有序命令。 |
| D：Arena与准入 | allocator/L1 frame/L2池、Grid Route、分层亲和性与R lease | 保留L2 Arena的L1切换安全；等待/回滚零泄漏，各Tile独立补位。 |
| E：配置执行与Backend | 目标Domain门禁、maintenance、outstanding、ACK、generation、取消/reset | L1只更新本层配置；L2切换排空完整闭包；缺命令或错误ACK不得继续。 |
| F：入口与案例迁移 | CLI/builders/现有MLIR及测试fixture、事件/return约束 | 普通await语法不变，所有运行入口消费显式编译产物；无隐藏兼容编译路径。 |
| G：证据与文档 | 执行IR dump、trace/PMU、边界与异常案例、IR_SPEC和使用文档 | 能解释一次等待、一次改档以及每条依赖由谁生成；不以metadata成功代替数值正确。 |

A先固定接口；B提取静态编译职责，C在此基础上生成新控制序列；D/E按同一产物协议集成；F/G统一迁移和验证。当前只更新本文，不创建代码框架或运行功能回归。后续并行实现时共享文件由集成owner负责，代码合并后统一format/lint/type/test。

## 8. 验证、bring-up 和验收标准

### 8.1 参考验收意图的本地落点

沿用参考提案T01–T32编号对照资源、安全与异常意图，以本文的分层合同、普通await和编译配置命令为准。

| 验证组 | 参考测试 | 本计划要求 |
|---|---|---|
| 静态合法性 | T01、T02、T04、T06、T26、T27、T32 | 每层允许集合、实际调用跨层能力、逐Bank/R×L1/布局正确；不按ID或总容量猜兼容。 |
| 准入与补位 | T03、T05、T07、T08、T12 | L2/L1分别判定SAME/COMPATIBLE，原子最后资源竞争、同类FIFO/碎片、逐Tile补位与回滚。 |
| 有限工作退休 | T09、T10、T11、T13、T14 | ordinary await保持旧工作推进；Route非空不能报告Grid完成；跨等待点生产者环拒绝。 |
| 可见性 | T15、T24、T25 | source consumed不等于目的可见，Cache hint不扣账，旧行场景由真实byte/版本oracle验证。 |
| Drain闭包 | T16、T17、T18、T19、T23、T31 | L1/L2分别重配，两方向、只读refill、L1维护产生下游写入、完成队列满均不提前Commit。 |
| 原子配置与故障 | T20、T21、T22、T28、T29、T30 | 等待与配置命令顺序、全成员ACK、旧代隔离、同配置不重Commit、唯一写者和取消确认。 |

### 8.2 分层配置与等待专项案例

| Case | 场景 | 预期 |
|---|---|---|
| RV01 | Context含L2 mode，两个Task分别请求L1.mode0/2 | 两级字段独立，Context不锁定整个调用的L1 mode。 |
| RV02 | Task请求不在程序L1允许集合 | 编译/加载拒绝，不自动改请求或换程序。 |
| RV03 | 两层各自存在，但Task需要的L2路径不受父配置支持 | 跨层调用拒绝，不由Task覆盖父L2 mode。 |
| RV04 | 同Context等旧grid_done后改L1，父L2 Arena仍live | L1完整事务可执行，L2 handle/generation/容量账本不变。 |
| RV05 | 根仍持有L2 Arena却在体内要求改L2 | 拒绝；编译器需拆根并使用Device等待/数据交接。 |
| RV06 | 另一个根仍有活跃或可继续投递的L1工作 | 本地await不能授权全层改档；Device序列化证明缺失就拒绝。 |
| RV07 | 只等input_released/output_ready | 不能据此假定Task/L1已退休，切换准备验证失败。 |
| RV08 | 普通await已完成，但Cache refill/下游事务/ACK尚未结束 | 配置命令保持阻塞，后继Task不准入；不改普通await语义。 |
| RV09 | 可执行产物有mode变化但无前置await或配置命令 | 拒绝；validator不合成缺失操作。 |
| RV10 | Task基准0，当前L1为2且程序允许2 | 编译绑定/运行准入可兼容复用，不切回0；真实依赖仍保留。 |
| RV11 | Task多Grid竞争同一父调用R名额 | 跨所有Grid计数，retire/安全取消才归还，free不提前归还。 |
| RV12 | 取消已开始的配置事务，随后迟到ACK/DMA | 错误收敛和隔离完成前不开门、不复用地址。 |
| RV13 | 初始配置/Registry与编译入口假设不一致 | 拒绝或走显式初始化/恢复流程，不伪造空await。 |
| RV14 | 只改一个层级 | 只改变该层配置generation；访问闭包包含必要上下游事务，但不要求无关存储无条件退休。 |

### 8.3 编译器与 validator 边界专项案例

| Case | 场景 | 预期 |
|---|---|---|
| CB01 | 从执行产物删除一个必要RAW/WAR/WAW边 | Loader/只读verifier拒绝，运行前后依赖集合不被修改。 |
| CB02 | 同一Context以不同实参alias关系调用 | 调用点产物独立或采用明确保守图；共享模板不被后一个调用累积改写。 |
| CB03 | program_id/hash缺失或不匹配 | 拒绝，不使用程序名CRC或运行时自动编号回退。 |
| CB04 | 普通await含1个或多个事件 | 保留原事件语义和1:1 WAIT/WAITALL解码；不是补全另一组等待。 |
| CB05 | 配置事务未出现在编译dump中 | Runtime不得生成，即使当前硬件空闲或目标容量更大。 |
| CB06 | 相同产物多次launch | 静态指令/依赖/hash不变；仅owner、launch generation、Slot/地址等合法动态字段不同。 |
| CB07 | 实际Host binding违反编译alias假设 | 加载拒绝/要求显式重编译，不在run中重建依赖。 |
| CB08 | 给validator高层源IR而不是指定编译产物 | 正式入口拒绝；显式开发工具的compile/run两步必须可观察。 |

### 8.4 完成门槛

- 普通 await 的源码形式与事件等待规则保持一致，Context 内L1切换和Device层L2切换均有正确示例与真实执行证据。
- 同层Bank配置一致，L1/L2独立、层级generation正确、资源和取消路径不泄漏，模式变化前完整事务确实等待所有必要ACK。
- 完整编译产物可审计，validator运行时不生成缺失的依赖/等待/配置/程序身份。编译提取与新增功能分别验收。
- 数据可见性使用独立小规模数据oracle，调度性能与正确性分开；不能以固定profile或metadata成功宣称真实Cache命中率/数值正确。

本轮只验证设计、源码边界事实、例子及验收覆盖，不声称已实现分层mode或新的编译器。功能落地后再在现有conda环境统一运行项目门禁和实际CLI场景。

## 9. 风险、取舍和后续细化方向

### 9.1 保持 await 简单的代价

源 IR 更容易读，但必须能检查编译器最终生成的配置序列；不能把真实安全动作藏回validator。普通await成功与全域静默分开会增加一条可观察的配置控制操作，这是硬件必需的事务，不是新的程序分组。

### 9.2 全层统一配置与并发

L1 mode由Task声明，不表示每Task或每Tile拥有自己的mode。V1在跨根L1改档时选择编译器显式序列化，可能损失并发；未经证明的局部await不能换取全域切换。若以后放宽，需要明确跨根同步/服务边界，而非直接扩大await含义。

### 9.3 资源、Cache与模型局限

Arena保留、striped padding和同类FIFO有容量/HOL代价，Profile亲和性不保证最优延迟。共享Cache存在干扰、驱逐、带宽竞争与隔离风险，无每Context容量/命中/QoS保证。真实Bank端口、mapping、Cache组织、系统预留、维护能力和外部Master参与协议均待目标冻结。

### 9.4 不可删的运行时工作

把静态lowering移给编译器，不等于删除真实资源竞争、动态地址绑定、事件代数、return退休检查、barrier/await注册fence或fault/reset。这些硬件行为决定程序能否安全完成；它们必须与编译产物共同验证，不能为了“Runtime简单”用固定延迟、无条件成功或默认全空状态替代。
