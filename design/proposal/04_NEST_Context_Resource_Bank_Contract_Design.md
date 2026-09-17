# NEST SRAM Profile、资源合同与 simulator_experiment 实现

> 状态：`simulator_experiment` 实现已落地，**验收进行中**。完整回归、示例 corpus、Perfetto 审计以及 T/RV/CB 逐 ID 证据由集成 owner 在门禁结束后统一签署；本文当前只描述实现合同和完成门槛，不把尚未挂接的证据标记为最终接受。
> 依据：[NEST_SRAM_Bank_Runtime_IR_Proposal_v0.1.md](../../review/NEST_SRAM_Bank_Runtime_IR_Proposal_v0.1.md)、当前 `pipeline_validator` 实现及用户 review。**L2 mode 由 `nest.context` 指定，L1 mode 由 Task 启动操作指定；await 保持普通事件等待写法。编译器生成显式等待和配置执行序列，Loader 只读验证，Runtime 只执行已加载产物，不现场补依赖或配置步骤。**
> 软件接口为 contract v0；本文列出的 mode bytes、系统预留、维护能力、粒度和控制时序是可复现实验目标，真实物理编码与数值仍由后续硬件规格冻结。

## 1. 定位、目标和 First Silicon cutline

### 1.1 当前设计

| 议题           | 决策                                                                                                                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 模式定义       | L1/L2 各自有 N 个字节配置 mode；每项明确 SPM/Cache bytes，不使用百分比输入。                                                                                                                          |
| L2 选择        | `nest.context.resource_contract` 声明 `l2_mode`、本层 `allowed_profiles` 和 L2 Arena 需求。一个根 Context 的 L2 mode 在其生命周期内固定。                                                             |
| L1 选择        | Task 启动操作 `nest.dispatch.tasks.async` 显式携带 `l1_mode`；被调用 `tile.program` 的资源合同声明 L1 `allowed_profiles` 和每 Task 资源需求。                                                         |
| 等待写法       | 保持 `nexus.await %context_event`、`nest.await %grid_event`、`tile.await %engine_event`，不要求用户在 await 上填写配置域、完成等级或切换授权。                                                        |
| Context 内切换 | `nest.context` 内可以用 `nest.await` 等待旧 Grid，为后续 Task 的 **L1-only** 切换准备；配置未变的父 L2 Arena 可继续存活。                                                                             |
| L2 切换        | 在 Device 控制流等待相关根 Context 退休后进行。仍持有旧 L2 Arena 的根不能在自己体内改变 L2 mode。                                                                                                     |
| 完整切换等待   | 普通 await 等指定事件；编译器另输出有序的 Profile 配置事务，执行全域排空、Cache 维护及 Commit ACK。事件完成与可安全改档不是同一条件。                                                                 |
| 配置范围       | 同层 Domain 的所有 Bank 始终使用同一 mode；所有 Tile 的 L1 同属配置域。Task 的 mode 不是私有硬件分区。                                                                                                |
| 补位           | 安全、依赖、资源和基础优先级先行；同等条件优先 SAME，其次 COMPATIBLE，同类 FIFO。Context 比较 L2，Task 比较 L1。                                                                                      |
| 软件边界       | `compiler.compile_program` 产生完整、可检查且不可变的执行产物；`loader.load_program` 只读验证并绑定实际 HBM；`Simulator.run` 只接收 `LoadedProgram`，不隐式运行依赖推导、Profile 规划或程序身份生成。 |

**本合同采用保留 await 原写法的方案。** 不将普通 await 偷偷升级为全设备 Barrier，也不把它当作只要存在就能切换的标记。完整安全性由“编译器插入的事件等待 + 编译器生成的完整配置事务 + Runtime 的真实计数/ACK检查”共同保证。

### 1.2 范围与约束

- 保留 ready-action、Tile eligible-head、SPMD task 身份、placement、显式 UCE pin、`bindings/ins/outs` 和正常异步事件模型。
- 不增加自动 Profile/Tiling 优化、live SRAM migration、抢占恢复或跨 Context 私有 Cache 配额。
- 不因源码里出现不同 mode，就由 validator 自动加一条等待、补一条数据依赖或现场生成配置事务。
- `nest.task.range` 只定义逻辑任务索引，不是启动操作；本文的 **Task op 指 `nest.dispatch.tasks.async`**。
- 同一层统一配置不等于统一访问类型：各 Bank 可同时服务其 SPM/Cache Region 中的不同请求。不同 Tile 的 L1 容量不能合并给某一个 Tile 使用。
- SPM 使用 Context/Task-lifetime Arena 硬预留；局部 Buffer release 不向其他 Context 提前返还 Arena 配额。

## 2. 职责、非职责和 ownership

### 2.1 对象与配置责任

| 对象                        | 拥有的内容                                                      | 不拥有的内容                         |
| --------------------------- | --------------------------------------------------------------- | ------------------------------------ |
| L2 Profile Domain           | 目标定义的全部相关 L2 Bank 的统一 mode/generation               | 各 Context 的独立 mode 寄存器        |
| L1 Profile Domain           | 全部 Tile L1 Bank 的统一 mode/generation                        | 每个 Task 单独切自己所在 Tile 的配置 |
| Allocation Pool             | 每 Group 的 L2 池、每 Tile 的 L1 池及独立资源账本               | 跨 Tile 借用容量                     |
| `nest.context`              | L2 mode、L2 允许集合、L2 Arena、子任务并发上限                  | 固定整个 Context 的 L1 mode          |
| `nest.dispatch.tasks.async` | 本次 Task/Grid 的 L1 基准请求 `l1_mode`、程序引用和依赖         | L2 mode 覆盖、配置寄存器写权限       |
| `tile.program`              | 固定代码/布局、L1 允许集合、L1 Arena/控制预算、跨层访问能力要求 | 第二个独立的默认 L1 mode 选择入口    |

L1/L2 选择独立，但**一个具体调用的跨层组合仍需验证**。例如 Task 要求 L2 Cache 路径，而父 Context 的已绑定 L2 mode 无 Cache 且无合法 Bypass，则该调用非法。Task 不通过携带另一个 `l2_mode` 来覆盖父合同。

### 2.2 编译器、Loader 与 Runtime

| 组件 / 当前路径                                      | 已实现责任                                                                                                                            | 禁止承担                                                       |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `profiles.py`、`config.py`、`hardware_config.yaml`   | 解析 schema v2 的 typed target/profile source，构建字节模式、几何、系统预留、组织/能力及 Registry hash                                | 执行中改变模式表或从旧 Cache 容量字段取得第二份容量            |
| `compiler/api.py`                                    | `compile_program` 串联 source verify、调用点 lowering、资源布局、Profile/维护 pass、event frontier、程序身份、hash 与语义复核         | 把动态 Slot、物理地址或运行时竞争冻结成假静态结果              |
| `compiler/lowering.py`                               | 从干净源 Context 为每个 submit 生成独立 `binding_id`，按调用点 actual→formal alias 专化依赖、role、descriptor 与 release 闭包         | 原地改写共享源模板，或让前一次调用的 alias 边累积到后一次调用  |
| `compiler/resources.py` / `compiler/profile_pass.py` | striped Arena 布局、逐 Bank/R/Slot/事件预算、分层 mode 绑定、普通 await/排他边、配置及范围维护序列                                    | 运行时试档、按两个 allowed 列表临时做跨层组合                  |
| `compiled_program.py`                                | frozen `CompiledProgram`/`LoadedProgram`、严格 JSON allowlist codec、持久化 hash 与可读执行 dump；本模块不导入 compiler               | pickle/eval/dynamic import、未知字段或未知类型透传             |
| `execution_verifier.py` / `loader.py`                | 不改图地重建访问效果、必要 hazard/maintenance/frontier、资源与 relocation 检查，并核对实际 binding、权限、alias 与 target fingerprint | 修补非法输入、补依赖、重排程序或重新编译                       |
| Device / Group 控制器                                | 执行 submit/await/return、已编译的 Device/Group 控制项与 ready-action                                                                 | 根据未来 Context 或当前空闲状态偷改 L2/L1                      |
| `memory/arena.py` 与 Group/Tile Admission            | 当前容量、连续 stripe、Slot、R lease、pin、精确 plan/commit/rollback，以及 view/Arena 两级退休                                        | 从程序体重新求静态 bytes，或在局部 free 时提前返还 owner Arena |
| `memory/profile_controller.py`                       | 一个 TileGroup 的唯一配置/维护 writer、门禁、真实成员总线 ACK、generation、取消隔离和显式恢复                                         | 因发现 mode 不同而自行插入程序步骤，或用固定延迟伪造全成员成功 |
| `memory/cache.py` / `transfer.py` / `byte_store.py`  | 零容量 Cache bypass、真实 refill/clean/维护闭包、代数检查；`full_memory` 下提供稀疏字节 oracle                                        | 从 Cache 容量推导命中率，或为未建模 tensor 算术制造结果        |

普通 await 不持有配置锁。配置事务在其已编译执行点获取唯一控制权；等待期间不持有会阻止旧请求返回或资源释放的 allocator 锁。Device 与 Group 经同一 `ProfileController` 提交，运行命令身份还包含 run/owner launch generation，静态 `command_id` 不会复用上一次执行的成功状态。

## 3. 微架构和状态机

### 3.1 从源 IR 到执行产物

```text
Source MLIR: Context(L2) -> Task(L1), ordinary await/events
                         |
 compiler.compile_program(ModuleOp, HardwareConfig, SimConfig, ...)
                         |
 compiler/lowering.py -> resources.py -> profile_pass.py
                         |
 CompiledProgram(schema=1, ABI=v0, frozen entry/calls/layout/proofs/hashes)
                         |
 serialize -> JSON / parse -> loader.load_program(read-only verification)
                         |
                    LoadedProgram
                         |
 Simulator.run -> Device PC -> Group ready-action -> Tile admission/execution
                         |
 ProfileController / ArenaPool / cache+transfer / completion+fault handling
```

源码模式 CLI 也严格执行上述两阶段流程并持久化中间产物；`--compiled-file` 分支不导入 `pipeline_validator.compiler`。执行 dump 必须能看到每条 WAIT、配置/维护事务、后继 dispatch、完整 descriptor 步骤及 `source_ref`，这些内容不能只存在于注释或 Runtime 推导分支。

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

| 层                 | 基准                 | 合法允许集合                                                       | SAME / COMPATIBLE                                              |
| ------------------ | -------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------- |
| Group Context 补位 | Context 的 `l2_mode` | Context 的 L2 `allowed_profiles`，并满足该调用的子任务计划         | 当前 L2 等于基准为 SAME；不同但已验证可运行为 COMPATIBLE。     |
| Tile Task 补位     | 启动 op 的 `l1_mode` | 被调用 Tile Program 的 L1 `allowed_profiles`，结合该调用的跨层检查 | 当前 L1 等于请求为 SAME；不同但可在当前 L1 执行为 COMPATIBLE。 |

同等基础优先级下 SAME 优于 COMPATIBLE，同类按 ready 序号 FIFO。只对可比较桶队首做无副作用 plan；SAME 队首不能完整准入而 COMPATIBLE 可以时，允许后者补位。同桶不越过资源阻塞队首挑更小请求。Profile 优先级不越过数据依赖、R 上限、配置命令或 await。

集合外请求不能作为普通容量等待无限挂起：编译器必须给出先等待再切换的路径。Runtime 不改变已经选择的程序、Tiling、布局和允许集合。无限流式提交的公平性需要独立的 aging/配额设计；当前只保证有限合法工作在既定公平服务前提下可结束。

## 4. 接口、descriptor、寄存器和协议

本节描述已实现的 `simulator_experiment` contract v0。资源属性和 `l1_mode` 是源 IR 必填项；await 仍使用原语法，不增加配置属性或新的操作数种类。

### 4.1 显式字节模式

每个字段以**单 Bank**为单位。L1/L2 模式表独立；ID 是本层索引，不可按大小比较容量或跨层混用。

```yaml
memory:
  target:
    # bundled simulator_experiment 值；由后续硬件规格冻结。
    profile_command_timeout_cycles: 2000000
    l1:
      system_reserved_spm_per_bank: 2048
      reset_mode: 0
      spm_mapping_id: striped_arena_v0
      cache_org_id: profiled_lru_v0
      cache_write_policy: read_only
      maintenance_caps: [invalidate_range, clean_invalidate_all, bypass]
    l2:
      system_reserved_spm_per_bank: 4096
      reset_mode: 0
      spm_mapping_id: striped_arena_v0
      cache_org_id: profiled_lru_v0
      cache_write_policy: read_only
      maintenance_caps: [invalidate_range, clean_invalidate_all, bypass]
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

`NestContextOp.resource_contract`、`TileProgramDefOp.resource_contract` 和 `NestDispatchOp.l1_mode` 在 parser/constructor 中均为必填；缺失即源 IR 非法。Context 的可选 `l2_cache` 与 Tile Program 的可选 `l1_cache`/`l2_cache` 若出现，必须完整给出 `required/access/bypass/target_bytes` 四项。容量只来自 active `MemoryProfile`；latency/MSHR 等控制参数仍来自 target，但不再有另一份可写 Cache data capacity。

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

`arena_v0` 已由 `memory/arena.py::ArenaPool` 实现为分层硬预留：根 Context 持有 L2 Arena，Task 持有本 Tile 的 L1 Arena。每个编译布局有确定的 alignment、stripe、逐 Bank `per_bank_bytes`、Buffer slot/offset/有效 segment 与 layout hash；padding 属于 Arena 预留但不暴露为逻辑字节。

设某父调用的 Task 并发上限为 R，许可子程序在对应 L1 Profile 下的最大 Arena 为 E：

```text
R <= physical Tile context slot count
R × E <= this Tile's user-usable L1 SPM
for every bank b:
    R × max(child Arena bytes[b]) <= user-usable L1 SPM[b]
```

还要检查 Slot、控制预算、连续跨度和 alignment。R=4 却只有一个 Task 能放下的合同在编译期拒绝，不由 Runtime 偷降 R。L2 也逐 Bank 检查空池可行性，不要求所有根同时驻留。

Runtime 使用 per-`(parent_invocation_id, launch_generation, tile_id)` 的 Task lease 计数；覆盖全部 Grid 的 pending-activation、active、retiring Task。Commit 扣除，安全 retire/cancel 确认才归还；失败事务整体回滚，`input_released`、`tile.free` 或取消请求本身不归还名额。

`tile.free/nest.release` 失效逻辑 Buffer view，可在同 Arena 的已验证布局内复用；Arena 配额在其 owner 退休后才返还 Pool。完整调用闭包、真实最后访问、pin、generation 和 SlotFrame 检查均保留。

L1-only 切换要求旧 L1 用户 Arena 全部退休，但**不要求父 L2 Arena 或不涉及旧 L1 访问的 L2 生命周期 pin 归零**。`ArenaPool` 的一个实例只代表一个层级/Pool，L1 `reconfigure` 没有修改 L2 Pool 的路径；父 L2 handle、profile generation、有效 view 和字节继续存活。禁止后续 Task 使用旧 L1 handle；改变 L2 则必须先让相关根退出，不能保留旧 L2 Arena 后原地改边界。

### 4.5 普通 await：保留什么、不能保证什么

保留原写法和普通事件等待语义：

```mlir
nexus.await %context_done
nest.await %grid_done
nest.await %grid_a, %grid_b
tile.await %load_done
```

| 写法          | 等待内容                                                           | 单靠它不能推出                                  |
| ------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| `nexus.await` | 指定根 Context 的成功完成/退休，阻塞后续 Device 提交               | 所有无关根均已结束、Cache/Fabric 已完成全域维护 |
| `nest.await`  | 指定本 Context 内事件；保留注册 fence，阻塞本 Context 后续动作登记 | 其他根的 Grid 已结束、全层 L1 已静默            |
| `tile.await`  | 当前 UCE context 的指定引擎事件                                    | 本 Task 已退休、其他 UCE context 已退出         |

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

这不要求所有 Context 永远串行：不改变 L1 mode、或全部 Task 可在当前配置下兼容运行的多个根仍可重叠并补位。更精细的跨根同步需另行证明；本合同不增加跨 owner 的隐式 Group Barrier。

若父 Context 后半段需要不同 L2 mode，源 IR/上层编译必须把工作表达为不同根，并通过已有合法 HBM backing/Store/Prefetch 交接仍需保存的数据；当前编译器对仍持有 L2 Arena 的根内改档要求直接拒绝，不在此处发明新根或地址空间。

### 4.7 可运行的 Profile reconfiguration 示例

仓库中的可运行源码是 [`examples/scenarios/profile_reconfiguration.mlir`](../../examples/scenarios/profile_reconfiguration.mlir)，统一 runner 名为 `profile-reconfiguration`。它不是伪语法片段：parser 要求两个资源合同及每次 dispatch 的 `l1_mode`，CLI 可以把它编译、持久化，再由独立 `--compiled-file` 进程加载执行。

源码保留普通 await 写法，关键控制流为：

```mlir
%g0, %r0, %w0 = nest.dispatch.tasks.async @task_a
  l1_mode = 0 tasks(%tasks) globals() bindings() ins() outs()
  signal_policy {} : (!nest.event<"g0">, !nest.event<"">, !nest.event<"">)
nest.await %g0

%g1, %r1, %w1 = nest.dispatch.tasks.async @task_b
  l1_mode = 2 tasks(%tasks) globals() bindings() ins() outs()
  signal_policy {} : (!nest.event<"g1">, !nest.event<"">, !nest.event<"">)
%g2, %r2, %w2 = nest.dispatch.tasks.async @task_a
  l1_mode = 0 tasks(%tasks) globals() bindings() ins() outs()
  signal_policy {} : (!nest.event<"g2">, !nest.event<"">, !nest.event<"">)
nest.await %g1, %g2
```

因此三个 Task 的**基准请求**为 `0/2/0`；编译器把实际绑定冻结为 `0/2/2`：`g1` 前执行一次 L1.mode0→mode2，`g2` 因 `task_a.allowed_profiles=[0,2]` 兼容复用 mode2，不切回 0。`ctx_tasks` 的 L2 请求和实际值均为 mode0，其 4096 B 父 L2 Arena 在内部 L1-only 切换期间保持同一 handle/generation。R=1 覆盖同一父调用的全部 Grid，故 `g2` 可登记但在 `g1` 安全退休归还 lease 前不能准入。

Device 源控制流仍是普通事件：

```mlir
%c0 = nexus.submit_context.async @ctx_tasks : !nexus.event<"c0">
nexus.await %c0
%c1 = nexus.submit_context.async @ctx_l2_2 : !nexus.event<"c1">
nexus.await %c1
```

编译产物在 `g0` 的 WAIT 后包含完整 L1 `PROFILE_RECONFIG`，在 `c0` 的 WAIT 后包含完整 L2 `profile_reconfig`。每条描述符固定展开：

```text
ACQUIRE -> CHECK_FRONTIER -> CLOSE_ISSUE -> DRAIN_REFERENCES
-> CLEAN_INVALIDATE -> DRAIN_DOWNSTREAM -> PREPARE -> WAIT_READY_ACK
-> COMMIT -> WAIT_COMMIT_ACK -> OPEN_ISSUE -> RELEASE
```

描述符还绑定 `expected_mode/target_mode`、Registry hash、普通等待 instruction ID、事件 frontier、目标层全部 member ID、排他 binding 与 `source_ref`。Loader 删除任一必要 await、frontier、成员或步骤都会拒绝；Runtime 不从源码注释或 mode 差异补发命令。启动空 frontier 只接受真实初始化证明，warm 入口 mode 不同必须走显式恢复，不能伪造无事件 await。

这个场景验证控制/资源生命周期，不宣称 tile 程序完成 tensor 数值计算。真实 64 B 字节保持证据使用独立 `full_memory` copy smoke，并核对 L1 改档前后父 L2 identity、generation 和 bytes。

### 4.8 接口与错误边界

- 正式 API 是 `compiler.compile_program(ModuleOp, hw, sim, ...) -> CompiledProgram`、`serialize_compiled_program`/`parse_compiled_program`、`loader.load_program(..., actual_bindings=...) -> LoadedProgram` 与 `Simulator.run(LoadedProgram)`；源码、裸 `CompiledProgram` 或执行 DTO 传给 `run` 均拒绝。
- `CompiledProgram` 固定 `schema_version=1`、`compiler_abi="v0"`，深冻结 entry/prefix、调用绑定、relocation、source map、依赖证明、entry/exit profile、静态效果、资源预算、binding guard 和 workload metadata。动态 launch ID、Slot 与物理地址只通过 allowlist relocation 在运行实例中绑定，不写回共享模板。
- `source_hash`、Registry hash、target hash 与 artifact hash 均基于 canonical 内容。target fingerprint 包含拓扑、存储/控制容量、Profile target 与 context 数等静态可行性字段，不绑定 `max_cycles`、trace 或 scheduler policy；同 mode ID 改 bytes、预留、能力或几何会让旧产物加载失败。
- 严格 JSON codec 拒绝重复 key、未知字段/类型/opcode、非有限数字、未知版本和缺失身份；artifact hash 正确后仍执行语义验证，不把 hash 当作合法性证明。
- 每个 model submit 有独立 `callsite_id`/`binding_id` 和从干净源模板生成的 alias 专化。Loader 用实际 HBM 范围、权限与 alias guard 核对晚绑定；不满足就拒绝，不在 `run` 中重建图。
- 列表外 mode、跨层能力不符、入口配置不符、缺必需 await/配置/维护、跨根 L1 排他证明缺失和旧代请求均是明确错误，不作为普通容量等待。

## 5. 数据流、控制流和时序路径

### 5.1 Context 内 L1-only 切换

1. 编译器确认父 L2 绑定与数据生命周期稳定，收集旧 Grid 及其它会访问 L1 的已发请求；源中已有等待则保留，缺少覆盖时插入普通 `nest.await`。
2. await 是本 Context 的注册 fence；旧任务、返回、完成和释放路径继续推进。事件成功后才到达编译好的 L1 配置命令，而不是直接准入新 Task。
3. `CHECK_FRONTIER` 消费本次 run/owner/PC 对应的普通等待证明，并核对设备级排他、旧 L1 Task/Route/Frame/Arena 与仍可投递生产者；旧 launch 的成功 await 不能复用。
4. `CLOSE_ISSUE` 后收敛旧 L1 miss/refill、写缓冲、引擎、DMA、NoC、维护及其下游；L1 clean 产生的 L2/HBM 工作也必须真实完成。
5. 目标 L1 Profile 的**每个 member**都完成 Prepare ACK 后才发 Commit；每个 Commit ACK 都到齐后才发布新 L1 generation、重建本层 Arena free map/Cache/MSHR 并开门。L2 mode/generation、父 Arena 和 view 不变。
6. 后继 Task 才可进行 L1 Arena/Slot/Frame/R lease 的原子准入；未 commit 计划无资源副作用，commit 后异常按 Grid fault/cancel-confirm 收敛。

“完整”针对实际重配域及它的访问闭包，不等于要求无关层的全部用户存储都归零。旧 L1 handle不能逃逸到后继Task；不能只看 `tile.free` 或某一个 UCE Slot空闲就提交新配置。

### 5.2 Device 层 L2 切换

需要跨根保留的数据必须由源 IR 中已有合法 HBM backing 与 Store/Prefetch 显式交接；编译器验证访问、依赖和可见性，不凭空创造 HBM 地址或数据搬运。Device 侧用普通 `nexus.await` 等待全部相关根 `context_done`；所有受影响 L2 Arena 必须退休，所有可能向该 L2 发请求的 Tile/L1 工作也纳入闭包。

随后执行已编译的 L2 配置事务：获取唯一控制权、关闭新用户请求、必要上游/下游维护、DMA/NoC/refill 归零、全成员 ACK，成功后只增加实际改变配置的 Domain generation。L1 数值可以不变，但其旧请求和依赖不能遗漏。

在仍持有旧 L2 Arena 的 Context 内，仅等自己的 Grid 不能授权 L2 重配。源/前端必须使用不同根及 Device 等待表达这种变化；否则编译拒绝。

### 5.3 完成事件与资源退休

| 完成对象         | 保证                                                 | 不保证                                         |
| ---------------- | ---------------------------------------------------- | ---------------------------------------------- |
| `input_released` | 指定输入的消费者读取结束                             | Task计算结束、L1 Arena已退、整个Grid完成       |
| `output_ready`   | 指定层级输出可用                                     | 所有Task退休或Host/HBM可见                     |
| `grid_done`      | 全部预期Task正常结束，相关L1 frame/资源按协议退休    | 其他根的Grid完成、L2父Arena释放或全域Cache维护 |
| `context_done`   | 根的子任务、必需输出与资源退休完成                   | 整个共享Domain已经完成配置维护                 |
| 配置命令完成     | 目标Domain访问闭包静默、维护及所有成员Commit ACK完成 | 后续另一次切换可省略await                      |

若代码只尝试释放而仍有 pin/release-pending，不能据事件完成位忽略残留账本；配置命令必须再次核对真实资源/在途状态。错误事件、超时或取消不能转成成功等待记录。

源读取结束不等于目的可见。HBM 导出必须来自已有合法 Store；编译器基于真实 Global view 生成必要的范围失效/一致性维护，不能认为普通 DMA completion 自动清理所有共享 Cache 旧副本。目标只有全域静默维护能力时，相应交接还需排空相关工作并生成全域维护；无必要 backing 或能力则拒绝，而不是靠一个 await 名词兜底。

编译生成的 `MEMORY_MAINTENANCE` 同样不是隐式副作用，其固定序列为：

```text
ACQUIRE -> WAIT_DEPENDENCIES -> BLOCK_RANGE_ISSUE
-> DRAIN_RANGE_REFERENCES -> CLEAN_INVALIDATE -> DRAIN_DOWNSTREAM
-> ACK -> UNBLOCK_RANGE_ISSUE -> RELEASE
```

范围绑定实际 Global input index、offset、bytes 与受影响层；无 range capability 时只能使用已编译的全域范围及对应静止证明，Runtime 不临时扩大或缩小范围。

### 5.4 配置事务、锁与故障

每条配置/维护命令是有完成边界的控制项，也是 Device PC 或 Group 注册/发射 fence；完成前后继不能越过。`memory/profile_controller.py::ProfileController` 是一个 TileGroup 的唯一 writer，Device 与 Group 都经同一实例提交，控制存储与成员完成通道独立于用户 SPM。

L1 的成员集合覆盖全部 Tile L1 Bank，L2 覆盖 Group L2 Bank。总线每 Tick 至多服务一个真实请求/响应；Ready/Commit 分开核对 runtime transaction、member、stage 与 generation。少任一受影响成员 ACK 都不开门，重复、旧代、未知成员不计成功。初始化与显式恢复同样逐成员 Prepare/Commit，不用总定时器批量设成功。

事件 await 和配置命令之间的证明绑定当前 run generation、owner launch generation 与 PC 消费序号。新用户活动若破坏编译前提则 fault；相同 Profile 不重复 Commit，但显式数据依赖等待照常执行。配置中途失败保持门关闭并进入受控隔离/reset；取消仅是请求，已发配置、DMA 和迟到 ACK/返回在隔离确认前不能释放地址、返 credit 或开放新工作。

### 5.5 有界执行与进展

保留 CPU-before-Group、完成后收集、依赖次 Tick 可见等协同原则；普通 await 阻塞后继登记，不停止旧工作所需的引擎、返回、维护和完成推进。

事件存储按**编译证明的 live frontier**计账，而不是为全部历史事件永久留槽。`compiler/resources.py::finalize_event_resources` 在 Profile/维护动作生成后计算每个事件的未来引用次数与峰值 frontier，写入 `event_uses` 和 `ResourceBudget.event_frontier`；只读 verifier 独立重建并要求精确一致。Runtime `EventTable` 先按 frontier 预留 quota，再在事件真正产生/最后一次引用后占用/回收 live slot；同时受 Group `event_capacity`、`context_action_quota`、dispatch/Grid、Frame Slot 与引擎控制容量约束。

等待图不能形成“父持 L2 等子、子需要另一个未准入根才能结束”或“await 前生产者依赖 await 后提交”的环。编译器证明切换点可到达及相关工作有限；Loader 验证证明和预算，Runtime 只处理真实竞争。Group 等待不能引用另一个根 owner 的事件；跨根排空使用 Device 事件和已编译提交顺序。

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
| ----------- | ----------: | ------------: | ------: | -------: | ----------------: | ----------------: |
| L1.mode0    |       65536 |             0 |      16 |  1048576 |                 0 |           1015808 |
| L1.mode1    |       57344 |          8192 |      16 |   917504 |            131072 |            884736 |
| L1.mode2    |       49152 |         16384 |      16 |   786432 |            262144 |            753664 |
| L2.mode0    |      524288 |             0 |      16 |  8388608 |                 0 |           8323072 |
| L2.mode1    |      458752 |         65536 |      16 |  7340032 |           1048576 |           7274496 |
| L2.mode2    |      393216 |        131072 |      16 |  6291456 |           2097152 |           6225920 |

L1.mode1改为53248/12288 B后，单Tile用户SPM为 `16×(53248−2048)=819200 B`。49152+8192 B不守恒，65535+1 B不满足64 B粒度，均拒绝。模式比较不按ID大小，也不按占比近似猜合法。

在参考提案的测试geometry（每Tile 4×32 KiB、每Bank用户SPM22 KiB）中，R=4、每Task6 KiB/Bank需要24 KiB/Bank，空池仍不可能，加载即拒绝。一个Bank需要23 KiB、其余只需1 KiB，总量虽小也非法。

### 6.2 Cache 与性能边界

每层 Cache 是 pool-shared，`required/access/bypass` 是能力要求，`target_bytes` 是工作集 hint，不按 Context 相加扣 Arena。`DeterministicLRUCache` 使用 allocation ID、allocation generation、line offset 与 profile generation 标识行，支持 range invalidate、clean 和 `read_only`/`write_back`；普通 Context 退出不无条件清全池。

零容量是合法 Profile：Runtime 跳过该层 lookup/fill/MSHR，按已编译 `bypass_levels` 走下一层或 HBM；显式 L1/L2 hit 指向 disabled Cache 会拒绝。编译器检查完整程序中的 Gather outcome 与合同，不会因前半段未用 Cache 就放过后半段 hit，也不会为零容量临时补一行。

`ByteStore` 只在注入 `Simulator(..., byte_store=...)` 且 `fidelity=full_memory` 的小规模正确性路径提供数值证据。它是稀疏页+有效位 oracle：Host 可先 `seed_hbm`，但运行前所有 seed 必须落在实际 binding；设备读写仍检查 binding 权限、边界和 allocation generation，未初始化读取报错，不补零。Transfer 按 resolved view 的逻辑 segment 顺序复制有效字节，padding 不可见，源腿结束不代表目的已可见。

非 oracle Gather 只有 allocation-qualified 的保守 whole-view provenance；范围维护会失效所有相交 provenance，不能据此声称精确命中。oracle Gather 必须用 `(binding_id, request_id, source_offset)` 为每次调用绑定实际源范围，并要求请求/Cache line 都在 resolved source view 内；预置 Cache line 也检查 line 对齐、容量、权限和 write-back dirty 能力。ByteStore 不实现 BOA/EVU/USE tensor 算术，Cache bytes 也不预测命中率；所有 fidelity 执行合同、预算、门禁与代数检查，真实字节结论只来自 `full_memory`。

### 6.3 可观察输出

- 分别记录 Context 请求/实际L2 mode、Task 请求/实际L1 mode、各层Registry/hash/generation，不用一个混合Profile字段掩盖归属。
- 记录 Await指令及事件、配置命令开始/结束、等待旧工作/维护/Fabric/ACK的时间、与源op的映射。
- 同时报告Arena预留、live Buffer、系统预留、共享Cache和R lease；L1改档前后L2 handle/generation连续性应可核对。
- 编译产物哈希和指令/依赖清单应随报告保留，以证明哪些依赖、等待或配置来自编译器，而不是validator临时生成。

## 7. 软件实现与 RTL 映射边界

### 7.1 编译器已实现职责（独立清单）

| ID  | 当前实现责任                                                        | 产物 / 证明                                                                                                  |
| --- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| C01 | `profiles.py` + `compiler/resources.py` 解析 Registry 与分层合同    | Context L2、Tile L1 允许集合，模式/组织/版本、Cache/Bypass 与跨层调用合法性                                  |
| C02 | 确定 striped layout 和静态预算                                      | 含 padding/复用生命周期的逐 Bank Arena、Slot/Grid/engine 预算、R×L1 空池证明                                 |
| C03 | `compiler/lowering.py` 从干净模板做访问效果与调用点 alias 专化      | L2/global RAW/WAR/WAW、release/return 闭包、每次 submit 独立 `binding_id`；源码 digest 前后不变              |
| C04 | `resources.py` / `profile_pass.py` 绑定每次根调用与 Task 的实际模式 | L2/L1 请求与 resolved 值、完整程序能力检查、许可跨层 `ProfileState` 集合                                     |
| C05 | 维护完整 Grid/root frontier，生成必要普通 await 与跨根排他顺序      | 等待可达性、owner/generation、旧生产者集合；不把 `input_released/output_ready` 当退休                        |
| C06 | 生成 `ProfileReconfigDesc` / `MemoryMaintenanceDesc`                | 固定步骤、等待 instruction、事件 frontier、全 member、范围、依赖、源因果与 HBM 可见性顺序                    |
| C07 | 生成 frozen 低层执行对象                                            | role/dispatch/release 索引、engine/DMA/Gather descriptor、view、WAIT/WAITALL、return 闭包与 relocation       |
| C08 | 分配确定的 program ID/version/hash 并封装不可变产物                 | source/Registry/target/artifact 指纹、调用绑定、重定位 allowlist；无 0 ID/名称 CRC/默认 text bytes 回退      |
| C09 | 持久化与诊断                                                        | 源 IR、`.compiled.mlir.txt`、完整 `.exec.txt`、严格 JSON artifact、`.target.yaml`、`source_ref` 与可定位错误 |

编译器的最终一步调用只读 verifier 做自检，但这不把 Loader 变成第二个编译器。正式加载验证语义而不修改 graph；artifact hash 即使被攻击者重算，也不能让缺 hazard、等待、维护或配置步骤的产物通过。

### 7.2 当前实现所有权

| 当前路径                                                                | 所有权                                                                                                                    |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `pipeline_validator/compiler/__init__.py`、`compiler/api.py`            | 唯一公共 `compile_program` 入口；内部顺序固定为源验证/lowering→资源→Profile/维护→依赖与 event 终结→身份/hash→只读语义复核 |
| `compiler/lowering.py`                                                  | 内部 source→execution lowering；model 每个 submit 重新从源 Context 构造，绑定 actual 索引/alias，不发布为 Runtime API     |
| `compiler/resources.py`                                                 | Arena layout、生命周期复用、逐 Bank/R/Slot/Grid/事件/engine 静态可行性                                                    |
| `compiler/profile_pass.py`                                              | SAME/COMPATIBLE 绑定、L1/L2 配置、普通等待、跨根排他及 global range maintenance                                           |
| `compiled_program.py`                                                   | compiler-free frozen DTO、严格 codec、artifact sealing 与完整执行 dump                                                    |
| `execution_verifier.py`、`loader.py`                                    | 重建访问/预算/控制证明，验证 target 与 actual binding，返回只读 `LoadedProgram`                                           |
| `simulator.py`                                                          | 重新核对 LoadedProgram target 与当前入口 mode，只执行已给定 entry/prefix；不导入 lowering，不分配程序身份                 |
| `runtime/relocation.py`、`tile_group.py`、`tile.py`                     | 为每次 launch 复制允许动态字段，实例化 owner/event/queue/实际 HBM；共享程序、descriptor 和依赖模板不写回                  |
| `memory/arena.py`、`runtime/event_table.py`                             | Arena/view 生命周期、精确 extent、profile/allocation generation 与编译 event quota                                        |
| `memory/profile_controller.py`、`memory/cache.py`、`memory/transfer.py` | 唯一配置 writer、全成员协议、维护、Cache、真实事务/取消隔离与代数                                                         |
| `memory/byte_store.py`                                                  | 可选 full-memory 稀疏字节 oracle 与每调用 Gather 源范围绑定                                                               |
| `cli.py`                                                                | 源模式显式 compile→persist→load→run；compiled-file 模式只 parse/load/run                                                  |

`tile.signal input_released/output_ready` 的 “phase” 只指既有读/写里程碑及其 aggregate policy，与 Profile 分组或模式选择无关。

### 7.3 正式 API 与独立重放

```python
compile_program(module, hw, sim, *, binding_assumptions=None,
                source_name="<memory>", workload_info=None) -> CompiledProgram
serialize_compiled_program(program) -> str
parse_compiled_program(text) -> CompiledProgram
load_program(program, hw, sim, *, actual_bindings=None) -> LoadedProgram
Simulator(hw, sim, byte_store=optional_oracle).run(loaded_program) -> SimResult
```

源码 CLI 的 `--compile-only --compiled-output PATH` 同时写 `PATH`、同 stem 的 `.exec.txt`、`.compiled.mlir.txt` 和 `.target.yaml`；默认路径按 artifact hash 内容寻址，已有不同内容拒绝覆盖。`--profile-bytes LEVEL:MODE=SPM:CACHE` 只修改本次源码编译目标并重建 Registry/hash，不能用于 `--compiled-file` 篡改旧产物。独立重放使用 artifact 对应 target YAML 和实际 input bindings，不读源文件、不导入 compiler；编译/加载错误退出 2，执行失败或超时退出 1。

Runtime 每次 launch 都重核当前实际入口 profile。首次 reset-mode 初始化逐成员取 ACK；warm launch 若当前 mode 与产物入口不同直接拒绝，caller 必须调用显式恢复或加载匹配入口的产物。`Simulator.run` 不自动 reset Profile、清 Cache 或复用上次 await 授权。

### 7.4 实现与验收状态

| 范围                  | 当前实现状态                                                                                         | 验收状态                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| A：分层 Schema        | schema v2 target/source、必填 L2/L1 合同和 dispatch `l1_mode` 已接入 parser/builders/config          | 正反例与完整门禁证据待集成 owner 统一挂接                    |
| B：显式编译产物       | compiler/codec/Loader/Simulator clean boundary、调用点 alias、不可变身份与 target fingerprint 已落地 | compile→persist→独立重放证据待最终索引签署                   |
| C：Profile 与维护编译 | mode 绑定、await/frontier、排他、Profile/maintenance descriptor 已落地                               | §4.7 执行与篡改拒绝证据待最终索引签署                        |
| D：Arena 与准入       | Arena/view 两级生命周期、R lease、Grid Route、逐 Tile plan/commit 与分层亲和性已落地                 | L1-only 父 L2 稳定性、字节 copy 与边界回归待最终索引签署     |
| E：Backend 与 oracle  | 单 writer、逐成员 ACK、门禁/generation/cancel、Cache/Transfer/ByteStore 已落地                       | 恢复、DMA/维护/Gather coherence 及故障压力证据待最终索引签署 |
| F：CLI 与案例迁移     | compile-only、compiled-output/file、profile-bytes、四份 artifact 和 runner 已落地                    | compiler-free replay 与完整 examples corpus 待最终索引签署   |
| G：证据与文档         | trace/report/dump 已暴露 Profile、Arena、lease、hash 与 source_ref                                   | Perfetto SQL、完整回归及 T/RV/CB 逐项链接尚未最终签署        |

“已落地”表示当前源码存在并可由定向真实场景执行，不等于全部验收完成，也不等于 `simulator_experiment` 数值已经成为物理硬件承诺。

## 8. 验证、bring-up 和验收标准

### 8.1 参考验收意图的本地落点

沿用参考提案 T01–T32 编号，对照当前分层合同、普通 await、显式编译产物和真实配置/维护命令。下表是必须全部收口的验收矩阵；状态以最终逐 ID 证据索引为准，不能因单个 smoke 通过而整组标记完成。

| 验证组         | 参考测试                          | 当前合同要求                                                                         |
| -------------- | --------------------------------- | ------------------------------------------------------------------------------------ |
| 静态合法性     | T01、T02、T04、T06、T26、T27、T32 | 每层允许集合、实际调用跨层能力、逐Bank/R×L1/布局正确；不按ID或总容量猜兼容。         |
| 准入与补位     | T03、T05、T07、T08、T12           | L2/L1分别判定SAME/COMPATIBLE，原子最后资源竞争、同类FIFO/碎片、逐Tile补位与回滚。    |
| 有限工作退休   | T09、T10、T11、T13、T14           | ordinary await保持旧工作推进；Route非空不能报告Grid完成；跨等待点生产者环拒绝。      |
| 可见性         | T15、T24、T25                     | source consumed不等于目的可见，Cache hint不扣账，旧行场景由真实byte/版本oracle验证。 |
| Drain闭包      | T16、T17、T18、T19、T23、T31      | L1/L2分别重配，两方向、只读refill、L1维护产生下游写入、完成队列满均不提前Commit。    |
| 原子配置与故障 | T20、T21、T22、T28、T29、T30      | 等待与配置命令顺序、全成员ACK、旧代隔离、同配置不重Commit、唯一写者和取消确认。      |

### 8.2 分层配置与等待专项案例

| Case | 场景                                                 | 预期                                                                               |
| ---- | ---------------------------------------------------- | ---------------------------------------------------------------------------------- |
| RV01 | Context含L2 mode，两个Task分别请求L1.mode0/2         | 两级字段独立，Context不锁定整个调用的L1 mode。                                     |
| RV02 | Task请求不在程序L1允许集合                           | 编译/加载拒绝，不自动改请求或换程序。                                              |
| RV03 | 两层各自存在，但Task需要的L2路径不受父配置支持       | 跨层调用拒绝，不由Task覆盖父L2 mode。                                              |
| RV04 | 同Context等旧grid_done后改L1，父L2 Arena仍live       | L1完整事务可执行，L2 handle/generation/容量账本不变。                              |
| RV05 | 根仍持有L2 Arena却在体内要求改L2                     | 拒绝；源/前端需表达为不同根并使用Device等待与合法数据交接，Runtime不现场拆根。     |
| RV06 | 另一个根仍有活跃或可继续投递的L1工作                 | 本地await不能授权全层改档；Device序列化证明缺失就拒绝。                            |
| RV07 | 只等input_released/output_ready                      | 不能据此假定Task/L1已退休，切换准备验证失败。                                      |
| RV08 | 普通await已完成，但Cache refill/下游事务/ACK尚未结束 | 配置命令保持阻塞，后继Task不准入；不改普通await语义。                              |
| RV09 | 可执行产物有mode变化但无前置await或配置命令          | 拒绝；validator不合成缺失操作。                                                    |
| RV10 | Task基准0，当前L1为2且程序允许2                      | 编译绑定/运行准入可兼容复用，不切回0；真实依赖仍保留。                             |
| RV11 | Task多Grid竞争同一父调用R名额                        | 跨所有Grid计数，retire/安全取消才归还，free不提前归还。                            |
| RV12 | 取消已开始的配置事务，随后迟到ACK/DMA                | 错误收敛和隔离完成前不开门、不复用地址。                                           |
| RV13 | 初始配置/Registry与编译入口假设不一致                | 拒绝或走显式初始化/恢复流程，不伪造空await。                                       |
| RV14 | 只改一个层级                                         | 只改变该层配置generation；访问闭包包含必要上下游事务，但不要求无关存储无条件退休。 |

### 8.3 编译器与 validator 边界专项案例

| Case | 场景                                  | 预期                                                                              |
| ---- | ------------------------------------- | --------------------------------------------------------------------------------- |
| CB01 | 从执行产物删除一个必要RAW/WAR/WAW边   | Loader/只读verifier拒绝，运行前后依赖集合不被修改。                               |
| CB02 | 同一Context以不同实参alias关系调用    | 调用点产物独立或采用明确保守图；共享模板不被后一个调用累积改写。                  |
| CB03 | program_id/hash缺失或不匹配           | 拒绝，不使用程序名CRC或运行时自动编号回退。                                       |
| CB04 | 普通await含1个或多个事件              | 保留原事件语义和1:1 WAIT/WAITALL解码；不是补全另一组等待。                        |
| CB05 | 配置事务未出现在编译dump中            | Runtime不得生成，即使当前硬件空闲或目标容量更大。                                 |
| CB06 | 相同产物多次launch                    | 静态指令/依赖/hash不变；仅owner、launch generation、Slot/地址等合法动态字段不同。 |
| CB07 | 实际Host binding违反编译alias假设     | 加载拒绝/要求显式重编译，不在run中重建依赖。                                      |
| CB08 | 给validator高层源IR而不是指定编译产物 | 正式入口拒绝；显式开发工具的compile/run两步必须可观察。                           |

### 8.4 完成门槛与证据归档

最终接受必须同时满足：

- 普通 await 的源码形式与事件规则保持一致；Context 内 L1 切换、Device 层 L2 切换、同模式显式数据等待均有源码、compiled dump 与 trace 对应证据。
- 同层 Bank 配置一致，L1/L2 独立；每次实际改档等待全部受影响成员 ACK，generation、取消、Arena、lease、pin、Route 和在途事务无泄漏。
- 完整编译产物可审计；独立进程不导入 compiler 也能 parse/load/run，Runtime 不生成缺失依赖、等待、维护、配置或程序身份。
- 数据可见性使用小规模 `full_memory` ByteStore oracle；调度时序与数值正确性分开，不能用 metadata 或固定 profile 宣称真实 Cache 命中率、tensor 算术或 RTL 正确。
- 完整 regression、指定正负 example corpus、Perfetto parse/单位/无丢失审计和 T01–T32、RV01–RV14、CB01–CB08 逐项索引全部通过并归档。

本次文档更新按约束不运行验证命令，也不把集成过程中的定向结果改写为本文的完成声明。显式编译/持久化/独立重放、恢复、跨 L1 改档字节保持、DMA/维护/Gather coherence、完整 regression/corpus 和 Perfetto 审计都必须由集成 owner 在最终验收索引中链接真实命令、artifact、trace 与 oracle 输出；在该索引签署前不记录最终通过数，也不宣告整体接受。

## 9. 风险、取舍和后续细化方向

### 9.1 保持 await 简单的代价

源 IR 更容易读，但必须能检查编译器最终生成的配置序列；不能把真实安全动作藏回validator。普通await成功与全域静默分开会增加一条可观察的配置控制操作，这是硬件必需的事务，不是新的程序分组。

### 9.2 全层统一配置与并发

L1 mode由Task声明，不表示每Task或每Tile拥有自己的mode。V1在跨根L1改档时选择编译器显式序列化，可能损失并发；未经证明的局部await不能换取全域切换。若以后放宽，需要明确跨根同步/服务边界，而非直接扩大await含义。

### 9.3 资源、Cache与模型局限

Arena 保留、striped padding 和同类 FIFO 会带来容量/HOL 代价，Profile 亲和性不保证最优延迟。共享 Cache 仍有干扰、驱逐、带宽竞争与隔离风险，不提供每 Context 容量/命中/QoS 保证。ByteStore 只证明被实际 Transfer 覆盖的有效字节和已显式绑定的 Gather 源范围，不模拟 tensor 算术；非 oracle 路径使用保守 provenance。真实 Bank 端口、mapping、Cache 宏/Tag/ECC 面积、系统预留、维护能力和外部 Master 参与协议仍由后续硬件规格冻结。

### 9.4 不可删的运行时工作

静态 lowering 位于 `compiler/`，不等于可以删除真实资源竞争、动态 HBM 绑定、launch/event/profile 代数、return 退休、await registration fence、逐成员 ACK、取消隔离或 fault/reset。`execution_verifier.py` 能拒绝不完整产物，但只有 Runtime 真实计数与事务推进才能证明本次执行安全结束；禁止用固定延迟、无条件成功、默认全空或重放上一次成功状态代替。
