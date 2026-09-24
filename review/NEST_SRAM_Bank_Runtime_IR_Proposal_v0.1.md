# NEST/HSTI SRAM Bank Profile V1

## 调度器、Runtime 与 IR 资源合同设计 Proposal

| 项目         | 内容                                                                       |
| ------------ | -------------------------------------------------------------------------- |
| 文档版本     | v0.1 · 待评审提案                                                          |
| 日期         | 2026-09-15                                                                 |
| 主要范围     | SRAM Bank Profile 的执行管理、两级 Admission、安全切换、资源合同和验证接口 |
| 不在本版范围 | 编译器全局优化、Profile/Tiling 联合搜索、自动阶段合并、自动选择最优配比    |
| 设计优先级   | 数据正确性与可验证性 > 有界实现复杂度 > 混合执行性能                       |
| 交付性质     | 设计与实现约束，不是已经实现或已经通过 RTL/实机验证的系统                  |

> **核心方案：软件预先指定执行阶段的 Profile 和程序版本；阶段内部由 Runtime 动态 Admission；真正改变 SRAM 分区时，通过 HBM 物化、完成等待和排空建立安全边界。**
>
> `allowed_profiles` 用于判断程序版本能否在指定配置下运行，不授权 Context 自行修改物理分区。L2 与 L1 分别管理实际内存，Cache 默认共享，SPM 使用可核算的硬预留。

---

## 0. 阅读方式与决策状态

本文中的“必须”“禁止”是**本提案的规范性要求**。其中既有已经讨论一致的原则，也有为形成可实现 V1 而补充的具体选择；后者不冒充已经确认的历史决策。

### 0.1 沿用的已确认原则

| ID  | 原则                                                                                                       |
| --- | ---------------------------------------------------------------------------------------------------------- |
| B01 | 同一层级的统一配置范围内，所有 Bank 在同一时刻采用相同的 SPM/Cache Region 比例和边界。                     |
| B02 | SPM Region 与 Cache Data Region 同时存在；不同 Bank 可以分别服务两类请求，不需要每周期同步访问同一类区域。 |
| B03 | L1 与 L2 的 Profile 相互独立，不要求两层比例相同。                                                         |
| B04 | `nest.context` 具有自己的 L2 资源合同；Tile Program 具有自己的 L1 资源合同。                               |
| B05 | L2 可以知道 Tile 的资源需求，但实际 L1 地址和本地 Context Slot 由 Tile-local Admission 分配。              |
| B06 | SPM 容量属于硬资源；普通共享 Cache 的工作集预算不等于私有容量或命中保证。                                  |
| B07 | 改档时允许通过切开程序、将后继所需数据写回 HBM 并等待，换取简单、明确的数据安全边界。                      |
| B08 | V1 不要求携带存活 SRAM 数据在线重分区，不做活跃数据迁移。                                                  |
| B09 | 固定阶段 Profile 不妨碍多 Context 完成后动态补位；不改成固定批次执行。                                     |
| B10 | 当前先固定程序版本和执行计划，不讨论编译器如何寻找最优版本、最优 Profile 或最优阶段。                      |

### 0.2 本提案补充的 V1 实现选择

| ID  | 建议采用的具体选择                                                                                                 | 主要取舍                                         |
| --- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| V01 | 单个 Runtime Coordinator 是 Profile 配置的唯一写入者。                                                             | 避免多个 Context 或多个 Host 线程竞争改档。      |
| V02 | 每个配置作用域同一时刻只执行一个预先指定的 Memory Phase；Phase 内可有多个 Group/Tile Context。                     | 不在运行中跨 Phase 混排，简化配置一致性。        |
| V03 | 以有限、不可变的 Phase Manifest 描述本阶段根 Context；关闭登记后继续执行已登记工作。                               | 避免排空时禁止必要后续工作而自锁。               |
| V04 | SPM 采用 Context-lifetime Arena 硬预留；局部 Buffer 可复用，但 V1 不向其他 Context 提前借出尚未结束的 Arena 配额。 | 保守使用容量，换取无后续扩容的简单合同。         |
| V05 | 同阶段、依赖已就绪的资源准入队列使用确定性 FIFO；资源不足的队首不被后续候选越过。                                  | 接受 Head-of-line 性能损失，不引入在线选择优化。 |
| V06 | 真正改档前，保守排空受影响的执行闭包；包括可能继续向该层发请求的上游工作。                                         | 即使只改 L2，也不能忽略 Tile/L1 请求。           |
| V07 | 新 Profile 通过 Prepare/Commit/ACK 发布；所有成员确认前保持发射门关闭。                                            | 接口稍多，但避免部分 Bank 使用新配置。           |
| V08 | 提供按阶段、Context、资源等待原因划分的 Trace 和错误码。                                                           | 为后续优化留下可测量依据。                       |

`V04`、`V05` 是降低第一版实现复杂度的选择，不是架构永久限制。本文不为其性能最优性作保证。

---

## 1. 非目标与禁止隐含引入的能力

本版不实现：自动 Profile/Tiling 优选、任意百分比配置、每个 Context 独立改变 Bank 比例、Bank 内活数据迁移、共享 Cache 的按 Context 硬分区、一般化跨 Context 共享逻辑 L2 Buffer、一般化跨 Tile 协作屏障和抢占恢复。

保留现有的两级 Multi-context、SPMD Tile Program 和异步事件表达。**Memory Phase 不是新增的 Stream，也不是 Tile Grid 的 SPMD Epoch。**三者不混用。

同一物理 L2 资源池由多个 Context 使用，并不表示它们共享同一个逻辑 Buffer。V1 中不同 Group Context 之间的依赖数据默认通过 HBM 交接；同一 Group Context 内的多个 Dispatch 可以使用其自有 L2 Arena，但必须遵守事件和生命周期约束。

---

## 2. 四个必须分开的概念

| 概念            | 定义                                                                       | 例子                                         |
| --------------- | -------------------------------------------------------------------------- | -------------------------------------------- |
| Profile Domain  | 必须统一提交分区配置的所有 Bank 的集合。                                   | 本设计约定的整个 L2 配置域；整个 L1 配置域。 |
| Allocation Pool | 实际进行 SPM 容量分配的资源池。                                            | Group 的 L2 Pool；每个 Tile 自己的 L1 Pool。 |
| Memory Phase    | 使用预先指定的 Profile 向量、有限根 Context 集合和安全结束条件的执行阶段。 | `dense_0`、`mixed_1`。                       |
| Program Variant | 已确定代码、Tiling、布局和资源合同的程序版本。                             | `matmul_v0`，执行中不修改其 Tile 大小。      |

### 2.1 配置域与分配池不必一一对应

以所有 Tile 的 L1 Bank 使用统一配置为例：

```text
L1 Profile Domain：统一选择 L1_P75
    Tile 0 的 L1 Allocation Pool：独立分配本地 Arena
    Tile 1 的 L1 Allocation Pool：独立分配本地 Arena
    Tile 2 的 L1 Allocation Pool：独立分配本地 Arena
    Tile 3 的 L1 Allocation Pool：独立分配本地 Arena
```

所有 Tile 的比例相同，但各自剩余容量、已驻留 Context 和当前请求可以不同。**不能把四个 Tile 的空闲 L1 相加，然后用总数给某一个 Tile 做 Admission。**

V1 的 Domain 成员由目标硬件描述固定，不能由 Context 随意缩小。若当前约定是“同一级别全部 Bank 统一”，实现必须把该层所有相关 Bank 纳入成员集合，不能悄悄降级为每 Tile 独立 Profile。

### 2.2 L1 与 L2 独立配置，不等于切换时互不影响

一个阶段可以指定：

```text
L2 = L2_P100
L1 = L1_P75
```

只改变 L2 时可以保持 L1 Profile 的数值不变，但仍需排空会访问 L2 的 Tile 请求。只改变 L1 时，本版也不暂停正在执行的 Group Context 后直接重排其子任务；先使受影响的根 Context 正常结束。

**配置的独立性是数值独立；Drain 的范围由访问关系决定。**

### 2.3 标识符

| 字段                             | 用途                                     |
| -------------------------------- | ---------------------------------------- |
| `domain_id` / `pool_id`          | 分别定位配置域和实际分配池。             |
| `phase_id`                       | 区分执行阶段，不能用 Profile ID 替代。   |
| `profile_id`                     | 索引目标支持的离散物理配置。             |
| `profile_generation`             | 每次真正提交该 Domain 的新配置后递增。   |
| `context_id` / `tile_context_id` | 根 Context 与 Tile-local Context 身份。  |
| `lease_id` / `lease_generation`  | SPM Arena 与资源 Ticket 的生命周期身份。 |

两个相邻 Phase 可以使用相同 Profile，但仍有不同 `phase_id`。只有真正改档才增加 `profile_generation`。计数回绕必须在完全空闲时处理，不能让旧 Token 与新 Token 发生 ABA 混淆。

---

## 3. 硬件 Profile Registry

### 3.1 Registry 使用字节数和组织 ID，不使用运行时百分比运算

百分比是人类可读名称。实际 Profile 至少包含：

| 字段                                     | 语义                                              |
| ---------------------------------------- | ------------------------------------------------- |
| `profile_id`, `level`, `domain_class`    | 配置身份和适用范围。                              |
| `bank_bytes`, `bank_count_per_pool`      | 目标数据阵列几何结构。                            |
| `spm_bytes_per_bank`                     | 每个 Bank 内 SPM Region 的结束边界。              |
| `cache_bytes_per_bank`                   | Cache Data Region 大小。                          |
| `spm_mapping_id`, `cache_org_id`         | SPM 地址映射、Cache 索引/Tag/回填组织的合法配置。 |
| `allocation_granule`, `alignment`        | Arena 分配粒度和布局对齐要求。                    |
| `system_reserved_spm_per_bank`           | 不能分配给用户 Context 的 SPM 区域。              |
| `cache_write_policy`, `maintenance_caps` | Cache 写策略和可用清理/失效能力。                 |
| `registry_abi_version` / `registry_hash` | 防止程序镜像与设备配置表不匹配。                  |

必须满足：

```text
SPM 数据容量 + Cache 数据容量 = 可配置数据阵列容量
用户可用 SPM = SPM 数据容量 - 系统预留
```

Tag、MSHR、配置状态、完成队列等控制存储应单独核算。用于完成 Drain 和配置提交的控制状态不能依赖马上要失效的用户 SPM 区域。

90/10 等名称只能对应 Registry 中已经实现的离散档位。若边界不满足 SRAM/Cache 组织粒度，不得通过四舍五入在 Runtime 临时制造一个档位。

### 3.2 示例几何结构：只用于说明和测试

| 层级 | Pool 组织               | Profile | 每 Bank SPM | 每 Bank Cache | 每 Bank 系统 SPM 预留 |
| ---- | ----------------------- | ------- | ----------: | ------------: | --------------------: |
| L2   | 8 Bank × 64 KiB         | L2_P100 |      64 KiB |             0 |                 4 KiB |
| L2   | 同上                    | L2_P75  |      48 KiB |        16 KiB |                 4 KiB |
| L1   | 每 Tile 4 Bank × 32 KiB | L1_P100 |      32 KiB |             0 |                 2 KiB |
| L1   | 同上                    | L1_P75  |      24 KiB |         8 KiB |                 2 KiB |

这些容量、Bank 数量和比例不是最终硬件规格。下文所有数值例子均沿用此测试几何结构。

### 3.3 P100 的含义

P100 只表示用户数据阵列配置为 SPM、该层 Cache Data Region 为零；它不表示“所有指令和访存路径都自动支持 Cache 被关闭”。

镜像加载时必须验证该阶段涉及的程序、隐式访存路径及底层控制功能不依赖被禁用的 Cache。只检查当前执行的是不是 MatMul 不够。

---

## 4. 调度器与 Runtime 架构

```text
Host / Device Runtime
    Plan Loader + Validator
    Phase Coordinator（唯一 Profile 写入者）
    HBM Buffer / Completion 管理
                |
                v
Group Scheduler
    Phase/Dependency 登记队列
    L2 Admission + L2 Arena Allocator
    Multi-context Sequencer / Ready-action 发射
    Grid 生命周期与事件聚合
                |
                v
Tile Scheduler（每 Tile）
    Tile-local Admission + L1 Arena Allocator
    Tile Context Slot + 本地程序执行
    LSU / DMA / MMA 等请求与完成事件
                |
                v
Memory Controllers / Fabric
    Bank 仲裁、Cache/Tag/MSHR、DMA、HBM 事务
    Outstanding Accounting + Maintenance + Profile Commit ACK
```

### 4.1 责任矩阵

| 组件              | 必须负责                                                       | 不负责                                    |
| ----------------- | -------------------------------------------------------------- | ----------------------------------------- |
| Plan Loader       | 目标 ABI、Profile/版本兼容性、资源边界、Phase 依赖合法性检查。 | 不搜索最优 Tiling。                       |
| Phase Coordinator | 阶段顺序、关闭登记、Drain、Profile 事务、故障收敛。            | 不给 Tile 分配实际 L1 地址。              |
| Group Admission   | 原子预留 L2 Arena、Group Slot、必要元数据。                    | 不提前占住所有 Tile 的 L1。               |
| Group Sequencer   | 只发射依赖和资源条件满足的动作，维护 Context PC 和等待状态。   | 不直接修改 Bank Profile。                 |
| Tile Admission    | 原子预留本地 Slot、L1 Arena 和本地控制资源。                   | 不替全层选择 Profile。                    |
| Memory Controller | 处理访存、完成统计、Cache Maintenance、配置提交。              | 不解析整个算子 DAG 或决定下一阶段跑什么。 |

### 4.2 与 Multi-context / Ready-action 的衔接

一个 Group Context 的 `await` 只挂起该 Context 的推进，不是整个 Group 的 Barrier。其他 Context 的就绪动作仍可发射。

Ready-action 只解决“哪个动作现在可以执行”；它不能绕过 Profile、Arena 生命周期和阶段门禁。原型可保留每 Group 每周期发射一条控制动作的基线；发射宽度是目标参数，不影响本文协议。

V1 不要求跨越未分析的副作用重排同一 Context 中的动作。动作是否可越过前项，由现有 IR 的依赖和顺序语义决定，而不是只看某个 Event 已经就绪。

---

## 5. 执行计划与 Memory Phase

### 5.1 执行计划由软件显式给出

```text
Plan
  Phase E0：{L2_P100, L1_P75}，运行已指定的 MatMul 版本
  Phase E1：{L2_P75,  L1_P75}，运行已指定的 Gather 和 MatMul 版本
```

配置可由手工、配置文件或现有工具生成。Runtime 只验证并执行，不因“当前空闲比较多”临时切到另一个 Profile，也不临时选择更大 Tile。

外部可以提交下一阶段的描述符，但在该阶段激活前，它们只能占用有界的元数据空间，不能占用当前阶段的 L2/L1 Arena、执行 Slot 或数据面请求资源。

### 5.2 Phase Manifest

每个 Phase 至少包含：

```text
phase_id
profile_vector：每个相关 Domain 的确切 Profile ID
root_contexts：有限的 Context 调用、程序版本与依赖
profile_domain_members：来自目标描述，不允许 Context 更改
boundary_policy = HBM_QUIESCENT
export_bindings：需要保留的 HBM 输出及范围
next_phase_id / terminal
```

Manifest 安装后不可修改。Dynamic Shape 的实际参数可以在启动前绑定，但必须通过既有资源上界检查；不允许运行到一半后突破合同。

### 5.3 阶段内动态补位

阶段内可同时有多个 Group Context；每个 Group Context 又可以投递多个 Tile Task。一个 Task 完成后，相应 Tile 可以接纳后续 Task，不要求等其他 Tile 一起结束。

**四个 Group Context 不等于四个 Tile Context。**例如四个根 Context 各向四个 Tile 发一个 Task，可以形成 16 个 Tile Context；它们的 L2、L1 和 Slot 必须分别计账。

### 5.4 `seal` 不等于立即禁止所有 Admission

这是本提案中最重要的调度语义之一：

```text
OPEN：可以登记本 Phase Manifest 中尚未登记的根 Context。
SEALED_EXECUTING：不再接受新的根 Context；
                 但已登记、尚未准入的根 Context 继续准入；
                 已准入 Context 的 Prefetch、Tile Dispatch、Store 继续推进。
DRAINING：本阶段逻辑工作已经全部结束后，
          才彻底关闭新工作发射，只保留完成和 Maintenance 路径。
```

否则会产生：关闭准入 → 已登记消费者不能启动 → 旧工作无法完成 → 永远无法切换。

V1 要求有限 Manifest 在 `seal` 前登记完成。已经接受的工作不得因为后续有一个 Profile 切换请求而被悄悄丢弃或转移到新阶段。

---

## 6. 资源合同的语义

### 6.1 程序合同与调用绑定分离

| 对象                     | 存放的内容                                                      |
| ------------------------ | --------------------------------------------------------------- |
| Program/Context Contract | 资源需求、合法 Profile、布局、所用 Cache 路径、输出完成要求。   |
| Invocation Binding       | `phase_id`、本次选定的 Profile 向量、具体程序版本、参数、依赖。 |
| Runtime State            | 当前物理 Profile、Generation、剩余资源、队列和租约。            |

`preferred_profile` 可以保留为注释/未来提示，但 V1 的执行逻辑不读取它作选择。真正执行依据是 Phase 的指定配置。

### 6.2 `allowed_profiles` 的含义

`allowed_profiles` 表示这个**具体编译版本及其布局**在对应档位下合法。它可由静态检查或手工配置加验证得到，不要求逐个档位实机试跑。

兼容性必须检查逐 Bank 容量、对齐、映射、隐式访存需求和动态参数边界。实机 Profiling 决定性能偏好，不替代资源合法性检查。

Profile ID 没有大小兼容关系。SPM 增大同时可能使 Cache 缩小，因此禁止使用 `target_profile >= required_profile` 这样的比较。

多层环境若存在组合约束，不能简单对 L1/L2 的允许集合做笛卡尔积。程序镜像应提供已验证的 `compatible_environment` 表，或者显式证明两层约束可独立组合。该表只做合法性检查，不包含优化算法。

### 6.3 SPM 合同：V1 使用完整 Arena 预留

| 字段                                           | 要求                                                       |
| ---------------------------------------------- | ---------------------------------------------------------- |
| `spm_bytes_per_bank` / `spm_bank_requirements` | 包含输入、输出、临时值、缓冲份数、Padding 和仍在途的数据。 |
| `allocation_kind`                              | V1 建议 `striped_arena`。                                  |
| `alignment`, `layout_id`                       | 与程序访问方式一致，不能只检查总字节数。                   |
| `reservation_lifetime`                         | `group_context` 或 `tile_context`。                        |
| `growth_policy`                                | V1 固定为 `forbidden`。                                    |
| `max_live_grids`, `event_slots`                | 必需的有界控制元数据。                                     |

以每 Bank 相同长度的 Striped Arena 为最简单实现：所有 Bank 分配同一段 Row 范围，程序在 Arena 内使用确定布局。若需求不均匀，V1 可按最大 Bank 需求向上取整后统一预留，接受内部浪费。

分配器可以使用固定粒度 Bitmap/First-fit；实际成功条件包含连续区间和对齐。总空闲字节足够但没有合适跨度时，应报告 `WAIT_FRAGMENTATION`，不能错误承诺一定能分配。

### 6.4 `nest.release` 与 Arena 配额

本提案明确区分两个动作：

```text
Buffer release：该逻辑 Buffer 不再可访问，其内部跨度可以供同 Arena 后续使用。
Arena release ：Context 全部退出、在途引用归零后，配额交回共享 Admission Pool。
```

V1 中 `nest.release` 不意味着其他 Context 可以立即借走整个预留预算。这样 Context 后续阶段无需再次申请更大的 Arena，不会因扩容等待持有其他资源。

这是本提案为简化实现增加的保守策略。若现有实现已经支持 Buffer 级全局回收，需显式区分两种 Allocator 策略；不能在同一合同下混用、双重扣账。

### 6.5 Cache 合同：默认共享，不做虚假的容量保证

| 字段                       | 语义                                                  |
| -------------------------- | ----------------------------------------------------- |
| `cache_required`           | 是否要求该层 Cache 路径启用。                         |
| `cache_access`             | `none`、`read` 或目标确实支持的 `read_write`。        |
| `cache_target_bytes`       | 性能预算/工作集提示，V1 不作为私有硬配额。            |
| `cache_budget_kind`        | V1 固定为 `hint`。                                    |
| `sharing`                  | `pool_shared`。                                       |
| `max_outstanding_requests` | 可选的有界发射上限；不是 MSHR 或 Cache 容量独占保证。 |

多个 Context 共用同一 Pool 的物理 Cache；L1 共用范围是对应 Tile 的本地 Context，L2 共用范围是实际共享该 Pool 的 Context。统一 Profile 不会把多个 Tile 的物理 L1 合成一个共享数据池。

`cache_target_bytes` 不进行私有容量扣减，不保证命中，也不能简单相加后当作正确性门槛。硬分区/驻留控制需要额外机制；Linux resctrl 将共享、独占分配和 Pseudo-Locking 区分为不同能力，可作为语义区分的参考。[R4]

### 6.6 跨层级 Cache 需求

Tile Program 通过 L1 Cache → L2 Cache 路径访问 HBM 时，合同不能只写“需要 L1 Cache”而忽略 L2。程序的环境约束必须明确哪一层要求非零 Cache、哪些层允许 Bypass。

本版不自动推导新的 Bypass 实现。目标没有声明某种路径时，Loader 必须拒绝，而不是假设关闭该层 Cache 仍能工作。

---

## 7. Group/L2 Admission 协议

### 7.1 Context 状态

```text
REGISTERED
    ├─ WAIT_PHASE：所属 Phase 未激活
    ├─ WAIT_DEPENDENCY：前驱的 context_done / HBM 可见事件未到
    └─ READY_FOR_ADMISSION
              ↓
       WAIT_CAPACITY / WAIT_FRAGMENTATION / WAIT_SLOT
              ↓ 原子预留成功
       PENDING_ACTIVATION
              ↓
       ACTIVE ↔ WAIT_EVENT / WAIT_TILE / WAIT_ENGINE
              ↓
       RETIRING（最终写回与引用收敛）
              ↓
       DONE

任何合法状态均可能转入 FAULT；故障收敛不等同于正常 DONE。
```

等待准入的根 Context 只能占用登记元数据，不得预先扣 L2/L1 Arena、执行 Context Slot 或 DMA 发射资源。`PENDING_ACTIVATION` 已持有提交成功的 Ticket，必须计入所有容量和 Drain 统计。

### 7.2 检查顺序

1. 检查 Phase 是否已经激活、程序/合同/参数是否匹配。
2. 检查本次指定 Profile 是否属于该版本允许集合，以及多层组合是否合法。
3. 检查跨 Context 的数据依赖是否完成。
4. 在 Group Admission 锁或等价硬件串行事务内，计划 L2 Arena、Group Slot、Event/Grid 元数据。
5. 全部可满足才 Commit；任何失败都回滚，不发布可执行 Context。
6. 将 Ticket 放入 Activation Queue。原型可规定最早下一个调度 Tick 发射，保持周期行为可重复。

当前 Phase 指定配置与实际硬件配置不同，应报告 `CONFIG_MISMATCH` 并停止发射；不能把它当作普通容量等待。

### 7.3 原子准入伪代码

```python
# 协议伪代码：plan_bundle/commit_bundle 由实际 Allocator 实现。
def admit_group(invocation, runtime):
    if invocation.phase_id != runtime.active_phase_id:
        return WAIT_PHASE
    if not invocation.dependencies_satisfied():
        return WAIT_DEPENDENCY

    with runtime.group_admission_transaction():
        # 同时检查 Manifest 成员身份、尚未准入状态及阶段门禁。
        # OPEN_EXECUTING / SEALED_EXECUTING 可准入；Drain 后禁止。
        if not runtime.may_admit_registered_root(invocation):
            return INVALID_INVOCATION_STATE
        if runtime.hardware_profile_vector != invocation.assigned_profiles:
            return CONFIG_MISMATCH
        if not invocation.contract.accepts(invocation.assigned_profiles):
            return INVALID_PLAN

        bundle = runtime.plan_bundle(
            l2_arena=invocation.contract.spm,
            group_slot=1,
            control_records=invocation.contract.control_resources,
        )
        if not bundle.possible:
            return bundle.wait_reason

        ticket = runtime.commit_bundle(bundle)
        # Ticket 绑定 phase_id、profile_generation、lease_generation。
        runtime.pending_activations.publish(invocation, ticket)
        return ADMITTED
```

Publish 与 Commit 必须构成可回滚或不可分割的事务。不能出现 Arena 已分配，但队列插入失败后无人释放的路径。

### 7.4 不可能满足与暂时不足要分开

| 情况                                          | 处理                                        |
| --------------------------------------------- | ------------------------------------------- |
| 请求在空 Pool 中也超过目标 Profile 的可用容量 | `RESOURCE_EXCEEDS_PROFILE`，加载/提交失败。 |
| 合同与 Bank 数量或映射不符                    | `INVALID_LAYOUT`，不能等待。                |
| 其他 Context 暂时占用容量                     | `WAIT_CAPACITY`。                           |
| 总字节足够，但当前没有对齐的连续跨度          | `WAIT_FRAGMENTATION`。                      |
| Slot 或有界控制元数据暂时不足                 | `WAIT_SLOT` / `WAIT_CONTROL_RESOURCE`。     |

资源等待队列按容量/Slot/元数据释放事件重新检查，不要求每个 Tick 对所有等待者忙轮询。Activation 和资源释放可以触发局部唤醒。

---

## 8. Tile/L1 Admission 与 Dispatch

### 8.1 Group 不替 Tile 做实际 L1 分配

Group Dispatch 携带：程序版本、参数、L2 Buffer 句柄、依赖、L1 合同引用、目标 Tile 集合和父 Phase 身份。

不携带由 L2 分配的实际 L1 地址。继续保留“Tile 自己发起 Load”的边界；不因为引入 Profile 而让 L2 保存每个任务的 SPMD ID。逻辑工作索引仍由现有 SPMD 下发路径提供。

### 8.2 每个 Tile 的两阶段提交

```text
PLAN：
    选择空闲 tile_context_id
    检查父 Phase、Profile Generation 和依赖
    计划完整 L1 Arena、Frame 和本地事件资源

COMMIT：
    分配 L1 Arena
    Pin 该 Task 必需的父 L2 Buffer
    绑定 Tile Context Slot / 程序 Frame
    发布可执行任务
```

任一步骤失败，撤销本轮全部分配和绑定。等待 L1 的 Task 可以持有有界 Route 元数据，但不能部分持有 L1 Arena 或 Tile Slot。

父 Group Context 已取得自己的 L2 Arena 并等待 Tile，是允许的。前提是 Tile Task 在启动前已经取得完整本地资源，并且已运行 Task 不会等待尚未准入的其他根 Context 才能退出。

### 8.3 V1 的前向进展边界

本版只支持独立可完成的 SPMD Tile Task，Group 在外部聚合其结果。禁止单个 Tile Task 占住 Slot 后，执行必须等待一个尚未驻留 Tile Task 的跨 Tile Barrier。

每个已启动 Tile Task 所需的输入、输出和临时空间必须位于已批准的资源合同中。不能执行到后半段再要求额外 L1 扩容。

新 Profile 的工作不得占用旧 Phase 必需的 Route/Completion 资源。完成事件、最终 Store 和 Maintenance 必须具有可前进的控制/事务通道；不能让完成消息排在无法发射的未来阶段任务之后。

### 8.4 根 Context 与子任务计数

根 Context 必须统计未投递、已投递未启动、正在执行以及正在退休的子任务。`grid_done` 只有在其所有 Task 都已退休或发生明确失败时才能解析。

不能仅用“当前活动 Tile Slot 为零”判断 Grid 已完成：Route Queue 中可能仍有尚未启动的任务。

---

## 9. 队列与发射规则

### 9.1 队列分离

| 队列/状态集            | 内容                                   | 是否持有执行资源                           |
| ---------------------- | -------------------------------------- | ------------------------------------------ |
| Future Phase Metadata  | 后续阶段调用描述符。                   | 否。                                       |
| Dependency Wait Set    | 当前阶段前驱尚未完成的根调用。         | 否。                                       |
| Ready Admission FIFO   | 依赖已满足、等待 L2 Admission 的调用。 | 否。                                       |
| Pending Activation     | 已提交资源 Ticket，等待真正开始执行。  | 是。                                       |
| Active Group Contexts  | 正在运行或等待事件的 Group Context。   | 是。                                       |
| Tile Route Queue       | 已登记的 Tile Task。                   | 仅有界队列元数据；未准入前不占 L1/Slot。   |
| Completion/Maintenance | 返回、回收、排空和错误处理。           | 使用保留的控制资源，不被普通提交队列堵塞。 |

FIFO 规则针对已经进入 Ready Admission FIFO 的候选。依赖未满足的调用不占据该 FIFO 队首；这样不会因为源文件中的登记顺序而阻塞其尚未登记/尚未执行的生产者。

资源不足的就绪队首不被后来的更小请求越过。此处选择确定性和简单性，明确接受可能的空闲容量和 Head-of-line 等待。

### 9.2 同时到达事件的确定顺序

软件模拟器建议固定以下 Tick 顺序，用于可重复验证：

```text
1. 处理 Fabric/Engine 返回，更新依赖和资源释放。
2. 使之前提交成功的 Activation 生效。
3. 已活动 Group/Tile 发射满足条件的动作。
4. 重试被资源变化唤醒的 Admission，成功项进入下轮 Activation。
5. 更新 Phase 计数，推进 Drain/Maintenance 状态机。
```

具体 RTL 可以采用不同流水划分，但可见事件顺序必须符合等价协议。本文不要求五个步骤在一拍组合完成。

---

## 10. HBM 数据交接与事件语义

### 10.1 完成事件不是同一个东西

| 事件             | 必须表示什么                                                               | 不能推出什么                             |
| ---------------- | -------------------------------------------------------------------------- | ---------------------------------------- |
| `input_released` | 对指定输入的所有相关消费者已不再访问；跨 Grid 时必须聚合全部 Task。        | 不能仅凭第一个 Tile 的完成释放共享输入。 |
| `output_ready`   | 声明范围的输出已在声明的目的层级可被后续动作读取。                         | 不自动代表 HBM 写回完成。                |
| `tile_done`      | Tile Task 的计算与相关本地请求完成，L1 引用和 Slot 可按规则退休。          | 不自动代表整个根 Context 完成。          |
| `grid_done`      | Grid 全部 Task 退休，聚合事件已经收敛。                                    | 不等于 Host 可消费最终结果。             |
| `hbm_visible`    | 指定结果已完成必要写回和可见性处理，后继可从 HBM 路径正确读取。            | 不代表整个配置域没有其他在途请求。       |
| `context_done`   | 该根 Context 全部子任务结束、必需 HBM 输出可见、用户 Arena/Ticket 已释放。 | 不等于可以立刻修改全域 Profile。         |
| `phase_done`     | 全阶段登记工作完成，Drain/Maintenance 结束，受影响域满足静默条件。         | 后续改档仍需 Prepare/Commit/ACK。        |

必要时把 `source_consumed` 与 `destination_visible` 表达成不同完成等级。DMA 已读完源 Buffer，不等于目的端已经可供消费者观察。CUDA 异步复制文档也区分了源侧读取完成等待和目的数据可用的同步场景；这一点只作为事件语义设计的参考。[R3]

### 10.2 HBM 边界的精确定义

“全部写回 HBM”指：**所有后继或 Host 需要的数据都已经持久于本次 SRAM 重配置之外的后备存储路径，并对目标消费者可见**。不要求把已死亡的临时值和只读 Cache 副本也复制一遍。

这里的“持久”仅指不依赖马上失效的 SRAM 数据，不是断电持久化承诺，也不要求软件观察 DRAM 芯片单元何时完成内部写入。

Runtime 不能猜测哪些 Buffer 是输出。HBM 地址、有效范围、写回来源和完成事件由调用参数及合同明确提供。

调用绑定还必须声明 HBM 读写范围及必要依赖。同阶段多个 Context 可以只读同一份 HBM 数据，但不能无依赖地重叠写同一范围，或在生产者未完成时读取其输出。V1 假设受信任的同一地址空间，或由既有地址翻译/权限机制隔离；共享 Cache 本身不是访问控制机制。

内存 Fence 主要处理顺序，不能单独替代写回完成和可见性保证。CUDA 官方文档对 Fence 也作了这一明确区分。[R2]

### 10.3 普通 DMA 写回与共享 Cache 一致性

Producer 通过 DMA 更新 HBM 后，Consumer 的 Cache 可能仍持有该地址的旧副本。**改档协议正确，不等于跨 Context 的数据交接已经自动一致。**

目标 Runtime Backend 必须提供以下二者之一：

- 已定义且经过验证的一致性路径；或
- 对输出范围执行 Clean/Invalidate 与 Release/Acquire 等目标相关操作，使 `hbm_visible` 在声明的消费者范围内成立。

如果硬件只有整 Cache 清理能力，并且无法在其他 Context 运行时安全清理，V1 必须把这样的 Producer–Consumer 交接拆到不同 HBM-Quiescent Phase。不能在同一 Phase 的示例中默认存在未实现的细粒度一致性。

L1/L2 Cache 的写策略尚需目标明确：只读 Cache 不需要脏行写回，但仍需处理旧 Tag、在途 Miss 和回填；Write-back Cache 还必须清理脏行。合同不能隐含选择写策略。

---

## 11. Phase 收敛与 Drain 状态机

### 11.1 状态定义

| 状态                | 允许的动作                                                         | 退出条件                                    |
| ------------------- | ------------------------------------------------------------------ | ------------------------------------------- |
| `CONFIGURING`       | 只有配置/控制路径。                                                | 所有 Profile 提交成功。                     |
| `OPEN_EXECUTING`    | 当前 Manifest 根登记、Admission、子任务和普通执行。                | 收到合法 Seal。                             |
| `SEALED_EXECUTING`  | 不接受新根登记；继续已登记根的 Admission、已有工作的全部后续动作。 | 当前 Phase 根调用全部正常退休，或转 Fault。 |
| `EXPORT_DRAIN`      | 收敛被跟踪的最终 HBM 发布、Store 队列和完成事件。                  | 必需输出全部可见。                          |
| `CACHE_MAINTENANCE` | 只允许旧 Generation 的 Clean/Invalidate 和其事务。                 | 所有上游及下游 Cache Maintenance 完成。     |
| `FABRIC_DRAIN`      | 消费残留返回与确认，不发射用户请求。                               | 所有相关请求、回填、写缓冲、队列统计归零。  |
| `QUIESCENT`         | 发射门保持关闭，允许开始下一配置事务。                             | 发布 `phase_done`，等待下一阶段。           |
| `FAULT`             | 停止新用户工作，仅执行诊断、受控收敛或复位。                       | 明确恢复成功或终止请求。                    |

`EXPORT_DRAIN` 在所有 `context_done` 已具备严格 HBM 可见语义时应很快通过，但仍保留为防止计数遗漏的检查层。

### 11.2 可结束执行阶段的条件

```text
sealed == true
registered_root_count == expected_root_count
completed_root_count == expected_root_count
failed_root_count == 0
waiting_dependency_count == 0
ready_admission_count == 0
pending_activation_count == 0
active_group_count == 0
unissued_child_count == 0
queued_or_active_tile_task_count == 0
live_user_l2_lease_count == 0
live_user_l1_lease_count == 0
```

失败不能通过减少 `expected_root_count` 假装成功。取消的 Phase 应产生独立的失败/取消完成结果。

### 11.3 真正改档前的额外静默条件

```text
all_required_exports_visible
all_source_consumption_refs_released
dma_queued_or_outstanding == 0
cache_miss_and_refill_outstanding == 0
write_buffer_outstanding == 0
fabric_request_and_response_refs == 0
maintenance_active == false
all_member_controllers_quiescent_ack == true
```

计数必须在请求对外发布前增加，在其真实完成/取消确认后减少。只统计“已经进入 HBM Controller 的请求”不够：还在 LSU、DMA Queue、NoC 或回填流水中的请求也受影响。

### 11.4 上下游 Maintenance 顺序

V1 按请求可能流动的方向处理：先阻止新的用户生成，再收敛上游，随后清理下游，最后等待 HBM 可见和全路径归零。

例如允许 L1 脏数据写到 L2 时，不能先宣布 L2 Clean 完成，再让 L1 开始把脏行写回 L2。保守流程为：

```text
全部相关根 Context 结束
    → L1 Clean/Invalidate 完成
    → 等待其下游事务收敛
    → L2 Clean/Invalidate 完成
    → 等待 HBM 写回与 Fabric 全部收敛
```

若目标只读或 Write-through，Backend 可以使相应 Clean 步骤成为空操作，但必须保留完成和统计语义。

### 11.5 SPM 增大和减小使用同一协议

```text
SPM 减小：回收原 SPM，不能仍有存活 Buffer 或 DMA 引用。
SPM 增大：回收原 Cache，不能仍有旧 Miss 回填或脏数据。
```

V1 不根据增减方向省略同步。CUDA Runtime 文档也说明不同 Cache 配置偏好可能插入设备侧同步点；这里只说明配置切换不应被视为天然免费，不推断其内部 Bank 实现。[R1]

---

## 12. Profile 切换事务

### 12.1 单写入者与配置所有权

同一个 Profile Domain 必须只有一个 Coordinator 持有配置 Lease。多个 Host 请求可以排队，但不能并发执行冲突的 Phase。Host 软件、Group Sequencer 和 Tile Program 不得绕过 Coordinator 写 Profile 寄存器。

配置 Lease 只保护写入权，不等于持有 Allocator 的临界区锁。等待 Drain 或硬件 ACK 时不得持有会阻塞资源释放、完成队列或旧任务推进的锁。同一 Coordinator 连续执行下一阶段时可沿用已有配置所有权，不再次竞争自己的 Lease。

如 CPU、GPU、调试 DMA 或其他外部 Master 也能访问该域，必须加入 Quiescence 协议，或者由系统保证本执行期间它们不访问。未计入的外部写入会使 Drain 证明失效。

### 12.2 Prepare / Commit / ACK

```text
1. Acquire：获取所有相关 Domain 的配置所有权，按固定 Domain 顺序防止锁环。
2. Validate：检查目标 Profile、环境组合和硬件 Registry ABI。
3. Quiesce：取得旧 Phase 的 phase_done / 初始空闲证明。
4. Prepare：写入所有成员的 Shadow 配置，不允许用户请求观察新配置。
5. Ready ACK：各 Bank/控制器确认能提交。
6. Commit：提交新配置；用户发射门仍关闭。
7. Commit ACK：确认所有成员已使用新边界、映射和 Cache 组织。
8. Publish：更新 profile_generation，建立新 Allocator 容量视图。
9. Open：激活新 Phase，开放其 Admission 和发射。
```

首次启动也必须取得经过验证的 Reset/空闲状态或完成一次维护，不能仅因尚未运行本 Plan 就假定 Cache 和在途请求为空。

物理配置寄存器不必在同一纳秒翻转；**所有成员确认前保持用户路径关闭**，即可保证外部可见的原子性。门禁和跨时钟域 ACK 仍需硬件验证。

L1 与 L2 可以在同一停止执行窗口内分别 Commit，不要求它们选择相同比例。整个新阶段必须等其所有必需 Domain 都确认后再开放。

### 12.3 无实际配置变化

若前后阶段 Profile 向量相同，跳过 Profile Register Commit，不增加 Generation。

但显式指定的 `HBM_QUIESCENT` Phase 边界仍然执行；V1 不自动消除或合并它。若手工计划希望保持连续动态补位，应把兼容工作放在同一个 Phase，而不是期待 Runtime 偷偷跨越显式安全边界。

### 12.4 失败处理

Prepare 前失败可以释放配置 Lease 并拒绝新阶段。已经开始 Commit 后出现部分成员失败，必须保持发射门关闭，报告 `PROFILE_COMMIT_FAILED`，进入受控 Domain Reset/重初始化。

不得直接写回旧 Profile 然后假装旧数据仍然存在。旧 Cache 可能已经失效，映射可能部分更新，恢复需要明确的协议。

所有 Ticket 和相关请求携带 Generation。发现旧代请求不是“丢掉日志就算完成”：若它仍可能写内存，应隔离并进入 Fault，防止旧回填破坏新 SPM。Generation 检查是检测保护，不是替代排空。

---

## 13. 进展保证与死锁边界

### 13.1 本版要消除的等待环

| 风险                                                 | 本版规则                                               |
| ---------------------------------------------------- | ------------------------------------------------------ |
| 旧 SPM 数据要等新 Cache 模式消费者才能释放。         | 后继数据先物化到 HBM，Phase 边界不保留用户 SRAM 引用。 |
| 关闭阶段后禁止必要子任务启动。                       | Seal 仅关闭新根登记；已登记工作继续执行。              |
| 等待资源的根 Context 已占部分 Slot/SPM。             | 未成功 Commit 的 Admission 不持有执行资源。            |
| Context 后半段要求额外内存，持有前半段资源等待扩容。 | 完整 Arena 预留，V1 禁止增长。                         |
| 活跃 Tile 等待未驻留 Tile 的 Barrier。               | V1 不接受需要 Gang Residency 的 Tile 协作语义。        |
| 后续 Phase 队列堵住旧工作的完成消息。                | 元数据、就绪任务与完成/维护路径分离。                  |
| 旧 Phase 依赖未来 Phase 才会生成的事件。             | Loader 拒绝逆向 Phase 依赖和未闭合的事件来源。         |

### 13.2 仍然依赖的硬件和程序前提

有限 Manifest、可终止的程序、合法依赖、最终可返回的内存事务、公平或有界服务的仲裁、可前进的完成/写回路径，是本方案正常结束的前提。

Capacity Admission 本身不能证明整个 Fabric 无死锁。NoC/LSU/Cache 的信用环路、协议重试和错误恢复仍需单独验证。建议为 Completion、Maintenance 及必要的最终写回保留逃生资源，且不得被未来 Phase 的普通工作耗尽。

如果最老等待项在 Pool 空闲时仍不能准入，或当前 Phase 没有可执行工作且等待事件不会再被产生，应报告诊断错误，而不是无限等待。

### 13.3 超时和取消

超时只触发诊断/取消请求，不能立即回收仍可能被 DMA 访问的 Arena。必须获得取消完成、隔离确认或足够范围的 Reset ACK，才能释放资源并复用地址。

取消后未完成的输出标记无效，后继不发射。V1 不承诺恢复取消时的中间 SRAM 状态。

---

## 14. IR 资源合同与阶段操作

> 本节是**拟议的 NEST/Nexus 自定义 IR 语义草案**。操作名、属性名和汇编形式尚未注册到现有 MLIR Dialect；示例不能直接交给现有 `mlir-opt` 解析。实现时应保留语义，再确定 ODS/Assembly Format。

### 14.1 新增对象的最小集合

| IR 对象                       | 作用                                              | 执行者                             |
| ----------------------------- | ------------------------------------------------- | ---------------------------------- |
| `nest.memory_profile`         | 引用/描述目标 Registry 中一个合法档位。           | Loader 验证，不由 Context 执行。   |
| `nexus.environment`           | 显式 L2/L1 Profile 向量。                         | Phase Coordinator。                |
| `nest.resource_contract`      | SPM、Cache、布局和控制资源合同。                  | Group/Tile Admission。             |
| `nexus.phase.async`           | 描述有限 Phase Manifest，返回 `phase_done` 事件。 | Device Runtime。                   |
| `nexus.phase.seal`            | 关闭该 Phase 的根登记，保留已登记工作的推进权。   | Phase Coordinator。                |
| `nexus.submit_context.async`  | 把调用登记到所属 Phase，不代表资源已经分配。      | Device Runtime / Group Scheduler。 |
| `nest.store.async` 的完成等级 | 指定最终 Store 需要达到 `hbm_visible`。           | DMA/Memory Backend。               |

不提供用户 Context 可执行的 `set_bank_profile` 指令。实际改档是 Phase Coordinator 的受保护操作，不是普通 Tile/Group 指令。

### 14.2 Profile 与环境声明

```mlir
// 语义草案；Registry 还必须包含 cache_org、映射和 ABI 等字段。
nest.memory_profile @L2_P100 {
  level = "L2", bank_count = 8,
  bank_bytes = 65536, spm_bytes_per_bank = 65536,
  cache_bytes_per_bank = 0, system_spm_bytes_per_bank = 4096
}
nest.memory_profile @L2_P75 {
  level = "L2", bank_count = 8,
  bank_bytes = 65536, spm_bytes_per_bank = 49152,
  cache_bytes_per_bank = 16384, system_spm_bytes_per_bank = 4096
}
nest.memory_profile @L1_P75 {
  level = "L1", bank_count = 4,
  bank_bytes = 32768, spm_bytes_per_bank = 24576,
  cache_bytes_per_bank = 8192, system_spm_bytes_per_bank = 2048
}

nexus.environment @ENV_DENSE { l2 = @L2_P100, l1 = @L1_P75 }
nexus.environment @ENV_MIXED { l2 = @L2_P75,  l1 = @L1_P75 }
```

环境是数值组合，不改变 Domain 成员和实际分配责任。示例选择相同 L1 只是为了展示“只改 L2”也需要完成相关上游 Drain。

### 14.3 MatMul 与 Gather 的合同示例

```mlir
nest.resource_contract @matmul_l2 {
  level = "L2",
  allowed_profiles = [@L2_P100, @L2_P75],
  compatible_environments = [@ENV_DENSE, @ENV_MIXED],
  spm = {
    allocation_kind = "striped_arena",
    bytes_per_bank = 8192, bank_count = 8, alignment = 256,
    layout_id = "mm_l2_v0",
    reservation_lifetime = "group_context", growth_policy = "forbidden"
  },
  cache = {
    required = false, access = "none",
    target_bytes = 0, budget_kind = "hint", sharing = "pool_shared"
  },
  control = { group_slots = 1, max_live_grids = 1, event_slots = 8 }
}

nest.resource_contract @matmul_l1 {
  level = "L1",
  allowed_profiles = [@L1_P75],
  compatible_environments = [@ENV_DENSE, @ENV_MIXED],
  spm = {
    allocation_kind = "striped_arena",
    bytes_per_bank = 4096, bank_count = 4, alignment = 256,
    layout_id = "mm_l1_v0",
    reservation_lifetime = "tile_context", growth_policy = "forbidden"
  },
  cache = {
    required = false, access = "none",
    target_bytes = 0, budget_kind = "hint", sharing = "pool_shared"
  },
  control = { tile_slots = 1, event_slots = 4 }
}

nest.resource_contract @gather_l2 {
  level = "L2",
  allowed_profiles = [@L2_P75],
  compatible_environments = [@ENV_MIXED],
  spm = {
    allocation_kind = "striped_arena",
    bytes_per_bank = 8192, bank_count = 8, alignment = 256,
    layout_id = "gather_l2_v0",
    reservation_lifetime = "group_context", growth_policy = "forbidden"
  },
  cache = {
    required = true, access = "read",
    target_bytes = 65536, budget_kind = "hint", sharing = "pool_shared"
  },
  control = { group_slots = 1, max_live_grids = 1, event_slots = 8 }
}

nest.resource_contract @gather_l1 {
  level = "L1",
  allowed_profiles = [@L1_P75],
  compatible_environments = [@ENV_MIXED],
  spm = {
    allocation_kind = "striped_arena",
    bytes_per_bank = 4096, bank_count = 4, alignment = 256,
    layout_id = "gather_l1_v0",
    reservation_lifetime = "tile_context", growth_policy = "forbidden"
  },
  cache = {
    required = true, access = "read",
    target_bytes = 16384, budget_kind = "hint", sharing = "pool_shared"
  },
  control = { tile_slots = 1, event_slots = 4 }
}
```

这些容量仅说明合同形态，不是某个真实 MatMul/Gather 的测量数据。实际程序必须有与之匹配的 Buffer/Layout Plan。

`compatible_environments` 是保守合法组合表。在环境 ID 定义不可变且有 ABI 校验的前提下，它可下降为 Bitset；不要求 Runtime 运行约束求解器。

### 14.4 程序声明与合同绑定

```mlir
nest.context @matmul_v0(%A, %B, %C) attributes {
  resource_contract = @matmul_l2,
  permitted_tile_programs = [@matmul_tile_v0],
  completion = "exports_hbm_visible_and_resources_released"
} {
  // %arena 是 Admission 授予本 Context 的 L2 Arena，不是临时全局 malloc。
  %a = nest.arena.view "a" layout(@matmul_l2)
  %b = nest.arena.view "b" layout(@matmul_l2)
  %o = nest.arena.view "out" layout(@matmul_l2)

  %a_ready = nest.prefetch.async %A into %a
  %b_ready = nest.prefetch.async %B into %b

  %grid_done, %input_released, %output_ready =
      nest.dispatch.spmd.async @matmul_tile_v0
          ins(%a, %b) outs(%o)
          depends_on(%a_ready, %b_ready)

  nest.release %a depends_on(%input_released)
  nest.release %b depends_on(%input_released)

  %hbm_done = nest.store.async %o to %C
      depends_on(%output_ready)
      completion = "hbm_visible"

  nest.release %o depends_on(%hbm_done)
  nest.await %grid_done
  nest.await %hbm_done
  nest.return
}

nest.tile.program @matmul_tile_v0(%A_tile, %B_tile, %C_tile)
attributes { resource_contract = @matmul_l1 } {
  %ta = tile.load.async %A_tile
  %tb = tile.load.async %B_tile
  tile.await %ta, %tb
  %acc = tile.mma ...
  tile.signal input_released
  %ts = tile.store.async %acc to %C_tile
      completion = "destination_visible"
  tile.await %ts
  tile.signal output_ready
  tile.return
}
```

省略号和未展开的参数类型明确表示算子体/具体汇编格式未定义。`nest.return` 不能早于所有延迟 Release 和子任务退休完成就发布 `context_done`。该结束条件由 Runtime 再检查，不能只相信 PC 已到 Return。

Gather 的 irregular 数据路径仍是目标允许的 Tile → L1 Cache → L2 Cache → HBM 路径；不会因为需要 L2 资源合同，就强迫将 Gather 的全部输入物化到 L2 SPM。L2 Arena 可用于明确需要的输出/控制数据和后续规则计算数据。

### 14.5 MatMul → Gather → MatMul 的显式阶段计划

```mlir
nexus.program @run(%A, %B, %X_hbm, %indices_hbm,
                   %Y_hbm, %W, %Z_hbm) {
  // 静态 Region 描述有限 Manifest；Phase 激活后按协议登记并 Seal。
  %e0 = nexus.phase.async @dense_0 environment(@ENV_DENSE) {
    %m0 = nexus.submit_context.async @matmul_v0(%A, %B, %X_hbm)
    nexus.phase.seal
  }

  %e1 = nexus.phase.async @mixed_1 environment(@ENV_MIXED)
      after(%e0) {
    %g = nexus.submit_context.async
        @gather_v0(%X_hbm, %indices_hbm, %Y_hbm)
    %m1 = nexus.submit_context.async
        @matmul_v0(%Y_hbm, %W, %Z_hbm)
        depends_on(%g)
    nexus.phase.seal
  }

  nexus.await %e1
  nexus.return
}
```

`%g` 表示严格的 `context_done`，不是仅 `grid_done`。本例把 Gather 和第二个 MatMul 放在同一 P75 阶段，**这是手工指定的计划，不是 Runtime 优化结果**。

这个同阶段 HBM 交接示例要求第 10.3 节的一致性/范围维护能力。目标只有全域静默维护时，应将第二个 MatMul 放入新的 `ENV_MIXED` Phase；即使 Profile 相同，也显式等待前一阶段 `phase_done`。

如果希望全部三个程序都使用 P75，可以手工将其放入同一个 `ENV_MIXED` Phase，并提供 HBM 依赖。V1 不负责比较这些计划谁更快。

### 14.6 新增 Verifier，而非新增优化 Pass

建议分成以下验证层：

| 检查层                   | 检查内容                                                               |
| ------------------------ | ---------------------------------------------------------------------- |
| Profile Verifier         | 字节边界、合法组织、Domain 类别、对齐和 Registry ABI。                 |
| Contract Verifier        | 逐 Bank 需求、Arena 完整性、Cache 路径、控制资源和动态参数上界。       |
| Program Binding Verifier | 代码版本、布局、合同和环境组合一致；内部 Tile 程序可在阶段环境运行。   |
| Phase Verifier           | 有限 Manifest、无未来阶段反向依赖、选定 Profile 兼容所有可能执行路径。 |
| Lifetime/Export Verifier | 输入释放、输出 HBM 完成、Return、跨阶段没有活 SRAM 句柄。              |
| Runtime Recheck          | 实际目标、Shape、Generation、当前剩余容量与维护能力。                  |

MLIR ODS 支持 `hasVerifier`、`hasRegionVerifier`；涉及嵌套操作的检查可放在 Region Verifier，跨符号、事件和全计划的验证可放在独立只读 Validation Pass。[R5]

Verifier 不负责找到更好的 Profile，只负责拒绝不合法或无法证明满足合同的镜像。对于不能证明的动态情况，应使用明确的启动前 Guard；不能把“不知道”当成“允许”。

---

## 15. Runtime ABI 与 Backend 接口

### 15.1 最小面向调用方的 C API 草案

下面代码是**可编译的头文件声明草案**，不是函数实现。使用状态码和不透明句柄；实际 ABI 版本与错误码值仍需固化。

```c
#ifndef NEXUS_BANK_RUNTIME_V1_H
#define NEXUS_BANK_RUNTIME_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NexusRuntime NexusRuntime;
typedef struct NexusPlan NexusPlan;
typedef uint64_t NexusEvent;

typedef enum NexusStatus {
    NEXUS_OK = 0,
    NEXUS_NOT_READY = 1,
    NEXUS_INVALID_PLAN = 2,
    NEXUS_RESOURCE_EXCEEDS_PROFILE = 3,
    NEXUS_CONFIG_MISMATCH = 4,
    NEXUS_UNSUPPORTED_MEMORY_SEMANTICS = 5,
    NEXUS_TIMEOUT = 6,
    NEXUS_CANCELLED = 7,
    NEXUS_DEVICE_FAULT = 8
} NexusStatus;

typedef struct NexusLaunchBindings {
    uint32_t struct_size;
    uint32_t abi_version;
    const void *argument_data;
    size_t argument_bytes;
} NexusLaunchBindings;

NexusStatus nexus_plan_load(
    NexusRuntime *rt, const void *image, size_t image_bytes,
    NexusPlan **out_plan);

NexusStatus nexus_plan_submit(
    NexusRuntime *rt, const NexusPlan *plan,
    const NexusLaunchBindings *bindings, NexusEvent *out_done);

NexusStatus nexus_event_query(
    NexusRuntime *rt, NexusEvent event, NexusStatus *out_result);

NexusStatus nexus_event_wait(
    NexusRuntime *rt, NexusEvent event, uint64_t timeout_ns,
    NexusStatus *out_result);

NexusStatus nexus_request_cancel(
    NexusRuntime *rt, NexusEvent request_done);

NexusStatus nexus_plan_release(
    NexusRuntime *rt, NexusPlan *plan);

#ifdef __cplusplus
}
#endif
#endif
```

`event_wait` 超时不自动取消，也不释放正在使用的资源。`plan_release` 对在用 Plan 应拒绝或延迟到所有引用完成，不能直接释放程序/合同存储。

Host 参数数据的生命周期由 Submit 明确定义：建议 Submit 成功前复制必要的有界参数描述，或显式 Pin 调用方存储；不能读取已经失效的栈地址。

### 15.2 设备 Backend 的必需能力

| Backend 原语                              | 必须提供的结果                                          |
| ----------------------------------------- | ------------------------------------------------------- |
| `query_registry()`                        | Profile、几何结构、映射 ABI 和 Cache Maintenance 能力。 |
| `close_user_issue(domain_set)`            | 用户发射门已关闭的确认。                                |
| `snapshot_outstanding(domain_set)`        | 覆盖上下游队列/在途请求的统计快照。                     |
| `publish_hbm_ranges(exports)`             | 结果对声明消费者可见的完成事件。                        |
| `cache_maintain(domain, op)`              | Clean/Invalidate 的完成事件，包含其产生的事务。         |
| `prepare_profile(domain, profile, txn)`   | 全成员准备 ACK 或失败。                                 |
| `commit_profile(domain, txn)`             | 全成员提交 ACK 和新配置回读。                           |
| `open_user_issue(domain_set, generation)` | 只允许新代请求的发射。                                  |
| `isolate_or_reset(domain_set)`            | 确认旧请求不再能破坏被复用的地址。                      |

Backend 原语可以由固件轮询、MMIO + Interrupt 或 RTL 状态机实现。**完成事件的语义必须一致，不把某一种实现方式硬编码进 IR。**

### 15.3 序列化与热路径

镜像中使用固定宽度字段、显式端序、大小和版本；不能把含原生指针的 C 结构直接 `memcpy` 成设备 ABI。所有长度、字节数乘法与偏移相加需要溢出检查。

建议在 Plan 加载时完成合同归一化，下降为 Profile Bitset、逐 Bank 固定需求、事件索引和预编译 Layout ID。热路径只做比较、资源事务和队列推进，不解析 MLIR 文本、不扫描整张图、不运行性能搜索。

控制对象使用有界池；Pool 耗尽应返回错误或明确背压，不能在持有 Admission 锁时进行不可控的动态分配。

---

## 16. 错误分类与诊断

| 错误/等待原因                          | 含义                                              | 后续行为                               |
| -------------------------------------- | ------------------------------------------------- | -------------------------------------- |
| `WAIT_PHASE`                           | 合法的未来 Phase 尚未激活。                       | 只保留元数据。                         |
| `WAIT_DEPENDENCY`                      | 输入完成事件未满足。                              | 不占 SPM/执行 Slot。                   |
| `WAIT_CAPACITY` / `WAIT_FRAGMENTATION` | 当前 Arena 无法分配。                             | 在相应释放事件后重试。                 |
| `WAIT_TILE_SLOT`                       | 目标 Tile 暂无 Slot。                             | 已活动旧任务继续运行。                 |
| `PROFILE_NOT_ALLOWED`                  | 计划配置不在程序合法集合。                        | 拒绝计划，不自行换 Profile。           |
| `INVALID_INVOCATION_STATE`             | 重复准入、非 Manifest 成员或在 Drain 后尝试准入。 | 拒绝操作；内部违反状态机时转诊断错误。 |
| `RESOURCE_EXCEEDS_PROFILE`             | 空闲状态也放不下。                                | 立即失败，不无限等待。                 |
| `INVALID_DEPENDENCY_PHASE`             | 旧阶段等待未来阶段事件。                          | 拒绝计划。                             |
| `UNSUPPORTED_MEMORY_SEMANTICS`         | 缺少所需 HBM 可见性或 Cache 维护能力。            | 拒绝相关计划。                         |
| `STALE_GENERATION`                     | 旧配置/旧 Arena 的任务或返回出现。                | 隔离、记录、进入 Fault。               |
| `DRAIN_TIMEOUT`                        | 在规定诊断阈值内未能收敛。                        | 输出计数与阻塞链，受控取消/复位。      |
| `PROFILE_COMMIT_FAILED`                | 一部分成员未成功提交。                            | 保持门关闭，不能继续执行。             |

诊断至少包含：Plan/Phase/Context、目标 Domain/Pool、当前/目标 Profile、Generation、等待资源、持有者列表或可定位的 Lease ID、尚未完成的请求类型。

等待是正常状态，错误是无法继续遵守合同；两者不能使用同一个无区别的“Not Ready”日志掩盖。

---

## 17. 两个具体运行过程

### 17.1 已是混合 Profile：16 个 Tile Context 满载后补入 Gather

测试配置：4 个物理 Tile，每 Tile 4 个 Context Slot；Group Slot 至少 5 个。四个 MatMul 根 Context 各投递 4 个 Task，形成 16 个驻留 Tile Context。

使用 `ENV_MIXED`；每个根 Context 的 L2 需求为 8 KiB/Bank，每个 Tile Task 的 L1 需求为 4 KiB/Bank。

```text
L2_P75 每 Bank 用户可用容量：48 - 4 = 44 KiB
四个 MatMul 根 Context 占用：4 × 8 = 32 KiB/Bank
再加一个 Gather 根 Context：32 + 8 = 40 KiB/Bank，仍可放下

L1_P75 每 Tile 每 Bank 用户可用容量：24 - 2 = 22 KiB
四个 MatMul Task 占用：4 × 4 = 16 KiB/Bank
此时实际阻塞项是 4 个 Tile Slot 已满，不是 L1 总容量。
```

该例假设 Gather 的输入已可见，与尚未完成的 MatMul 没有必须等待的数据依赖。

| 时刻 | Group/L2                                       | Tile/L1                                           | Profile |
| ---- | ---------------------------------------------- | ------------------------------------------------- | ------- |
| T0   | 四个 MatMul 已准入。                           | 每 Tile 四个 Slot 占满。                          | 不变。  |
| T1   | Gather L2 合同通过，可取得自己的 L2 Arena。    | Gather Task 进入 Route 等待；不占 L1/Slot。       | 不变。  |
| T2   | 不要求四个 MatMul 根同时结束。                 | Tile 0 一个旧 Task 退休，释放本地 Slot/Arena。    | 不变。  |
| T3   | Gather 根继续跟踪剩余 Task。                   | Tile 0 准入一个 Gather Task；其他 Tile 各自补位。 | 不变。  |
| T4   | 根 Context 在最终 HBM 输出和子任务完成后退休。 | 完成一个就补一个，仍不固定分批。                  | 不变。  |

Gather 的 L2 Cache Target 是提示，不会在 T1 从共享 Cache 中切出私有块。若 Gather 必须等待某个 MatMul 的 HBM 输出，则它先留在 Dependency Wait Set，不能按照本例 T1 提前取得执行资源。

### 17.2 当前 P100，后续阶段需要 P75

```text
E0：ENV_DENSE，四个 MatMul 根 Context
E1：ENV_MIXED，Gather 及后续 MatMul
```

1. E1 描述符可以登记到未来阶段元数据，但不触发改档，也不占 E0 的 L2/L1 执行资源。
2. E0 Seal 后继续执行它已经登记的所有根 Context、Tile Task 和最终 Store。
3. E0 全部输出完成 HBM 交接，用户 Arena 全部释放。
4. 保守收敛 L1/L2 Cache 和 Fabric，获得 `phase_done(E0)`。
5. L2 提交 P100 → P75；L1 的 Profile 数值保持 P75。L1 不需要重复配置，但其相关访问已被排空。
6. 所有成员 ACK 后激活 E1。
7. E1 内后续 MatMul 使用其允许 P75 的既定版本，不自行申请切回 P100。

如手工计划明确添加 E2/P100，Runtime 才在 E1 的安全边界后切回。不能因为看见 MatMul 名称就自动调整比例。

---

## 18. Trace、计数器与性能解释

### 18.1 Trace 层级

```text
Device Request
  Plan
    Memory Phase
      Profile Domain Transaction
        Close / Export Drain / Cache Maintenance / Fabric Drain
        Prepare / Commit / ACK / Open
      Group Context
        L2 Admission Wait / L2 Lease / Group Actions
        Tile Grid
          Tile Context（含实际 Tile ID）
            L1 Admission Wait / L1 Lease / Load / Compute / Store
```

每条事件携带：`request_id, phase_id, domain_id, profile_id, profile_generation`；Context/Task 事件额外带 `context_id, tile_id, tile_context_id, lease_id`。

### 18.2 必需指标

| 类别  | 指标                                                                                                    |
| ----- | ------------------------------------------------------------------------------------------------------- |
| 阶段  | 当前 Profile、Phase 执行时间、Seal 后剩余执行时间、进入 QUIESCENT 的时间。                              |
| 配置  | Profile 真实 Commit 次数、Prepare/Commit/ACK 延迟、失败次数。                                           |
| 等待  | Dependency、L2 Capacity、L1 Capacity、Fragmentation、Group/Tile Slot、Engine、Phase 等待时长。          |
| 容量  | 每 Pool/Bank 的已预留 Arena、实际 Live Buffer、峰值、系统预留。                                         |
| Cache | 可获得时记录请求、Miss、回填、脏写回、Maintenance 行数/字节数；Context 归因是可选统计，不代表私有分区。 |
| 流水  | 活跃 Group/Tile Context、Tile 利用情况、DMA/Compute 的重叠情况。                                        |
| HBM   | 正常输出搬运与因显式边界产生的额外搬运，按原因分别标记。                                                |
| 安全  | Stale Generation、重复完成、重复释放、资源泄漏、Drain Timeout。                                         |

### 18.3 避免误判“切换成本”

Seal 后剩余 MatMul 的自然执行尾部，不应全部算成 Profile Register 切换开销。建议分别记录：

```text
旧工作执行尾部
必要 HBM 输出完成
边界额外维护/搬运
Fabric 静默等待
实际 Profile Prepare/Commit
新阶段启动
```

固定 Profile 计划与确定性队列可减少策略变化，但共享 Cache、访存和仲裁仍可能影响 Latency。V1 不承诺 WCET 或无抖动执行；未来若有硬实时目标，需要单独定义服务上界、隔离和可测量的最坏情况假设。

---

## 19. 关键不变量与断言

下列为建议写入模拟器和 RTL 验证环境的规范性不变量：

| ID  | 不变量                                                                                      |
| --- | ------------------------------------------------------------------------------------------- |
| I01 | 对开放发射的任一 Domain，所有成员 Bank 的有效 Profile 和 Generation 一致。                  |
| I02 | 任一已发布请求的 Phase/Generation 与其访问的域匹配。                                        |
| I03 | 任一 Pool 任一 Bank：用户已预留 SPM + 系统预留不超过当前 SPM Region。                       |
| I04 | 任一 Slot、Arena 和控制 Ticket 只有一个当前 Owner。                                         |
| I05 | 未成功准入的根 Context 不持有 L2/L1/执行 Slot。                                             |
| I06 | Tile Admission 失败后，L1 分配、父 Buffer Pin、Slot/Frame 绑定全部回滚。                    |
| I07 | `input_released` 聚合覆盖全部指定消费者；释放后不存在允许继续访问的引用。                   |
| I08 | `context_done` 之前必需 HBM 输出已可见，且该 Context 的全部执行资源已退休。                 |
| I09 | `phase_done` 之前 Manifest 根调用全部完成、Activation/Route/执行集合均已收敛。              |
| I10 | Profile Commit 之前用户 Lease 为零、维护完成、在途事务为零、成员 Quiescent ACK 齐全。       |
| I11 | SEALED_EXECUTING 仍允许本 Phase 已登记根的 Admission 和必要后续动作。                       |
| I12 | CACHE_MAINTENANCE/FABRIC_DRAIN 阶段不能生成新的用户请求；Maintenance 产生的请求仍纳入计数。 |
| I13 | 不接受旧 Phase 对未来 Phase 的生产事件依赖。                                                |
| I14 | Cache Hint 不生成伪私有容量 Ticket，不参与 SPM 容量扣账。                                   |
| I15 | Commit 任一成员失败时，用户发射门保持关闭。                                                 |
| I16 | 请求超时或取消后，未收到隔离/取消完成前不能复用其可能访问的地址。                           |
| I17 | 同一配置域不存在两个同时持有写权限的 Coordinator。                                          |
| I18 | 同 Profile 的不同 Phase 不复用旧 Phase 身份；Profile 不变不意味着可以忽略数据依赖。         |

这些断言是实现要求，不代表已经完成模型检查或 RTL 证明。

---

## 20. 验证矩阵与验收条件

### 20.1 必须覆盖的功能与异常测试

| Test | 场景                                             | 预期结果                                               |
| ---- | ------------------------------------------------ | ------------------------------------------------------ |
| T01  | P75 阶段的 MatMul 允许 P75/P100，但偏好 P100。   | 保持 P75，不生成配置事务。                             |
| T02  | P100 阶段提交只允许 P75 的 Gather。              | 计划验证失败；不能现场切换。                           |
| T03  | 16 个 Tile Slot 满，Gather 合同已通过 L2。       | 等待各 Tile Slot；不提前占 L1，不改 Profile。          |
| T04  | 空 Pool 中 SPM 请求也超过容量。                  | 立即报永久资源错误。                                   |
| T05  | 总空闲容量足够但连续跨度不足。                   | 返回 Fragmentation 等待或明确失败，不越界。            |
| T06  | 一个 Bank 需求过大，其他 Bank 空闲。             | 不能以跨 Bank 总容量补偿。                             |
| T07  | 两个候选同时争取最后一份 Arena。                 | 仅一个 Commit 成功，另一方无泄漏。                     |
| T08  | Tile COMMIT 中途发生绑定失败。                   | L1、Slot、Pin、Frame 全部回滚。                        |
| T09  | 已登记根还在 WAIT_DEPENDENCY 时 Seal。           | 前驱完成后该根仍能 Admission。                         |
| T10  | 根还需投递下一波 Tile Task 时 Seal。             | 后续任务继续投递，最终可退出。                         |
| T11  | 旧 Phase 依赖下一 Phase 的事件。                 | Loader 拒绝。                                          |
| T12  | 资源等待队首较大、后项较小。                     | 按 V1 FIFO，不隐式越过；记录 HOL 等待。                |
| T13  | 所有活动 Tile Slot 为零，但 Route Queue 非空。   | 不能发布 grid_done/phase_done。                        |
| T14  | Group PC 到 Return，但 HBM Store 未完成。        | 不能发布 context_done。                                |
| T15  | DMA 已读完源 Buffer，但目的端未可见。            | source_consumed 可触发局部释放；不能冒充 hbm_visible。 |
| T16  | P100 → P75，所有根输出仍有在途写回。             | 不允许 Commit。                                        |
| T17  | P75 → P100，旧 Cache Miss 尚未回填。             | 不允许 Commit。                                        |
| T18  | Cache 为只读，但旧回填未归零。                   | 仍必须等待，不因无脏数据而跳过。                       |
| T19  | L1 Clean 产生新的 L2 写入。                      | L2 不能提前宣布维护结束。                              |
| T20  | 一个 Bank 的 Prepare/Commit ACK 延迟。           | 全域发射门保持关闭。                                   |
| T21  | 一个 Bank Commit 失败。                          | Domain Fault，不尝试继续使用混合配置。                 |
| T22  | 延迟返回使用旧 Generation。                      | 检测、隔离并报告；不能写入新 SPM。                     |
| T23  | 下一 Phase 队列堆满。                            | 旧 Phase 的完成、写回和 Maintenance 仍能前进。         |
| T24  | 多个 Gather 的 Cache Target 合计超过容量。       | 不按私有配额报容量错误；可能有性能干扰。               |
| T25  | Producer DMA 更新 HBM，Consumer Cache 留有旧行。 | 通过一致性/范围维护保证新值；缺能力则拒绝同阶段交接。  |
| T26  | 同一个 Tile Program 内后半段才需要 Cache。       | 整个版本的合同必须要求 Cache，不能只检查首个 MatMul。  |
| T27  | 手工合同遗漏双缓冲/输出存储。                    | 静态验证或运行时越界保护报错，不能无声覆盖。           |
| T28  | 相邻 Phase Profile 相同。                        | 无 Register Commit，但保留显式 HBM_QUIESCENT 边界。    |
| T29  | Host 线程同时提交冲突 Domain 的两个计划。        | 配置所有权串行化。                                     |
| T30  | 取消请求后迟到 DMA。                             | 资源在隔离完成前不复用。                               |
| T31  | 只有 L2 档位变化，L1 数值不变。                  | L1 相关访问仍纳入 Drain；L1 不做无意义的重复 Commit。  |
| T32  | L1/L2 各自合法但组合不在兼容环境表中。           | 拒绝组合，不能推断笛卡尔积合法。                       |

### 20.2 推荐测试层次

先建立离散事件的软件参考模型：随机延迟 Load/Store/回填、随机触发队列背压和准入失败，逐步检查 I01–I18。随后进行控制器模块仿真，再接入真实 DMA/Cache 模型，最后在 FPGA/目标平台上核对事件和数据。

测试目标分别是：数据正确、资源不泄漏、错误不冒充完成、有限合法任务最终可结束。吞吐和尾延迟在功能验收后测量；不把没有卡死的单次运行当成完整死锁证明。

### 20.3 最小可接受交付

V1 只有在以下条件满足后，才可以宣称支持 Profile Runtime：

```text
至少两个真实 Profile 可加载和提交；
同 Profile 多 Context 动态补位正常；
两个方向的改档均经过静默边界；
跨边界 HBM 数据正确，包括 Cache 旧行场景；
所有用户资源最终归零，故障路径没有提前复用；
能通过 Trace 解释一次 Context 为什么等待、一次改档为什么等待。
```

---

## 21. 实施拆分：不包含优化器

| 阶段              | 交付内容                                                       | 完成标准                                           |
| ----------------- | -------------------------------------------------------------- | -------------------------------------------------- |
| A. 静态描述       | Registry、Domain/Pool 拓扑、合同 Schema、有限 Phase Manifest。 | 手工计划可验证，非法档位/容量可拒绝。              |
| B. 两级 Admission | L2/L1 Arena、Slot、事务和回滚、资源唤醒。                      | 单 Profile 下通过容量/回滚/动态补位测试。          |
| C. 生命周期       | 精确的输入释放、输出完成、Context/Grid/Phase 计数。            | 不将 source_consumed 或 grid_done 当作 Host 完成。 |
| D. 安全改档       | Seal/Drain/Maintenance、Prepare/Commit/ACK、Generation。       | P100↔P75 两个方向均正确。                          |
| E. 数据可见性     | HBM 导出和 Cache 一致性/维护 Backend。                         | Producer–Consumer 新值可见；缺能力时明确拒绝。     |
| F. 异常与 Trace   | Timeout、取消、Reset、细粒度等待原因和计数。                   | 故障收敛可诊断，无未经确认的地址复用。             |
| G. IR 接入        | 自定义 Op/Attribute、Verifier、静态描述符 Lowering。           | 同一手工计划通过 IR 与直接描述符产生等价行为。     |

步骤 C、D、E 在集成时需要共同验证；表中拆分是开发职责划分，不意味着缺少可见性处理的改档已经可用。

### 21.1 建议模块目录

```text
runtime/
  plan_loader.*                 # ABI、目标与合同检查
  phase_coordinator.*           # Seal/Drain/配置事务
  group_admission.*             # L2 资源事务
  tile_admission.*              # L1 本地资源事务
  spm_arena_allocator.*          # 固定粒度 Arena
  event_tracker.*               # Grid/Context/Phase 完成语义
  memory_backend.*              # 可见性、维护、MMIO/控制器桥接
  diagnostics.*                 # 错误与 Trace

ir/
  MemoryProfile.*
  ResourceContract.*
  PhaseOps.*
  VerifyResourcePlan.*          # 验证，不进行模式优选
  LowerResourceDescriptors.*

tests/
  admission/
  lifecycle/
  profile_transition/
  visibility/
  fault_injection/
```

---

## 22. 需要由硬件目标补齐的参数

以下属于目标规格缺口，不需要现在引入编译器优化，但在 RTL/Runtime 接口冻结前必须给出：

| 项目          | 必须明确的内容                                                  |
| ------------- | --------------------------------------------------------------- |
| 实际拓扑      | L1/L2 Bank 数、容量、配置域成员和 Pool 边界。                   |
| Bank 服务能力 | 端口数、读写限制和 SPM/Cache 竞争规则；容量比例不等于带宽保证。 |
| Profile 组织  | 真实档位、映射、Cache Set/Way/Tag 组织、合法边界粒度。          |
| 资源上限      | Group/Tile Slot、Grid/Event/Route 容量、Arena 粒度。            |
| 访存语义      | Cache 写策略、Bypass 能力、HBM 完成等级、可见性范围。           |
| 维护能力      | 范围或全域 Clean/Invalidate、上游依赖、Maintenance 完成 ACK。   |
| 排空可观测性  | 哪些引擎计数、哪些队列需要覆盖，Quiescent ACK 的保证。          |
| 保护能力      | Arena 边界检查、Generation 传递、旧请求隔离、复位范围。         |
| 外部 Master   | 是否还有 CPU/GPU/调试 DMA 访问同域，以及如何停止或计入。        |

不能仅以“可改一个比例寄存器”视为硬件已经支持本 Runtime；可验证的静默、完成和发布机制是同等重要的接口。

---

## 附录 A：最小资源预检查参考代码

下面是可独立运行的 Python 参考代码，仅演示**合法 Profile/容量的预检查**。它不实现实际连续 Arena 分配、队列、Drain、RTL 或完整 Runtime。返回 `OK_TO_PLAN` 也不是已经准入；真实 Commit 仍需第 7、8 节的原子事务。

```python
from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class Profile:
    profile_id: str
    bank_count: int
    spm_per_bank: int
    cache_per_bank: int
    system_spm_per_bank: int


@dataclass(frozen=True)
class Contract:
    allowed_profiles: FrozenSet[str]
    spm_per_bank: Tuple[int, ...]
    cache_required: bool
    cache_target_bytes: int = 0  # Hint only; never a private reservation.


def precheck(
    profile: Profile,
    contract: Contract,
    reserved_per_bank: Tuple[int, ...],
) -> str:
    if profile.bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if len(contract.spm_per_bank) != profile.bank_count:
        return "INVALID_CONTRACT_LAYOUT"
    if len(reserved_per_bank) != profile.bank_count:
        raise ValueError("runtime ledger bank count mismatch")
    if min(
        profile.spm_per_bank,
        profile.cache_per_bank,
        profile.system_spm_per_bank,
        contract.cache_target_bytes,
        *contract.spm_per_bank,
        *reserved_per_bank,
    ) < 0:
        raise ValueError("negative resource value")

    usable = profile.spm_per_bank - profile.system_spm_per_bank
    if usable < 0:
        return "INVALID_PROFILE"
    if any(used > usable for used in reserved_per_bank):
        return "CORRUPT_RUNTIME_LEDGER"
    if profile.profile_id not in contract.allowed_profiles:
        return "PROFILE_NOT_ALLOWED"
    if contract.cache_required and profile.cache_per_bank == 0:
        return "CACHE_DISABLED"
    if any(need > usable for need in contract.spm_per_bank):
        return "RESOURCE_EXCEEDS_PROFILE"
    if any(
        need + used > usable
        for need, used in zip(contract.spm_per_bank, reserved_per_bank)
    ):
        return "WAIT_CAPACITY"
    return "OK_TO_PLAN"


def self_test() -> None:
    kib = 1024
    p75 = Profile("L2_P75", 8, 48*kib, 16*kib, 4*kib)
    p100 = Profile("L2_P100", 8, 64*kib, 0, 4*kib)
    mm = Contract(frozenset({"L2_P75", "L2_P100"}), (8*kib,)*8, False)
    gather = Contract(frozenset({"L2_P75"}), (8*kib,)*8, True, 64*kib)

    assert precheck(p75, mm, (32*kib,)*8) == "OK_TO_PLAN"
    assert precheck(p75, mm, (40*kib,)*8) == "WAIT_CAPACITY"
    assert precheck(p100, gather, (0,)*8) == "PROFILE_NOT_ALLOWED"

    # Even a wrongly broad allow-list cannot bypass the Cache-path check.
    bad_gather = Contract(mm.allowed_profiles, (8*kib,)*8, True)
    assert precheck(p100, bad_gather, (0,)*8) == "CACHE_DISABLED"

    too_large = Contract(mm.allowed_profiles, (45*kib,)*8, False)
    assert precheck(p75, too_large, (0,)*8) == "RESOURCE_EXCEEDS_PROFILE"

    # Cache targets are hints: a large target alone is not a capacity error.
    hint = Contract(gather.allowed_profiles, (8*kib,)*8, True, 1024*kib)
    assert precheck(p75, hint, (0,)*8) == "OK_TO_PLAN"

    one_bank_blocked = (40*kib,) + (0,)*7
    assert precheck(p75, mm, one_bank_blocked) == "WAIT_CAPACITY"


if __name__ == "__main__":
    self_test()
    print("Resource precheck self-tests passed.")
```

该参考代码采用逻辑资源快照，不处理并发；实际 Runtime 必须在分配事务中重读并验证状态，不能把这里的先检查再提交当成无锁安全实现。

---

## 附录 B：评审时最需要确认的新增选择

1. 是否采用本提案的 Context-lifetime Arena，接受内部释放不立即归还全局配额的保守策略。
2. 是否采用依赖就绪后的严格 FIFO Admission，接受不做资源就绪候选越过。
3. 是否采用本版保守的根 Context 级 Drain 闭包，不对活跃 Group 做仅 L1 的局部重配置。
4. 目标能否提供同 Phase HBM 交接所需的可见性/范围维护；缺少时统一拆为全域静默边界。
5. L1/L2 的实际 Profile Domain 成员、Cache 写策略和 Quiescent ACK 能否在目标描述中明确。

这五项是实现评审问题，不是要求先实现编译器优化，也不改变已经对齐的 Bank 内统一 Region 分区原则。

---

## 参考资料与证据范围

本提案的 NEST/Nexus IR、Admission 和 Phase 协议是针对当前讨论提出的设计，不是下列外部系统已有的 API。外部资料只支持明确标注的通用语义区分，不能据此推断硬件内部采用完全相同的 SRAM 布局。

核验日期：2026-09-15。

- **[R1] NVIDIA CUDA Runtime API — Execution Control。**参考点：硬资源使用量与 Carveout Preference 的区别；不同配置偏好可能引入设备侧同步。不是 NEST 切换协议的实现证明。
- **[R2] NVIDIA CUDA Programming Guide — C/C++ Language Extensions，Memory Fence。**参考点：顺序约束不等于独立的数据可见性保证；NEST 的 HBM 完成语义仍需自身 Backend 明确定义。
- **[R3] NVIDIA CUDA Programming Guide — Asynchronous Data Copies。**参考点：异步复制、源侧读取完成与数据可用性需要适当的完成/同步语义。
- **[R4] Linux Kernel Documentation — resctrl。**参考点：共享 Cache、资源分配、独占分配与 Pseudo-Locking 是不同能力。普通共享 Cache 预算不能冒充驻留保证。
- **[R5] MLIR — Operation Definition Specification。**参考点：`hasVerifier`、`hasRegionVerifier` 及其验证顺序；不代表本文自定义操作已经存在。

[R1]: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html
[R2]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
[R3]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
[R4]: https://docs.kernel.org/filesystems/resctrl.html
[R5]: https://mlir.llvm.org/docs/DefiningDialects/Operations/

---

## 结论

本版把可变因素限制在明确边界内：

> **软件指定 Phase 和 Profile；Context 用合同申请资源；Group/L2 与 Tile/L1 分别原子准入；旧阶段全部输出在 HBM 可见并排空后，统一提交新 Profile。**

阶段内的多 Context 仍可动态补位。Runtime 不根据算子类型改档，不在线调整 Tiling，不把 Cache Hint 当成私有资源。第一版接受更保守的 Arena 生命周期和 HBM 边界，以换取可以实现、可以诊断和可以验证的安全执行模型。

后续优化只能建立在这些语义不变量之上；应先通过 Trace 找到实际瓶颈，再决定是否引入早释放、候选越过、局部 Drain 或更复杂的 Profile 规划。
