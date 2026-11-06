# Nexus 共享 AM V1：需求规格与设计改造指南

- 版本：1.0
- 日期：2026-10-26
- 适用对象：Nexus/Elenor 的 tile context、UCE、矩阵阵列组、AM、模拟器及后续 RTL。
- 交付目的：供其他设计或实现模型直接使用，形成改动方案、实现任务和验收依据。
- 状态：需求与建议设计。尚未据此修改或验证用户的 Nexus 仓库；本文不声称已有这些接口、模块或测试结果。
- 术语约定：MUST/必须为 V1 正确性要求；SHOULD/建议为推荐实现；参数示例不代表已完成 PPA 定型。

## 0. 给实施模型的首要指令

先读取目标仓库的 AGENTS.md、现有架构规格、配置入口、调度器、事件模型和测试。建立“本文概念 → 仓库实际对象/路径”的对应表，再修改实现。本文出现的模块名、接口名和配置键都是建议名称，不应被当作仓库已经存在的符号。

必须保持以下两个用户指定的约束：

1. **每条输出累加链保持原 engine affinity。** V1 在链成功建立时选择一个 engine，直到该链释放 AM，都不得迁移到其他 engine。
2. **每个 tile context 同时最多使用两个 AM slot。** 配额跨该 context 使用的所有 engine 统计，不是“每个 engine 各有两个”。

本规范采用的解释是：affinity 作用于“一条输出块的完整 K 累加链”，不把整个 tile context 永久绑定到一个 engine。同一 context 的两个独立输出链可以绑定不同 engine。

两个 slot 是同时占用上限，不是为每个 context 永久预留两个物理槽，也不是把全局可驻留 context 数量固定为四个。

不要通过以下方式规避约束：

- 把 per-context 配额改成 per-engine 配额。
- 为新任务增加不计数的“临时 AM slot”。
- 让 one-shot GEMM 绕过配额但仍占用同一 AM。
- 让同一 handle 隐式由多个 engine 并发更新。
- 让已绑定链在原 engine 忙时自动迁移。
- 通过增加大队列、无限缓冲或零延迟 AM，掩盖资源与时序问题。

## 1. 设计目标、基线与边界

### 1.1 设计目标

- 多个矩阵阵列共享阵列组内部的 AM 容量。
- 多 K-step 的 partial C 留在 AM 中累加，避免在每个 K-step 边界经过 Tile L1 读写。
- UCE 继续执行 tile program，保留 wait/event/context switch。
- 设备侧用有限状态表和 ready 仲裁处理已经提交的工作。
- 编译器只管理两个逻辑 AM 位置及相关依赖，不负责轮询物理 bank 或固定物理阵列编号。
- 在有限资源下给出明确的进展条件，不以“工作负载很少出现死锁”作为正确性依据。

### 1.2 推荐实现基线

| 项目                                 | V1 基线                                | 性质             |
| ------------------------------------ | -------------------------------------- | ---------------- |
| 矩阵 engine 数量                     | 4                                      | 沿用本轮讨论场景 |
| AM group 数量                        | 4                                      | 推荐物理组织     |
| 每组逻辑 slot 数量                   | 2                                      | 推荐固定区域划分 |
| 全局 AM slot 总数                    | 8                                      | 与上述组织一致   |
| 每 tile context 的 live AM slot 上限 | 2                                      | 用户硬约束       |
| AM 分配                              | 按输出链动态分配；成功时完整保留资源   | 推荐合同         |
| engine affinity                      | 输出链建立时绑定，整条链不可变         | 用户约束的明确化 |
| 同一 handle 的更新                   | 每次仅一个 K-step 更新                 | V1 正确性要求    |
| 同一 context 的两条链                | 允许独立并行、允许不同 engine          | 推荐合同         |
| 结果输出                             | FINAL 后 drain 到已准入的 L1 输出区域  | 保守 V1 合同     |
| one-shot GEMM                        | 使用一个短生命周期 AM slot，计入配额   | V1 基线          |
| AM 地址/所有权                       | 阵列组内部管理，软件看到 opaque handle | 推荐合同         |

阵列尺寸、accType、精度、频率、SRAM 容量和延迟应优先沿用目标仓库中已确认的配置。本文不会将 FP32、8×8 或 1 GHz 示例擅自变成产品约束。

### 1.3 V1 不包含的能力

以下能力不得作为满足本规范的隐含前提：

- K-step 跨 engine 迁移。
- 多 engine 对同一 C 的 split-K 并发更新。
- 面向软件的任意地址 atomic add。
- AM 状态抢占、spill、eviction、在线搬迁或压缩。
- 动态改变一个驻留 context 的 memory profile。
- 不计入两个 slot 配额的旁路 AM。
- 将整个 tile program 改成通用 DAG 执行器。
- 将 EVU 直接读取 AM 并长期持有 AM 引用作为默认输出合同。

如果仓库已经支持相关功能，应标明它与 V1 路径的关系；不得为实现此规格而无关地删除原功能。

## 2. 对象、身份与所有权

| 对象               | 定义                                                   | 不应混同的对象             |
| ------------------ | ------------------------------------------------------ | -------------------------- |
| Physical Tile      | 包含 UCE、矩阵阵列组、向量设备和本地存储的硬件单元     | 编译器的输出分块           |
| tile context       | 一次 tile program 的驻留执行实例，带独立 context epoch | engine、AM slot、单条 GEMM |
| output tile        | 编译器划分的一个逻辑 C 子块                            | 物理 PE 阵列尺寸           |
| accumulation chain | 对同一个 output tile 完成全部 K-step 的链              | 整个 tile context          |
| logical AM slot    | 某 context 的逻辑位置 0 或 1                           | 固定物理 bank              |
| physical AM slot   | 共享 AM 中分配给一条链的独占地址区域                   | SRAM 端口或 engine         |
| AM group           | 提供独立存储与累加服务的模块，可包含多个 SRAM bank     | 一个软件 context           |
| engine             | 一份矩阵计算资源                                       | AM 所有者                  |
| acc_handle         | 带 generation 的链/slot 引用                           | 可永久复用的裸 slot 编号   |

### 2.1 推荐身份关系

软件看到：

```text
ctx_key    = (tile_context_id, context_epoch)
logical_am = 0 或 1
acc_handle = opaque(entry_id, generation)
```

硬件表记录：

```text
acc_handle ->
    owner_ctx_key
    logical_am
    output_chain_id
    physical_am_group
    physical_region
    bound_engine
    shape / acc_format / layout
    lifecycle_state
```

不要求将这些字段全部逐条暴露给 ISA。实现可以压缩编码，但必须保留等价的身份区分能力。

### 2.2 三个生命周期分离

- AM 区域：链建立成功后保留，直到安全输出完成及所有旧访问结束。
- engine：仅在实际执行步骤及完成必要交接时占用；链等输入时不独占 engine。
- 完成事件：可比 AM 区域存活更久，直到事件已被消费或按协议退休。

硬件 context 切换不释放 AM，也不改变 owner 或 affinity。

## 3. 两个 slot 的准确计数规则

令 L(ctx) 表示某 context 尚未安全释放的 AM lease 集合：

```text
used_am_slots(ctx) = |L(ctx)|
0 <= used_am_slots(ctx) <= 2
sum_ctx used_am_slots(ctx) = global_live_leases <= 8
```

这里的全局等式适用于本 V1 仅由上述 context lease 使用共享槽的基线。

### 3.1 必须计入配额的状态

成功取得 lease 后，下列状态均占一个配额：

- RESERVED：AM 已分配，但 FIRST 尚未提交或执行。
- WAIT_STEP / WAIT_INPUT：等下一步描述符或输入。
- RUNNING：矩阵计算进行中。
- COMMITTING：结果尚在 FIFO、仲裁、RMW 或写回中。
- FINAL_READY：最终 AM 结果已就绪，等待输出。
- DRAINING：最终结果正在输出。
- FAULT_QUIESCING：错误收尾中，尚不能安全复用。

只有满足 slot_free 的全部条件后，才递减计数。

### 3.2 不计入配额的对象

- 尚未获得 lease、并已明确返回 BUSY 的申请。
- 已释放 AM 后单独保留的完成事件记录。
- 不使用 AM 的 UCE、MFE 或向量工作。

one-shot GEMM 在本 V1 中仍经 AM，其 RESERVED 到 DRAINING 过程均计入两个槽。

### 3.3 动态配额，不是永久预留

默认不在 context admission 时无条件预留两个物理槽。

例如：

- 4 个 context 各占 2 槽，可以填满 8 槽。
- 8 个 context 各占 1 槽，也可以填满 8 槽；前提是 UCE 等其他资源允许这些 context 驻留。
- 某 context 占 2 槽时，即使全局仍空着 6 槽，也不能再成功取得第三槽。
- 某 context 释放一槽后，可再次动态申请一个共享槽；不要求获得相同物理位置或相同 engine。

若未来某类程序必须同时保有两槽才能推进，应显式增加相应的成组资源申请合同。该能力不是 quota=2 的自动推论，也不是本 V1 的默认行为。

### 3.4 对单 context 并行度的影响

本 V1 规定：

```text
一条链 -> 一个 live slot -> 一个固定 engine
一个 context -> 最多两条 live 链
```

所以，一个 context 同时执行其矩阵工作所占用的 engine 数至多为 2。

四个 engine 同时计算，至少需要两个有足够 ready 工作的 context，并且链分配覆盖不同 engine、输入与 AM 带宽可用。这只是必要条件，不是利用率保证。

FINAL_READY/DRAINING 仍计配额，因此两个槽都被占用时，实际正在计算的阵列数也可能为零。不得把该现象直接标成硬件死锁。

## 4. 规范需求索引

| ID  | 必须满足的行为                                                                        |
| --- | ------------------------------------------------------------------------------------- |
| R01 | 四个 engine 可访问共享 AM 中合法分配给其链的区域，不保留隐含的每 engine 两槽硬限制。  |
| R02 | per-context live lease 上限为 2，覆盖全部占用状态，硬件/模拟器均校验。                |
| R03 | 同一 handle 的 bound_engine 从建立到释放不可变。                                      |
| R04 | 不得仅因两条链属于同一 context 而增加顺序依赖；实际发射仍遵守数据依赖与共享资源限制。 |
| R05 | 一个 slot 只有一个链 owner；同一 handle 同时最多一个更新中的 K-step。                 |
| R06 | TRY_OPEN 成功时完整取得槽、配额、必要元数据与完成记录；失败无持久副作用。             |
| R07 | 新申请不得阻断已有链的 STEP、FINAL、输入推进和输出释放。                              |
| R08 | 步骤接收、发射、计算结束、AM 提交、输出可见和槽释放必须区分。                         |
| R09 | 发射不可暂停的计算块前，必须取得覆盖其在途结果的接收能力。                            |
| R10 | AM 端口、bank、adder 吞吐分别建模，同地址 RMW 必须互锁。                              |
| R11 | INIT/ADD、有效 mask、累加精度和最终转换时机符合数学与数据格式合同。                   |
| R12 | handle/generation/context epoch 校验防止旧消息影响新 owner。                          |
| R13 | 事件持久保存，等待原子地检查状态并登记，不能丢唤醒。                                  |
| R14 | input_release 以真实最后读取及全部读者结束为依据，独立于 C 的生命周期。               |
| R15 | STEP 接收后描述符和执行配置语义已固定，不受软件后续改写影响。                         |
| R16 | FINAL 自动触发输出与安全释放；正常释放不依赖 UCE 轮询。                               |
| R17 | slot_free 后归还配额；context retire 前必须完成其未结束的资源与引用收尾。             |
| R18 | 提供按原因分解的等待计数与关联 trace，完成指定验收场景。                              |

## 5. 硬件组织建议

### 5.1 顶层结构

![共享 AM 顶层结构：UCE 经共享 AM 表分发到 Engine 0-3，结果经路由仲裁写入 AM group 0-3，再由输出 drain 写回 L1](./design/image/nexus_shared_am_top_structure.svg)

每个 AM group 包含 SRAM、局部 RMW 累加通路、请求控制和写回状态。group 可以内部拆分多个物理 SRAM bank，逻辑 slot 不必对应一块宏。

### 5.2 最小模块职责

| 建议逻辑模块       | 职责                                                |
| ------------------ | --------------------------------------------------- |
| SharedAmAllocator  | 共享区域分配、逻辑槽映射、context 配额和 generation |
| ChainTable         | owner、affinity、步骤顺序、配置及生命周期           |
| StepMailbox        | 为每条已建立链保留一个下一步骤描述符位置            |
| MatrixReadyArbiter | 只在绑定 engine 上选择满足条件的链                  |
| ResultCapture      | 接收阵列结果，维护预留 credit 与在途记录            |
| AmRouter           | 将合法结果送到目标 AM group                         |
| AmRmwController    | 读旧值、局部累加、写回与地址互锁                    |
| AmDrainController  | 最终结果输出、目的地可见性和释放                    |
| EventTracker       | 粘性事件、等待者、完成记录和安全退休                |

这些模块可以合并或嵌入仓库现有模块，不能只为名称一致而复制重复的 scheduler/event 系统。

### 5.3 存储布局

推荐 V1 使用固定大小 slot，一个输出块完整落在一个 AM group 内。每组两个固定区域，槽内采用稳定布局。

每个 slot 的容量计算必须包含：

- M/N 与 stride；
- accType 字节数；
- 对齐、padding；
- 实际分配粒度。

若 shape 超出单槽容量，明确返回 unsupported/size error，由编译器重新 tiling 或采用已有合法路径。不得越界占用邻槽，也不得在不计配额的情况下隐式扩槽。

示例：64×64、32 bit 累加数据需要 16 KiB；八槽共 128 KiB。该数字只说明容量关系。

### 5.4 AM group 与 engine 独立

一个合法示例：

| Context/逻辑槽 | 物理 AM 位置     | 固定 engine |
| -------------- | ---------------- | ----------- |
| A / 0          | group 2 / slot 0 | engine 0    |
| A / 1          | group 0 / slot 1 | engine 2    |
| B / 0          | group 3 / slot 0 | engine 0    |
| B / 1          | group 1 / slot 0 | engine 3    |

A/0 和 B/0 可以在 engine 0 上交替执行，但不能同一时刻占用同一份计算资源。A/0 等输入时，不妨碍 engine 0 执行 B/0。

## 6. 建议的软件/设备接口合同

本节名称是协议伪接口，不是已确定的 RISC-V opcode 或仓库 API。

### 6.1 TRY_OPEN：非阻塞建立输出链

```text
TRY_OPEN(
    ctx_key,
    logical_am,        // 0 或 1
    chain_id,
    output_shape,
    acc_format,
    layout,
    output_destination,
    chain_config
) -> GRANTED(handle, bound_engine, completion_token)
   | BUSY(reason, retry_event_snapshot)
   | ERROR(reason)
```

成功的线性化点必须原子完成：

1. 检查 context 身份与逻辑位置。
2. 检查该 context 未达到两个 live lease。
3. 选取全局空闲物理 slot。
4. 取得 chain table、mailbox、最终完成状态记录等必要资源。
5. 设置 owner/generation，并固定 bound_engine。
6. 将 slot 置为 RESERVED，计数加一。
7. 返回成功的 handle。

bound_engine 在此处是 affinity，不是对该 engine 整条链生命周期的独占预约。可依据兼容性和负载选择暂时忙碌的 engine，后续等待实际执行机会。

completion_token 对应本链的 chain_done：最终输出已可见、旧 AM 访问已结束、slot 已释放且 quota 已归还。提前通知结果可读时使用独立的 output_visible 事件。完成记录可以提前保留/写入，但该 token 只有在上述条件全部成立后才被标为满足；故障则记录显式失败状态，不能伪装成功。

推荐选择策略：兼容的 engine 中优先选择已绑定待运行工作较少者；AM group 优先选择空闲且预计争用较低者。该策略不要求每次产生唯一或全局最优放置。

失败合同：

- BUSY 不占 AM、不增加配额、不留下半初始化 handle。
- 区分本 context 配额不足、逻辑槽仍占用、全局 AM 不足、元数据/事件容量不足。
- 操作可受有限接口仲裁延迟影响，但不得因缺 AM 长期阻塞 UCE，必须返回 BUSY。
- 传输请求入队不等于 GRANTED。
- 重试必须明确仅针对“未成功”的请求；不得重复执行已经成功的 OPEN。若互连允许重试/重放，应具备 request identity 或等价的 exactly-once 处理。

BUSY 的失败判断与 retry_event_snapshot 必须在同一个资源线性化点取得，或使用等价的电平式“资源现在可用”等待条件。不能先判失败、等别的请求释放资源后，再读取较新的credit版本让调用方等待下一次变化。credit唤醒只表示允许重试，不保证资源已为该请求保留。该要求同样适用于mailbox、完成记录及MFE信用等待。

成功后暂不要求物理 SRAM 清零；必须在第一次合法写入时正确初始化，且初始化前禁止读取该链有效数据。

### 6.2 TRY_SUBMIT_STEP：提交同一链的下一 K-step

```text
TRY_SUBMIT_STEP(
    handle,
    seq,
    A_descriptor,
    B_descriptor,
    input_ready_tokens,
    input_usage_handles,
    k_extent,
    INIT_or_ADD,
    FINAL
) -> ACCEPTED(step_token)
   | BUSY(mailbox_credit_event)
   | ERROR(reason)
```

要求：

- 验证 owner、generation、context epoch、affinity 和步骤序号。
- seq 从 0 连续递增；拒绝重复、跳号、FINAL 后再次提交等非法行为。
- seq=0 为 INIT；后续步骤为 ADD；只有最后一步带 FINAL。
- 单步 GEMM 使用 INIT|FINAL，并占一个槽。
- 第一次接收 FINAL 后关闭该 handle 的新步骤入口。
- ACCEPTED 表示描述符已被快照或完整转存到受保护的硬件存储；软件可复用提交描述符本体。
- A/B 数据缓冲不能据此释放，必须等 input_release。
- 可以接收尚未 input-ready 的描述符；依赖 token 由设备等待。
- 既有链的 mailbox/提交路径不能被等待新 AM 的 OPEN 队列占满。

推荐每条链一个当前执行记录加一个 next-step mailbox。mailbox 满返回 BUSY，不能扩成无限步骤队列。seq 计数应区分 submitted、issued、committed，避免仅用一个游标混淆接收与完成。

### 6.3 等待接口

使用现有事件系统的等价能力，不强制新增具体 ISA：

```text
WAIT(event_token)
WAIT_ANY(event_tokens)
```

等待必须原子地完成“检查是否已完成”和“登记等待者”。事件先发生、后执行 WAIT 时，WAIT 应立即满足。

token 必须唯一标识一个事件实例。可以使用独立 event ID，也可以通过 token 与内部表等价关联 (ctx_key, handle entry_id, handle generation, 步骤序号, 事件种类)。只含 epoch/generation 的数值不足以区别不同 context 或不同 entry；不能只依赖裸物理 slot 编号或一个周期的 done 脉冲。

当 AM 被复用时，旧完成事件不能被覆盖。可以使用独立完成表或有序完成计数等实现；必须证明保留期限与有界复用正确，不能仅以 generation 位数较大来代替协议。

### 6.4 正常释放与 context 结束

正常路径不要求 UCE 显式 FREE：

```text
FINAL step committed
    -> drain to admitted destination
    -> output visibility established
    -> old AM accesses quiescent
    -> record completion
    -> slot_free and quota decrement
```

输出完成记录应在 OPEN 成功时已有容量保证，避免完成队列满阻止所有 slot 释放。

context 返回时如仍有合法已提交 FINAL/drain，进入 FINISHING，继续接收完成并等待资源归还。若存在未提交 FINAL 的活链，属于程序/协议错误，不能把 partial C 当作正常结果，也不能直接清零配额。

V1 不使用后台 detach。context ID/epoch 只在所有命令、输入引用、AM 访问与事件引用完成或按明确合同退休后复用。

FINISHING 的硬件收尾不得依赖已经返回的原 context 再运行并亲自消费完成事件。完成必要的通知和外部引用处理后，相关记录按既定协议自动退休；有外部等待者时仍须保证其能够观察完成。

## 7. 状态机与完成边界

### 7.1 每条链的状态

![AM slot 每条链的状态机：FREE、RESERVED、WAIT_STEP、WAIT_READY、RUNNING、COMMITTING、FINAL_READY、DRAINING](./design/image/nexus_shared_am_slot_state_machine.svg)

next-step mailbox 可以在 RUNNING/COMMITTING 时预先收到下一步。状态机里的 WAIT_STEP 表示当前链没有可发射步骤；若已有 mailbox，前步提交后可直接转 WAIT_READY。

### 7.2 事件的严格定义

| 事件           | 含义                                                | 不能据此推导                |
| -------------- | --------------------------------------------------- | --------------------------- |
| step_accepted  | 描述符已安全接收                                    | 输入已读取、计算已开始      |
| compute_done   | 本步骤全部结果已产生并被受保护的接收路径接住        | AM 写入已提交               |
| step_committed | 本步骤全部 AM 更新完成，满足下一步读取/累加的可见性 | 最终输出已到 L1             |
| input_release  | 指定输入缓冲在指定层级已无剩余读取者                | C 已提交或 slot 已释放      |
| output_visible | FINAL 结果已在约定 L1 目的区域达到可见性条件        | 所有其他资源引用均已退休    |
| slot_free      | 本代 AM 区域可安全再分配，配额已归还                | 旧事件可以覆盖新 generation |

保守 V1 可让 output_visible 和 slot_free 出现在同一个完成处理阶段，但逻辑定义仍分开。共享 bank 端口和事件记录可能使其周期不同。

### 7.3 步骤提交条件

```text
step_committed =
    end_of_step_seen
    AND all_expected_results_received
    AND pending_am_updates == 0
```

pending_am_updates 必须包含结果接收、路由、队列、RMW 流水线和待写回中的未完成更新。不能在某一周期计数恰好为零、但阵列仍会继续输出时提前完成。

同一 handle 的下一步发射必须等待前一步 step_committed。V1 可保守地在 step_committed 时释放当前 engine 的步骤占用；不能等到整条链 FINAL drain 才允许 engine 执行其他链。

### 7.4 错误与安全回收

- 非法 handle、owner、seq、shape 或 mask 应在产生 AM 副作用之前拒绝。
- 中途故障应停止该链新发射，完成或安全终止已接受的在途访问，再回收资源。
- 不得在 timeout 后直接将 slot 标成 FREE，让旧写回落到新 owner 上。
- generation 能帮助拒绝旧消息，不能替代在途访问的 quiescence。
- V1 不要求实现任意时刻的细粒度抢占；若仅支持受控排空/复位恢复，应显式报告限制与错误状态。
- 故障链不得产生正常成功的 output_visible。

完成记录必须区分成功与失败的终止状态。建议使用 chain_terminal(status)：成功仅在 chain_done 条件成立后记录 SUCCESS；故障/取消则在旧访问安全结束、slot及配额回收后记录 ERROR/CANCELLED，不要求产生成功的 output_visible。软件不得只等待成功事件来回收失败链。

context发生终止性错误时，停止生成新工作，通知设备按受控收尾协议处理其余已拥有链和已接受操作，安全回收后退休事件和context。未完成的partial C可以在故障收尾后丢弃，但不能报告成功结果。若实现只能通过受控复位恢复，必须显式暴露该状态，不得永远停留在正常WAIT中。

## 8. UCE、编译器与设备调度的职责

### 8.1 编译器/软件负责

- 每个 context 使用逻辑位置 0/1，保证最多两个 live handle。
- 为同一输出链提交正确的 seq、INIT/ADD 和唯一 FINAL。
- 在真正依赖结果的位置 WAIT，允许独立工作先提交。
- 维持输入缓冲的 producer/consumer 引用，不提前复用。
- 安排两槽 prologue、steady state、epilogue，处理不同 K 长度与尾块。
- 不在一个新 OPEN 上阻塞已有链的续算生产者。

不要求软件指定物理 AM bank、固定 engine 编号或对共享 SRAM执行 atomic。

### 8.2 UCE context scheduler 负责

- 执行当前 context 的标量指令与提交操作。
- WAIT 条件未满足时保留 PC/寄存器状态，切换其他可运行 context。
- 正确处理事件先到、等待后登记的情况。
- 不把 context 切换视作 AM 释放、链终止或 affinity 变更。

### 8.3 阵列组 ready 仲裁负责

一个已提交步骤可发射，至少要同时满足：

```text
handle有效
AND owner/seq合法
AND 前一步已committed
AND 输入ready
AND 绑定engine可接收
AND 所需AM/结果接收能力已取得
AND 配置切换处于合法边界
```

调度器不能绕过 affinity 去选择其他空闲 engine，也不能执行尚未提交的步骤。

已有链继续执行不需要新 AM slot。新 OPEN 不得占满 continuation 路径。相同 context 的两条链可独立 ready，不得仅因共享 context 而人为增加程序顺序依赖；实际发射仍受 bound_engine、AM bank/端口、结果 credit 及其他共享资源限制。

建议在竞争时优先保障能释放状态的 drain 与已驻留链推进，并使用轮转/年龄等公平策略，避免无限饿死。优先级只在 ready 候选中选择，不能空等一个未 ready 的高优先级任务。

## 9. 进展合同：两个槽不自动消除死锁

### 9.1 每条已建立链必须能够独立释放

V1 支持的普通输出累加链在取得一个 AM slot 后，必须能够使用这同一个 slot 完成：

```text
INIT -> ADD ... -> FINAL -> DRAIN -> RELEASE
```

完成该链不能依赖另一次新的 AM 申请成功。输入可以逐段到达，不要求完整 K 的全部数据同时驻留。

若某个算法必须同时持有两条链才能让任何一条链释放，应使用另外显式定义的成组资源合同或调整分块/输出路径，不能偷偷套入普通动态分配。

### 9.2 配额引起的自锁反例

即使全局还有六个空槽，也可能出现：

```text
context A:
    open C0                  // 使用1槽
    open C1                  // 使用2槽
    submit C0.K0
    submit C1.K0
    blocking_open C2          // 因本context配额满而停止
    submit C0.FINAL           // 永远无法到达
    submit C1.FINAL
```

因此，TRY_OPEN 必须返回 BUSY；调用方也不能无限重试同一个新 OPEN 而不服务已有链。

UCE 切换只能让其他 context 执行，不能替当前 context 生成尚未提交的 FINAL。设备 Ready 仲裁也不能调度尚未生成的动作。

### 9.3 已有链的生产与提交必须可继续

- 已有链的 next-step mailbox 不由新 OPEN 共享并耗尽。
- 本 V1 的 input_ready token 应指向已经 ready 的输入，或已经被 MFE/producer 系统受理的生产操作。
- 不允许先提交对任意未来软件信号的等待，再把生产该信号的操作放到可能阻塞的位置之后，除非已有明确的进展证明。
- 先服务已有链的供数/续步，再用空闲逻辑位置尝试新链。
- mailbox 空间不足时，等待能使该 mailbox 腾出的已接受工作；不能丢失或重复提交步骤。
- 不提前占用 engine 或 bank 服务资格，然后无限等待未提交的输入。

### 9.4 延续现有 memory admission

沿用 NEST / tile 的既有 memory contract：

- 一个驻留 context 只绑定一个 memory profile。
- 已建立链具有后续输入窗口的推进条件。
- FINAL 的 L1 目的区域在准入合同中可用。
- 不让等 AM 的新任务预取耗尽旧链完成所必需的 L1 输出空间。
- DMA/MFE、结果接收和最终输出有可推进的请求/响应通路。

不必额外永久保留一个空 AM slot。需要保护的是旧链的执行和释放路径。

### 9.5 条件化的进展论证

在以下前提下，可以论证满槽只造成背压：

1. 每条已建立链只有有限个 K-step。
2. 已受理输入生产最终完成。
3. UCE 会继续生成已有链的合法续步，不被新申请阻断。
4. ready 步骤与 drain 获得公平服务。
5. 最终输出通路可推进。
6. 硬件安全释放不依赖 UCE 轮询。

同一链无需新 AM 即可逐步到达 FINAL 并释放；因而至少一条有进展条件的链最终归还槽。上述任一前提被破坏时，不得声称仅凭 quota=2 就保证无死锁。

## 10. 两槽软件流水参考模板

以下是协议伪代码，可由编译器静态展开为 prologue/steady state/epilogue，不要求新增一个通用软件 DAG 调度器。

```text
live[0] = empty
live[1] = empty

while 存在未开始输出 or 任一live位置非empty:
    changed = collect_completed_events()

    for i in [0, 1]:
        if live[i]存在 and chain_terminal(live[i])为ERROR或CANCELLED:
            return fail_context_and_quiesce_owned_work(live)
        if live[i]存在 and chain_done(live[i]):
            reap_completion_record(live[i])
            live[i] = empty
            changed = true

    # 优先服务已经持有AM的链，两个位置轮转
    for i in round_robin([0, 1]):
        if live[i]存在:
            changed |= drive_chain_nonblocking(live[i])

    # 仅对空的软件逻辑位置尝试新任务
    for i in [0, 1]:
        if live[i]为空 and 存在未开始输出:
            r = TRY_OPEN(ctx, logical_am=i, next_output_descriptor)

            if r == GRANTED:
                live[i] = new_chain_state(r.handle, r.bound_engine)
                consume_next_output_descriptor()
                changed = true
                changed |= drive_chain_nonblocking(live[i])
            elif r == BUSY:
                remember_retry_snapshot(i, r)
                # 不循环重试到成功，不阻止旧链推进
            else:
                return fail_context_and_quiesce_owned_work(live)

    if changed:
        continue

    wait_set = unresolved_progress_events_of(live)
    wait_set += relevant_open_retry_events_for_empty_positions()

    assert wait_set不为空
    WAIT_ANY(wait_set)
```

drive_chain_nonblocking 的职责：

1. 处理该链的输入读取完成和输入缓冲释放。
2. 在缓冲合法可用时，尝试启动下一段 MFE 输入生产。
3. MFE 请求已受理后，才把对应 ready token 放入 STEP。
4. 尝试提交 seq 指定的下一步；只有 ACCEPTED 才增加提交游标。
5. seq=0 使用 INIT；最后一步同时带 FINAL。
6. FINAL accepted 后停止生成新步骤，只追踪完成。
7. 任意资源不足都返回调用方继续服务另一条链。

资源BUSY与终止性ERROR必须区别。drive_chain_nonblocking遇到同步ERROR时，转入同一fail_context_and_quiesce_owned_work路径并结束正常提交循环；实现可用显式结果类型或受控异常表达。该错误路径不等待失败链的成功chain_done，而是驱动已有操作安全收尾，观察terminal状态并退休相关记录。

### 10.1 完成记录的有限实现建议

可为每个 context 的两个逻辑 AM 位置各保留一个 chain-event record：

- OPEN 成功时占用记录。
- 保存此代身份、已提交进度、最终完成或错误。
- AM 可以在安全 drain 后归还共享池。
- 记录保留到软件 reap，随后允许同一逻辑位置建立下一条链。
- 若物理 AM 已释放而记录尚未 reap，新 OPEN 可返回“完成记录未空闲”，不能伪装成 AM quota 仍被占用。

chain_done 可以作为以下条件的便捷组合，不要求新增另一套硬件完成队列：

```text
chain_done = FINAL输出已可见 AND slot已安全释放
```

如果已有事件系统能保留多代记录，可以复用它。无论采用哪种实现，物理 slot 的复用不能清掉旧完成状态。

### 10.2 软件 live 状态与硬件 quota 的关系

硬件可能先释放 AM，而 UCE 稍后才观察 chain_done。此期间软件仍保守认为逻辑位置占用，但硬件 used_am_slots 已减一。

这是允许的；不得因为软件尚未 reap 就推迟物理槽回收，也不得让软件在未确认旧逻辑位置可安全重用时直接覆盖事件记录。

## 11. AM 累加、结果信用与输出硬件

### 11.1 局部 RMW

推荐每个 AM group 的累加流程：

```text
新partial结果 -> 路由到目标group
group内部读旧C -> 向量adder -> 写回同一AM位置
```

旧 C 不需要回送到计算阵列。四个 group 应具有明确、可独立计量的累加吞吐。

单个 group 的能力取决于 SRAM 端口、adder lane 数、流水线和访问冲突。增加 bank 数量不能在模型中自动增加共享 adder 的吞吐。

### 11.2 INIT、内部微块与精度

- 新链必须覆盖/初始化全部有效输出位置，不读取上一个 owner 的残留数据。
- 如果 INIT 所代表的一个逻辑 K-step 内部仍会多次更新同一输出位置，只有该位置首次更新是覆盖，后续微块必须累加。
- bias/初始 C 的语义应沿用既有数学合同，只应用一次。
- 中间结果保持规定的 accType；激活、缩位、量化等最终处理按合同执行，不能每个 K-step 都提前应用。
- 同一链的 shape、布局和累加类型保持一致。
- 浮点参考模型必须采用规定的运算与分组顺序；固定 seq 不自动等于单次完整 FMA 的逐 bit 结果。
- 整数溢出、饱和/回绕和舍入策略沿用已冻结规格；不得由实施模型自行选择。
- 尾块 mask、padding、空任务和零 K 必须显式处理。可以由编译器 lower 为已有合法路径，但不能等待永远不会到达的 last beat。

### 11.3 地址互锁

同一 handle 仅一个 K-step 更新，不代表内部永远没有同地址 hazard。

必须处理：

- 同一行的旧写尚未提交，下一次 RMW 已到达。
- FINAL 读取与最后一批写回重叠。
- 物理地址相同但控制位不同的请求。
- 部分 mask 与未写入位置。
- 不同 slot 的物理区域不得重叠。

可采用已有 bank 内互锁、行 busy 标记或 forwarding；V1 不要求通用多 writer 原子单元。

### 11.4 输出接收 credit

若启动后会产生不可任意暂停的输出，则发射前必须证明有接收空间或保证服务。

以某个实际结果FIFO为例，令 O 为已经到达且仍占用FIFO的beat数，R为已为未来不可撤回输出预留但尚未到达的beat数，Q为该FIFO容量，必须始终满足 O + R <= Q。发射时增加R，结果到达时将对应计数从R转到O，安全出队时减少O。下一级必须同时具备自己的接收保证；不能把不同级不可互换的缓冲容量简单相加后宣称安全。

实现也可以保守地把端到端credit保持到AM提交，但必须给出与实际容量对应的计数不变量，并将其与pending_am_updates区分。

推荐从最小不可暂停的物理计算块预留，而非一次预留整个大逻辑 output tile：

- 新微块发射前预留其最坏输出量。
- 计入所有已经在途、尚未产生的结果。
- FIFO 当前占用数不能替代总在途预留数。
- 归还 credit 必须对应真实容量释放或向另一份有保证的缓冲转移。
- 一个输出块的最坏结果量超过全部保证容量时，必须进一步在合法边界分块或拒绝该发射。
- 若采用保证 bank 服务的方式代替完整缓存，必须明确保证窗口和最坏冲突，不能仅检查当前周期 bank 空闲。

### 11.5 bank/group 服务与公平性

第一版可以在一个最小输出块发射时取得目标 AM group 的服务资格，完成该块后归还。不要为整条等待输入的 K 链永久锁住整个 group。

多个 group 的资源如需一起取得，应全成或全败，避免“拿住部分 group 等其余 group”。

输出 drain 在共享读端口上必须获得服务。连续 ACCUM 不能无限饿死 drain，否则占用槽无法释放。也不能全局串行化所有 compute 和 drain；不同端口/不同 group 允许的重叠应由模型表达。

### 11.6 带宽核算示例

定义 E=阵列数，q=输出阶段每阵列每周期结果元素数，b=累加元素字节数。

```text
新结果流量     = E × q × b  B/cycle
ACCUM旧值读取  = E × q × b  B/cycle
ACCUM结果写回  = E × q × b  B/cycle
最终drain读取  = 另外计入
```

例：E=4，q=8，b=4，则新结果128 B/cycle，AM读旧值128 B/cycle、写回128 B/cycle。

这是输出阶段的接口需求推导，不是测量数据，不等于平均流量，不等于L1/HBM需求。实际吞吐必须来自所实现的数据流、K分块和请求轨迹。

## 12. 功能模型、周期模型与 RTL 的改造范围

### 12.1 先建立实际仓库映射

实施模型先输出以下表，使用真实路径和符号：

| 概念                | 现有对象/路径 | 当前语义 | 需要变更的语义           | 兼容性影响 |
| ------------------- | ------------- | -------- | ------------------------ | ---------- |
| tile context与epoch | 待仓库勘察    | 待确认   | 加入/绑定AM配额所有者    | 待确认     |
| GEMM提交与K链       | 待仓库勘察    | 待确认   | handle/seq/FINAL合同     | 待确认     |
| engine选择          | 待仓库勘察    | 待确认   | 链级固定affinity         | 待确认     |
| AM或累加结果模型    | 待仓库勘察    | 待确认   | 共享槽、RMW、提交事件    | 待确认     |
| event与wait         | 待仓库勘察    | 待确认   | 持久记录、原子等待、代际 | 待确认     |
| MFE与input_release  | 待仓库勘察    | 待确认   | 供数进展、真实最后读取   | 待确认     |
| 输出与资源回收      | 待仓库勘察    | 待确认   | drain/visible/free边界   | 待确认     |

不得把本表的“待确认”当成实现时可以跳过的内容。

### 12.2 功能模拟器

必须表达：

- context quota 和全局 slot ownership；
- 不可变 affinity；
- 合法/非法步骤序列；
- INIT/ADD/FINAL 数据语义或明确的抽象结果合同；
- one-shot 的相同配额规则；
- 完成和释放的不同事件；
- 描述符快照、epoch/generation 与安全重用；
- 失败无副作用、错误收尾不提前复用。

功能模拟器可以没有准确周期，但不能把 compute_done、step_committed、output_visible、slot_free 合并为一个含糊的 done。

### 12.3 周期模型

在功能合同上增加：

- 每个 engine 的执行占用和配置切换边界；
- 不可暂停结果的预留 credit；
- 有限 FIFO、路由竞争；
- AM bank读写端口和adder吞吐；
- RMW/writeback 延迟与同地址 hazard；
- drain 的端口、目的地反压及完成延迟；
- UCE 发令/等待及 mailbox 容量限制。

不要求功能模型与 RTL 逐周期一致，但必须以同一事件依赖合同解释结果。

### 12.4 RTL

建议边界：

- 复用矩阵计算单元与兼容的单引擎执行控制。
- 把AM区域映射、所有权和跨engine依赖放在共享层。
- 将现有单执行器的AM连接改为有来源身份的多执行器访问。
- 所有返回/完成信息能够找到正确的handle和context。
- 结果端增加真实接收保障；不能仅在下游加ready而假定整条阵列可停。
- 保持每个engine的配置在安全边界切换；必要字段在描述符/当前步骤中快照。
- 最终提交确认来自共享AM路径的真实完成，而不是原始mesh输出末尾。

若目标仓库尚无 RTL，本次改造应先完成已有模拟器和规格接口，不要编造“已接入RTL”的成果。

## 13. 参数与配置建议

以下键是语义示意，不是可以直接覆盖现有仓库的配置文件：

```yaml
matrix_group:
  num_engines: 4

  shared_am:
    enabled: true
    num_groups: 4
    slots_per_group: 2
    total_slots: 8
    allocation_policy: dynamic_per_chain
    max_live_slots_per_tile_context: 2
    physical_slot_mapping: fixed_region_in_one_group

  execution:
    affinity_scope: output_chain
    bind_affinity_at: successful_open
    allow_engine_migration: false
    allow_parallel_steps_same_handle: false
    allow_split_k_shared_output: false

  submission:
    open_semantics: try_open_all_or_nothing
    step_semantics: try_submit_snapshot
    next_step_mailboxes_per_live_chain: 1

  output:
    one_shot_uses_am_slot: true
    automatic_final_drain: true
    destination_contract: admitted_l1_region
    release_after: output_visible_and_old_accesses_quiescent

  events:
    persistent_completion: true
    atomic_check_and_wait: true
```

以下参数必须由真实配置/建模决定，不在本文虚构固定值：

- slot_capacity_bytes、slot alignment；
- accType、输入/输出精度与舍入行为；
- SRAM宏端口、子bank数、读写延迟；
- group adder lane 数与延迟；
- ResultCapture 容量、最大在途微块与credit粒度；
- UCE context容量、事件表容量；
- drain宽度、L1写入可见性点与MFE资源配额。

配置验证应拒绝 total_slots 与分组组织不一致、slot容量不足、缺失接收保障，以及将用户配额改成大于2的V1配置。

## 14. 验收矩阵

“通过”必须来自实际执行和观察；仅新增测试文件或打印预期日志不算通过。

| 测试ID | 场景                                                                                             | 必须观察到的结果                                                                                                   | 对应需求    |
| ------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ----------- |
| T01    | 同一context连续取得两槽后申请第三槽，全局仍有六槽空闲                                            | 第三次立即返回BUSY且无副作用；原两链可继续FINAL并释放                                                              | R02,R06,R07 |
| T02    | 多context竞争全局槽；包含八个context各一槽                                                       | 全局不超过8；不隐含最多4个context或每engine两槽                                                                    | R01,R02     |
| T03    | RESERVED、WAIT_INPUT、FINAL_READY、DRAINING、one-shot                                            | 全部正确计入配额；仅slot_free递减一次                                                                              | R02,R08,R16 |
| T04    | 同一context两链分配到不同engine                                                                  | 两链可独立推进；每条链全部K-step仍在其绑定engine                                                                   | R03,R04     |
| T05    | 原engine忙、其他engine空闲                                                                       | ready链等待原engine，不自动迁移；报告affinity相关等待                                                              | R03,R18     |
| T06    | 一个engine服务多个context/超过两个全局handle                                                     | 无隐含两个本地AM元数据上限；按步骤公平交替                                                                         | R01,R04     |
| T07    | AM先填垃圾，再执行单段、多段、尾块和有初值场景                                                   | INIT不读残留；bias只一次；mask/accType/最终转换正确                                                                | R11         |
| T08    | 重复seq、跳号、重复INIT、FINAL后续步、错误owner                                                  | 拒绝且不修改AM、不泄漏资源、不重复累加                                                                             | R05,R11,R12 |
| T09    | 最后一行已产生，但FIFO/AM写回故意延迟                                                            | compute_done可以发生；step_committed及最终输出不能提前                                                             | R08         |
| T10    | 四engine最大结果突发，credit不足/恰好够/多个块在途；含Q=8、O=6、R=2                              | 不足不发射；该满额示例不能再预留新结果；到达时R转O而非释放总容量；无溢出和重复归还                                 | R09         |
| T11    | 不同AM group并行与同group热点竞争                                                                | 吞吐不超过实际bank/adder能力；冲突等待可观测且公平                                                                 | R10,R18     |
| T12    | 同地址RMW、未提交写后的读取、部分mask                                                            | 不读旧值、不丢更新；控制位不能绕开物理地址互锁                                                                     | R10,R11     |
| T13    | 全局八槽满，新OPEN持续到达                                                                       | 已有链mailbox、供数、FINAL与drain仍可推进，最终释放                                                                | R07,R16     |
| T14    | 一链mailbox满、另一链可推进；提交后覆盖原描述符                                                  | 满者返回BUSY且seq不变；另一链推进；已接受参数不变                                                                  | R04,R15     |
| T15    | 输入多个读者、延迟最后读取、释放后立即覆写输入                                                   | 释放时机正确，后续结果不受覆写影响                                                                                 | R14         |
| T16    | 事件早到、WAIT登记边界到达、多事件不同顺序完成；资源释放位于失败判断与BUSY返回之间且之后不再变化 | 无丢唤醒；BUSY快照不能遗漏这次释放，等待仍允许重试；已处理事件不导致忙循环                                         | R13         |
| T17    | slot复用后注入旧generation结果、释放和完成消息                                                   | 新链数据/计数不受影响；原owner仍能观察旧完成                                                                       | R12,R13     |
| T18    | 不同context交错stride/transpose/scale/activation                                                 | 配置不串扰，不能修改仍有在途访问依赖的配置                                                                         | R15         |
| T19    | 正常context退出时仍在drain；另测缺FINAL退出、中途故障及部分成功部分失败                          | 正常进入FINISHING；错误安全收尾后产生失败terminal并回收；不等永远不会到来的成功chain_done；不提前清quota或复用身份 | R17         |
| T20    | 单context四个独立输出，与两个context各两输出对比                                                 | 前者至多两个engine同时计算；后者仅在资源满足时可覆盖四engine                                                       | R02,R03,R18 |
| T21    | OPEN过程中元数据/事件容量不足                                                                    | 全成或全败；无占槽后拿不到完成记录的半成功状态                                                                     | R06,R13     |
| T22    | FINAL drain受到L1反压，同时有持续ACCUM                                                           | 反压不丢数据；在下游恢复且公平仲裁下最终完成，释放不需UCE轮询                                                      | R07,R08,R16 |

T10/T11/T12/T22 在只有功能模型时不能被声称完成周期/RTL验证。交付报告必须标注各测试验证到的层级。

### 14.1 最小不变量断言

- 每context live lease数在0..2，全局live lease数在0..8。
- 分配映射一一对应，无两个live owner重叠物理区域。
- 每次成功OPEN恰好加一，每次安全释放恰好减一，失败不改变计数。
- 对live handle，bound_engine恒定。
- 同一handle最多一个issued但未committed步骤。
- committed_seq不超过submitted_seq，步骤按规定次序推进。
- slot_free时没有待处理步骤、在途AM访问或未完成drain。
- 每个接收缓冲的尚未到达预留量R加实际占用O不超过容量Q；结果到达时R减、O增，不重复计数或提前归还。
- 旧generation消息不能改变当前代的状态或AM数据。
- 正常context retire时不存在未收尾的AM所有权。

## 15. 性能观测与判断标准

新增或映射以下统计，不要求使用本文的精确名称：

| 类别       | 建议观测                                                     |
| ---------- | ------------------------------------------------------------ |
| 配额       | per_context_quota_block、global_am_full、logical_slot_in_use |
| 元数据     | mailbox_full、event_record_full                              |
| engine     | 每engine有效计算周期、等待输入、等待绑定engine               |
| AM         | 每group更新量、bank/adder冲突、读写端口占用                  |
| 结果接收   | result_credit_block、预留量与FIFO高水位                      |
| 输出       | final_ready驻留时间、drain反压、slot释放延迟                 |
| 链生命周期 | OPEN至FIRST、步骤间空洞、FINAL至FREE、总lease驻留时间        |

性能对比必须保持总MAC、AM容量、输入/输出带宽、时钟与精度一致。不要把增加SRAM或端口带来的收益误归因于“共享调度”。

必须至少比较：

1. 私有AM与共享AM，在输出到达/驻留不均衡时的容量利用。
2. 低冲突与热点AM group的差异。
3. 单context两槽与多context填满四engine的差异。
4. FINAL排空占据配额时的短任务等待。

不预先承诺共享AM一定加速。若负载规则、原私有布局已经平衡，互连与仲裁可能增加延迟；收益应通过实际轨迹体现。

## 16. 建议实施顺序与交付物

### 阶段A：仓库勘察和规格对齐

产出实际路径映射、现有行为摘要、与R01..R18的差距。确认当前engine完成事件、输入释放和context身份的真实含义。

不得仅修改配置名称而不改变资源语义。

### 阶段B：功能合同与状态机

先实现动态lease、两槽quota、固定affinity、顺序/身份校验和事件边界。完成T01..T09中功能层适用部分，以及T14/T16/T17/T19/T21。

此阶段可以使用抽象AM，不必先写SRAM RTL。

### 阶段C：两槽程序生成与进展

调整IR验证、lowering或现有tile program模板，使OPEN失败不会阻塞已有链，描述符成功接收后可安全复用。

验证两链不同K长度、one-shot与长链混合、FINAL排空、input-ready延迟和两槽局部配额反例。

### 阶段D：周期与访问资源

加入有限结果credit、bank/adder服务、AM提交延迟、drain反压和原因统计，完成相关周期验收。

不得用无限FIFO或把AM更新视作零成本来通过性能测试。

### 阶段E：RTL连接与差分检查

仅在项目确有RTL目标时进行。对齐真实Gemmini接口、微块发射边界和输出保障。功能模型、周期模型、RTL使用同一组对象身份与事件依赖合同。

首轮重点确认结果不丢、顺序正确、回收安全；性能结论单独报告。

### 实施模型最终应提交

1. 实际修改文件和对应职责。
2. R01..R18逐项实现状态及实际代码位置。
3. T01..T22的执行结果与验证层级；未运行项目明确标注。
4. 关键trace片段或统计，足以复核quota、affinity和提交/释放边界。
5. 已知限制与尚未定型的参数。
6. 兼容性影响和对现有配置/程序的迁移办法。
7. 明确区分“设计完成、代码完成、模拟通过、RTL通过、PPA评估”，不得混用。

## 17. Gemmini复用依据与边界

本节源码依据固定到：

**ucb-bar/gemmini，commit 8c3f9923a44a2fe2c7930587be297d6d4f8c09ca。**

相关文件在本轮讨论中已读取并核对。以下是局部接口事实，不是对四引擎组合进行过完整RTL审计的声明。

| 编号 | 已核实事实                                             | 对本设计的影响                                              |
| ---- | ------------------------------------------------------ | ----------------------------------------------------------- |
| G1   | MeshWithDelays结果口为Valid，缺少输出ready             | 需要发射前接收保障，不能直接依赖下游反压                    |
| G2   | Scratchpad中多个AM bank接入一个AccPipeShared           | bank数量和独立累加吞吐必须分别设计                          |
| G3   | ExecuteController对不能被接收的有效AM写入有断言        | 新仲裁层必须保证所有已产生结果有落点                        |
| G4   | mesh输出最后一行时会生成执行完成，而AM后续仍有写流水线 | 原execute completion不能直接等同新step_committed            |
| G5   | AccumulatorMem已有未完成写相关的地址互锁               | 可参考/复用局部hazard机制，但需增加多engine owner与顺序管理 |
| G6   | scratchpad与accumulator具有不同的数据宽度/转换语义     | 不能用低精度SP存储直接替代宽位partial C                     |

可核查链接：

- G1：[MeshWithDelays.scala，结果接口](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/MeshWithDelays.scala#L58-L65)。
- G2：[Scratchpad.scala，AccPipeShared连接](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/Scratchpad.scala#L685-L708)；[AccumulatorMem.scala，共享向量加法器](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/AccumulatorMem.scala#L73-L89)。
- G3：[ExecuteController.scala，AM写入](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/ExecuteController.scala#L946-L962)。
- G4：[ExecuteController.scala，执行完成](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/ExecuteController.scala#L970-L979)；[AccumulatorMem.scala，写流水线](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/AccumulatorMem.scala#L111-L118)。
- G5：[AccumulatorMem.scala，地址互锁](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/AccumulatorMem.scala#L378-L395)。
- G6：[固定版本README，Scratchpad and Accumulator](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/README.md#scratchpad-and-accumulator)。

目标仓库可能使用不同Gemmini版本或自定义模型。实施前应重新定位等价逻辑，不能按行号盲改。

## 附录A：可直接交给其他模型的任务说明

```text
请按《Nexus共享AM V1：需求规格与设计改造指南》修改当前项目。

首先检查仓库实际结构、AGENTS.md、配置、scheduler、tile context、GEMM提交、
事件、MFE和AM相关逻辑，并输出概念到实际路径的映射。不要假定文档中的建议模块名已经存在。

必须遵守：
1. 四个matrix engines共享AM池；基线四个AM groups、全局八个slots。
2. 每个tile context同时最多两个live AM leases，跨所有engines统计。
3. 配额包含RESERVED、等待输入、计算、COMMITTING、FINAL_READY、DRAINING和one-shot。
4. 每条输出链在TRY_OPEN成功时固定engine affinity，整条链不能迁移。
5. 同context两条链可以绑定不同engine；不能把整个context绑定到一个engine。
6. 配额是上限，不是每context永久预留两槽；不把context总数固定为四。
7. TRY_OPEN成功即完整取得物理槽、配额、必要metadata与完成记录；BUSY无副作用且不阻塞UCE。
8. 已有链的STEP/FINAL/供数/drain不能被新OPEN挡住。
9. 同handle仅一个K-step更新；下一步等前一步AM提交，不引入atomic或split-K。
10. 保留UCE wait/context switch，只增加/复用设备端有限ready仲裁。
11. STEP成功接收后描述符语义固定；完成事件持久且带代际。
12. 发射前保障不可暂停的结果接收；不能假定当前Gemmini输出支持任意反压。
13. compute_done、step_committed、input_release、output_visible、slot_free语义分开。
14. 最终输出自动drain到已准入的L1区域，再安全释放；不用UCE轮询来触发释放。
15. 维持一个驻留context一个memory profile的既有约束。

实施按功能合同、两槽软件流水、周期资源、RTL接口逐层推进。
先修复影响正确性的资源/事件语义，再评估性能。
一个context在本V1中至多同时驱动两个matrix engines，这不是实现缺陷。

必须针对文档R01..R18说明实现，并按T01..T22报告实际验证结果。
不要虚构测试通过、RTL接入或PPA数字。
对尚未支持的项明确列出，不通过改变约束来伪造完成。
```

## 附录B：评审时的快速检查

- [ ] 两槽限制是否按tile context统计，而非按engine统计？
- [ ] RESERVED和DRAINING是否仍计入配额？
- [ ] one-shot是否使用并计入同一AM池？
- [ ] 全局空槽是否能分配给任意合法engine的链？
- [ ] engine中的元数据是否仍暗中限制为两条驻留链？
- [ ] handle是否同时绑定一个输出链、一个AM区域和一个不可变engine？
- [ ] 第三次申请失败后，原两条链是否仍能得到软件续步？
- [ ] OPEN是否全成或全败，且没有成功句柄继续等待AM的状态？
- [ ] STEP接收是否快照描述符，失败是否保持提交游标不变？
- [ ] last结果是否可能仍在FIFO/RMW中却被提前标为committed？
- [ ] 事件是否能跨物理slot复用保存，WAIT登记是否避免丢唤醒？
- [ ] 是否给不可暂停的全部在途结果预留了容量？
- [ ] bank与adder吞吐是否独立建模，drain是否能够取得服务？
- [ ] 输入释放是否依据最后实际读取及所有读者，而非submit或MAC done？
- [ ] FINAL排空和slot释放是否不依赖UCE轮询？
- [ ] context退出、异常和generation复用是否等待旧访问安全结束？
- [ ] 性能报告是否区分配额、affinity、输入、bank、credit和drain等待？
- [ ] 是否清楚标记哪些是设计建议、哪些已实现、哪些实际验证通过？
