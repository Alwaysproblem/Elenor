# Tile Group Sequencer 设计文档

## 1. 定位、目标和 First Silicon cutline

Tile Group Sequencer 规格现在描述**每 root 的注册/完成控制器**，而不是一个串行 action-index 执行器。当前参考实现中，每个已准入 root Context 拥有一个 `TileGroupSequencer`，它保存该 root 的 action registration cursor、`nest.await` / `nest.barrier` registration fence、queued/inflight action 和 completion 状态；**issue 带宽不属于它**——Group 共享的 `GroupScheduler` 在有限 action 表和扫描窗口中，每 cycle 最多 ISSUE 一条并 REGISTER 一条 action。

当前主执行链是 `source xDSL → compile_program → immutable CompiledProgram → independent verify/load(actual bindings) → LoadedProgram → Simulator.run`。`ExecTileGroupTask` 是编译产物中 root Context 的底层表示；它的 action 列表在编译期确定，本层只按注册顺序登记，不做 runtime lowering、动态生成或 graph 解释。

本层职责合同：

- 按 root registration cursor 顺序登记 action；`WAIT_EVENT` 类 action（`nest.await` lower 产物）作为 fence，暂停该 root 的后续 registration，直到完成。
- 不等待 action 执行完成才推进 cursor；register 与 completion 解耦，action 可长期 inflight。
- `nest.barrier` lower 为 `BARRIER_GROUP`，只等待同一 Context 更早 action 完成，不是隐式全 Group/Device barrier。
- `dispatch_role` 只登记有限 Grid Route；Task 由每 Tile 独立 admission commit（L1 Arena、Frame、UCE pin、父 L2 pin、R lease），不由本层串行推进。
- root 完成条件是本 Context 全部 action 完成、全部 Grid Task 安全退休且 L2 ownership/claim 收敛；不是“跑完 action list”本身。

与旧基线的差异：旧文把本层描述为“按 action index 逐条 issue 的串行控制器”，并把 `dispatch.role` 视为立即的 prepared-tile-task 派发。当前源码不是这样：本层不拥有 issue，role dispatch 由共享 Group scheduler issue 后经 Grid Route/Tile admission 落地。旧文同时声称“Device Runtime 将 graph schedule lowering 成 TileGroupTask”为运行时动作——这在当前主线中不存在，lowering 只发生在编译期。

硬件映射（action RAM、cursor/fence 状态、CSR）仍是未冻结草案；`GroupScheduler` 属于单个 Group，不是芯片级多 Group 调度器。

First Silicon / 硬件 cutline 草案：

| 项目     | 当前参考语义                                  | 硬件/后续规格                        |
| -------- | --------------------------------------------- | ------------------------------------ |
| Task     | 编译期固定 action 列表、无自修改              | action RAM 容量、编码、保护机制      |
| Register | per-root cursor + fence + quota，与执行解耦   | cursor/fence/quota 寄存与流控        |
| Issue    | Group 共享单条 ISSUE / REGISTER（不同仲裁层） | issue pipeline、多 issue 的 PPA 取舍 |
| Dispatch | bounded Grid Route + 每 Tile 独立 Task commit | Route 表、Tile admission 接口        |
| Sync     | WAIT_EVENT fence / BARRIER_GROUP / 显式依赖   | event/generation/sequence 域冻结     |
| PMU      | registration/issue/completion 分层计数        | counter ID/CSR/trace 编码冻结        |

所有 action op 编码、表容量、descriptor window、timeout 默认值由后续规格冻结。

## 2. 职责、非职责和 ownership

每个已准入 root 的 `TileGroupSequencer` owns：

- Registration cursor：本 root action 的顺序登记位置；register 成功即前进，不等 action 执行。
- Registration fence：`WAIT_EVENT` / `BARRIER_GROUP` / profile / maintenance action 会暂停本 root 后续 registration，直到完成解除。
- Queued/inflight action 记录：本 root 已登记/已 issue action 的状态与完成归属；issue 由 Group 共享 `GroupScheduler` 执行。
- Grid Route 与逐 Tile Task commit：`dispatch_role` 建立有限 Route 后，每 Tile 独立完成 Task 准入（L1 Arena、Frame、UCE pin、父 L2 pin、R lease）。
- root 完成判定：本 Context action 完成 + 全部 Grid Task 安全退休 + L2 ownership/claim 收敛；随后归还 Group slot。
- 事件身份：本 root 事件的 owner 标识、launch generation 与 sequence 参与 EventTable 的生产/消费。

不负责：

- Group 共享 ready-action issue 带宽、scan 窗口、S0/S1 policy（`GroupScheduler` owns）。
- CPU `nexus.*` 提交/依赖、DevicePort 消息协议、pending root metadata 或 Group root 资源准入。
- Tile-local UCE PC、L1 Arena/Frame、engine launch、数据相关 MFE 访存。
- runtime graph lowering、动态 action 生成、编译期已证明资源的现场修复。
- 尚未建模的 group-level loop/branch ISA、preemption、跨 Group barrier。

Ownership 约束：

1. registration cursor 按 source 顺序推进；不允许乱序登记或回退。
2. fence 未解除时本 root 不得登记后续 action；其他 root 的登记不受影响。
3. 每个 action 的 completion 归属其 owner root；旧 generation completion 不得唤醒新 root。
4. 本层不直接写 Tile-local L1/UCE 状态；Task commit 经 Grid Route 协议。
5. `fault_record_slot` 由 launch metadata 指定，本层写 root 级 fault 摘要。
6. Stream Queue 内部 token/credit 由 Stream Queue Engine owns；本层仅登记显式 action 或观察事件。

## 3. 微架构和状态机

### 3.1 内部结构（硬件映射草案）

```text
per-root TileGroupSequencer
├── registration cursor
├── registration fence (WAIT_EVENT / BARRIER / control)
├── queued / inflight action records
├── root completion / fault state
└── event owner identity (context, generation, sequence)
        |
        v
Group-shared GroupScheduler
├── bounded action table
├── scan window / S0 | S1 policy
├── adapter / inflight / control credit
└── one ISSUE + one REGISTER per cycle, completion poll first
```

本层没有 group-level program text、loop counter 或 PC；action 列表是编译产物中的静态序列。硬件 CSR/位宽/队列深度由后续规格冻结。

### 3.2 Root 生命周期状态机

```text
ROOT_ACTIVE
  -> register next action (cursor++ / fence may pause)
  -> actions issued by shared Group scheduler
  -> dispatch actions build Grid Routes
  -> per-Tile Tasks commit and run
  -> all actions done
  -> all Grid Tasks safely retired (Arena/Frame/pin/R lease released)
  -> root L2 ownership / claims / events converge
  -> ROOT_COMPLETE

fault / cancel:
  -> root fault record
  -> accepted work drain / cancel-confirm
  -> ROOT_ERROR or ROOT_CANCELLED
```

与旧文差异：旧 `ACCEPT→VALIDATE→…→ISSUE_OR_WAIT` 串行状态机把 validate/residency/init 描述为每 root 的执行阶段。当前模型中，编译/加载/独立验证在进入运行前完成；root 准入时资源一次性提交。若未来硬件 bring-up 需要保留 per-root validate/residency 状态，它们属于硬件草案的流水化选择，不是当前软件参考语义。

### 3.3 Register / issue / completion 语义

- `note_registered`：按 cursor 顺序登记 action，记录 fence；cursor 前进。
- `issue_registered`（由共享 scheduler 调用）：提交一条已 eligible action；`DISPATCH_ROLE` 生成 Grid Route，`WAIT_EVENT` 记 await，`BARRIER_GROUP` 等待本 Context 更早 action。
- `note_completion`：独立于 action 表回收 credit/event；action 表满不阻塞完成。
- completion 与 registration 解耦：root 可同时存在已登记未 issue、已 issue 未完成 action。

### 3.4 Wait/fence 状态机

```text
FENCE_SET (await/barrier/control action registered)
  -> shared scheduler issues it when eligible
  -> completion observed
  -> FENCE_CLEAR -> registration resumes
```

`WAIT_EVENT` 等待显式 source event（含其依赖闭包）；`BARRIER_GROUP` 等待本 Context 更早 queued/inflight action 全部完成。二者都不阻塞其他 root。

## 4. 接口、descriptor、寄存器和协议

本节先给出当前软件动作集与硬件草案的边界；C 结构与 CSR 均为硬件 v0 草案，不是软件 artifact 编码。

### 4.1 Action 语义（当前编译产物动作集）

当前 `ExecGroupActionOp` 包括普通编译期 binding/transfer action、dispatch/event/control action；完整动作集合为：`INIT_STREAM`、`DMA_PREFETCH`、`DMA_STORE`、`BIND_L2_VIEW`、`BIND_L2_IMPORT`、`PUBLISH_L2`、`DISPATCH_ROLE`、`WAIT_EVENT`、`BARRIER_GROUP`、`COLLECTIVE_RUN`、`SIGNAL_EVENT`、`RELEASE_L2`、`PROFILE_RECONFIG`、`MEMORY_MAINTENANCE`。下表是软件动作的语义映射，不是硬件 opcode 编码：

| 动作                                      | 来源 / 产生方式                 | 本层行为                                            |
| ----------------------------------------- | ------------------------------- | --------------------------------------------------- |
| `INIT_STREAM`                             | stream descriptor / action list | 初始化显式声明的 Stream Queue overlay               |
| `BIND_L2_VIEW`                            | compiler resource lowering      | 绑定当前 root 的 L2 view                            |
| `BIND_L2_IMPORT`                          | compiler shared-input lowering  | 绑定只读 shared backing claim                       |
| `DMA_PREFETCH` / `DMA_STORE`              | 显式 transfer action            | 登记 transfer、依赖与 completion event              |
| `DISPATCH_ROLE`                           | `nest.dispatch.*` lowering      | 登记 Grid Route；completion=Grid done               |
| `WAIT_EVENT`                              | `nest.await`                    | fence 本 root 后续 registration                     |
| `BARRIER_GROUP`                           | `nest.barrier`                  | 等待本 Context 更早 action                          |
| `PUBLISH_L2`                              | `nest.publish`                  | 发布只读 backing / claim 元数据                     |
| `RELEASE_L2`                              | `nest.release`                  | 使 L2 view 失效；owner Arena/claim 仍依生命周期收敛 |
| `COLLECTIVE_RUN`                          | collective action               | 登记 collective completion event                    |
| `SIGNAL_EVENT`                            | `nest.return`                   | signal root completion event                        |
| `PROFILE_RECONFIG` / `MEMORY_MAINTENANCE` | compiler pass                   | 显式控制事务，等待完整 ACK/generation               |

普通 action 的 `dependencies` 只等待真实生产者事件；显式 await 与数据依赖是不同约束。硬件 action 编码、立即数、descriptor window 与 timeout 机制由后续规格冻结。

### 4.2 Grid Route 与 Task 合同

`dispatch_role` 携带 dispatch request（binding、tile mask、Grid id、`input_released`/`output_ready` 事件等）。本层登记有限 Grid Route；每 Tile 独立 commit Task。`requested_contexts_per_tile=R` 以 `(parent binding_id, launch_generation, tile_id)` 为 key 覆盖父 Context 全部 Grid。dispatch 的 `context=N` pin 所选各 Tile 的同号 UCE context。

`input_released` / `output_ready` / Grid done 的语义见 Tile Group 规格与本层 §5；它们不等于 Arena 退休或 HBM 持久化。硬件 Grid Route 表容量、Task 描述符编码由后续规格冻结。

### 4.3 硬件 launch descriptor（v0 草案，示意）

```c
typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t flags;
    uint16_t priority;
    uint32_t context_id;
    uint32_t task_id;
    uint32_t group_id;
    uint32_t role_count;
    uint32_t tile_mask_union;
    uint64_t group_task_iova;
    uint32_t group_task_bytes;
    uint64_t role_binding_iova;
    uint32_t role_binding_bytes;
    uint64_t engine_desc_iova;
    uint32_t engine_desc_bytes;
    uint64_t stream_desc_iova;
    uint32_t stream_desc_bytes;
    uint64_t wait_ref_iova;
    uint32_t wait_ref_count;
    uint32_t wait_ref_crc_or_zero;
    uint32_t signal_event;
    uint32_t signal_sequence;
    uint16_t residency_hint;
    uint16_t cache_policy;
    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_group_task_launch_desc_v0_t;
```

这是既有硬件 v0 descriptor 字段的草案示意，未冻结宽度/对齐/CRC/endianness/编码；不是当前软件 `CompiledProgram` 的二进制格式，也不增加新的字段或硬件实现承诺。当前软件对象是 `ExecTileGroupTask`，schema/ABI 命名空间独立。

### 4.4 CSR 草案

| CSR 草案            | 目标观察内容                                   |
| ------------------- | ---------------------------------------------- |
| `TGS_REG_CURSOR`    | 本 root registration cursor 与 fence 状态      |
| `TGS_ACTION_STATUS` | queued/inflight action 计数与最老未完成 action |
| `TGS_ROOT_STATUS`   | root 生命周期、fault/cancel 状态               |
| `TGS_GRID_STATUS`   | Grid Route 登记、每 Tile admission/retirement  |
| `TGS_WAIT_REASON`   | 当前 fence/wait 原因                           |
| `TGS_FAULT_CODE`    | 最近 fault code                                |

CSR 地址、位宽、权限由后续规格冻结。`ACTIVE_CONTEXT_TASK`/`TGS_ACTION_INDEX` 等旧草案名与当前 per-root cursor + 共享 issue 结构不对应，已由上表替换。

## 5. 数据流、控制流和时序路径

### 5.1 控制流（当前参考模型）

```text
compile_program → load/verify → LoadedProgram
CPU submit / depends_on / await
  → DevicePort → GroupPort pending metadata
  → root 完整准入（Group slot + L2 + 事件/控制预算）
  → 本 root Sequencer 按 cursor 注册 action（fence 可暂停）
  → Group 共享 scheduler：poll completion/control → ISSUE → REGISTER
  → DISPATCH_ROLE 建立有限 Grid Route
  → 每 Tile 独立 Task commit（Arena/Frame/UCE pin/父 L2 pin/R lease）
  → Tile UCE eligible-head RR issue → engines/transfers
  → 完成 → Task 安全退休 → Grid done → root done
  → CPU 下一 cycle harvest
```

本层只在控制面登记/等待；不搬运 payload，不执行 Tile Program。Group step 的完成最早在下一 CPU phase 可见，这是模型推进顺序而非硬件时钟承诺。

### 5.2 与 Grid / Tile 的关系

- Grid Route 登记 dispatch 的 Task 集合与 phase 事件；Task commit/retirement 归 Tile Group/Tile 层。
- `input_released`/`output_ready`/Grid done 分别表示输入读取结束、L1→L2 输出完成、全部 Task 安全退休。
- root 等待本 Context action 与事件，且 L2 ownership/claim 收敛后才完成；producer 退休不等于 backing 释放。

### 5.3 时序路径（硬件草案）

需要收敛的硬件路径：cursor/fence 更新、action 表 issue 仲裁、Grid Route 写入、event owner/generation 比对、completion fan-in。具体 pipeline/时钟由后续规格冻结；软件模型不提供这些物理时序。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 模型资源

本层自身不持有 SRAM 数据区；root L2 Arena 属于 Group 层。相关模型容量：`group.action_capacity`（默认 16）、`context_action_quota`（默认 8）、`scan_width`（默认 4）、`group.active_context_capacity`（默认 8）、`context_pending_capacity`（默认 32）。这些是探索参数，硬件 action RAM/队列深度由后续规格冻结。

### 6.2 性能模型

root 推进受 registration 速率（每 cycle 至多一条 REGISTER，受 quota/表容量/fence）、共享 issue 速率（每 cycle 至多一条 ISSUE）、Grid/Tile Task admission 与完成收敛共同限制。多 root 并存时共享 issue 是 Group 级瓶颈之一；S0/S1 只改变 ready 排序，不提供全局公平或无饥饿保证。

### 6.3 PMU / snapshot 观测

模型观测包括 `tgs_dispatch_role`、`tgs_wait_event`、`tgs_barrier`、`tgs_collective_run`、`tgs_store_dma` 等事件计数，以及 Group scheduler 的 registration/issue idle、action table full、context quota full、dependency wait candidate、ordering stall、barrier wait、adapter/inflight credit wait 等 cycle 类。这些名称是模型 snapshot/trace 类别，不是硬件 PMU counter ID。

硬件 PMU/CSR 编号、trace 格式与跨时钟对齐由后续规格冻结；不同层并发 stall 不相加为唯一 cycle。

## 7. RTL/软件实现建议

RTL 映射草案：

- per-root cursor/fence 状态寄存 + root 完成判定；不实现 group-level PC。
- 与共享 issue scheduler 的接口：registered action 表、credit、completion 通知。
- Grid Route 表 + 每 Tile admission 握手（有界 Route、R ledger、cancel/confirm）。
- event owner/generation/sequence 比对与 fault/cancel-confirm 路径。

软件侧：编译产物已包含 action 列表与资源预算；本层不做 runtime lowering 或修复。硬件 action 编码、CSR、SRAM/时序均待规格冻结。

## 8. 验证、bring-up 和验收标准

### 8.1 调度合同验证点

- Registration cursor 顺序推进；fence 解除前本 root 不登记新 action，且不影响其他 root。
- 同一 cycle 内 register 与 issue 至多各一条；register 成功不依赖 action 完成。
- completion/credit 回收独立于 action 表满；无丢失完成或隐式补同步。
- `WAIT_EVENT` 只等待显式 source event；`BARRIER_GROUP` 只等待本 Context 更早 action；无隐式 Group/Device barrier。
- `DISPATCH_ROLE` 登记 Grid Route；每 Tile 独立 commit；单 Tile 阻塞不撤销其他 Tile 已提交 Task。
- root 完成当且仅当 action 完成、Grid Task 全部安全退休、L2 ownership/claim 收敛。
- S0/S1 在有限 scan 窗口内行为可区分；S2 当前等同 S1。

### 8.2 参考 bring-up 场景

1. 单 root 空 action：准入、完成、slot 归还。
2. await fence：后继 action 在 fence 完成前不登记；其他 root 不受影响。
3. barrier：仅本 Context 更早 action 完成后通过。
4. dispatch + Grid Route：多 Tile Task 独立 commit，单 Tile 资源不足不阻塞其他 Tile。
5. 多 root 并存：共享 action 表、quota、scan 的有界推进与 wait attribution（不承诺全局无饥饿）。
6. fault/cancel：pending root 取消与已准入 work 的 drain/cancel-confirm 可区分。
7. PMU/snapshot：registration/issue/completion 计数与实际调度轨迹一致。

### 8.3 验收边界

上述为当前参考模型合同与硬件 bring-up 目标；本层硬件 CSR/RTL、多 Group、物理时序与 PMU 编码均未实现或冻结。不能以本规格文本宣称芯片验收或性能数字。

## 9. 风险、取舍和后续细化方向

风险：

- Action op 范围失控：Tile Group Sequencer 若承担 workload-specific 算子语义，会变成难验证的小 CPU。
- Wait/event/stream epoch 不严谨：reset 后 stale event 可导致错误完成。
- Descriptor window 与 BOA/MFE hot bank 冲突。
- Role completion scoreboard 对多 role in flight 支持不足，导致 dispatch 被迫串行化。
- Timeout policy 不明确，可能掩盖 deadlock 或误杀长 latency DMA。
- Issue ownership 漂移：若把 Group 共享 ready-action issue 下放回本层，会重新引入 per-root 串行 action-index 执行器并丢失多 root 并发。

取舍：

- 使用 action list + descriptor-driven engines，而不是硬件解释 graph 或取指 fetchable group-level program text。
- First Silicon V1 先固定有限 action op 和静态 role binding，优先验证 command/event/DMA/stream/PMU。
- role 内 Tile-SPMD 由 Tile UCE 推进，不在 Tile Group Sequencer 引入 per-role PC 或 per-tile dynamic dispatch 模型。
- 未来硬件若需要 per-root validate/residency 流水，只作为准入前的流水化选择，不改变 per-root cursor + 共享 issue 的合同。

后续需要冻结：

- Binary action op encoding、action list alignment、action 数量上限。
- `elenor_group_task_launch_desc_v0_t` 字段宽度、role binding descriptor ABI、CSR map。
- Timeout 默认值、zero timeout 语义和 fault code。
- Group task 最大 role 数、queue/stream policy、multi-consumer 行为。
- PMU counter 编号、overflow、read/clear 行为。
