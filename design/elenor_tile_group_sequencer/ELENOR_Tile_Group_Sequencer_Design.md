# Tile Group Sequencer 设计文档

## 1. 定位、目标和 First Silicon cutline

Tile Group Sequencer 是 Tile Group 内的 Group Task 执行控制器。Device Runtime 将 graph schedule lowering 成 `TileGroupTask` 并提交 `ELENOR_CMD_LAUNCH_GROUP_TASK`；Tile Group Sequencer 接收 group task，维护 action index，初始化 stream/event/barrier 资源，issue Group DMA、dispatch role binding、等待 event/stream/barrier，并最终 signal group task done。

`TileGroupTask` 是 group-level dispatch descriptor，不是可取指的 program/ISA，也不是 subgraph。它绑定 `task_id`、`group_id`、descriptor window、stream descriptors、Group DMA prefetch/store descriptors、collective/barrier metadata、completion event、fault record slot、timeout、residency hint，以及一个或多个 `TileRoleBinding`。Tile Group Sequencer 解释的是 action list，不是 fetchable group-level program text，也不是 Tile Program ISA。

Tile Group Sequencer 的目标不是成为通用微控制器，而是成为可验证的 device group task 推进器：

- 执行 command/descriptor/action，不解释高层 graph。
- 在 group 内以 role binding / block token 粒度推进 role dispatch。
- 控制 HBM/DDR/LPDDR 到 L2 的 Group DMA 预取和 storeback。
- 协调 Stream Queue、Barrier/Event、Collective 与 Tile Dispatcher。
- 为 PMU 提供准确的 primary stall owner。

控制层级（canonical）：

```text
Graph Schedule PC / Group Task Iterator -> Tile Group Sequencer action index -> Tile PC / Tile UCE -> Engine
```

硬件执行边界只涉及 `command buffer、descriptor、TileGroupTask 和 Tile Program`。`TileRoleBinding` 是 group task 内的静态 dispatch metadata，每个 role 内仍是 Tile-SPMD：通过 `tile_id`、`group_id`、descriptor offset 和 slot/frame binding 区分数据，不引入 per-tile dynamic role dispatch。
First Silicon V1 cutline：

| 项目     | First Silicon V1                                          | V1.x / V2 reserved                    |
| -------- | --------------------------------------------------------- | ------------------------------------- |
| Task     | 固定 action list、有限 action op、无自修改                | 压缩编码、动态 action 生成            |
| Dispatch | 静态 role binding、role 顺序 dispatch、stream wait/credit | 动态 role 创建、priority scheduling   |
| DMA      | 1D/2D/strided Group DMA prefetch/storeback                | gather list、multicast DMA            |
| Control  | action index 线性推进 + bounded wait；无 branch/loop ISA  | 通用分支预测或复杂异常恢复            |
| Sync     | wait/event/barrier/stream/EOS/error                       | cross-group barrier、preemption       |
| PMU      | issue、wait、stall、timeout、fault attribution            | trace sampling 和 feedback scheduling |

所有 action op 编码、action 数量上限、descriptor window 容量、最大 role 数、最大 queue 数、timeout 默认值由后续规格冻结。

## 2. 职责、非职责和 ownership

Tile Group Sequencer owns：

- Group task accept / validate：`elenor_group_task_launch_desc_v0_t` 的 ABI version、size、range、`role_count`、`tile_mask_union`、event/queue id、SRAM window 检查。
- Action index 推进：按 action list 顺序解释每条 `GroupAction`，不维护 program-counter semantics at group level、不取指 fetchable group-level program text、不维护 group-level program register file。
- Role binding lookup：`dispatch.role role_id=<n>` 时按 `task_id` 上下文查找 `TileRoleBinding`，组装 prepared tile task 交给 Tile Dispatcher。
- Stream Queue 初始化、reset、drain command issue；`wait.event` / `barrier.group` / collective 完成等待。
- Group DMA prefetch / store descriptor issue、completion wait、DMA fault propagation。
- Tile Dispatcher issue：role binding 的 `tile_mask`、tile program id、resident local handle、descriptor patch window、stream binding、role_id、timeout/fault slot。
- Barrier/Event wait/signal sequencing 与 group event timeout。
- Collective command issue 和 completion wait。
- PMU event generation（`tgs_*` 前缀）：active、issue、wait reason、fault action index、timeout。

Tile Group Sequencer 不负责：

- BOA/EVU/MFE/USE datapath 微循环或 tile-local engine launch；这些由 Tile UCE 和 engine sequencer 负责。每个 role 内仍是 Tile-SPMD：同一份 Tile Program template 在该 role 的 `tile_mask` 上运行，通过 `tile_id`、`group_id`、descriptor offset 和 slot/frame binding 区分数据。
- Tile Program residency。Tile Program 的 lookup/fetch/verify/install/wake 由 Program Residency Manager 按 `TileRoleBinding` 管理；本模块没有 fetchable group-level program text residency、没有 fetchable group-level program text fetch、没有 program-counter semantics at group level。
- L2 到 L1 的 Tile DMA；Tile Group Sequencer 只准备 L2 buffer、descriptor 和 stream token。
- MFE 的 page/segment 数据相关动态地址生成；Tile Group Sequencer 只 dispatch 或等待相关 role/event。
- USE 的 state ownership、recurrence、checkpoint/restore。
- Runtime queue scheduling、context priority、global graph dependency。
- Cache coherency；Shared SRAM/L2 通过显式 descriptor 和 fence 管理。

Ownership 约束：

1. Tile Group Sequencer 是 group 内 group task control owner；Tile UCE 是 tile-local program control owner。
2. Tile Group Sequencer 可写 group event/status 区，但不能直接写 tile-local L1 状态，除非通过 Tile Dispatcher/Tile UCE 协议。
3. Tile Group Sequencer issue 的 Group DMA completion event 必须唯一映射到 descriptor sequence。
4. Tile Group Sequencer 不应直接解释 Stream Queue payload 内容，只使用 token metadata。
5. `fault_record_slot` 由 launch command 指定，Tile Group Sequencer 负责写入 group-task-level fault 摘要。
6. Stream Queue 的 token/credit/EOS/error 内部状态由 Stream Queue Engine 拥有；Tile Group Sequencer 只 init/reset/drain 并通过 Event/Fault 观察。

## 3. 微架构和状态机

### 3.1 内部模块

```text
Tile Group Sequencer
├── Task Accept / Descriptor Validator
├── Program Residency Manager Interface (per TileRoleBinding)
├── Action Decode / Issue
├── Action Index Register
├── Role Binding Lookup
├── Stream Init/Reset/Drain Unit
├── Stream Wait/Credit Unit
├── Event/Barrier Wait Unit
├── DMA Issue Queue
├── Tile Dispatch Queue
├── Collective Issue Queue
├── Fault / Timeout Unit
└── PMU Event Encoder
```

Tile Group Sequencer 不维护 group-level program register file、loop counter 或 stage scoreboard；action list 是顺序结构，由 action index 线性推进。action list 中不存在 branch/label/region begin/end opcode。寄存器数量和字段宽度由后续规格冻结。

### 3.2 Group task 状态机

```text
RESET
  -> IDLE
  -> ACCEPT
  -> VALIDATE
  -> RESIDENCY_LOOKUP
  -> RESIDENCY_FETCH_ON_MISS
  -> RESIDENCY_VERIFY
  -> INIT
  -> ACTION_DECODE
  -> ISSUE_OR_WAIT
  -> COMPLETE
  -> IDLE

fault/timeout from VALIDATE..ISSUE_OR_WAIT:
  -> FAULT_RECORD
  -> DRAIN
  -> SIGNAL_ERROR
  -> IDLE 或 RESET_REQUIRED
```

状态说明：

- `ACCEPT`：从 group launch queue 取 group task，锁存 `context_id` / `task_id` / `group_id`。
- `VALIDATE`：检查 descriptor size、range、queue/event count、`role_count`、`tile_mask_union`、SRAM window。
- `RESIDENCY_LOOKUP`：对每个 `TileRoleBinding` 按 `context_id + tile_program_id + tile_program_version` 查询 Program Residency Manager。注意 residency 的对象是 Tile Program，不是 group task 本身——group task 是 descriptor，不是可取指 program。
- `RESIDENCY_FETCH_ON_MISS`：warm launch 命中 cache 时跳过；cold launch 通过 Group DMA 拉取对应 Tile Program，并把等待中的 group task 挂到相同 key。没有 fetchable group-level program text fetch。
- `RESIDENCY_VERIFY`：核对 Tile Program hash/CRC/ABI/epoch；失败则写 fault record，禁止进入 READY。
- `INIT`：初始化 stream queue、barrier epoch、PMU epoch；按 `init.stream` action 配置 queue。
- `ACTION_DECODE`：从 action list 取当前 action index 的 `GroupAction`，decode op 与 args。
- `ISSUE_OR_WAIT`：对 DMA、dispatch.role、collective、barrier、stream 产生 command；对 `wait.event` 进入等待。action index 在 issue 成功或 wait 完成后递增。
- `COMPLETE`：所有 in-flight role dispatch / DMA / collective 清空，signal group task done。

### 3.3 Role dispatch semantics

Tile Group Sequencer 通过 `dispatch.role role_id=<n>` 推进 role dispatch。role 是 `TileGroupTask` 内的 `TileRoleBinding`，不是高层 op，也不是 stage。数据以 block token / descriptor window 流动：

```text
for action in task.actions:
  init.stream     配置 producer/consumer queue
  dma.prefetch    HBM -> L2，准备 role 0 输入
  dispatch.role   role_id=0  ->  Tile Dispatcher 组装 prepared tile task
  dispatch.role   role_id=1  ->  Tile Dispatcher 组装 prepared tile task
  wait.event      等待 role 1 完成
  dma.store       L2 -> HBM
  group_task.complete
```

Tile Group Sequencer 必须支持的 dispatch 行为：

- role 顺序 issue：action list 中 `dispatch.role` 按 role_id 查找 `task.role_bindings[role_id]`；缺失 role_id 抛出 fault（`unknown role_id`）。
- prefetch ahead：在 role 计算当前 block 时，Group DMA 预取 next block。
- role overlap：同一 group task 内不同 role 可位于不同完成阶段，由 event id 区分。
- stream backpressure：queue full 时停止 signal producer role 或停止 prefetch 进入该 queue 对应 buffer。
- event order：同一 role 的 dispatch completion 按 event sequence 更新；允许不同 role overlap。
- EOS drain：role 完成后由 Tile UCE / Stream Queue Engine 处理 EOS，Tile Group Sequencer 在 drain 阶段等待 queue 空。
- error short-circuit：任一 role error token 或 tile error event 触发 group task fault policy。

### 3.4 Wait 状态机

```text
WAIT_ENTER
  -> CHECK_READY
  -> PARK
  -> WAKEUP
  -> REVALIDATE
  -> RETIRE
```

wait source 包括 event、stream token、credit、barrier、DMA completion、collective completion。`REVALIDATE` 必须比较 epoch 和 sequence，避免 reset 后 stale wakeup。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Action list 语义

Tile Group Sequencer 解释的 action op 固定为以下集合，不存在 fetchable group-level program text opcode、program-counter、branch 或 region begin/end：

| Action op             | 作用                                       | 必要 args                                             |
| --------------------- | ------------------------------------------ | ----------------------------------------------------- |
| `init.stream`         | 初始化 stream queue                        | queue_id、depth、producer_mask、consumer_mask、policy |
| `dma.prefetch`        | issue Group DMA HBM 到 L2                  | dma_desc_id、dst_l2、event_id                         |
| `dma.store`           | issue Group DMA L2 到 HBM                  | dma_desc_id、src_l2、event_id                         |
| `dispatch.role`       | 向 Tile Dispatcher 派发 prepared tile task | role_id                                               |
| `wait.event`          | 等待 event done/error/timeout              | event_id、timeout                                     |
| `barrier.group`       | group participant barrier                  | participant_mask、event_id                            |
| `collective.run`      | issue group collective                     | collective_desc_id、event_id                          |
| `signal.event`        | signal group/tile event                    | event_id、status                                      |
| `group_task.complete` | drain 并完成 group task                    | signal_event                                          |

`dispatch.role` 的 args 为 `(role_id,)`。Tile Group Sequencer 查找 `task.role_bindings[role_id]`，记录 event id（`ins.dst` 或 `ev_role<role_id>`），调用 `TileGroup.dispatch_role(binding, cycle, event_id=event_id)`。缺失 role_id 触发 `unknown role_id <id>` fault。

action op 只包含上表所列——不包含 branch/label、block advance、显式 EOS 注入或 stream/credit 等待 op（stream token/credit 等待通过 `wait.event` 绑定到对应 event；EOS 注入由 Tile UCE 或 Stream Queue Engine 在 role 执行内完成）。action op 编码、立即数宽度、operand 格式由后续规格冻结。

### 4.2 TileGroupTask pseudo-flow 示例

```text
group_task.accept  task=attention_task
    init.stream   s0, depth=3, producer=role0, consumer=role1, policy=all_eos

    dma.prefetch  desc=input_qk, dst=l2_buf0 -> ev_dma0
    wait.event    ev_dma0

    dispatch.role role_id=0           # QK producer, tile_mask=0x03, out_stream=0
    dispatch.role role_id=1           # softmax+AV consumer, tile_mask=0x0C, in_stream=0
    wait.event    ev_role1

    dma.store     desc=output_av, src=l2_out -> ev_dma_out
    wait.event    ev_dma_out
group_task.complete signal=ev_group_task_done
```

每个 role 的 Tile Program 由 Program Residency Manager 按 `TileRoleBinding` 安装到 resident local slot；Tile Group Sequencer 不取指 Tile Program，也不存在 fetchable group-level program text。

### 4.3 Launch descriptor (runtime ABI)

Group task launch 使用 `ELENOR_CMD_LAUNCH_GROUP_TASK = 1` 和 `elenor_group_task_launch_desc_v0_t`。本字段集为 v0 draft；未冻结宽度由后续规格冻结，不在本文档伪造二进制编码。

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

    uint32_t wait_event_base;
    uint16_t wait_event_count;
    uint16_t signal_event;

    uint16_t residency_hint;
    uint16_t cache_policy;
    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_group_task_launch_desc_v0_t;
```

字段语义：

- `task_id` / `group_id` / `context_id` 标识本次 group task 实例。
- `role_count` / `tile_mask_union` 描述 group task 的 role 数与所有 role binding 的 tile mask 并集；`tile_mask_union` 必须是当前 group 可用 tile 的子集。
- `group_task_iova` / `role_binding_iova` / `engine_desc_iova` / `stream_desc_iova` 指向 action list、role binding 表、engine descriptor、stream descriptor 的 backing memory。group task 本身是 descriptor，不是可取指 program。
- `wait_event_base` / `wait_event_count` / `signal_event` 定义 group task 的事件命名空间。
- `residency_hint` / `cache_policy` 是 hint，不是 correctness 语义；正确性由 `tile_program_id + version + hash + epoch` 保证。
- `timeout_cycles == 0` 的语义由后续规格冻结，RTL 不应假设无限等待。
- `fault_record_slot` 由 Tile Group Sequencer 在 fault 路径写入 group-task-level 摘要。

Protocol：

- Descriptor validation failure must not partially initialize queues or dispatch tiles。
- `tile_mask_union` 必须是当前 group 可用 tile 的子集；每个 `TileRoleBinding.tile_mask` 必须是 `tile_mask_union` 的子集。
- `*_iova` 只作为 backing store；运行态 action 解释来自 resident descriptor window，group task 不被取指。
- `residency_hint/cache_policy` 是 hint，不是 correctness 语义；正确性由 `tile_program_id + version + hash + epoch` 保证。
- `timeout_cycles == 0` 的语义由后续规格冻结，RTL 不应假设无限等待。

### 4.4 CSR 建议

| CSR                     | 描述                                            |
| ----------------------- | ----------------------------------------------- |
| `TGS_CONTROL`           | enable、soft_reset、drain、single_step          |
| `TGS_STATUS`            | idle/running/waiting/draining/error             |
| `TGS_ACTION_INDEX`      | 当前 action index 或 fault action index         |
| `TGS_WAIT_REASON`       | event、stream、credit、barrier、DMA、collective |
| `ACTIVE_CONTEXT_TASK`   | context_id、task_id、role_id                    |
| `TGS_TIMEOUT_REMAINING` | 当前 wait timeout 计数                          |
| `TGS_FAULT_CODE`        | 最近 fault code                                 |
| `TGS_PMU_EVENT`         | debug view of PMU event encoder                 |

CSR 命名不包含旧式 group-level PC 或 context-region 前缀。CSR 地址映射、位域宽度由后续规格冻结。

## 5. 数据流、控制流和时序路径

### 5.1 Control flow

```text
Runtime command buffer
  -> Group launch queue
  -> Tile Group Sequencer ACCEPT/VALIDATE
  -> RESIDENCY_LOOKUP / FETCH_ON_MISS / VERIFY (per TileRoleBinding Tile Program)
  -> ACTION_DECODE
  -> issue Group DMA / dispatch.role prepared tile task / wait event / collective
  -> Tile UCE and group engines produce events
  -> Tile Group Sequencer retires wait, advances action index
  -> group_task.complete signals runtime event
```

Tile Group Sequencer 只在 group control plane 上 issue，不在 data path 上搬 payload。Tile Program backing store 解析和 install 由 Program Residency Manager + Group DMA 完成；payload movement 由 Group DMA、Tile DMA、MFE、Collective 完成。

### 5.2 Data flow relationship

- HBM -> L2：Tile Group Sequencer issue Group DMA prefetch/store descriptor。
- L2 -> L1：Tile Group Sequencer dispatch role，Tile UCE 根据 tile program 和 descriptor issue Tile DMA。
- Role -> Role：Tile Group Sequencer 初始化 Stream Queue 并等待 token/credit event；payload 地址通常指向 L2 或 tile-local slot，由 descriptor 指明。
- Tile -> Tile：Tile Group Sequencer issue Collective command 或 barrier，不直接读取 partial data。
- State update：USE 管理 state；Tile Group Sequencer 只等待 USE/tile event 或调度包含 USE op 的 tile program。

### 5.3 Timing paths

关键 RTL paths：

- Action index + decode -> issue valid -> target ready backpressure。
- Event scoreboard CAM/compare -> wait wakeup。
- Stream credit/occupancy compare -> wait wakeup。
- Role completion fan-in reduction by event id。
- Timeout counter compare -> fault transition。

建议将 event/stream/barrier wakeup registered，避免 Tile Group Sequencer issue path 和 queue occupancy path 形成长组合环。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 SRAM/L2 assumptions

Tile Group Sequencer 使用的 SRAM 区域：

| 区域              | 用途                                 | 备注                             |
| ----------------- | ------------------------------------ | -------------------------------- |
| descriptor window | group task/role/queue/dma desc cache | 需 range/CRC validation          |
| event/status      | event scoreboard、role status        | 避免与 BOA operand hot bank 冲突 |
| stream metadata   | queue head/tail/credit、policy       | payload buffer 可在独立 L2 bank  |
| fault/PMU epoch   | fault record、counter snapshot       | runtime 可读                     |

Tile Group Sequencer 不持有 program cache——Tile Program residency 由 Program Residency Manager 按 `TileRoleBinding` 管理。descriptor/event region 不应与 BOA operand 或 MFE stream buffer 争用同一组 bank。注意此处 "region" 指 SRAM 逻辑分区（memory region），不是已删除的 group-level subgraph 执行层。

### 6.2 性能模型

Tile Group Sequencer 本身不应成为 throughput bottleneck。issue rate 模型：

```text
Issue_bw_required = action_per_task * tasks_in_flight / T_task
Issue_bw_available = f_tgs / cycles_per_action
```

必须满足：

```text
Issue_bw_available >= Issue_bw_required
```

Group task control latency：

```text
T_group_task_control =
  T_init
+ Σ wait_overhead(event/stream/credit/barrier)
+ T_issue_dma_dispatch_collective
+ T_drain
```

Group control overhead 应小于主要 BOA/EVU/MFE/Group DMA 计算和搬运时间；否则 PMU 应显示 `tgs_issue_stall` 或 `tgs_wait_*` 异常偏高。

### 6.3 PMU counters

必需 counter（`tgs_*` 前缀，替代 `rs_*`）：

- `tgs_active_cycles`。
- `tgs_residency_stall_cycles`。
- `tgs_decode_issue_cycles`。
- `tgs_issue_backpressure_cycles` by target：DMA、Tile Dispatcher、Queue、Barrier、Collective。
- `tgs_wait_event_cycles`。
- `tgs_wait_stream_cycles`。
- `tgs_wait_credit_cycles`。
- `tgs_wait_barrier_cycles`。
- `tgs_wait_dma_cycles`。
- `tgs_wait_collective_cycles`。
- `tgs_timeout_count`。
- `tgs_fault_count` by fault_code。
- `tgs_dispatch_role_count`。
- `tgs_task_completed_count`。

Stall attribution：

- Tile Group Sequencer 正在 issue 且 target not ready：owner 为 target backpressure。
- wait event/stream/credit/barrier：owner 为对应 wait reason。
- Tile Program residency miss/cold load：owner 为 program/descriptor stall 或 DMA memory，不能同时计两次 primary stall。

PMU counter 编号、overflow、read/clear 行为由后续规格冻结。

## 7. RTL/软件实现建议

RTL：

- 使用显式 `accepted`/`retired` handshake 追踪每条有副作用 action；fault 时可判断是否需要补偿或 drain。
- action decoder 保持 small and boring；复杂行为放 descriptor，不在 decoder 中嵌入 workload-specific special case。
- role completion scoreboard 按 `role_id` 和 `event_id` 组织，避免无法处理多 role in flight。
- Wait Unit 分离 source-specific ready 和 common timeout/epoch check。
- 所有 event write 带 producer_id、sequence、timestamp、status。
- Tile Program residency miss 通过 Group DMA 路径完成，复用 DMA completion event 和 fault model；不存在 fetchable group-level program text fetch 路径。
- CSR single_step 仅用于 debug，不进入 production scheduling contract。
- action index 只允许线性递增，不实现 branch/loop 通用通路。

Software/compiler：

- compiler 输出 group task action list、role binding 表、queue descriptor、DMA descriptor、collective descriptor，以及 `tile_program_id/tile_program_version/hash/cache hint`。
- runtime patch context-level IOVA、shape、descriptor base、event id 和 Tile Program backing store metadata，不在 hot path 生成 per-tile program 或显式 `program.load`。
- firmware validation 应在 launch 前检查 ABI version、size、range、producer/consumer mask、`tile_mask_union`、`role_count`、tile program version/hash、timeout policy。
- group task 尽量使用固定 Tile Program template 和 descriptor auto-patch，减少 residency miss 和 program cache 压力。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA/formal verification points

- Action index safety：action index 始终落在 action list bounds；不存在 branch target 越界。
- Residency readiness：`dispatch.role` 只有在 Program Residency Manager 对该 `TileRoleBinding` 返回 READY、descriptor window ready 且 epoch 匹配时才允许产生 tile 副作用。
- Side-effect retirement：未 accepted 的 issue 不得产生 event/DMA/dispatch 副作用；accepted action 必须最终 retire、fault 或被 reset epoch kill。
- Wait freshness：wait wakeup 的 event/stream/barrier epoch 必须等于当前 group task epoch。
- Timeout liveness：wait 超过 timeout 后必须进入 fault path 并 signal ERROR/TIMEOUT。
- Role dispatch safety：dispatch 的 `tile_mask` 必须是 `tile_mask_union` 与 available_tile_mask 的子集。
- DMA exactly-once：每个 Tile Group Sequencer accepted DMA descriptor 必须产生 exactly one completion status，reset kill 除外且必须可观测。
- Deadlock bounded proof：在 queue/DMA/tile/collective fairness 假设下，group task action list 要么推进到 `group_task.complete`，要么 timeout。

### 8.2 Bring-up

1. Empty group task：仅 `group_task.accept; group_task.complete`。
2. DMA prefetch wait：HBM -> L2 completion event。
3. Stream producer/consumer roles：role 0 producer (`out_stream`) + role 1 consumer (`in_stream`)，单 group task 内两 role 通过 stream queue 流动。
4. Collective issue/wait：reduce add completion。
5. Barrier：participant mask release 和 timeout case。
6. Fault injection：bad descriptor、unknown role_id、DMA timeout、queue error token。
7. Reset/drain：RUN 和 WAIT 状态下 soft reset，验证 event RESET、credit 回收、action index 停止。

### 8.3 验收标准

- Cold launch 和 warm launch 均可执行同一 group task，且不需要 compiler/runtime 发显式 `program.load`。
- Tile Group Sequencer 能完整推进 `command -> Program Residency Manager (per TileRoleBinding) -> Group DMA -> stream -> prepared tile dispatch -> event -> group task done`。
- PMU counters 能区分 issue backpressure、wait event、wait stream、wait DMA、timeout。
- 所有 SVA/formal safety properties 在配置 depth 下通过。
- Fault path 不产生 stale done event，不泄漏 stream credit，不留下 orphan DMA descriptor。

## 9. 风险、取舍和后续细化方向

风险：

- Action op 范围失控：Tile Group Sequencer 若承担 workload-specific 算子语义，会变成难验证的小 CPU。
- Wait/event/stream epoch 不严谨：reset 后 stale event 可导致错误完成。
- Descriptor window 与 BOA/MFE hot bank 冲突。
- Role completion scoreboard 对多 role in flight 支持不足，导致 dispatch 被迫串行化。
- Timeout policy 不明确，可能掩盖 deadlock 或误杀长 latency DMA。

取舍：

- 使用 action list + descriptor-driven engines，而不是硬件解释 graph 或取指 fetchable group-level program text。
- First Silicon V1 先固定有限 action op 和静态 role binding，优先验证 command/event/DMA/stream/PMU。
- role 内 Tile-SPMD 由 Tile UCE 推进，不在 Tile Group Sequencer 引入 per-role PC 或 per-tile dynamic dispatch 模型。

后续需要冻结：

- Binary action op encoding、action list alignment、action 数量上限。
- `elenor_group_task_launch_desc_v0_t` 字段宽度、role binding descriptor ABI、CSR map。
- Timeout 默认值、zero timeout 语义和 fault code。
- Group task 最大 role 数、queue/stream policy、multi-consumer 行为。
- PMU counter 编号、overflow、read/clear 行为。
