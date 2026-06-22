# ELENOR USE 设计文档

## 1. 定位、目标和 First Silicon cutline

USE（Unified State Engine）是 ELENOR 的 tile-local state compute 和 state lifecycle 功能组件。它面向模型状态、循环状态、scan、recurrence、checkpoint/restore、token routing metadata update 和 tile-local event assist。

USE 不是通用 CPU，不是 Tile Program 主控制器，也不是 memory processor。Tile UCE 和 USE 可以共享同一个 tile-local RISC-V / micro-controller 或等价 micro-sequencer 实现，但功能边界必须保持清晰：

```text
Tile UCE = program control and engine orchestration
USE      = state compute and state lifecycle
MFE      = most data-related dynamic memory access
```

UCE 负责 Tile Program PC、launch/wait/branch、stream token、descriptor patch 和 engine 编排；USE 负责 state register/cache、scan、recurrence、checkpoint/restore 和 local state/event fast path；MFE 负责 page walk、segment decode、address generation、prefetch、reorder 和 stream fill。USE 可以读取 MFE 产生的 metadata stream，也可以更新 token routing metadata，但不接管 page/segment 数据流主路径。

Architecture V1 目标：

- 对 SSM/Mamba/RWKV 类 recurrence、prefix scan、simple state update 建立 tile-local 硬件路径。
- 对 MoE token routing、dynamic shape branch assist 和 local event assist 提供可验证的快速路径。
- 与 Slot Frame、Stream Queue、Tile UCE、MFE、Event 和 PMU 保持 descriptor-driven contract。
- 支持 checkpoint/restore，使 fault/reset path 下 state lifecycle 行为确定。

First Silicon V1 cutline：

| 类别           | 必须实现                                                             | 可预留或后续实现                                  |
| -------------- | -------------------------------------------------------------------- | ------------------------------------------------- |
| State storage  | state register file、state cache tag/valid/dirty、state slot binding | 多级 state cache、跨 tile state migration         |
| Scan           | prefix sum、prefix max、associative scan 基础模式                    | 更复杂 monoid、自定义 combine 函数                |
| Recurrence     | affine recurrence、gated update、loop counter update                 | 高级 recurrence transform、compiler 自动 chunking |
| Checkpoint     | checkpoint/restore descriptor、dirty tracking、fault path 行为       | 多版本 rollback、跨 command speculative state     |
| Event assist   | local wait/signal/barrier assist，不替代 UCE control                 | event dependency compression                      |
| Token metadata | expert offset/counter、top-k reorder assist                          | 复杂 token routing policy、rollback policy 扩展   |
| PMU            | USE active、state cache hit/miss、state stall、checkpoint bytes      | sampled state trace                               |

未冻结值写作 `由后续规格冻结`、`由 SRAM profile 冻结` 或 `由 PPA exploration 冻结`。

## 2. 职责、非职责和 ownership

### 2.1 职责

USE 负责：

1. 小状态和 scalar metadata 的 state register file。
2. 大状态块的 tile-local state cache，例如 SSM hidden state。
3. Prefix sum、prefix max 和基础 associative scan。
4. Affine recurrence、gated update 和 loop-carried state update。
5. State checkpoint、restore、dirty tracking 和 commit ordering。
6. Dynamic shape branch assist 的 tile-local scalar 快路径，例如局部 loop bound、active token count。
7. Token routing metadata update，例如 expert offset、per-expert counter、top-k result reorder assist。
8. Local event/barrier assist，例如 state-dependent signal 或 local wait aggregation。
9. PMU state bottleneck 指标和 error syndrome 输出。

### 2.2 非职责

USE 不负责：

- Tile Program fetch/decode、Tile PC 和主控制流。
- 常规 BOA/EVU/MFE/DMA launch sequence。
- Page table walk、segment address generation、KV prefetch、reorder 和 stream fill。
- HBM/DDR/LPDDR 全局 memory policy。
- 高层 graph schedule、dynamic graph 解释、operator partition。
- 完整通用 CPU 任务或 OS 级控制。
- 大规模 dense GEMM 或 vector elementwise 主路径。

### 2.3 ownership matrix

| 对象                   | owner                          | USE 权限                           | 说明                                                             |
| ---------------------- | ------------------------------ | ---------------------------------- | ---------------------------------------------------------------- |
| State register file    | USE                            | read/write                         | loop state、small scalar、metadata                               |
| State cache            | USE                            | read/write/dirty/evict             | DMA 只能通过 checkpoint/restore 或 explicit command 修改         |
| State slot in L1 frame | USE lifecycle，UCE binds frame | read/write state data              | slot role 必须标记为 state                                       |
| Checkpoint buffer      | USE                            | write/read                         | backing store 地址由 descriptor/runtime 提供                     |
| Tile Program PC        | UCE                            | none                               | USE 只通过 event/condition result 影响 UCE branch                |
| Stream token           | Stream Queue Engine            | metadata read/update when launched | credit protocol 仍由 UCE/Stream Queue 管理                       |
| Page/segment metadata  | MFE                            | read 或 metadata assist            | 数据地址生成 owner 仍是 MFE                                      |
| Event state            | Local Event Unit               | assist signal/wait request         | 不取代 UCE wait 主路径                                           |
| PMU primary stall      | PMU                            | source signal                      | state stall 归 USE，event/stream stall 按 primary owner 规则归因 |

## 3. 微架构和状态机

### 3.1 USE 微架构

```text
+------------------------------------------------------------------+
| USE                                                              |
|                                                                  |
| +----------------+    +----------------+    +------------------+ |
| | Task/Desc Port | -> | State Scheduler| -> | Operand Align    | |
| +----------------+    +-------+--------+    +--------+---------+ |
|                             |                      |             |
|                             v                      v             |
|                    +----------------+     +-------------------+  |
|                    | State Reg File |     | State Cache       |  |
|                    +-------+--------+     +---------+---------+  |
|                            |                        |            |
|          +-----------------+----------+-------------+----------+ |
|          |                            |                        | |
|          v                            v                        v |
| +----------------+          +----------------+        +---------+|
| | Scan Unit      |          | Recurrence Unit|        | Router  ||
| | sum/max/assoc  |          | affine/gated   |        | Assist  ||
| +--------+-------+          +--------+-------+        +----+----+|
|          |                           |                     |     |
|          +--------------+------------+----------+----------+     |
|                         v                       v                |
|                 +---------------+       +-------------------+    |
|                 | Checkpoint    |       | Event/PMU/Error   |    |
|                 | Commit/Restore|       | Assist            |    |
|                 +---------------+       +-------------------+    |
+------------------------------------------------------------------+
```

### 3.2 USE task 状态机

```text
RESET
  -> IDLE
  -> DESC_FETCH
  -> DESC_VALIDATE
  -> STATE_ACQUIRE
  -> INPUT_READY
  -> EXECUTE
  -> COMMIT
  -> CHECKPOINT_OPTIONAL
  -> EVENT_SIGNAL
  -> IDLE
```

异常路径：

```text
任意 active 状态
  -> ERROR_CAPTURE
  -> STATE_ROLLBACK_OR_MARK_DIRTY
  -> EVENT_ERROR
  -> IDLE 或 RESET
```

状态说明：

| 状态                | 行为                                                                              | 退出                 |
| ------------------- | --------------------------------------------------------------------------------- | -------------------- |
| RESET               | 清 task valid、pipeline、temporary result；state cache 是否保留由 reset mode 决定 | reset release        |
| IDLE                | 等待 UCE launch.use 或 MMIO task                                                  | launch accepted      |
| DESC_FETCH          | 读取 USE descriptor 和 state view                                                 | descriptor ready     |
| DESC_VALIDATE       | 检查 version、size、state slot、dtype、update_type、checkpoint_policy             | valid 或 fault       |
| STATE_ACQUIRE       | 获取 state lock/tag，检查 cache hit/miss，必要时请求 DMA/restore path             | state ready          |
| INPUT_READY         | 等待 input slot、MFE metadata stream 或 EVU/BOA event                             | operands ready       |
| EXECUTE             | scan、recurrence、metadata update 或 event assist                                 | compute done         |
| COMMIT              | 写回 state register/cache/output slot，更新 dirty/valid                           | commit done          |
| CHECKPOINT_OPTIONAL | 按 policy 写 checkpoint buffer 或标记 checkpoint required                         | checkpoint done/skip |
| EVENT_SIGNAL        | 向 Local Event Unit signal done/error                                             | event accepted       |
| ERROR_CAPTURE       | 捕获 desc id、state slot、update_type、fault code                                 | syndrome ready       |

### 3.3 Scan pipeline

```text
Input vector / state chunk
  -> align and mask
  -> local prefix network
  -> chunk summary output
  -> optional summary scan input
  -> fixup add/max/combine
  -> output/state commit
```

First Silicon 至少支持：

- prefix sum。
- prefix max。
- fixed associative combine mode，由 descriptor 指定。
- mask/tail，由 descriptor 或 EVU mask metadata 提供。

复杂自定义 combine 函数由后续规格冻结；V1 不要求 arbitrary callback。

### 3.4 Recurrence pipeline

典型形式：

```text
S[t + 1] = A[t] * S[t] + B[t] * X[t]
```

或 gated update：

```text
S[t + 1] = gate[t] * candidate[t] + (1 - gate[t]) * S[t]
```

Pipeline：

```text
state load -> input/gate align -> multiply/add or select -> clamp/round -> commit
```

dtype、rounding、saturation 和 approximation 由 descriptor 明确；未冻结编码由后续规格冻结。

### 3.5 Checkpoint/restore 状态机

```text
CHECKPOINT_IDLE
  -> SNAPSHOT_REQUEST
  -> FLUSH_DIRTY_STATE
  -> WRITE_CHECKPOINT_BUFFER
  -> MARK_CLEAN_OR_VERSIONED
  -> CHECKPOINT_DONE
```

Restore：

```text
RESTORE_REQUEST
  -> INVALIDATE_OR_LOCK_STATE
  -> READ_CHECKPOINT_BUFFER
  -> REBUILD_STATE_CACHE
  -> RESTORE_DONE
```

要求：

- checkpoint/restore 必须由 descriptor 或 explicit command 发起。
- fault/reset path 下，dirty state 必须是 committed、rolled back 或 marked invalid 三者之一。
- checkpoint pointer 可以由 USE / Tile UCE patch，但 state lifecycle owner 是 USE。
- DMA 参与 checkpoint 数据搬运时必须通过 event 同步。

## 4. 接口、descriptor、寄存器和协议

### 4.1 USE state descriptor

```c
typedef enum {
    ELENOR_USE_OP_PREFIX_SUM      = 1,
    ELENOR_USE_OP_PREFIX_MAX      = 2,
    ELENOR_USE_OP_AFFINE_REC      = 3,
    ELENOR_USE_OP_GATED_UPDATE    = 4,
    ELENOR_USE_OP_CHECKPOINT      = 5,
    ELENOR_USE_OP_RESTORE         = 6,
    ELENOR_USE_OP_TOKEN_METADATA  = 7,
    ELENOR_USE_OP_EVENT_ASSIST    = 8,
} elenor_use_op_v0_t;

typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t op;
    uint16_t flags;

    uint32_t context_id;
    uint32_t state_view_slot;
    uint32_t input_slot;
    uint32_t output_slot;

    uint32_t state_dim;
    uint32_t seq_len;
    uint32_t batch;
    uint32_t tile_offset;

    uint16_t dtype_state;
    uint16_t dtype_input;
    uint16_t dtype_accum;
    uint16_t rounding_mode;

    uint32_t checkpoint_policy;
    uint32_t checkpoint_slot;
    uint32_t event_signal;
    uint32_t fault_record_slot;

    uint32_t param_slot;
    uint32_t reserved0;
} elenor_use_state_desc_v0_t;
```

字段语义：

- `state_view_slot` 指向 `elenor_state_view_t` 或 slot-based state view。
- `input_slot/output_slot` 可为无效值，表示纯 state op 或 event assist。
- `param_slot` 保存 affine/gate 参数、scan mode 或 metadata update 参数。
- `checkpoint_policy` 指定 none、before update、after update、on fault、explicit。
- `tile_offset` 用于 Tile-SPMD 切分 state chunk，不等同于 page table walk。

### 4.2 StateView

```c
typedef struct {
    uint16_t abi_version;
    uint16_t view_size;
    uint32_t flags;

    uint64_t base;
    uint32_t state_dim;
    uint32_t num_slots;
    uint32_t dtype;

    uint32_t slot_stride;
    uint32_t checkpoint_base_slot;
    uint32_t owner_context;
    uint32_t version;
} elenor_state_view_v0_t;
```

StateView 描述 state backing store 和 version；实际 L1 state slot 仍由 Tile Slot Frame 绑定。`base` 的 address space、IOMMU domain 和 residency 由 runtime/firmware 管理。

### 4.3 Tile UCE launch.use 请求

```c
typedef struct {
    uint16_t engine_kind;
    uint16_t flags;
    uint32_t use_desc_slot;
    uint32_t event_id;
    uint32_t timeout_cycles;
    uint32_t dependency_event_mask;
} elenor_use_launch_req_v0_t;
```

语义：

1. UCE 完成 descriptor patch 和 fence。
2. UCE 发送 launch request。
3. USE 获取 state lock/tag，执行 descriptor。
4. USE signal done/error event。
5. UCE wait event 后继续 Tile Program。

USE 不直接推进 UCE PC；它只能通过 event result、condition result 或 fault 影响 UCE 的后续 branch。

### 4.4 自定义指令示例

若采用 RISC-V 共享实现，USE 可作为 custom functional unit 或 MMIO task。示例语义，不冻结编码：

```asm
# UCE control path
patch.desc       d_use, slot[STATE], slot[INPUT], slot[OUTPUT]
launch.use       d_use -> e_use
wait             e_use
br.err           e_use, state_fault

# 可选 custom instruction 形态
use.scan.sum     rd_event, rs_desc
use.rec.affine   rd_event, rs_desc
use.ckpt         rd_event, rs_state_view
use.restore      rd_event, rs_state_view
```

这些指令只提交 USE task 或读取 result token，不允许把 USE 扩展成通用 scalar/vector compute engine。

### 4.5 USE CSR / MMIO

| CSR                                | 语义                                           |
| ---------------------------------- | ---------------------------------------------- |
| `use_status`                       | idle/busy/wait_state/execute/checkpoint/fault  |
| `use_active_desc`                  | 当前 descriptor slot                           |
| `use_state_tag`                    | 当前 state view/version/tag                    |
| `use_fault_status`                 | fault code、state slot、op、dtype              |
| `use_cache_status`                 | hit/miss/dirty/valid summary                   |
| `use_checkpoint_status`            | checkpoint in-flight、bytes、version           |
| `use_pmu_select` / `use_pmu_value` | PMU 读数                                       |
| `use_debug_ctrl`                   | halt at task boundary、snapshot state metadata |

CSR 地址和 bit field 由后续规格冻结。

### 4.6 Event、Stream、Slot 协议

Event：

- USE task completion 通过 Local Event Unit signal。
- Event assist 只能优化 local wait/signal 聚合，不改变 UCE 是主控制流 owner 的事实。
- Timeout 由 UCE watchdog 或 USE local watchdog 捕获；fault record 必须包含 owner。

Stream：

- USE 可读取 token metadata 或更新 routing metadata，但 token credit acquire/release 仍由 UCE/Stream Queue 协议负责。
- USE 产生 error 时，如果当前 tile stage 属于 stream pipeline，UCE 负责把 error token 推送到下游。

Slot：

- state slot 必须标记 `STATE` role。
- USE 写 state slot 时需要 write permission 和 owner match。
- DMA 只有在 checkpoint/restore descriptor 或 explicit state load/store command 下可接触 state backing buffer。

## 5. 数据流、控制流和时序路径

### 5.1 SSM / Mamba / RWKV 数据流

```text
input token / chunk
  -> BOA projection
  -> EVU local activation / gate prep
  -> USE recurrence update
  -> optional USE checkpoint
  -> BOA output projection
```

USE 只处理 state update：projection dense compute 归 BOA，activation/tail 归 EVU，input chunk stream 归 MFE/DMA。

### 5.2 Chunked scan 数据流

```text
sequence -> chunks -> local scan -> chunk summary scan -> fixup -> output
```

分层 owner：

- UCE 启动每个 scan/fixup task 并等待 event。
- USE 执行 local scan、summary combine、fixup。
- Stream Queue 连接 stage token。
- MFE/DMA 负责大块输入输出移动。

### 5.3 MoE token routing assist

```text
router logits -> EVU/BOA top-k prep
  -> USE expert counter / offset update
  -> MFE Segment Stream token grouping
  -> BOA expert GEMM
  -> EVU/Collective combine
```

USE 可维护 per-expert counter、prefix offset、capacity flag 和 local routing metadata；实际 token gather、segment stream、expert weight fetch 归 MFE/DMA。

### 5.4 Paged attention 中的 USE 边界

Paged attention 默认 owner：

```text
MFE: page table walk / block table decode / physical address generation / KV prefetch / reorder
USE: streaming state / local counters / checkpoint / optional token metadata update
UCE: Tile Program control / launch / wait / stream EOS/error branch
```

USE 不做 page walk 主路径。若 metadata 需要 state update，USE 产生更新后的 metadata 或 counter，MFE 仍负责后续数据流。

### 5.5 时序路径

USE launch 时序：

```text
UCE patch desc
  -> launch.use accepted
  -> USE desc validate
  -> state cache lookup
  -> input/event ready
  -> execute scan/recurrence/update
  -> commit state
  -> optional checkpoint
  -> signal event
  -> UCE wait retires
```

关键性能条件：

```text
T_use_state_update <= T_neighbor_compute_or_stream_overlap
```

当 USE 成为瓶颈，PMU 应显示 `use_active_cycles` 高、`use_state_cache_miss` 或 `use_state_stall` 增加，而不是被错误归因到 UCE wait。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置项

| 配置                        | 说明                            | 冻结方式                |
| --------------------------- | ------------------------------- | ----------------------- |
| State register file entries | 小状态和 scalar metadata 数量   | 由后续规格冻结          |
| State cache capacity        | tile-local state cache 大小     | 由 SRAM profile 冻结    |
| State cache associativity   | tag/evict 策略                  | 由 SRAM profile 冻结    |
| Scan lane width             | scan datapath 宽度              | 由 PPA exploration 冻结 |
| Recurrence units            | affine/gated update 并行度      | 由 PPA exploration 冻结 |
| Checkpoint bandwidth        | state checkpoint 吞吐           | 由 SRAM profile 冻结    |
| Supported dtype             | INT/BF/FP/state dtype           | 由后续规格冻结          |
| Event assist entries        | local event aggregation entries | 由后续规格冻结          |

### 6.2 性能模型

USE state path 一阶模型：

```text
T_use = T_desc
      + T_state_lookup
      + T_input_ready
      + T_update_compute
      + T_commit
      + T_checkpoint_optional
```

State cache 影响：

```text
T_state_lookup = hit_rate * T_hit + (1 - hit_rate) * T_miss
```

系统性能上限：

```text
Perf = min(BOA_compute, EVU_compute, MFE_stream, USE_state, Memory_bw)
```

对 recurrence workload：

```text
Throughput_state = state_updates / max(T_use, T_input_stream, T_output_consume)
```

对 MoE routing：

```text
imbalance = max(tokens_per_expert) / avg(tokens_per_expert)
U_boa = 1 / imbalance
```

USE 的 routing assist 目标是降低 dispatch overhead 和改善 expert offset 生成，不保证单独解决 expert imbalance。

### 6.3 PMU counter

| Counter                     | 说明                                |
| --------------------------- | ----------------------------------- |
| `use_active_cycles`         | USE 执行 active 周期                |
| `use_task_count`            | 完成 task 数                        |
| `use_scan_count`            | scan task 数                        |
| `use_recurrence_count`      | recurrence task 数                  |
| `use_checkpoint_count`      | checkpoint task 数                  |
| `use_restore_count`         | restore task 数                     |
| `use_state_cache_hit`       | state cache hit 次数                |
| `use_state_cache_miss`      | state cache miss 次数               |
| `use_state_stall`           | 等 state lock/cache/fill 的周期     |
| `use_input_wait`            | 等 input slot/event/metadata 的周期 |
| `use_commit_stall`          | 等 writeback/checkpoint 的周期      |
| `use_event_assist_count`    | event assist 次数                   |
| `use_token_metadata_update` | token metadata update 次数          |
| `use_fault_count`           | USE fault 次数                      |
| `use_checkpoint_bytes`      | checkpoint/restore bytes            |

Primary attribution：

- USE 正在执行 state compute：`engine_active` secondary tag USE。
- USE 等 state cache fill：`engine_wait_operand` 或 `dma_wait_memory`，根据 miss owner 归因。
- USE 等 stream metadata：`stream_credit_empty_or_full` 或 `engine_wait_operand`，由具体阻塞源决定。
- UCE wait USE event 的周期不重复计入 USE active；可作为 UCE wait secondary tag。

## 7. RTL/软件实现建议

### 7.1 RTL 模块拆分

```text
use_state_engine
├── use_launch_ingress
├── use_desc_fetch_validate
├── use_state_scheduler
├── use_state_regfile
├── use_state_cache
├── use_operand_align
├── use_scan_unit
├── use_recurrence_unit
├── use_token_router_assist
├── use_checkpoint_unit
├── use_event_assist
├── use_fault_unit
└── use_pmu_source
```

接口：

- UCE -> USE：launch request、descriptor ref、event id、timeout。
- USE -> Event Unit：done/error/timeout signal。
- USE -> L1 SRAM：state cache read/write、checkpoint buffer read/write。
- USE -> Stream metadata：token metadata read/update request。
- USE -> PMU：active/stall/hit/miss/fault source。

### 7.2 Shared RISC-V / micro-controller 实现

共享实现建议：

```text
tile-local RISC-V / micro-controller
├── shared fetch/debug/CSR/exception shell
├── UCE front-end: Tile Program PC, launch/wait/branch/stream/patch
└── USE back-end: state FU, state cache, scan/recurrence/checkpoint
```

实现纪律：

- Signal 命名和 RTL ownership 区分 `uce_*` 与 `use_*`。
- RISC-V exception cause 必须能区分 UCE control fault 和 USE state fault。
- USE state FU 不执行 arbitrary memory walk；page/segment 数据访问仍经 MFE。
- Debug halt 可停在 task boundary；读取 state cache metadata 不应破坏 dirty/valid。

### 7.3 Clock/reset/debug/exception

Clock：

- First Silicon 可与 tile core clock 同步。
- Scan/recurrence datapath 可后续独立 clock gate；跨域策略由 PPA exploration 冻结。
- State cache SRAM clock 由 SRAM profile 冻结。

Reset：

| Reset                | 行为                                                                            |
| -------------------- | ------------------------------------------------------------------------------- |
| hard reset           | 清 state cache valid/dirty、regfile、pipeline、fault、PMU running state         |
| tile soft reset      | 停止新 task，当前 task 进入 rollback/mark invalid 策略，signal reset event      |
| state preserve reset | 保留 selected state cache line，标记 version，需要后续 restore/validate         |
| USE local reset      | 清 USE pipeline，不改变 UCE PC；必须向 UCE 返回 deterministic error/reset event |

Exception：

| fault              | 触发                                               | syndrome                |
| ------------------ | -------------------------------------------------- | ----------------------- |
| invalid descriptor | version/size/op/dtype/slot 不合法                  | desc id、op、field      |
| state permission   | 非 state slot、write denied、owner mismatch        | slot id、owner、context |
| state cache fault  | tag conflict、dirty eviction illegal、ECC optional | state view、tag、way    |
| checkpoint fault   | checkpoint address/range/event timeout             | checkpoint slot、bytes  |
| arithmetic fault   | unsupported dtype、overflow mode illegal           | op、dtype、rounding     |
| event assist fault | invalid event id 或 illegal barrier mode           | event id、mode          |

Debug：

- read state metadata：valid/dirty/version/tag。
- snapshot small state register file。
- halt at task boundary。
- expose active descriptor and operation.
- expose PMU counters.

### 7.4 软件和 compiler 建议

Compiler pass：

- 识别 scan/recurrence/state update pattern。
- 生成 USE descriptor template 和 StateView。
- 将 dense projection 分给 BOA，elementwise/gate prep 分给 EVU，state update 分给 USE。
- 对 chunked scan 生成 local scan、summary scan、fixup 的 region/tile pipeline。
- 初期依赖 kernel library，不要求 compiler 自动发明复杂 recurrence transform。

Runtime/firmware：

- 管理 state backing store、checkpoint buffer 和 version。
- 在 context switch/reset/fault 时调用 checkpoint/restore command 或标记 state invalid。
- 读取 PMU 时区分 USE active、state miss、UCE wait。

## 8. 验证、bring-up 和验收标准

### 8.1 单元验证

| 单元                | 覆盖点                                                                |
| ------------------- | --------------------------------------------------------------------- |
| Descriptor validate | op/dtype/slot/size/version/reserved field                             |
| State regfile       | read/write hazard、reset、debug snapshot                              |
| State cache         | hit/miss、dirty、evict、protected region、ECC optional                |
| Scan unit           | prefix sum、prefix max、mask/tail、chunk summary、fixup               |
| Recurrence unit     | affine、gated update、rounding/saturation、sequence boundary          |
| Checkpoint unit     | before/after/on-fault policy、restore、version mismatch               |
| Event assist        | wait/signal aggregation、invalid event、timeout                       |
| Token metadata      | expert counter、offset update、capacity flag、duplicate handling mode |
| Fault unit          | syndrome completeness、event error、reset path                        |
| PMU                 | active/stall/hit/miss/fault counter and primary attribution           |

### 8.2 Golden tests

1. Prefix sum random tensor，覆盖 small/medium/long length、mask/tail。
2. Prefix max，覆盖负值、重复最大值、tail mask。
3. Affine recurrence，与 Python golden 逐步比对。
4. Gated update，覆盖 gate 0、gate 1、中间值和 saturation mode。
5. Chunked scan：local scan + summary + fixup 与整段 golden 对齐。
6. Checkpoint/restore：正常 restore、fault 后 restore、reset 后 restore。
7. MoE routing metadata：per-expert counter、offset、capacity overflow 行为。
8. Event assist：local wait/signal 和 timeout。
9. Fault injection：invalid state slot、unsupported dtype、checkpoint range fault、state cache dirty eviction illegal。

### 8.3 Bring-up 顺序

1. launch.use 空 task -> event done。
2. State register file read/write task。
3. Prefix sum small vector。
4. Prefix max + mask/tail。
5. Affine recurrence single batch。
6. Gated update with EVU-prepared gate/input slots。
7. State cache hit/miss PMU。
8. Checkpoint/restore descriptor。
9. SSM/Mamba/RWKV recurrence golden trace。
10. MoE token metadata assist + MFE Segment Stream integration。
11. Fault/reset path with state rollback or invalidation。

### 8.4 验收标准

- USE task 必须通过 UCE launch/use descriptor/event 路径执行，不能由 testbench 直接驱动 datapath 作为唯一证据。
- Scan/recurrence/checkpoint/restore 必须与 Python golden 对齐。
- USE 不执行 page walk 主路径；paged attention trace 中数据相关动态地址 owner 应为 MFE。
- UCE 与 USE 共享物理实现时，debug、exception、event ownership 清晰可观测。
- State slot 权限、checkpoint version、dirty/valid 状态在 reset/fault 后确定。
- PMU 能解释 state bottleneck：active、state miss、input wait、commit stall 不混淆。
- Phase 5 验收需要 Mamba/RWKV 类 recurrence golden、checkpoint/restore fault path 和 USE state cache counter 同时通过。

## 9. 风险、取舍和后续细化方向

### 9.1 风险

| 风险                            | 影响                                               | 缓解                                                             |
| ------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------- |
| USE 范围膨胀成通用控制器        | 验证面失控，和 UCE 重叠                            | V1 绑定 prefix scan、affine recurrence、checkpoint、event assist |
| 与 MFE ownership 混淆           | page walk 或 dynamic memory path 进入 state engine | 明确 MFE 拥有数据相关动态访存，USE 只更新 state/metadata         |
| State cache 一致性复杂          | reset/fault 后 state 不确定                        | version、dirty、checkpoint policy、explicit restore              |
| Checkpoint 成本过高             | recurrence 性能被写回拖慢                          | checkpoint policy 可配置，PMU 记录 bytes/stall                   |
| Shared RISC-V debug 混淆        | bring-up 难定位 fault owner                        | UCE/USE CSR、fault cause、PMU source 分离                        |
| Compiler 过早依赖复杂 transform | First Silicon 难收敛                               | 先 kernel library + fixed pattern，再扩展自动 chunking           |

### 9.2 取舍

- Prefix/simple recurrence 优于通用状态机：覆盖 V1 关键 workload，验证边界清晰。
- State cache protected region 优于完全共享 scratch：降低 state 被 DMA 临时数据破坏风险。
- Descriptor-driven state task 优于任意 firmware loop：便于 ABI、PMU、verification 对齐。
- Checkpoint/restore 显式化优于隐式 state persistence：fault/reset 行为可测试。

### 9.3 后续细化方向

- USE descriptor binary encoding、op code、dtype 编码：由后续规格冻结。
- State register file entries、state cache capacity/associativity/line size：由 SRAM profile 冻结。
- Scan/recurrence datapath 宽度、latency、clock gating：由 PPA exploration 冻结。
- Checkpoint policy 枚举和 rollback semantics：由后续规格冻结。
- Event assist 的 exact barrier/wait aggregation 规则：由后续规格冻结。
- MoE token metadata duplicate/capacity overflow 行为：由后续规格冻结。
- Debug CSR 地址和 fault syndrome layout：由后续规格冻结。
