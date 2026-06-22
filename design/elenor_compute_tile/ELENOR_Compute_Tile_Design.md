# ELENOR Compute Tile 设计文档

## 1. 定位、目标和 First Silicon cutline

Compute Tile 是 ELENOR 的 tile-local kernel 执行域。它不调度高层 graph，也不消费 tensor algebra；它接收 Region Sequencer 下发的 tile task，执行已经由 compiler/runtime 降低好的 Tile Program、slot frame、descriptor 和 stream token。

核心定位：

```text
Compute Tile = Tile UCE 控制面
             + L1 SRAM / Tile DMA 数据工作集
             + BOA dense compute
             + EVU irregular vector compute
             + MFE tile stream port
             + USE state compute/lifecycle
             + local event / PMU / debug
```

目标是把一个 tile 内的 kernel pipeline 闭环：L2 到 L1 搬运、descriptor auto-patch、engine launch、event wait、stream pop/push、L1 slot 生命周期、state update 和 PMU 归因。Compute Tile 只处理 tile-local 工作；跨 group 调度、全局 memory policy、多模型 QoS 和 host ABI 由上层 runtime / Region Sequencer / driver 负责。

Architecture V1 的目标形态：

- 同一份 Tile-SPMD program template 可在多个 tile 上运行，通过 `tile_id`、`group_id`、descriptor offset 和 slot frame binding 区分数据。
- Tile UCE 和 USE 是两个功能组件；实现上可共享一个 tile-local RISC-V / micro-controller 或等价 micro-sequencer。
- UCE 负责 program control、engine launch、event wait、stream token 和 descriptor patch。
- USE 负责 state register/cache、scan、recurrence、checkpoint/restore 和 state lifecycle。
- MFE 负责大多数数据相关的动态内存访问，包括 page/segment metadata walk、address generation、prefetch、reorder 和 stream fill。
- BOA、EVU、MFE、USE 都通过 descriptor/task 进入执行，不让 datapath 解释高层 graph。

First Silicon V1 cutline：

| 范围     | 必须闭环                                                                                     | 可预留字段或后续实现                                        |
| -------- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 控制面   | tile task receive、prepared local program handle check、PC 推进、wait/fence、tile done event | priority、preemption、多上下文 tile-local time slicing      |
| L1 / DMA | slot frame binding、1D/2D/strided L2->L1、L1->L2、async event                                | multicast、gather list、复杂 layout transform               |
| BOA      | INT8/BF16 GEMM、QK/AV 基础路径、split-K reduce 接口                                          | 复杂 epilogue fusion、稀疏 matmul                           |
| EVU      | elementwise、mask/tail、softmax/norm、基础 gather                                            | full scatter、atomic update、复杂 permutation               |
| MFE port | Page Stream minimal token 到 L1 stream slot、error/EOS 传播                                  | Segment Stream full update、Sparse Block、Persistent Stream |
| USE      | state register/cache 接口、prefix scan、simple recurrence、checkpoint/restore                | 高级 recurrence transform、复杂 token routing rollback      |
| PMU      | engine active/stall、DMA bandwidth、SRAM bank conflict、stream wait、event wait              | sampled trace、完整 feedback scheduler                      |

未冻结数值全部以 `由后续规格冻结`、`由 SRAM profile 冻结` 或 `由 PPA exploration 冻结` 标注，不在 Compute Tile 文档中伪造二进制编码或物理宏参数。

## 2. 职责、非职责和 ownership

### 2.1 模块职责

| 模块                       | owner               | 职责                                                                                                                         | 非职责                                                     |
| -------------------------- | ------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Compute Tile top           | Tile integration    | Tile task ingress、clock/reset/debug 集成、L1/NoC/PMU 汇聚                                                                   | graph schedule、global queue policy                        |
| Tile Command Queue         | Tile UCE            | 接收 Region Sequencer 发来的 prepared tile task，保存 context、program id、local handle、frame id、stream id                 | host command ring 解析                                     |
| Tile UCE                   | Tile control        | Tile PC、从 local program slot/I-cache fetch/decode、launch/wait/branch/fence、descriptor patch、stream token、Tile DMA 编排 | state update 算术、page table walk、高层 graph 解释        |
| USE                        | Tile state          | state register/cache、scan、recurrence、checkpoint/restore、local event assist                                               | Tile Program 主 PC、常规 engine launch、大多数动态数据访存 |
| Tile DMA                   | Tile data movement  | L2<->L1、slot 到 slot copy、async completion event、basic stride                                                             | HBM 访问调度、page/segment walk                            |
| L1 SRAM / Slot Frame       | Tile memory         | program/descriptor/event region、operand、accumulator、vector temp、stream buffer、state cache                               | cache coherence、global allocation                         |
| BOA Cluster                | Dense compute       | GEMM、Conv lowering、attention QK/AV、expert MLP                                                                             | elementwise、fine-grained gather/scatter                   |
| EVU                        | Irregular compute   | elementwise、activation、norm、softmax、mask/tail、基础 gather/scatter                                                       | 大规模 dense matmul 主路径                                 |
| MFE Tile Port              | Memory flow ingress | 接收 MFE stream token/payload，写入 L1 stream/metadata slot，向 UCE/EVU/BOA/USE 暴露 ready                                   | 任意图遍历、程序控制流                                     |
| Local Event / Barrier Unit | Tile sync           | engine completion、tile done、timeout、fault、local barrier                                                                  | group barrier 的全局 arbitration                           |
| PMU / Trace                | Observability       | primary stall owner、engine utilization、queue/stream/SRAM/NoC counter                                                       | 调度策略本身                                               |

### 2.2 ownership 边界

关键 ownership 必须唯一：

- Tile Program PC：Tile UCE 唯一 owner。
- state slot 内容：USE 管理生命周期；DMA 只能在明确 checkpoint/restore 或 UCE 发起的数据搬运路径下修改。
- metadata/page-list slot：MFE 可写入；UCE/USE 可读取；同一 slot 的写 owner 由 frame/descriptor 指定。
- stream token credit：Stream Queue Engine owner；Tile UCE 通过 pop/push/acquire/release 协议使用，不私自改 credit 计数。
- fault record：产生 fault 的模块写本地 syndrome，Local Event Unit 分配或引用 fault record slot，上报给 Region Sequencer。
- PMU primary stall：每个 cycle 只能归给一个 primary owner；secondary tag 仅用于 debug。

### 2.3 非目标

Compute Tile 不承担：

1. 高层 graph 解释和动态 tensor algebra 推理。
2. 全芯片多模型 QoS、preemption 和 global priority arbitration。
3. HBM/DDR/LPDDR 全局 memory policy。
4. 完整 GPU SIMT 编程模型、per-thread PC、warp scheduler 或 reconvergence stack。
5. 通用 CPU 任务、RTOS、driver 逻辑。

## 3. 微架构和状态机

### 3.1 Top-level 数据通路

```text
              Region Sequencer / Stream Queue / Event Fabric
                               |
                               v
+-------------------------------------------------------------------+
| Compute Tile                                                      |
|                                                                   |
|  +----------------+       +------------------+                    |
|  | Tile Cmd Queue | ----> | Tile UCE front   |                    |
|  +----------------+       | PC/launch/wait   |                    |
|           |               +--------+---------+                    |
|           |                        |                              |
|           v                        v                              |
|  +----------------+      +-------------------+      +----------+  |
|  | Local Event    | <--> | Descriptor Patch  | <--> | PMU/CSR  |  |
|  | Barrier/Fault  |      +-------------------+      +----------+  |
|  +-------+--------+                |                              |
|          |                         v                              |
|          |        +---------------------------------------------+ |
|          |        | L1 SRAM / Slot Frame                        | |
|          |        | program/desc/event/input/output/state/stream| |
|          |        +--+---------+---------+---------+--------+---+ |
|          |           |         |         |         |        |     |
|          v           v         v         v         v        v     |
|      Tile DMA      BOA       EVU      MFE port    USE    Router   |
|          |                                           |            |
+----------+-------------------------------------------+------------+
```

### 3.2 Tile task 状态机

```text
RESET
  -> IDLE
  -> TASK_ACCEPT
  -> RESIDENCY_CHECK
  -> FRAME_BIND
  -> PROGRAM_RUN
  -> DRAIN
  -> COMPLETE
  -> IDLE

任意状态出现不可恢复 fault:
  -> FAULT_CAPTURE
  -> FAULT_SIGNAL
  -> DRAIN_OR_RESET
  -> IDLE 或 RESET
```

状态说明：

| 状态                | 进入条件                                 | 行为                                                                                                                     | 退出条件                        |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| RESET               | reset asserted 或 reset command          | 清空 valid bit、停止 launch、回收本地 event、使 pending token 进入 drain policy，并失效本 tile 可见的 program handle tag | reset release 且 CSR 初始化完成 |
| IDLE                | 无 active tile task                      | clock gating eligible，保留 validated local program/state tag                                                            | command queue 非空              |
| TASK_ACCEPT         | Region Sequencer dispatch                | latch context_id、program_id、frame_id、stream binding、local handle、timeout                                            | command 合法                    |
| PREPARED_TASK_CHECK | task accepted                            | 校验 `program_local_slot/program_version/program_epoch`、descriptor cache、frame generation；不向 HBM 发 program load    | handle/frame ready              |
| FRAME_BIND          | prepared metadata ready                  | 校验 slot permission、alignment、bank policy、owner                                                                      | frame valid                     |
| PROGRAM_RUN         | frame bound                              | UCE 从 local program slot/I-cache 取指并推进 Tile Program，launch engines，处理 stream/event                             | program END 或 fault            |
| DRAIN               | program done 或 reset/drain              | 等待 outstanding DMA/engine/token 进入确定状态                                                                           | pending count 为 0 或 timeout   |
| COMPLETE            | drain complete                           | signal tile done event，snapshot PMU，可写 status                                                                        | event accepted                  |
| FAULT_CAPTURE       | invalid desc、timeout、ECC、engine fault | freeze syndrome、pc、desc id、slot id、event id                                                                          | fault record 写入完成           |

### 3.3 Engine task 状态机

每个 engine 接口采用一致的 ready/valid + event 模型：

```text
ENGINE_IDLE
  -> DESC_FETCH
  -> DESC_VALIDATE
  -> OPERAND_READY_WAIT
  -> ISSUE
  -> RUN
  -> WRITEBACK
  -> EVENT_SIGNAL
  -> ENGINE_IDLE
```

- `DESC_VALIDATE` 检查 ABI version、size、slot permission、dtype/layout 支持、alignment、reserved bit。
- `OPERAND_READY_WAIT` 可等待 DMA event、MFE stream token、stream queue credit 或 USE state lock。
- `RUN` 内部由 BOA/EVU/MFE/USE 自己的 micro-sequencer 推进；UCE 只持有 outstanding event id。
- `EVENT_SIGNAL` 必须同时更新 event table 和 PMU completion counter。

### 3.4 L1 SRAM 与 arbitration

L1 SRAM 逻辑分区：

| 分区                         | First Silicon 建议                      | 访问者                            | 规则                                      |
| ---------------------------- | --------------------------------------- | --------------------------------- | ----------------------------------------- |
| Program / Descriptor / Event | 128 KB 级样例；最终由 SRAM profile 冻结 | UCE、descriptor patch、event unit | 不与 BOA operand hot path 共 bank         |
| BOA operand                  | 由 SRAM profile 冻结                    | DMA、BOA                          | double buffer，bank-aware layout          |
| BOA accumulator              | 由 SRAM profile 冻结                    | BOA、EVU optional                 | accumulator role 必须由 slot 标记         |
| EVU vector temp              | 由 SRAM profile 冻结                    | EVU、DMA                          | 支持 mask/tail replay                     |
| MFE stream buffer            | 由 SRAM profile 冻结                    | MFE port、BOA/EVU/USE             | ping-pong，token payload 与 metadata 分离 |
| USE state cache              | 由 SRAM profile 冻结                    | USE、checkpoint/restore DMA       | protected region，不被临时 DMA 覆盖       |
| DMA staging/shared           | 由 SRAM profile 冻结                    | DMA、UCE                          | 只作为显式 workspace 使用                 |

Arbiter 至少应区分：BOA operand read、BOA accumulator RMW、EVU LSU、MFE stream write/read、DMA load/store、USE state、UCE program/descriptor/event。带宽峰值和端口数由 SRAM profile 冻结；文档只冻结归因和隔离原则。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Tile task descriptor

```c
typedef struct {
    uint16_t abi_version;
    uint16_t desc_size;
    uint16_t flags;
    uint16_t priority;

    uint32_t context_id;
    uint32_t region_id;
    uint32_t group_id;
    uint32_t tile_id;

    uint32_t program_id;
    uint32_t template_id;
    uint32_t program_version;
    uint32_t program_local_slot;
    uint32_t program_local_offset;
    uint32_t program_bytes;
    uint32_t program_epoch;

    uint32_t desc_window_base;
    uint32_t desc_window_bytes;
    uint32_t frame_id;
    uint32_t frame_generation;
    uint32_t event_base;
    uint32_t stream_base;

    uint32_t timeout_cycles;
    uint32_t fault_record_slot;
} elenor_prepared_tile_task_v1_t;
```

字段语义：

- `abi_version`、`desc_size` 用于二进制兼容。
- `program_id/template_id/program_version` 标识期望执行的 Tile Program；真正执行入口由 `program_local_slot + program_epoch` 绑定。
- `program_local_slot/program_local_offset/program_bytes` 是 Tile Dispatcher 下发的 resident local handle；UCE 不在运行态解引用 global `program_iova`。
- `desc_window_base/bytes` 约束 descriptor patch / fetch 的可见窗口。
- `frame_id/frame_generation` 绑定 L1 slot frame；program text 不允许在 running 状态被 patch。
- `event_base/stream_base` 提供 tile-local event / stream 命名空间基址。

### 4.2 Slot Frame 接口

```c
#define ELENOR_TILE_SLOT_COUNT_V0 16

typedef struct {
    uint32_t base;
    uint32_t size;
    uint16_t layout;
    uint16_t role;
    uint16_t alignment;
    uint16_t bank_policy;
    uint32_t flags;
    uint32_t owner;
} elenor_tile_slot_v0_t;

typedef struct {
    uint16_t abi_version;
    uint16_t frame_size;
    uint32_t frame_id;
    uint32_t slot_count;
    elenor_tile_slot_v0_t slots[ELENOR_TILE_SLOT_COUNT_V0];
} elenor_tile_frame_v0_t;
```

校验规则：

1. `base + size` 必须落在 tile L1 可访问范围内。
2. `alignment` 必须满足 engine descriptor 的访问粒度。
3. `role` 与 engine descriptor 必须匹配，例如 BOA output 不能写 metadata-only slot。
4. `owner` 与写入路径一致；多个写 owner 必须通过 explicit handoff event。
5. `bank_policy` 可作为 compiler hint；实际 bank 数由 SRAM profile 冻结。

### 4.3 Tile UCE 指令示例

Compute Tile 文档只要求语义，不冻结编码。示例：

```asm
tile_stage:
    stream.pop       r_tok, S_IN
    br.eos           r_tok, tile_done

    patch.desc       d_load, frame.slot[input], r_tok.payload
    launch.mfe       d_load -> e_load
    wait             e_load

    patch.desc       d_boa, frame.slot[input], frame.slot[acc]
    launch.boa       d_boa -> e_boa
    wait             e_boa

    launch.evu       d_epilogue -> e_evu
    wait             e_evu

    launch.use       d_state_update -> e_use
    wait             e_use

    stream.acquire   r_out, S_OUT
    patch.desc       d_store, frame.slot[output], r_out.payload
    launch.mfe       d_store -> e_store
    wait             e_store
    stream.push      S_OUT, r_out
    stream.release   S_IN, r_tok
    br               tile_stage

tile_done:
    stream.eos       S_OUT
    signal.tile.done
    ret
```

### 4.4 Engine descriptor 引用原则

- BOA/EVU/MFE/USE descriptor 应优先引用 slot id，而不是裸物理地址。
- context-level base IOVA 由 runtime/firmware patch。
- tile_id/group_id/slot offset 由 UCE auto-patch。
- page list/segment offset 由 MFE 管理。
- state slot/checkpoint pointer 由 USE 管理生命周期，控制动作由 UCE 发起。

### 4.5 CSR / MMIO 寄存器窗口

| CSR                      | 访问者                     | 语义                                 |
| ------------------------ | -------------------------- | ------------------------------------ |
| `tile_status`            | firmware/debug             | idle/running/drain/fault/reset 状态  |
| `tile_pc`                | debug only 或 trap handler | 当前 Tile Program PC                 |
| `active_context`         | UCE                        | 当前 context id                      |
| `active_program`         | UCE                        | 当前 program id / resident tag       |
| `frame_base`             | UCE                        | 当前 frame descriptor base           |
| `event_head`             | event unit                 | local event ring head                |
| `fault_status`           | event/debug                | fault code、engine id、slot id valid |
| `pmu_select`/`pmu_value` | PMU                        | counter readout                      |
| `reset_control`          | firmware                   | tile-local reset/drain 命令          |

寄存器位宽、地址偏移、endianness 由后续规格冻结。

### 4.6 Stream Queue 协议

Compute Tile 通过 UCE 执行 stream 操作：

```text
consumer: stream.pop -> inspect flags -> consume payload -> stream.release
producer: stream.acquire -> fill payload -> stream.push
```

必须支持：

- valid token、EOS token、error token。
- queue full stall producer，queue empty stall consumer。
- reset/drain 回收 credit 并清理 pending event。
- error token 携带 fault record index，并传播到 tile/region completion。
- multi-consumer 策略必须在 stream descriptor 中声明为 broadcast、refcount 或 independent queue。

## 5. 数据流、控制流和时序路径

### 5.1 Cold launch

```text
Host Runtime
  -> upload package / descriptors / weights
  -> register program section metadata
  -> ring doorbell
Device Runtime
  -> validate command
  -> issue region task
Tile Group Sequencer
  -> ensure region/tile program residency
  -> init streams
  -> dispatch prepared tile task
Compute Tile
  -> accept prepared task
  -> validate local program handle / frame generation
  -> bind slot frame / descriptor window
  -> run Tile Program from local slot
  -> signal tile done
```

### 5.2 Warm launch

Warm path 不 reload program，仅更新 descriptor/context/shape metadata：

```text
Runtime patches descriptors
  -> invalidates descriptor cache if needed
Tile Group Sequencer dispatches prepared task for resident tile kernel
  -> UCE validates unchanged local program handle
  -> UCE binds updated descriptor
  -> UCE runs Tile Program
```

Warm launch 的关键风险是 descriptor cache coherence 和 stale local handle；必须要求 running program text 不被 patch，descriptor patch 有 version/tag 或 invalidate 规则，reset/drain 后旧 handle 不能复用。

### 5.3 Dense GEMM tile 数据流

```text
L2 A/B tiles
  -> Tile DMA or MFE load into input slots
  -> BOA operand double buffer
  -> BOA accumulator slot
  -> EVU optional epilogue
  -> Tile DMA or MFE store to L2 output
  -> stream token / event signal
```

时序目标：

```text
T_dma_load_next <= T_boa_compute_current
T_evu_epilogue  <= T_store_or_next_load_overlap window
```

具体 latency target 由 PPA exploration 冻结。

### 5.4 Paged attention tile 数据流

```text
KV metadata stream
  -> MFE page walk / prefetch / reorder
  -> L1 MFE stream buffer
  -> BOA QK
  -> EVU scale/mask/softmax
  -> BOA PV
  -> MFE/DMA store output
```

Compute Tile 内部 owner：UCE 控制 program 和 launch；MFE 管理数据相关动态地址；USE 只处理需要状态更新的 metadata/counter/checkpoint；BOA/EVU 做 compute。

关键性能条件：

```text
T_prefetch <= T_qk
```

若条件不成立，PMU 应显示 MFE prefetch miss / stream backpressure 与 BOA operand stall 相关。

### 5.5 SSM / recurrence 数据流

```text
input projection -> BOA
local elementwise -> EVU
state load/update -> USE
checkpoint optional -> USE + DMA
output projection -> BOA
```

USE state cache 应避免与 BOA operand hot path 争 bank；state checkpoint 必须有 explicit event，不能隐式覆盖 persistent state slot。

### 5.6 CONV 数据流

```text
Conv = MFE WinGen + BOA GEMM
```

|           Conv 类型           | 预计效率                           |
| :---------------------------: | ---------------------------------- |
|           1x1 Conv            | 很高                               |
| 3x3 regular Conv, IC/OC 较大  | 高                                 |
|   stride 1, padding simple    | 高                                 |
|     batch 或 OH×OW 足够大     | 高                                 |
| OC 较大，可充分填满 BOA n 维  | 高                                 |
|        Depthwise Conv         | 低，K 太小，复用少，像 elementwise |
|    Group Conv groups 很多     | 低，GEMM 被切碎                    |
|      small channel Conv       | 低，BOA 填不满                     |
|    very small feature map     | 低，M 太小                         |
| dilation / irregular padding  | 低，WinGen 复杂，stream 不连续     |
| 过大的 im2col materialization | 低，L1/L2 带宽浪费                 |

数据流：

```text
Input L1 Slot
   |
   v
MFE Line Buffer / Window Generator
   |
   v
A Window Stream FIFO
   |
   v
BOA A Ping-Pong Buffer

Weight L1 Slot
   |
   v
BOA B Ping-Pong Buffer

BOA OPA Array
   |
   v
C Output Slot
```

性能判断公式：

```text
AI_effective = MACs / bytes_moved_by_MFE_and_BOA
```

初步计算策略：

```text
1x1 Conv:
  直接当 GEMM
  不需要复杂 WinGen

3x3 Regular Conv:
  MFE implicit window stream + BOA GEMM

5x5 Regular Conv:
  可支持，但要看带宽

Depthwise Conv:
  不走 BOA GEMM 主路径
  尽量fallback到 EVU / dedicated depthwise lane / vector reduce path

BOA Conv 主路径：
  1x1 / 3x3 / dense regular conv

EVU 或专用轻量路径：
  depthwise / small-channel / highly grouped conv / irregular conv
```

## 6. 配置、PPA、性能模型和 PMU

### 6.1 推荐配置参数

| 项                   | Architecture V1 示例     | First Silicon 建议      |
| -------------------- | ------------------------ | ----------------------- |
| L1 SRAM              | 1 MB 到 4 MB / Tile      | 由 SRAM profile 冻结    |
| SRAM banks           | 16 到 32                 | 由 SRAM profile 冻结    |
| EVU lanes            | 16 到 64                 | 由 PPA exploration 冻结 |
| BOA OPA 数           | 由性能目标决定           | 由 PPA exploration 冻结 |
| Tile DMA outstanding | 由后续规格冻结           | 由后续规格冻结          |
| Stream queue depth   | workload-dependent       | 由后续规格冻结          |
| UCE clock ratio      | 与 tile clock 同步或分频 | 由 PPA exploration 冻结 |

### 6.2 Tile roofline

Compute Tile 的一阶性能上限：

```text
Perf_tile = min(BOA_compute,
                EVU_compute,
                MFE_stream_bw,
                USE_state_bw,
                Tile_DMA_bw,
                L1_SRAM_bw,
                NoC_tile_port_bw)
```

SRAM 带宽需求：

```text
BW_sram_required =
  BW_boa_A_read
+ BW_boa_B_read
+ BW_boa_acc_read_write
+ BW_evu_lsu
+ BW_mfe_stream
+ BW_dma_load_store
+ BW_use_state
+ BW_uce_program_desc_event
```

约束：

```text
BW_sram_required <= BW_sram_peak * efficiency
BW_eff = BW_sram_peak * (1 - conflict_rate)
```

### 6.3 PMU counter

必需 counter：

| Counter                    | 归因对象 | 说明                                        |
| -------------------------- | -------- | ------------------------------------------- |
| `tile_active_cycles`       | tile     | PROGRAM_RUN 周期                            |
| `tile_idle_cycles`         | tile     | command queue empty 或 clock gated eligible |
| `uce_fetch_or_event_stall` | UCE      | program/descriptor/event SRAM stall         |
| `uce_desc_patch_stall`     | UCE      | patch unit 或 descriptor cache wait         |
| `dma_bytes_read/write`     | Tile DMA | L2/L1 bytes                                 |
| `dma_stall_cycles`         | Tile DMA | L2/NoC/SRAM backpressure                    |
| `boa_active_cycles`        | BOA      | dense compute active                        |
| `boa_operand_stall`        | BOA      | operand 或 stream 不足                      |
| `evu_active_cycles`        | EVU      | vector active                               |
| `evu_lsu_replay`           | EVU      | bank/index replay                           |
| `mfe_stream_stall`         | MFE port | stream buffer full/empty                    |
| `use_active_cycles`        | USE      | state compute active                        |
| `use_state_cache_hit/miss` | USE      | state locality                              |
| `stream_credit_empty/full` | Stream   | consumer/producer stall                     |
| `event_wait_cycles`        | Event    | wait event blocking                         |
| `sram_bank_conflict`       | SRAM     | bank conflict primary owner                 |
| `noc_vc_backpressure`      | Router   | VC-level pressure                           |

Primary stall hierarchy：

```text
engine_active
engine_wait_event
engine_wait_operand
stream_credit_empty_or_full
sram_bank_conflict
noc_backpressure
dma_wait_memory
uce_program_or_descriptor_stall
unknown_or_unclassified
```

每个 cycle 只能进入一个 primary counter；secondary tag 可记录 engine id、slot id、stream id。

## 7. RTL/软件实现建议

### 7.1 RTL 切分

建议 RTL 层级：

```text
elenor_compute_tile
├── tile_cmd_queue
├── tile_uce_core_or_sequencer
├── tile_desc_patch_unit
├── tile_event_unit
├── tile_dma_engine
├── tile_l1_sram_wrapper
├── tile_l1_arbiter
├── boa_cluster
├── evu_core
├── mfe_tile_port
├── use_state_engine
├── tile_pmu
├── tile_debug_csr
└── tile_router_port
```

关键接口采用 ready/valid + event id：

- UCE -> engine：`launch_valid`、`launch_ready`、`desc_ref`、`event_id`。
- engine -> event：`done_valid`、`error_valid`、`fault_code`、`event_id`。
- engine -> L1：banked SRAM request/response，带 owner id 供 PMU 归因。
- stream -> UCE：token valid、EOS、error、payload pointer、credit state。

### 7.2 Clock/reset

建议至少区分：

| 域              | 内容                        | 说明                                      |
| --------------- | --------------------------- | ----------------------------------------- |
| tile_core_clk   | UCE、event、descriptor、PMU | 可与 engine 同步，便于 First Silicon 验证 |
| tile_engine_clk | BOA、EVU、USE datapath      | 可后续按 PPA 分域                         |
| tile_sram_clk   | SRAM macro                  | 由 SRAM macro 要求决定                    |
| noc_clk         | router port                 | 需要 CDC FIFO 或同步假设                  |

First Silicon 可采用单 tile clock 降低 CDC/RDC 风险；多 clock 优化由 PPA exploration 冻结。

Reset 策略：

- `hard_reset` 清除本 tile 可见的 program handle tag、descriptor cache、event、PMU running state。
- `tile_soft_reset` 停止新 launch，drain 可安全完成的 DMA/engine，pending stream token 按 reset policy 回收，并使旧 `program_epoch` 失效。
- `engine_reset` 仅复位指定 engine，必须让 UCE/event unit 看到 deterministic completion 或 error。
- `state_preserve_reset` 可保留 USE protected state cache，但必须标记 dirty/valid，行为由后续规格冻结。

### 7.3 Debug/exception

Debug 必须支持：

- halt tile after current instruction 或 after current engine launch boundary。
- read Tile PC、active descriptor id、frame id、slot table、outstanding event bitmap。
- PMU snapshot。
- fault record readout。

Exception 分类：

| fault              | 触发                                    | 行为                                   |
| ------------------ | --------------------------------------- | -------------------------------------- |
| invalid descriptor | version/size/slot/flags 不合法          | stop affected task，signal event error |
| slot permission    | 写只读 slot、role mismatch              | capture slot id 和 descriptor id       |
| address/range      | L1/L2 越界或 IOMMU fault                | capture address syndrome               |
| timeout            | wait/event/engine 超过 timeout          | drain 或 reset tile                    |
| stream protocol    | credit leak、unexpected EOS/error token | propagate error token                  |
| engine internal    | BOA/EVU/MFE/USE fault                   | capture engine id                      |
| SRAM ECC optional  | ECC uncorrectable                       | escalate reset domain                  |

### 7.4 软件栈建议

- compiler 生成 tile kernel library selection、descriptor template、slot frame 和 command buffer。
- runtime/firmware 负责 package load、context-level patch、program section registry、descriptor cache invalidate 和 residency hint。
- Tile UCE firmware 或 microcode 只消费 prepared local program handle，不调用 host service，也不在运行态发起 global program load。
- golden trace 应记录 command id、region id、tile id、program id、descriptor ids、event transitions、stream token sequence 和 PMU snapshot。

## 8. 验证、bring-up 和验收标准

### 8.1 单元验证

| 单元               | 必测内容                                                                          |
| ------------------ | --------------------------------------------------------------------------------- |
| Tile Command Queue | overflow/underflow、context isolation、reset drain                                |
| UCE front-end      | prepared task check、program handle/epoch、branch、wait、fence、trap              |
| Descriptor Patch   | tile_id/group_id/slot offset、range check、coherence invalidate、desc window 边界 |
| Slot Frame         | permission、alignment、owner handoff、bank hint                                   |
| Tile DMA           | 1D/2D/strided、async event、timeout、range fault                                  |
| Local Event        | done/error/timeout/reset 状态转换                                                 |
| Stream             | credit、backpressure、EOS、error token、reset/drain                               |
| L1 Arbiter         | bank conflict、priority starvation、PMU attribution                               |
| PMU                | primary owner 唯一性、counter snapshot 一致性                                     |

### 8.2 集成 bring-up 顺序

1. Tile command -> event done 空 kernel。
2. Tile DMA 1D/2D copy -> event -> compare。
3. Prepared task handle + frame generation smoke。
4. BOA GEMM through UCE launch，不允许 testbench 直接驱动 datapath。
5. EVU elementwise/mask/tail through UCE launch。
6. Stream pop/push/EOS/error token 与 Region Sequencer 连接。
7. MFE Page Stream token 到 L1 buffer，再 BOA/EVU 消费。
8. USE scan/recurrence + checkpoint/restore。
9. Paged attention tile trace，验证 `T_prefetch <= T_qk` case 的 PMU 指纹。
10. fault injection：invalid/stale program handle、invalid descriptor、slot fault、timeout、stream error、engine fault。

### 8.3 验收标准

- 所有 engine 必须通过 command/event 路径触发，而不是旁路 testbench。
- 每个 program handle 或 descriptor fault 必须能定位 command id、program id、tile id、local slot/epoch、descriptor id、slot id 或 address syndrome。
- reset/drain 后 stream credit、event pending、DMA outstanding 和 engine busy 进入确定状态。
- BOA/EVU/MFE/USE/UCE/SRAM/NoC stall 能通过 PMU primary owner 解释。
- Python golden 或 workload trace 能复现 dense GEMM、softmax/norm、paged attention tile path、USE recurrence 四类路径。
- First Silicon 验收至少覆盖 Phase 1 control plane + BOA runtime skeleton；Phase 5 前不得宣称 USE state path 完整。

## 9. 风险、取舍和后续细化方向

### 9.1 风险

| 风险                         | 影响                                    | 缓解                                                                                  |
| ---------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| UCE/USE 共享实现导致职责混淆 | program control 与 state compute 难验证 | 文档、RTL 接口、PMU、fault owner 分离；共享 fetch/debug/CSR，不共享 ownership         |
| L1 SRAM 争用                 | BOA/EVU/MFE/USE 互相阻塞                | bank-aware layout、protected state region、PMU primary stall、compiler memory planner |
| descriptor patch 一致性复杂  | warm launch 读到 stale descriptor       | descriptor version/tag、running text 不可 patch、显式 invalidate                      |
| stream deadlock              | pipeline 停滞且难 debug                 | credit leak detection、timeout、EOS/error 形式化、cycle wait graph 检查               |
| MFE 与 UCE 数据访问边界不清  | page walk 被错误放入控制面              | 数据相关动态地址归 MFE，program control 归 UCE，state metadata 更新归 USE             |
| reset 粒度不清               | fault recovery 污染其他 context         | tile/group/device reset domain 明确，fault record 记录 context id                     |
| PMU 重复计数                 | 性能分析失真                            | primary owner 唯一规则，secondary tag 不进入 utilization 汇总                         |

### 9.2 取舍

- 选择 Tile-SPMD template，而不是 per-tile program：降低 program cache 压力和 compiler 复杂度。
- 选择 slot frame，而不是固定物理地址：支持 dynamic shape、paged attention、workspace 变化和 descriptor auto-patch。
- 选择 descriptor-driven engines，而不是把 BOA/EVU/MFE/USE 暴露成复杂通用指令流：降低验证面。
- First Silicon 优先 command/event/DMA/PMU/BOA，再扩 EVU/MFE/USE：系统稳定性优先于功能堆叠。

### 9.3 后续细化方向

以下值不在本文冻结：

- Tile UCE 指令编码、register file 大小和 trap ABI：由后续规格冻结。
- L1 SRAM 容量、bank 数、端口、宏类型、ECC 策略：由 SRAM profile 冻结。
- BOA OPA shape、EVU lane 数、Tile DMA outstanding、MFE stream buffer 深度：由 PPA exploration 冻结。
- Debug CSR 地址、PMU counter id、fault code 编码：由后续规格冻结。
- reset/drain 对 resident state 的保留策略：由后续规格冻结。
