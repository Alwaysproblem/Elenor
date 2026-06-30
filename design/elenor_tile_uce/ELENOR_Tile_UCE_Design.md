# ELENOR Tile UCE 设计文档

## 1. 定位、目标和 First Silicon cutline

Tile UCE 是每个 Compute Tile 内的 program control 和 engine orchestration 功能组件。它执行 Tile Program，管理 Tile PC、branch、wait/fence、descriptor template patch、Tile DMA、BOA/EVU/MFE/USE launch、stream token 和 tile done event。

Tile UCE 不解释高层 graph，不做全局调度，不承担 state compute，也不负责 page/segment 数据相关动态内存访问。高层 graph 已由 compiler/runtime 降低为 command、TileGroupTask、Tile Program、slot frame、descriptor 和 stream queue contract。UCE 的边界是 tile-local kernel pipeline。

核心决策：Tile UCE 和 USE 可以共享同一个 tile-local RISC-V / micro-controller 或等价 micro-sequencer 实现，但在架构上必须保持两个独立功能组件：

```text
Tile UCE = Tile Program control + engine launch + stream/event/descriptor control
USE      = state compute + state cache/register + scan/recurrence/checkpoint lifecycle
MFE      = page/segment dynamic data movement + stream fill/reorder/prefetch
```

Architecture V1 目标：

- 支持 Tile-SPMD program template，所有 tile 运行同一类 program，通过 tile id、group id、slot frame 和 descriptor patch 区分工作。
- 提供小而稳定的控制指令集合：control、engine launch、sync、stream、descriptor、profiling/error。
- 与 Slot Frame、Stream Queue、Event Model、PMU 和 fault record 形成二进制可验证 contract。
- 允许软件工具链使用 RISC-V firmware/custom instruction、ROM bytecode 或硬件 sequencer 三种实现路径，但外部语义一致。

First Silicon V1 cutline：

| 类别            | 必须实现                                                     | 后续可扩展                                         |
| --------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| Program control | fetch/decode、PC、branch、loop、ret/end、trap                | compressed encoding、call stack、复杂 predication  |
| Engine launch   | launch.dma、launch.boa、launch.evu、launch.mfe、launch.use   | priority launch、speculative prelaunch             |
| Sync            | wait、waitall、fence、timeout                                | event set algebra、advanced dependency compression |
| Issue window    | `window_size=1`，单 active Tile Program，event sequence 匹配 | V1.x/V2 可探索 `window_size=2~4` sliding window    |
| Stream          | pop、push、acquire、release、EOS/error branch                | multi-consumer refcount acceleration               |
| Descriptor      | load/validate/patch、slot offset、tile/group id patch        | richer relocation、descriptor cache hierarchy      |
| Debug/fault     | PC capture、fault record、halt at boundary                   | sampled trace、single-step across engine internals |
| PMU             | wait/branch/patch/fetch stall、launch count、stream wait     | feedback scheduling hooks                          |

未冻结的指令编码、CSR 地址、寄存器数量、cache 容量和 pipeline 深度均写作 `由后续规格冻结` 或 `由 PPA exploration 冻结`。

## 2. 职责、非职责和 ownership

### 2.1 职责

Tile UCE 的职责：

1. 维护 Tile Program PC 和 resident local program handle 对应的执行状态。
2. 接收 prepared tile task，校验 `program_local_slot/program_version/program_epoch`、frame generation 和 descriptor window。
3. 执行 descriptor template auto-patch：tile_id、group_id、slot base、slot offset、shape variant、event id。
4. 发起 L2<->L1 Tile DMA 和 MFE stream/dataflow task。
5. 启动 BOA、EVU、MFE、USE engine task，并分配或引用 completion event。
6. 执行 wait、waitall、fence、timeout 和 local barrier assist。
7. 处理 Stream Queue token：pop、push、acquire、release、EOS、error propagation。
8. 管理 outstanding event bitmap、event sequence 和 engine busy 状态。
9. 执行 issue-window admission 和 slot hazard 检查；First Silicon 固定 `window_size=1`。
10. 在 fault、timeout、reset/drain 时进入确定状态并上报 fault record。
11. 向 PMU 提供 UCE fetch、decode、branch、wait、descriptor patch、stream stall 和 window admission stall 归因信号。

### 2.2 非职责

Tile UCE 不负责：

- 高层 graph schedule、operator fusion 决策、tensor algebra 解释。
- Global command queue consume、multi-model scheduling、QoS policy。
- Tile-local multithreading、preemption 或多 context time slicing；sliding window 只表示 issue lookahead / preparation depth，不改变单 Tile Program 语义。
- USE 的 state update 算术、prefix scan、recurrence、checkpoint content 管理。
- MFE 的 page table walk、segment offset decode、physical address generation、prefetch/reorder。
- BOA/EVU 内部 micro-loop、operand scheduling、vector lane scheduling。
- HBM/DDR/LPDDR 全局访问策略。
- 通用 CPU 任务、RTOS 或 driver 逻辑。

### 2.3 ownership matrix

| 对象                | owner                                                                     | UCE 权限                     | 说明                                                                         |
| ------------------- | ------------------------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| Tile PC             | UCE                                                                       | read/write                   | trap/debug 可读，正常运行由 UCE 独占                                         |
| Tile Program text   | Tile Group Program Residency Manager / Runtime backing store，UCE execute | fetch only                   | running 状态不可 patch；UCE 只接受 local resident handle                     |
| Descriptor template | Compiler/Runtime                                                          | patch selected fields        | patch 失败产生 invalid descriptor fault                                      |
| Slot frame          | Runtime/Firmware bind，UCE enforce                                        | read/validate                | 权限、alignment、owner 校验由 UCE/descriptor unit 执行                       |
| Stream token        | Stream Queue Engine                                                       | protocol access              | UCE 不直接修改 credit counter                                                |
| Engine event        | Local Event Unit                                                          | allocate/wait/signal request | event 状态由 event unit 写入                                                 |
| Issue window entry  | UCE                                                                       | allocate/check/release       | V1 固定 1 entry；V1.x/V2 才允许多 entry，必须携带 frame/slot hazard metadata |
| USE state slot      | USE                                                                       | launch/control only          | state lifecycle 归 USE                                                       |
| MFE metadata stream | MFE                                                                       | launch/wait/read token       | 数据相关动态地址归 MFE                                                       |
| PMU primary stall   | PMU                                                                       | source signal                | UCE 不能重复归因 engine stall                                                |

## 3. 微架构和状态机

### 3.1 UCE 微架构

```text
+---------------------------------------------------------------+
| Tile UCE                                                      |
|                                                               |
| +-------------+   +--------------+   +----------------------+ |
| | Task Ingress|-->| Handle Check |-->| Decode / Issue       | |
| +-------------+   +--------------+   +----------+-----------+ |
|                                             |                 |
|       +-----------------+-------------------+----------------+|
|       |                 |                   |                 |
|       v                 v                   v                 v
| +------------+   +--------------+   +--------------+   +-------------+
| | Branch/PC  |   | Local Fetch  |   | Wait/Fence   |   | Stream Unit |
| +------------+   +------+-------+   +------+-------+   +------+------+
|                         |                  |                  |
|                         v                  v                  v
|                  +-------------+    +-------------+    +-------------+
|                  | Desc Patch  |    | Event Unit  |    | PMU/Fault   |
|                  +------+------+    +------+------+    +------+------+
|                         |                  |                  |
|                         v                  v                  v
|                   BOA/EVU/MFE/USE/DMA  local events       fault record
+---------------------------------------------------------------+
```

实现可以是：

- 小 RISC-V core + custom instruction / MMIO launch queue。
- 极简硬件 sequencer，直接 decode Tile Program bytecode。
- 架构仿真阶段的 ROM bytecode interpreter。

无论实现形态如何，对外语义必须一致：同一 Tile Program 在相同 descriptor、slot frame、stream token 和 event 初始状态下产生相同 engine launch 序列、event 序列和 fault 行为。

### 3.2 UCE task 状态机

```text
RESET
  -> IDLE
  -> ACCEPT_TASK
  -> CHECK_PROGRAM_HANDLE
  -> CHECK_FRAME_GENERATION
  -> CHECK_DESC_WINDOW
  -> BIND_FRAME
  -> FETCH
  -> DECODE
  -> ISSUE
  -> WAIT_OR_NEXT
  -> COMPLETE
  -> IDLE
```

异常路径：

```text
任意 active 状态
  -> TRAP_CAPTURE
  -> SIGNAL_ERROR_EVENT
  -> DRAIN_OUTSTANDING
  -> IDLE 或 RESET
```

状态定义：

| 状态                   | 行为                                                                                            | 退出                      |
| ---------------------- | ----------------------------------------------------------------------------------------------- | ------------------------- |
| RESET                  | 清 PC、valid、outstanding bitmap、local trap；可选择保留 local resident tag，策略由后续规格冻结 | reset deassert            |
| IDLE                   | 等待 tile task，允许 clock gating                                                               | task valid                |
| ACCEPT_TASK            | latch context/program/frame/stream/event/timeout                                                | command header valid      |
| CHECK_PROGRAM_HANDLE   | 校验 local slot valid、`program_id/program_version/program_epoch` 匹配；失败则 trap             | handle valid              |
| CHECK_FRAME_GENERATION | 校验 frame generation 和 owner，避免 reset/drain 后 stale frame                                 | frame current             |
| CHECK_DESC_WINDOW      | 校验 descriptor window bounds 和 patch/fetch visibility                                         | desc window valid         |
| BIND_FRAME             | 校验 slot frame 权限、对齐、bank hint、owner                                                    | frame valid               |
| FETCH                  | 仅从 local program slot 或 I-cache 取指                                                         | instruction valid         |
| DECODE                 | 解析 opcode、operand、descriptor ref、event ref                                                 | legal instruction         |
| ISSUE                  | 发起 branch/launch/wait/stream/patch 等动作                                                     | issue accepted            |
| WAIT_OR_NEXT           | 若指令需要等待则监听 event/stream；否则 PC advance                                              | wait satisfied 或 timeout |
| COMPLETE               | 写 tile done event 和 PMU snapshot                                                              | event accepted            |
| TRAP_CAPTURE           | 捕获 PC、opcode、desc id、slot id、engine id、fault code                                        | fault record ready        |
| DRAIN_OUTSTANDING      | 停止新 issue，等待或取消 outstanding engine/token                                               | drain complete 或 reset   |

### 3.3 Instruction issue pipeline

推荐简化 pipeline：

```text
IF -> ID -> EX(control/patch/launch/stream/wait) -> COMMIT
```

- IF 可被 program SRAM stall 阻塞，计入 `uce_fetch_or_event_stall`。
- ID 做 opcode legality、privilege/context 检查。
- EX 对 launch/stream/patch 走不同 functional unit。
- COMMIT 更新 PC、outstanding bitmap、event wait state 和 PMU retired counter。

First Silicon 可做单 issue in-order；不需要乱序、speculation 或多发射。RISC-V 实现时，Tile Program 可以是 firmware 函数加 custom instruction；hardware sequencer 实现时，Tile Program 可以是固定宽度 bytecode。编码由后续规格冻结。

### 3.4 Outstanding event scoreboard

UCE 持有 tile-local outstanding scoreboard：

| 字段        | 含义                                 |
| ----------- | ------------------------------------ |
| `event_id`  | local/global event table index       |
| `sequence`  | expected event sequence / generation |
| `window_id` | issue window entry id；V1 固定为 0   |
| `producer`  | DMA/BOA/EVU/MFE/USE/Stream/Event     |
| `state`     | pending/done/error/timeout/reset     |
| `desc_ref`  | descriptor id 或 table slot          |
| `slot_mask` | 该 task 可能读写的 slot              |
| `timeout`   | local deadline                       |

Scoreboard 行数由后续规格冻结。资源不足、`event_id + sequence` 不匹配或 slot hazard 未解除时，`launch.*` / window admission 必须 stall 或 trap，不能静默覆盖 outstanding event。

### 3.5 UCE Sliding Window issue model

`UCE window size` 定义为单个 tile 上允许同时处于 active / in-flight 状态的 Tile Program 数量上限。它控制的是 **issue lookahead depth** 和 preparation overlap，不是 multithreading、乱序提交或新的 runtime scheduling 层。

First Silicon V1 冻结为：

```text
uce_window_size = 1
```

含义是：一个 prepared tile task 进入 `ACCEPT_TASK -> ... -> COMPLETE -> IDLE` 后，下一个 Tile Program 才能进入 active window。这样保持当前 prepared task、frame binding 和 descriptor window contract 不变，降低 event reuse、slot lifetime 和 reset/drain 的验证面。

V1.x / V2 可以在保持 Tile Program ISA 和 public IR 不变的前提下探索 `window_size=2~4`：

```text
Window=1:
  P0 complete -> P1 start -> P2 start

Window=2:
  P0 compute/store in-flight
  P1 load/patch/compute may issue only if slot hazard table proves independence

Window=4:
  P0/P1/P2/P3 may occupy different load/compute/store/wait stages,
  but commit/fault visibility still follows event + sequence rules.
```

关键规则：

1. window size 不保证 correctness；correctness 来自 `event_id + sequence`、dependency scoreboard、Slot Frame owner/lifetime 和 buffer alias 检查。
2. UCE front-end 仍可保持 single-issue in-order；window 只允许多个 Tile Program context 挂在 scoreboard 中，不表示多发射或 per-thread PC。
3. `window_id` 只是 UCE 内部 admission/hazard tag；不得暴露为 compiler public IR 或改变 Tile-SPMD template。
4. V1.x 若开启 `window_size>1`，每个 window entry 必须记录：prepared task id、frame id/generation、descriptor window、active event sequence、read/write slot mask、buffer owner state、timeout epoch。
5. `P1 load` 可以早于 `P0 store` 完成发出，前提是 P1 不访问 P0 active buffer，P0 store visibility event 尚未被误认为 DONE，并且 MFE async LD/ST queue 有 credit。
6. 所有 writable slot alias、write-after-read、read-after-write、write-after-write hazard 必须 stall admission；除只读 const alias 外不能靠 software convention 放行。
7. SRAM 侧不引入 V1 硬件动态 allocator；compiler/runtime 通过 Slot Frame 预留 ping-pong / multi-buffer slot set，UCE 只做 owner/lifetime reuse 检查。
8. reset/drain 必须按 window entry 清理 outstanding event、stream token、slot owner 和 descriptor generation；旧 completion 若 sequence 不匹配必须被丢弃或转 fault。

建议 profile：

| Profile       | `uce_window_size`    | 说明                                                                 |
| ------------- | -------------------- | -------------------------------------------------------------------- |
| V1 bring-up   | 1                    | 单 active program，验证当前 command/event/slot contract              |
| V1.x balanced | 2                    | load/compute/store 基本 overlap，依赖 double buffering               |
| V2 high-util  | 4                    | 更高 MFE/BOA/EVU overlap，需要更强 scoreboard、PMU 和 SRAM bank plan |
| 8+            | 不建议 First Silicon | SRAM 压力、tag CAM、reset/drain 和 formal 面积显著上升               |

## 4. 接口、descriptor、寄存器和协议

### 4.1 Tile Program 语义对象

Tile Program 只引用架构对象：

- scalar register：保存 loop counter、tile id、descriptor id、event id、token id。
- descriptor reference：指向 descriptor table 或 L1 descriptor slot。
- slot reference：指向 current frame 的 slot。
- event reference：local event id + expected sequence 或 event table entry。
- stream reference：Tile Group Sequencer 初始化的 stream queue id。
- prepared program handle：`program_local_slot / program_local_offset / program_bytes / program_epoch`。

禁止在 Tile Program 中硬编码长期稳定物理地址；使用 slot frame 和 descriptor patch 计算 effective address。Tile UCE 不得在 task 执行路径解引用 global `program_iova`；global backing store 只由 Tile Group Sequencer / Program Residency Manager 使用。

### 4.2 指令分类和示例

```text
Control:
  nop
  mov
  add
  cmp
  br
  br.eq
  br.ne
  br.eos
  ret

Engine Launch:
  launch.dma
  launch.boa
  launch.evu
  launch.mfe
  launch.use

Sync:
  wait
  waitall
  fence

Stream:
  stream.pop
  stream.push
  stream.acquire
  stream.release
  stream.eos
  stream.err

Descriptor:
  patch.desc
  load.desc
  validate.desc

Profiling/Error:
  prof.begin
  prof.end
  trap
```

指令编码、立即数字段宽度和寄存器数量由后续规格冻结。

### 4.3 Tile Program 示例：matmul + epilogue

示例中的 `e0`、`e1`、`e2` 和 `e_*` 是符号化 completion handle；它们逻辑上绑定 launch 产生的 `(event_id, event_sequence)`，`wait` / `waitall` 比较的是这对 identity，而不是裸 `event_id`。

```asm
tile_matmul_relu:
    prof.begin      PMU_GROUP_KERNEL

    patch.desc      d_load_a, slot[A], tile_id, group_id
    patch.desc      d_load_b, slot[B], tile_id, group_id
    launch.mfe      d_load_a -> e0
    launch.mfe      d_load_b -> e1
    waitall         e0 | e1

    patch.desc      d_boa, slot[A], slot[B], slot[C]
    launch.boa      d_boa -> e2
    wait            e2

    patch.desc      d_relu, slot[C], slot[OUT]
    launch.evu      d_relu -> e3
    wait            e3

    patch.desc      d_store, slot[OUT], tile_id, group_id
    launch.mfe      d_store -> e4
    wait            e4

    prof.end        PMU_GROUP_KERNEL
    ret
```

### 4.4 Tile Program 示例：stream pipeline

```asm
stage_loop:
    stream.pop      r_in, S_IN
    br.eos          r_in, done
    br.err          r_in, stream_fault

    stream.acquire  r_out, S_OUT

    patch.desc      d_load, token.payload(r_in), slot[INPUT]
    launch.mfe      d_load -> e_load
    wait            e_load

    launch.boa      d_compute -> e_compute
    wait            e_compute

    patch.desc      d_store, slot[OUTPUT], token.payload(r_out)
    launch.mfe      d_store -> e_store
    wait            e_store

    stream.push     S_OUT, r_out
    stream.release  S_IN, r_in
    br              stage_loop

done:
    stream.eos      S_OUT
    signal.tile.done
    ret

stream_fault:
    stream.err      S_OUT, fault.from_token(r_in)
    trap            FAULT_STREAM_TOKEN
```

### 4.5 Descriptor patch descriptor

```c
typedef enum {
    ELENOR_PATCH_TILE_ID       = 1 << 0,
    ELENOR_PATCH_GROUP_ID      = 1 << 1,
    ELENOR_PATCH_SLOT_BASE     = 1 << 2,
    ELENOR_PATCH_SLOT_SIZE     = 1 << 3,
    ELENOR_PATCH_EVENT_ID      = 1 << 4,
    ELENOR_PATCH_STREAM_TOKEN  = 1 << 5,
    ELENOR_PATCH_STATE_SLOT    = 1 << 6,
} elenor_patch_kind_v0_t;

typedef struct {
    uint16_t abi_version;
    uint16_t patch_size;
    uint16_t patch_kind;
    uint16_t target_desc_slot;

    uint32_t target_byte_offset;
    uint32_t source_ref;
    uint32_t scale;
    uint32_t addend;

    uint32_t bounds_min;
    uint32_t bounds_max;
    uint32_t flags;
} elenor_desc_patch_v0_t;
```

Patch unit 必须：

1. 检查 target offset 在 descriptor size 内。
2. 检查 source slot 权限与 alignment。
3. 对 `base + tile_id * stride + group_id * stride + addend` 做溢出检查。
4. 对 state slot patch 保留 USE ownership；UCE 只写控制 descriptor，不直接改 state data。
5. patch 失败时产生 invalid descriptor fault，并关联 command id、program id、tile id 和 descriptor id。

### 4.6 Engine launch 接口

```c
typedef struct {
    uint16_t engine_kind;
    uint16_t flags;
    uint32_t desc_slot;
    uint32_t event_id;
    uint32_t event_sequence;
    uint32_t context_id;
    uint32_t timeout_cycles;
} elenor_engine_launch_req_v0_t;
```

握手：

```text
UCE launch_valid + launch_req
  -> engine launch_ready
  -> event state = PENDING
  -> engine runs descriptor
  -> engine done/error
  -> Local Event Unit writes DONE/ERROR/TIMEOUT
  -> UCE wait observes event
```

Launch accepted 后，engine 拥有 descriptor snapshot 或 descriptor cache line；descriptor coherence 规则由后续规格冻结。最小实现可以要求 launch 前 descriptor patch 已完成且 launch 后该 descriptor 不再修改。UCE 和 engine/event unit 必须把 `event_id + event_sequence` 一起作为 completion identity，避免 reset/reuse 后 stale completion 被误判为当前 event。

### 4.7 CSR / debug 寄存器

| CSR                    | 语义                                                    |
| ---------------------- | ------------------------------------------------------- |
| `uce_status`           | idle/fetch/decode/wait/drain/fault                      |
| `uce_pc`               | current PC                                              |
| `uce_trap_pc`          | trap PC                                                 |
| `uce_trap_cause`       | fault code                                              |
| `uce_active_desc`      | 当前 descriptor slot                                    |
| `uce_active_event`     | 当前 wait 或 launch event                               |
| `uce_outstanding_mask` | outstanding event bitmap                                |
| `uce_window_cfg`       | configured issue window size；V1 hardwired 1            |
| `uce_window_status`    | active entries、admission stall cause、hazard slot mask |
| `uce_stream_status`    | 当前 stream id、token id、EOS/error flags               |
| `uce_pmu_ctrl`         | PMU select/snapshot                                     |
| `uce_debug_ctrl`       | halt/resume/single boundary-step                        |

寄存器地址和 bit field 由后续规格冻结。

### 4.8 Event / Stream / Slot 协议要求

Event：

- `wait` 只能等待 PENDING/DONE/ERROR/TIMEOUT/RESET 明确定义的 event。
- timeout 必须写 fault record，不得永久 hang。
- `fence` 确保前序 descriptor patch、L1 write、engine completion 对后序 launch 可见；精确 memory ordering 由后续规格冻结。

Stream：

- `stream.pop` 消耗 token 可见性，但 payload owner 直到 `stream.release` 才归还。
- `stream.acquire` 获得 output token/credit，`stream.push` 后 consumer 可见。
- EOS 必须可区分 producer；多 producer policy 由 stream descriptor 声明。
- error token 必须携带 fault record index 并沿 pipeline 传播。

Slot：

- UCE 在 launch 前校验 slot permission，但 engine 内部仍应防御性检查 role/size。
- slot owner handoff 通过 event 或 explicit barrier。
- accumulator slot 必须单独标记，避免 DMA/EVU 错写。
- 若 V1.x 开启 `window_size>1`，UCE 必须在 admission 前检查 active window 的 read/write slot mask：read-only const 可 alias；任何 writable alias、WAR/RAW/WAW hazard 均 stall 或 fault。
- slot owner handoff 的可见性由 event/fence/stream release 共同确定；window entry 不得仅凭 window slot 空闲释放 L1 buffer。

## 5. 数据流、控制流和时序路径

### 5.1 控制流层级

````text
Graph Schedule PC
  -> Group Task Iterator
  -> Tile Group Sequencer action index
  -> Tile UCE PC
  -> Engine task descriptor
  -> Engine internal micro-loop

UCE 只推进第四层，不越权修改 Tile Group Sequencer action index 或 engine micro-loop。

### 5.2 UCE launch 时序

```text
cycle N:     decode launch.boa d_boa -> e2
cycle N+1:   descriptor patch/fence check
cycle N+2:   launch_valid to BOA
cycle N+k:   BOA accepts, event[e2]=PENDING
cycle M:     BOA done, event[e2]=DONE
cycle M+1:   wait e2 retires, PC advances
````

具体 cycle 数由 PPA exploration 冻结；架构要求是 event ordering deterministic。

### 5.3 Sliding Window issue 时序

V1 时序等价于严格串行：

```text
P0: load -> compute -> store -> release -> complete
P1:                                      load -> compute -> store
```

V1.x 若 `window_size=2`，只允许在 admission check 通过后重叠：

```text
cycle A:   P0 store issued, event[e_store, seq0]=PENDING
cycle A+1: UCE checks P1 read/write slot mask and buffer owner table
cycle A+2: P1 load issued if no alias with P0 active output/store buffer
cycle B:   P0 store visibility event DONE(seq0)
cycle C:   P0 buffer release retires; slot owner becomes reusable
```

架构不要求 UCE out-of-order commit。即使多个 window entry 同时 active，fault、DONE、RESET 的可见性也必须由 event sequence 和 dependency scoreboard 决定。

### 5.4 Wait/fence ordering

- `wait e`：当前 PC 阻塞到 handle `e` 对应的 `event_id + event_sequence` 为 DONE；ERROR/TIMEOUT/RESET 进入 trap。
- `waitall mask`：所有 handle 对应的 completion identity 都 done 才继续；任一 error 进入 trap。
- `fence desc`：保证 descriptor patch 对后续 engine launch 可见。
- `fence mem`：保证前序 tile-local writes 对后序 engine reads 可见。
- `fence stream`：保证 token payload 已写入后再 push。

### 5.5 Descriptor warm patch 时序

```text
Runtime patches context-level descriptor in memory
  -> descriptor cache invalidate command or version bump
Tile Group Sequencer dispatches prepared tile task
  -> UCE binds frame
  -> UCE patch tile/group/slot fields
  -> UCE validates descriptor
  -> engine launch
```

风险点：runtime patch 与 UCE descriptor cache stale。最小规则：descriptor cache line 带 version；warm launch 必须显式 invalidate 或更新 version。

### 5.6 Fault path

```text
fault detected
  -> stop new issue
  -> capture PC/opcode/desc/slot/event/engine
  -> write fault record or local syndrome
  -> signal event ERROR
  -> propagate stream error token if in stream stage
  -> drain outstanding safe operations
  -> return to IDLE or wait reset
```

Fault path 必须可重放：同一 illegal descriptor 在同一状态产生同一 fault code 和 syndrome 字段。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置项

| 配置                      | 说明                                                            | 冻结方式                             |
| ------------------------- | --------------------------------------------------------------- | ------------------------------------ |
| UCE 实现形态              | RISC-V、micro-sequencer、ROM interpreter                        | 由 PPA exploration 冻结              |
| Program SRAM/I-cache      | Tile Program resident capacity                                  | 由 SRAM profile 冻结                 |
| Descriptor cache lines    | hot descriptor 数                                               | 由 SRAM profile 冻结                 |
| Register file size        | Tile Program scalar/temp                                        | 由后续规格冻结                       |
| Outstanding event entries | 同时 in-flight engine 数                                        | 由后续规格冻结                       |
| UCE issue window size     | 同时 active 的 Tile Program 数；V1 固定为 1，V1.x 可探索 2 到 4 | V1 固定；后续由 PPA exploration 冻结 |
| Stream token registers    | 同时持有 token 数                                               | 由后续规格冻结                       |
| Watchdog width            | timeout counter 宽度                                            | 由后续规格冻结                       |

### 6.2 UCE 性能模型

UCE 不是高吞吐 datapath；它的目标是隐藏 control overhead，不让 BOA/EVU/MFE/USE 因控制路径饿死。

关键指标：

```text
T_control_per_tile_block =
  T_fetch_decode
+ T_desc_patch
+ T_launch
+ T_wait_bookkeeping
+ T_stream_token
```

约束：

```text
T_control_per_tile_block << max(T_boa_compute, T_evu_compute, T_mfe_stream, T_use_state)
```

若该约束不成立，PMU 应显示 `uce_desc_patch_stall`、`uce_fetch_or_event_stall` 或 `stream_credit_empty/full` 成为 primary stall。

开启 sliding window 后，性能模型增加 latency hiding 项：

```text
T_effective_tile =
  T_load + T_compute + T_store
- T_overlap(window_size, independent_slot_sets, mfe_async_credit)
```

`T_overlap` 只有在 Slot Frame 预留独立 ping-pong / multi-buffer slot、MFE LD/ST queue 有 credit、event sequence 可区分复用轮次时才计入。若 window admission stall 高于 compute stall，继续增大 window 只会增加 SRAM pressure 和验证复杂度。

### 6.3 PMU counter

| Counter                          | 说明                                                     |
| -------------------------------- | -------------------------------------------------------- |
| `uce_active_cycles`              | UCE 非 idle 周期                                         |
| `uce_retired_insts`              | retired Tile Program 指令数                              |
| `uce_branch_taken`               | taken branch 数                                          |
| `uce_launch_dma/boa/evu/mfe/use` | 各 engine launch 数                                      |
| `uce_wait_cycles`                | wait/waitall 阻塞周期                                    |
| `uce_wait_error`                 | wait 观察到 error/timeout/reset 次数                     |
| `uce_desc_patch_count`           | patch 指令数                                             |
| `uce_desc_patch_stall`           | patch unit/cache/range check stall                       |
| `uce_stream_pop_empty`           | pop 等待 input token                                     |
| `uce_stream_push_full`           | push/acquire 等待 output credit                          |
| `uce_fetch_stall`                | program fetch stall                                      |
| `uce_trap_count`                 | trap 次数，按 cause 可分组                               |
| `uce_timeout_count`              | watchdog timeout 次数                                    |
| `uce_window_active_high`         | active window entries high-watermark；V1 应为 1          |
| `uce_window_admission_stall`     | window full、event entry full 或 hazard table 阻塞       |
| `uce_slot_hazard_stall`          | slot alias / WAR / RAW / WAW hazard 导致 admission stall |
| `uce_store_visibility_wait`      | 等待 store visibility event 或 buffer release 的周期     |

PMU primary stall owner 规则：

- UCE 正在等待 engine event：归 `engine_wait_event`，并带 UCE wait secondary tag。
- UCE 因 descriptor SRAM 或 program SRAM 阻塞：归 `uce_program_or_descriptor_stall`。
- UCE 因 stream credit 阻塞：归 `stream_credit_empty_or_full`。
- UCE 因 SRAM bank conflict 不能 fetch/patch：归 `sram_bank_conflict`，secondary tag 为 UCE。

## 7. RTL/软件实现建议

### 7.1 RTL 模块拆分

```text
tile_uce
├── uce_task_ingress
├── uce_program_fetch
├── uce_decode
├── uce_pc_branch
├── uce_launch_unit
├── uce_wait_unit
├── uce_stream_unit
├── uce_desc_patch_unit
├── uce_scoreboard
├── uce_issue_window_ctrl
├── uce_buffer_hazard_table
├── uce_trap_fault
├── uce_debug_csr
└── uce_pmu_source
```

模块接口：

- `task_ingress` 与 Tile Command Queue ready/valid。
- `program_fetch` 与 L1 program SRAM 或 I-cache。
- `desc_patch_unit` 与 descriptor SRAM、slot frame table、event allocator。
- `launch_unit` 与 DMA/BOA/EVU/MFE/USE launch ports。
- `wait_unit` 与 Local Event Unit。
- `stream_unit` 与 Stream Queue Engine tile port。
- `issue_window_ctrl` 在 V1 hardwire `window_size=1`；V1.x 才管理多 active Tile Program entry。
- `buffer_hazard_table` 使用 Slot Frame shadow、read/write slot mask 和 event sequence 判定 admission 是否安全。
- `trap_fault` 与 Local Event/Fault Record。

### 7.2 RISC-V 共享实现建议

若采用小 RISC-V：

- RISC-V core 执行 UCE firmware 或 compiled tile bytecode loop。
- engine launch 可通过 custom instruction 或 MMIO launch FIFO。
- USE state functional units 作为 custom functional unit 或 MMIO task queue。
- debug、exception、CSR、instruction fetch 可与 USE 共享物理实现。
- 架构文档仍把 UCE 和 USE 分开描述，RTL 信号也应有 `uce_*` 与 `use_*` ownership。

禁止把 RISC-V 的通用 load/store 当成 MFE page walk fallback 常态路径；数据相关动态访问仍应由 MFE descriptor path 承担。RISC-V fallback 只用于 bring-up/debug 或明确的 slow path，是否保留量产路径由 PPA exploration 冻结。

### 7.3 Clock/reset/debug

Clock：

- First Silicon 建议 UCE、event、PMU 与 tile core 同 clock，减少 CDC。
- 若 UCE 与 BOA/EVU/MFE/USE 分 clock，launch/event 必须通过 CDC-safe handshake。
- program/descriptor SRAM clock 由 SRAM macro 冻结。

Reset：

- `uce_reset` 清 PC、scoreboard、decode state、stream held token valid、trap pending。
- `tile_soft_reset` 要求 UCE 停止新 issue 并驱动 drain。
- `debug_halt` 应停在 instruction boundary 或 launch boundary，不停在半个 stream credit handoff 中。
- outstanding event 在 reset 后必须进入 DONE/ERROR/RESET 中的确定状态。

Debug：

- 支持 halt/resume、read PC、read trap cause、read outstanding events、read current token。
- 单步粒度建议为 instruction boundary；跨 engine 内部单步由各 engine 自己 debug 机制处理。
- debug 访问不得破坏 stream credit 和 descriptor coherence。

### 7.4 软件工具链

- compiler 生成 Tile-SPMD program template 和 descriptor patch table。
- firmware/runtime 负责 program section registry、resident tag、descriptor cache invalidate、context binding；Tile UCE 只消费 prepared local handle。
- assembler/disassembler 应输出 symbolic descriptor/slot/event 名称，便于 trace 比对。
- golden trace 至少记录：PC、opcode、patched descriptor id、launch engine、event transition、stream token id、fault cause、PMU snapshot。
- compiler/runtime 若选择 V1.x sliding window profile，必须静态预留 ping-pong / multi-buffer slot set，并在 descriptor/trace 中标出 read/write slot mask；V1 profile 不需要改变 public IR。

## 8. 验证、bring-up 和验收标准

### 8.1 单元验证

| 验证项              | 覆盖点                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Prepared task check | local handle valid、program_id/version/epoch、frame generation、desc window bounds                               |
| Fetch/decode        | legal/illegal opcode、PC branch、ret/end、alignment fault、local slot OOB                                        |
| Descriptor patch    | offset bounds、slot permission、tile/group stride、overflow、invalidate                                          |
| Engine launch       | ready backpressure、event allocation、descriptor snapshot、timeout                                               |
| Wait/fence          | DONE/ERROR/TIMEOUT/RESET、ordering、multiple waitall                                                             |
| Stream unit         | pop empty、push full、EOS branch、error token propagation、credit release                                        |
| Scoreboard          | resource full、duplicate event、event sequence mismatch、window admission、slot hazard、reset drain、fault clear |
| Trap/fault          | syndrome 完整性、invalid/stale program handle、event error、stream error propagation                             |

### 8.2 Integration tests

1. 空 Tile Program：accept prepared task -> ret -> tile done。
2. Prepared handle only：不同 program_epoch/frame_generation 组合触发 hit/stale trap。
3. DMA copy kernel：UCE launch.dma -> wait -> compare。
4. BOA GEMM kernel：通过 UCE launch.boa，不旁路 datapath。
5. Stream relay kernel：pop input token、push output token、EOS/error path。
6. Paged attention control trace：launch.mfe -> launch.boa -> launch.evu -> launch.boa -> launch.mfe。
7. USE recurrence trace：launch.use scan/recurrence，验证 UCE 只控制 event，不修改 state data。
8. Fault injection：invalid/stale program handle、invalid descriptor、slot permission、stream credit leak、engine timeout。
9. Reset/drain：active launch 中 soft reset，所有 event/token/scoreboard 进入确定状态。
10. Sliding window profile：V1 验证 `window_size=1` 不重叠；V1.x 仿真 `window_size=2` 时验证 P0 store 与 P1 load 只在 slot 独立、event sequence 正确时 overlap。

### 8.3 验收标准

- Tile Program 的 PC、launch 序列、event 序列可由 golden trace 精确比对。
- UCE 与 USE 共享物理实现时，fault owner、debug CSR、PMU counter 和 RTL module interface 仍能区分 UCE control 与 USE state。
- Program handle 或 descriptor patch fault 能关联 context id、program id、tile id、local slot/epoch、descriptor id、slot id。
- Stream reset/drain 不泄漏 credit。
- `wait` 不会永久 hang；timeout 行为可验证。
- UCE overhead 在 canonical GEMM/paged attention/recurrence trace 中不会成为未解释瓶颈；若成为瓶颈，PMU 能指向 fetch、patch、stream 或 event wait。
- First Silicon `uce_window_size` 必须可观测为 1；若启用 V1.x window profile，必须证明 slot hazard stall、store visibility wait 和 event sequence mismatch 均可观测并可复现。

## 9. 风险、取舍和后续细化方向

### 9.1 风险

| 风险                            | 影响                                                               | 缓解                                                                                                             |
| ------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| 指令集过大                      | decoder、verification、toolchain 复杂                              | First Silicon 固定小集合，复杂 op 下沉到 descriptor-driven engine                                                |
| RISC-V fallback 被滥用          | 变成通用 CPU，性能和边界失控                                       | 明确 UCE 只做 control，MFE/USE/BOA/EVU 各自 owner 不变                                                           |
| descriptor patch/coherence 错误 | warm launch 产生错误地址                                           | version/invalidate/fence，patch bounds check                                                                     |
| stream token handling 出错      | deadlock 或 credit leak                                            | formal FIFO/credit/EOS/error/reset properties                                                                    |
| wait timeout 策略不清           | fault recovery 不可预测                                            | local watchdog + fault record + reset domain contract                                                            |
| Sliding window 过早打开         | 多 active program 放大 event reuse、slot alias 和 reset/drain 风险 | First Silicon 固定 `window_size=1`；V1.x 必须先冻结 event sequence、buffer hazard table 和 Slot Frame owner 规则 |
| UCE/USE debug 混淆              | bring-up 难定位                                                    | CSR 命名、fault code、PMU source 分离                                                                            |

### 9.2 取舍

- 小而明确的控制指令优于复杂 tile-local CPU ABI。
- Descriptor-driven engine launch 优于在 UCE 中暴露 datapath micro-op。
- Shared RISC-V 实现可减少面积和工具链成本，但不能合并 UCE/USE ownership。
- First Silicon 应先验证 command/event/DMA/BOA path，再扩展 MFE/USE 的复杂路径。

### 9.3 后续细化方向

- Tile Program binary encoding：由后续规格冻结。
- UCE register file、scoreboard entry 数、event id width：由后续规格冻结。
- UCE issue window entry 数、buffer hazard table 宽度和 event sequence 比较规则：V1 固定为 1；V1.x/V2 由后续规格冻结。
- Descriptor cache line size、coherence/invalidate protocol：由 SRAM profile 冻结。
- RISC-V custom instruction encoding 或 MMIO launch FIFO layout：由后续规格冻结。
- Debug CSR 地址、trap cause 编码、fault record binary layout：由后续规格冻结。
- UCE clock gating、power state、retention 策略：由 PPA exploration 冻结。
