# ELENOR EVU-MT 设计文档

## 1. 定位、目标和 First Silicon cutline

EVU-MT（Enhanced Vector Unit - Microthread Engine）是 ELENOR Compute Tile 内部的 **tile-local shared-PC microthread vector engine**。它服务于 BOA 不适合承接、但又不值得引入完整 GPU SM 的计算：elementwise、activation、softmax local phase、normalization、RoPE、layout pack/unpack、attention mask、dynamic shape tail、local indexed gather 和 small reduction。

核心定位：

```text
EVU-MT = shared-PC microthread execution
       + SIMT-like lane programming model
       + predicated lane execution
       + local slot-memory LSU
       + local reduction / shuffle / SFU
       + bank conflict replay
       + precise fault / PMU attribution
       - warp scheduler
       - per-thread PC
       - multi-warp residency
       - global memory programming model
```

EVU-MT 不是小 GPU SM。它只复用 SIMT 的 lane 编程模型，不引入 GPU 的 warp scheduling machinery。

### 1.1 设计目标

1. 为 compiler 提供一个稳定的 tile-local kernel target，而不是不断扩 descriptor-only 特殊算子集合。
2. 支持 predicated lane execution、mask/tail、local gather、small reduction、RoPE、softmax/norm 等 irregular path。
3. 保持 datapath 简洁：shared PC、单 kernel active、单 lane group 执行。
4. 只处理 local slot memory，不承担 HBM random latency hiding。
5. 保持 fault precise enough for bring-up：first-fault-wins、fault record 完整、event/fault 次序确定。
6. PMU 能区分 active、ifetch/decode/issue stall、LSU replay、SFU busy、writeback stall，而不是只给总 busy 周期。

### 1.2 First Silicon V1 cutline

| 优先级 | 能力                       | First Silicon V1 必须闭环                                             | 后续能力                   |
| ------ | -------------------------- | --------------------------------------------------------------------- | -------------------------- |
| P0     | shared-PC kernel execution | launch、ifetch、decode、shared PC、exit、event commit                 | 多 kernel resident policy  |
| P0     | predicate / tail           | lane_valid、active mask、predicate register、masked load/store        | 更深 mask stack            |
| P0     | local memory               | unit-stride、strided、masked load/store                               | wider LSU、coalescing hint |
| P0     | arithmetic                 | integer add/sub/mul/max/min、fp add/mul/max/min、compare、select、cvt | fma、bitwise 扩展          |
| P0     | reduction                  | reduce.sum、reduce.max、reduce.sumsq                                  | segment/prefix 扩展        |
| P0     | observability              | status CSR、fault record、PMU、basic debug readout                    | sampled trace              |
| P1     | indexed gather             | local index slot、bank replay、lane remap restore                     | reorder / coalesce hint    |
| P1     | SFU                        | exp.approx、rsqrt.approx、rcp.approx、gelu/silu.approx 子集           | 更高精度近似               |
| P1     | shuffle                    | pair shuffle、split-half shuffle、RoPE pair transform                 | small transpose            |
| P2     | structured mask control    | if_mask / else_mask / endif                                           | 更复杂 structured region   |
| P2     | scatter                    | 仅可选 conflict-free scatter                                          | ordered scatter            |
| P3     | atomic scatter             | 不进入 V1                                                             | V2/V3 研究                 |

V1 明确不支持：

```text
- warp scheduler
- per-lane PC
- multi-warp resident context
- random HBM gather
- page/segment metadata walk
- unordered duplicate scatter atomic
- arbitrary full crossbar permute
```

## 2. 职责、非职责和 ownership

### 2.1 EVU-MT 负责

#### 2.1.1 Predicated microthread arithmetic

EVU-MT 在 effective-active mask 下执行 lane arithmetic：

```text
- integer add / sub / mul / max / min
- floating add / mul / max / min
- compare
- select
- dtype convert
- approximate exp / rsqrt / reciprocal
- activation subset: relu / gelu.approx / silu.approx
```

语义：

```text
effective_active[i] = lane_valid_mask[i]
                    & active_mask[i]
                    & instr_predicate[i]
```

inactive lane：

```text
- 不更新 lane register
- 不发 memory request
- 不写 local memory
- 不参与 reduction
- 不触发地址 fault
```

#### 2.1.2 Mask / tail / predicate

EVU-MT 负责全部 lane-level predicate 语义：

```text
- dynamic shape tail
- padding tail
- attention mask
- sparse valid lane
- compare-generated predicate
- input mask load
- structured mask stack
```

tail 必须由硬件阻断：

```text
lane_valid_mask[i] = (block_base + i) < logical_elements
```

硬件不依赖 software padding 保证越界安全。

#### 2.1.3 Local vector reduction

EVU-MT 负责本 lane group 内部 small reduction：

```text
- reduce.sum
- reduce.max
- reduce.sumsq
- softmax row max / row sum local phase
- RMSNorm / LayerNorm sum / sumsq local phase
```

规则：

```text
- 只规约 active lane
- inactive lane 不参与 reduction
- all-inactive reduction 行为必须由 numerical policy 定义
- BF16/FP16 reduction 推荐使用 FP32 accumulate
```

EVU-MT 不负责跨 tile、跨 group 或全局 reduction。

#### 2.1.4 Local vector memory access

EVU-MT 负责 local slot memory 上的：

```text
- unit-stride load/store
- strided load/store
- masked load/store
- indexed gather
- conflict-free scatter optional
```

memory address 只能是 slot-relative：

```text
slot_id + byte_offset
```

EVU-MT 不直接访问 virtual address、HBM physical address、page table 或 global pointer。

#### 2.1.5 Shuffle / permute

V1 限制为低复杂度 lane shuffle：

```text
- even/odd pair shuffle
- split-half pair shuffle
- RoPE pair transform support
- optional 8x8 / 16x16 small transpose
```

V1 不支持 arbitrary full-crossbar permute。

#### 2.1.6 Bank conflict detection and replay

EVU-MT 负责本地 memory access 的 bank conflict detection 和 replay：

```text
- detect multi-lane bank conflict
- partial issue non-conflict lanes
- replay conflict lanes
- preserve logical lane mapping
- count replay cycles in PMU
```

replay 只解决 local SRAM / local slot memory 的短延迟冲突，不承担 HBM long latency hiding。

#### 2.1.7 Shared-PC instruction execution

EVU-MT 执行 microthread kernel ISA：

```text
- instruction fetch
- instruction decode
- shared PC update
- scalar register read/write
- lane register read/write
- predicate register read/write
- structured mask stack
- uniform branch
- exit / commit
```

控制约束：

```text
- 所有 lane 共享一个 PC
- 不支持 per-lane PC
- 不支持 arbitrary divergence reconvergence
- 小分支优先 if-conversion
- 分支只允许 structured mask control 或 uniform branch
```

#### 2.1.8 Fault / PMU

EVU-MT 负责本单元 fault detection、fault record 和 PMU attribution。

fault 类型至少包括：

```text
- invalid launch descriptor
- code slot fault / code OOB / unaligned PC
- illegal opcode
- unsupported dtype
- invalid register index
- slot permission fault
- active lane address OOB
- mask stack overflow / underflow
- replay timeout
- internal fault
```

PMU 至少包含：

```text
- active cycles
- instruction count
- ifetch stall
- decode stall
- issue stall
- LSU active cycles
- LSU replay cycles
- replay queue full cycles
- SFU active cycles
- reduction active cycles
- shuffle active cycles
- writeback stall cycles
- commit stall cycles
- masked lane count
- branch-mask cycles
- fault count by type
```

PMU primary stall attribution 必须互斥。

### 2.2 EVU-MT 不负责

EVU-MT 不负责：

```text
- dense matmul / conv 主路径
- page / segment metadata walk
- KV cache block table walk
- HBM random pointer chasing
- Tile Program PC
- engine-level orchestration
- state lifecycle / checkpoint / restore
- global unordered scatter atomic consistency
- cross-tile atomic update
- graph-level dynamic branch
```

边界原因：

| 非职责               | 原因                                           |
| -------------------- | ---------------------------------------------- |
| dense matmul / conv  | 属于 BOA 或其他 dense compute engine           |
| metadata walk        | 属于 MFE 或外部 memory flow logic              |
| Tile Program PC      | EVU-MT 只有 kernel 内部 shared PC              |
| engine orchestration | 由 Tile UCE 管理 launch/wait/event             |
| state lifecycle      | 内部状态只对当前 kernel 有效                   |
| global atomic        | 没有全局一致性模型                             |
| HBM latency hiding   | 无 warp scheduler，依赖外部预取到 local memory |

### 2.3 Ownership 表

| 对象 / 行为                    | Owner                       | EVU-MT 行为                          |
| ------------------------------ | --------------------------- | ------------------------------------ |
| microthread kernel binary      | compiler                    | EVU-MT fetch / decode / execute      |
| EVU-MT launch descriptor       | runtime / Tile UCE          | EVU-MT validate / latch / start      |
| shared PC                      | EVU-MT                      | 仅表示 kernel 内部 PC                |
| lane register file             | EVU-MT                      | per-lane temporary state             |
| scalar register file           | EVU-MT                      | uniform state                        |
| predicate register file        | EVU-MT                      | lane mask state                      |
| active mask / tail mask        | EVU-MT                      | 控制当前有效 lane                    |
| structured mask stack          | EVU-MT                      | 支持 if_mask / else_mask / endif     |
| local arithmetic / SFU         | EVU-MT                      | 执行算术和近似运算                   |
| local reduction                | EVU-MT                      | 只处理 lane-group local reduction    |
| local memory load/store/gather | EVU-MT                      | 访问 local slot memory               |
| scatter conflict policy        | compiler/runtime + EVU-MT   | 未证明 conflict-free 则 reject/fault |
| bank conflict replay           | EVU-MT                      | 检测、replay、PMU 归因               |
| metadata walk                  | MFE / external              | EVU-MT 不做                          |
| dense compute                  | BOA                         | EVU-MT 不做                          |
| tile-level scheduling          | Tile UCE                    | EVU-MT 不做                          |
| state lifecycle                | USE / external state engine | EVU-MT 不做                          |
| fault recovery policy          | Tile UCE / reset logic      | EVU-MT 按 fault_policy drain/kill    |
| PMU primary attribution        | EVU-MT                      | 负责本单元 active/stall/replay 归因  |

## 3. 执行模型和架构态

### 3.1 Kernel 执行模型

一条 EVU-MT kernel 的基本执行语义：

```text
one launch
  -> one shared PC
  -> one active lane group
  -> lanes execute same instruction under mask
  -> exit or fault
```

同一时间 EVU-MT 只运行一个 active kernel。V1 不做 multi-kernel context switch，不做 kernel preemption。

### 3.2 Register model

```text
r0-r15    scalar registers
v0-v15    lane registers
p0-p7     predicate registers
pc        shared PC
amask     active mask
```

其中：

- `r*` 对所有 lane 可见且一致。
- `v*` 是每 lane 一份。
- `p*` 每 bit 对应一条 lane。
- `amask` 是当前 structured control-flow 生效 mask。

### 3.3 Architectural state 草案

```c
typedef struct {
    uint32_t pc;

    uint32_t active_mask;
    uint32_t lane_valid_mask;
    uint32_t exec_mask;

    uint32_t scalar_reg[16];
    uint32_t pred_reg[8];

    uint32_t mask_stack[4];
    uint32_t mask_stack_ptr;

    uint32_t cmd_id;
    uint32_t event_id;

    uint32_t fault_code;
    uint32_t fault_pc;
    uint32_t fault_lane;
    uint32_t fault_slot;
    uint32_t fault_addr;
} evu_mt_arch_state_t;
```

### 3.4 Launch 后初始化规则

launch 成功后：

```text
pc               = entry_pc
lane_valid_mask  = tail_generate(block_base, logical_elements)
active_mask      = lane_valid_mask
pred_reg[*]      = 0
mask_stack_ptr   = 0
fault_code       = NONE
```

V1 不支持 architectural state save/restore；kernel 结束后内部状态可丢弃，只保留 debug CSR / fault syndrome / PMU snapshot。

## 4. 顶层接口和系统边界

### 4.1 Top-level SystemVerilog 草案

```systemverilog
module evu_mt #(
    parameter int LANES            = 32,
    parameter int LANE_REGS        = 16,
    parameter int SCALAR_REGS      = 16,
    parameter int PRED_REGS        = 8,
    parameter int MASK_STACK_DEPTH = 4,

    parameter int DATA_WIDTH       = 32,
    parameter int INST_WIDTH       = 64,
    parameter int ADDR_WIDTH       = 32,
    parameter int SLOT_ID_WIDTH    = 16,
    parameter int CMD_ID_WIDTH     = 16,
    parameter int EVENT_ID_WIDTH   = 16
) (
    input  logic               clk,
    input  logic               rst_n,

    input  logic               launch_valid,
    output logic               launch_ready,
    input  evu_mt_launch_req_t launch_req,

    output logic               commit_valid,
    input  logic               commit_ready,
    output evu_mt_commit_t     commit,

    output logic               imem_req_valid,
    input  logic               imem_req_ready,
    output evu_mt_imem_req_t   imem_req,
    input  logic               imem_resp_valid,
    output logic               imem_resp_ready,
    input  evu_mt_imem_resp_t  imem_resp,

    output logic               dmem_req_valid,
    input  logic               dmem_req_ready,
    output evu_mt_dmem_req_t   dmem_req,
    input  logic               dmem_resp_valid,
    output logic               dmem_resp_ready,
    input  evu_mt_dmem_resp_t  dmem_resp,

    input  logic               csr_req_valid,
    output logic               csr_req_ready,
    input  evu_mt_csr_req_t    csr_req,
    output logic               csr_resp_valid,
    input  logic               csr_resp_ready,
    output evu_mt_csr_resp_t   csr_resp
);
```

### 4.2 输入输出边界

输入：

```text
- launch request
- instruction memory response
- local data memory response
- CSR/debug request
```

输出：

```text
- instruction memory request
- local data memory request
- completion / fault commit
- CSR/debug response
```

EVU-MT 不关心：

```text
- launch 来自哪个上级调度器
- local memory 数据由谁准备
- output 被谁消费
- graph 如何 partition
- tile 如何全局调度
```

### 4.3 Launch / commit wrapper 语义

约束：

- `launch_ready=1` 仅在 `IDLE && !pending_commit && !fault_drain` 成立时允许。
- launch accepted 后，descriptor 全量 latch；后续 descriptor slot 被外部改写不得影响正在运行的 kernel。
- `commit_valid` 只在 done 或 fault 两种 terminal state 产生一次。
- fault record 对系统可见必须早于对应 fault event / error commit 可见。

### 4.4 Commit payload 草案

```c
typedef enum {
    EVU_MT_COMMIT_DONE  = 0,
    EVU_MT_COMMIT_FAULT = 1
} evu_mt_commit_kind_t;

typedef struct {
    uint16_t kind;
    uint16_t cmd_id;
    uint16_t event_id;
    uint16_t reserved0;
    uint32_t fault_record_slot;
    uint32_t pmu_snapshot_slot;
} evu_mt_commit_t;
```

V1 推荐 done/fault 统一走 commit interface，不引入单独 sideband interrupt path。

## 5. Launch Descriptor 和 slot contract

### 5.1 Launch descriptor v0 草案

```c
typedef struct {
    uint16_t version;
    uint16_t size_bytes;

    uint16_t cmd_id;
    uint16_t event_id;

    uint16_t code_slot;
    uint16_t arg_slot;
    uint16_t scratch_slot;
    uint16_t slot_table_slot;

    uint16_t num_lanes;
    uint16_t max_lane_regs;
    uint16_t max_scalar_regs;
    uint16_t max_pred_regs;

    uint16_t fault_policy;
    uint16_t priority;
    uint16_t flags;
    uint16_t reserved0;

    uint32_t entry_pc;
    uint32_t code_size_bytes;
    uint32_t arg_size_bytes;
    uint32_t logical_elements;

    uint32_t block_base;
    uint32_t reserved1;
} evu_mt_launch_desc_t;
```

### 5.2 字段语义

| 字段                     | 语义                                 | V1 约束                                               |
| ------------------------ | ------------------------------------ | ----------------------------------------------------- |
| `version` / `size_bytes` | ABI 兼容                             | 不匹配必须 reject/fault                               |
| `cmd_id`                 | 当前 kernel command id               | fault record 必须回填                                 |
| `event_id`               | completion event id                  | commit 时原样输出                                     |
| `code_slot`              | kernel binary 所在 local slot        | 必须具有 execute permission                           |
| `arg_slot`               | kernel 参数 slot                     | 只读                                                  |
| `scratch_slot`           | scratch / spill / temp slot          | permission 必须匹配                                   |
| `slot_table_slot`        | slot binding table 所在 slot         | 只读                                                  |
| `num_lanes`              | 本次启动有效 lane 数                 | `num_lanes <= LANES`                                  |
| `max_lane_regs`          | kernel 声明 lane RF 需求             | `<= LANE_REGS`                                        |
| `max_scalar_regs`        | scalar RF 需求                       | `<= SCALAR_REGS`                                      |
| `max_pred_regs`          | predicate RF 需求                    | `<= PRED_REGS`                                        |
| `fault_policy`           | fault 后 drain / kill 策略           | 编码由后续规格冻结                                    |
| `flags`                  | profile / numeric / debug flag       | reserved bit 必须为 0                                 |
| `entry_pc`               | kernel 入口 PC                       | instruction aligned                                   |
| `code_size_bytes`        | code 可见窗口                        | 必须覆盖 `entry_pc`                                   |
| `arg_size_bytes`         | 参数窗口                             | 0 合法，表示无参数                                    |
| `logical_elements`       | 逻辑元素总数                         | 用于 tail 生成，0 合法仅当 kernel 不访问 lane payload |
| `block_base`             | 当前 lane group 对应的逻辑起始 index | 用于 tail 和 index 基准                               |

### 5.3 Slot binding table v0 草案

descriptor 不直接内嵌大量物理地址；EVU-MT 通过 slot table 看到 kernel 可访问对象：

```c
typedef enum {
    EVU_MT_SLOT_ROLE_CODE = 0,
    EVU_MT_SLOT_ROLE_ARG,
    EVU_MT_SLOT_ROLE_INPUT,
    EVU_MT_SLOT_ROLE_OUTPUT,
    EVU_MT_SLOT_ROLE_INDEX,
    EVU_MT_SLOT_ROLE_MASK,
    EVU_MT_SLOT_ROLE_PARAM,
    EVU_MT_SLOT_ROLE_SCRATCH
} evu_mt_slot_role_t;

typedef struct {
    uint16_t slot_id;
    uint16_t role;
    uint16_t perm;
    uint16_t layout;
    uint32_t byte_size;
    uint32_t base_offset;
    uint32_t stride_bytes;
    uint32_t reserved;
} evu_mt_slot_binding_t;
```

约束：

- slot role 和 permission 必须与实际 opcode 匹配。
- `byte_size` 是 bounds checker 输入，不允许依赖外部“肯定不会越界”。
- `layout` 允许 compiler 传 bank-aware hint，但具体编码由后续规格冻结。
- slot table 自身只读；running kernel 不允许被 patch。

### 5.4 Descriptor validator 必须检查

```text
- version / size
- reserved bits == 0
- code_slot / arg_slot / slot_table_slot permission
- entry_pc alignment and range
- num_lanes / reg requirements 不超过硬件上限
- logical_elements / block_base 加法不溢出
- slot table 每项 role / perm / size 合法
```

不合法必须在 launch 阶段 reject，不能半执行后再发现结构性错误。

## 6. ISA profile 和控制流约束

### 6.1 Opcode 类别

| 类别        | 指令                                                                      |
| ----------- | ------------------------------------------------------------------------- |
| Control     | `exit`, `br.u`, `if_mask`, `else_mask`, `endif`                           |
| Lane        | `laneid`, `setvl`                                                         |
| Predicate   | `cmp`, `pred.and`, `pred.or`, `pred.not`, `mov.pred`                      |
| Memory      | `load`, `store`, `gather`, `scatter` optional, `load.scalar`, `load.mask` |
| Integer ALU | `add`, `sub`, `mul`, `max`, `min`, `select`                               |
| FP ALU      | `fadd`, `fmul`, `fmax`, `fmin`, optional `fma`                            |
| SFU         | `exp.approx`, `gelu.approx`, `silu.approx`, `rsqrt.approx`, `rcp.approx`  |
| Convert     | `cvt`, `round`, `sat`                                                     |
| Reduction   | `reduce.sum`, `reduce.max`, `reduce.sumsq`                                |
| Shuffle     | `shuffle.pair`, `shuffle.split_half`, optional `transpose.small`          |

### 6.2 Encoding 原则

V1 可使用固定 64-bit 指令宽度，降低 decode 复杂度：

```c
typedef struct {
    uint8_t  opcode;
    uint8_t  dst;
    uint8_t  src0;
    uint8_t  src1;
    uint8_t  pred;
    uint8_t  dtype;
    uint16_t imm16;
    uint32_t imm32;
} evu_mt_inst_t;
```

强约束：

```text
- illegal opcode 必须 fault
- unsupported dtype 必须 fault
- invalid register index 必须 fault
- branch target 必须 instruction aligned
- V1 不支持自修改代码
```

### 6.3 Shared-PC control flow

EVU-MT 不支持 per-lane PC。控制流只允许两种形式：

1. **uniform branch**：条件来自 scalar register 或 uniform compare。
2. **structured mask control**：`if_mask` / `else_mask` / `endif`。

典型 lowering：

```text
small branch        -> if-conversion + select/predicate
larger structured   -> if_mask / else_mask / endif
non-structured CFG  -> compiler 不可下放到 EVU-MT
```

### 6.4 Structured mask stack 规则

```text
if_mask Pn:
    push(active_mask)
    active_mask = active_mask & Pn

else_mask:
    top = mask_stack[top]
    active_mask = top & ~active_mask_before_else

endif:
    active_mask = pop()
```

要求：

- `MASK_STACK_DEPTH` 溢出 / 下溢必须 fault。
- `else_mask` 只能匹配最近一个未闭合 `if_mask`。
- compiler 必须保证 structured region 正确嵌套。

## 7. 内部模块划分和微架构

### 7.1 模块划分

```text
EVU-MT
├── Launch Frontend
├── Instruction Fetch Unit
├── Instruction Decode Unit
├── Shared PC / Control Unit
├── Predicate and Mask Engine
├── Scalar Register File
├── Predicate Register File
├── Lane Register File
├── Lane ALU
├── SFU
├── Reduction Unit
├── Shuffle Unit
├── Vector LSU
├── Replay Queue
├── Writeback Unit
├── Fault Unit
├── Commit Unit
└── PMU
```

### 7.2 模块职责

| 模块                         | 职责                                           | 关键点                              |
| ---------------------------- | ---------------------------------------------- | ----------------------------------- |
| Launch Frontend              | 接收 launch、读取并 latch descriptor           | launch 原子性、single-active-kernel |
| Instruction Fetch            | 从 code slot 取指                              | code range check、PC alignment      |
| Decode                       | opcode / dtype / reg index decode              | illegal reject 尽早化               |
| Shared PC / Control          | PC update、branch、exit、FSM 推进              | shared-PC only                      |
| Predicate / Mask             | lane_valid、active_mask、pred RF、mask stack   | tail first-class 处理               |
| Scalar RF                    | uniform operand                                | 可与 CSR path 分离                  |
| Lane RF                      | per-lane operand / result                      | inactive lane gating                |
| Vector LSU                   | unit/stride/masked/gather/scatter optional     | slot bounds、bank replay            |
| Replay Queue                 | 记录冲突 lane subset                           | 只 replay 局部 lane                 |
| ALU / SFU / Reduce / Shuffle | lane compute                                   | multi-cycle 单元 backpressure       |
| Writeback                    | lane RF、pred RF、scalar RF、local memory 提交 | 必须按 lane mapping 写回            |
| Fault Unit                   | first fault capture、fault record 生成         | fault record before event           |
| Commit Unit                  | done/fault commit                              | 单次 terminal commit                |
| PMU                          | 互斥 stall 分类和快照                          | primary owner 唯一                  |

### 7.3 建议 pipeline

```text
S0: LAUNCH / INIT
S1: IFETCH
S2: DECODE
S3: PREDICATE + RF READ
S4: EXECUTE / ADDR_GEN
S5: LSU / SFU / REDUCE / SHUFFLE
S6: WRITEBACK
S7: COMMIT / FAULT
```

### 7.4 各 stage 关键职责

#### S0: Launch / Init

```text
- 接收 launch request
- descriptor 基本合法性检查
- 初始化 cmd_id / event_id / PC
- 初始化 lane_valid_mask / active_mask
- 清空 pred RF / mask_stack state
```

#### S1: IFETCH

```text
- 根据 shared PC 读取 instruction word
- 检查 PC alignment
- 检查 code range
- 处理 imem fault / timeout
```

#### S2: DECODE

```text
- opcode decode
- src/dst register decode
- dtype decode
- imm decode
- 指令合法性检查
```

#### S3: Predicate + RF Read

```text
- 读取 scalar RF / lane RF / pred RF
- 计算 effective_active
- 生成 inactive gating
- branch / mask instruction 生成 next active state
```

#### S4: Execute / Addr Gen

```text
- ALU 执行
- dtype convert
- lane compare
- LSU 地址生成
- gather index address 生成
```

#### S5: LSU / SFU / Reduce / Shuffle

```text
- memory issue / replay enqueue
- SFU multi-cycle execute
- reduction tree execute
- shuffle datapath execute
```

#### S6: Writeback

```text
- 写 lane RF / scalar RF / pred RF
- 收集 gather response，恢复 lane mapping
- masked store 提交 byte enable
```

#### S7: Commit / Fault

```text
- exit 指令 commit
- first-fault capture
- PMU snapshot
- emit done/fault commit
```

## 8. 主状态机、fault 状态机和执行约束

### 8.1 Main FSM

```text
IDLE
  |
  | launch_valid && launch_ready
  v
LAUNCH_VALIDATE
  | invalid
  +---------> FAULT_COMMIT -> IDLE
  |
  v
INIT_STATE
  |
  v
IFETCH
  | imem fault
  +---------> FAULT_COMMIT -> IDLE
  |
  v
DECODE
  | illegal
  +---------> FAULT_COMMIT -> IDLE
  |
  v
ISSUE
  |
  +--> EXEC_ALU
  +--> EXEC_LSU -> LSU_WAIT -> LSU_REPLAY? -> WRITEBACK
  +--> EXEC_SFU -> SFU_WAIT -> WRITEBACK
  +--> EXEC_REDUCE -> REDUCE_WAIT -> WRITEBACK
  +--> EXEC_SHUFFLE -> WRITEBACK
  |
  v
PC_UPDATE
  | exit
  +---------> EVENT_COMMIT -> IDLE
  |
  +---------> IFETCH
```

### 8.2 Fault FSM

```text
FAULT_DETECT
  -> STOP_ISSUE
  -> DRAIN_OR_KILL_OUTSTANDING
  -> WRITE_FAULT_RECORD
  -> COMMIT_FAULT_EVENT
  -> IDLE
```

fault policy 约束：

- `kill`：直接阻止后续 issue，允许抛弃尚未可见写入的内部结果，但必须定义哪些请求已对外可见。
- `drain`：等待已发请求进入可确定状态后再提交 fault。
- V1 推荐默认 fail-fast + bounded drain；具体 timer 由后续规格冻结。

### 8.3 Single-active-kernel invariant

V1 必须满足：

```text
- EVU-MT 任一时刻最多一个 active cmd_id
- 一个 active cmd_id 最多一次 terminal commit
- fault 和 success commit 不得同时出现
```

## 9. Vector LSU、地址生成和 replay

### 9.1 Memory domain

EVU-MT 只访问 local slot memory：

```text
slot_id + byte_offset
```

不支持：

```text
- HBM physical address
- virtual address
- page table walk
- cache-coherent global pointer
```

### 9.2 Access modes

| 模式               | V1 支持     | 说明                               |
| ------------------ | ----------- | ---------------------------------- |
| unit-stride load   | required    | 连续读                             |
| unit-stride store  | required    | 连续写                             |
| strided load/store | required    | row/column/layout stride           |
| masked load/store  | required    | tail / predicate                   |
| indexed gather     | required    | local index slot -> data slot      |
| scatter            | optional    | 仅 conflict-free 或 ordered policy |
| atomic             | unsupported | V1 禁止                            |

### 9.3 Address generation

unit-stride：

```text
addr[i] = slot_base + base_offset + lane_id[i] * elem_size
```

strided：

```text
addr[i] = slot_base + base_offset + lane_id[i] * stride_bytes
```

gather：

```text
idx_addr[i] = index_slot_base + index_offset + lane_id[i] * index_elem_size
idx[i]      = load_u32(idx_addr[i])
addr[i]     = data_slot_base + base_offset + idx[i] * index_scale
```

规则：

```text
- inactive lane 不生成 memory request
- active lane OOB 必须 fault
- inactive lane OOB 不检查
- masked store inactive lane byte enable = 0
- gather index dtype V1 推荐 u32
- index_scale 推荐限制为 {1, 2, 4, 8, 16}
```

### 9.4 LSU request / response 草案

```systemverilog
typedef struct packed {
    logic [15:0] slot_id;
    logic [31:0] byte_offset;
    logic [63:0] byte_enable;
    logic [31:0] lane_mask;
    logic [7:0]  op_id;
    logic [7:0]  replay_tag;
    logic        is_gather;
    logic        is_store;
    logic [2:0]  access_size;
} evu_mt_dmem_req_t;

typedef struct packed {
    logic [7:0]   op_id;
    logic [7:0]   replay_tag;
    logic [31:0]  lane_valid;
    logic         fault;
    logic [5:0]   fault_lane;
    logic [15:0]  fault_code;
    logic [511:0] data;
} evu_mt_dmem_resp_t;
```

### 9.5 Replay 触发条件

```text
- SRAM bank conflict
- LSU structural hazard
- store port conflict
- dependency replay
```

### 9.6 Replay 规则

```text
- replay 不改变 logical lane order
- gather response 可以乱序返回
- writeback 必须恢复 lane mapping
- store replay 必须保持可见写入顺序
- replay queue full 会 backpressure pipeline
- replay timeout 必须 fault
```

### 9.7 Replay entry 草案

```c
typedef struct {
    uint8_t  op_id;
    uint8_t  replay_tag;

    uint32_t lane_mask;

    uint16_t slot_id;
    uint32_t base_offset;

    uint8_t  access_size;
    uint8_t  is_store;
    uint8_t  is_gather;
    uint8_t  reserved;
} evu_mt_replay_entry_t;
```

### 9.8 Scatter policy

scatter 只有在下列条件之一满足时才允许：

```text
- compiler/runtime 证明 conflict-free
- descriptor/launch policy 声明 ordered mode
- duplicate index 明确指定 fault-on-duplicate
```

V1 默认建议关闭 scatter。若启用但 duplicate policy 未声明，必须 fault，不能 silent last-writer-wins。

## 10. Datapath 单元和数值策略

### 10.1 ALU

基础 integer ALU：

```text
- add
- sub
- mul
- max
- min
- compare
- select
- dtype convert
```

基础 FP ALU：

```text
- fadd
- fmul
- fmax
- fmin
- optional fma
```

### 10.2 SFU

V1 SFU 子集：

```text
- exp.approx
- gelu.approx
- silu.approx optional
- rsqrt.approx
- rcp.approx
```

约束：

```text
- multi-cycle pipeline
- valid/ready backpressure
- approximation policy 必须冻结
- inactive lane 不进入 SFU
```

### 10.3 Reduction unit

支持：

```text
- reduce.sum
- reduce.max
- reduce.sumsq
```

规则：

```text
- inactive lane 不参与 reduction
- all-inactive reduction 返回 identity 或 fault
- FP16/BF16 输入推荐 FP32 accumulate
- reduction tree 分层多拍
```

### 10.4 Shuffle unit

基础支持：

```text
- pair even/odd shuffle
- split-half pair shuffle
- RoPE pair transform
```

可选支持：

```text
- 8x8 transpose
- 16x16 transpose
```

V1 不建议支持 arbitrary permute crossbar。

### 10.5 数值策略

必须冻结：

```text
- dtype_src / dtype_acc / dtype_dst
- rounding mode
- denorm / NaN / inf policy
- saturation or wrap policy
- exp / gelu / silu / rsqrt / rcp approximation error envelope
- all-inactive reduction identity
```

推荐 V1 规则：

- BF16/FP16 reduction 使用 FP32 accumulate。
- compare 结果写 predicate，不直接触发 branch。
- softmax / norm 的 golden tolerance 由后续规格冻结，但硬件路径必须固定而非实现依赖。

## 11. 关键 workload 映射

### 11.1 Fused elementwise / activation

典型 kernel：

```c
tid = block_base + lane_id;
if (tid < N) {
    y[tid] = gelu(x[tid] * scale + bias);
}
```

映射：

```text
laneid
cmp -> predicate
load x pred p0
fmul/add pred p0
gelu.approx pred p0
store y pred p0
exit
```

### 11.2 Local softmax

EVU-MT 只负责 tile-local softmax phase：

```text
score tile
  -> apply scale/mask/tail
  -> reduce.max
  -> exp.approx(score-max)
  -> reduce.sum
  -> divide normalize
```

跨 tile reduce、跨 sequence 合并仍由上层 collective/BOA/runtime 负责。

### 11.3 RMSNorm / LayerNorm

典型路径：

```text
load x
  -> reduce.sumsq 或 sum/sumsq
  -> rsqrt.approx / reciprocal path
  -> scale / bias
  -> optional activation
  -> store y
```

### 11.4 Local gather + activation

典型路径：

```text
load index slot
  -> gather input slot
  -> activation / convert
  -> store output slot
```

关键检查：

- index slot OOB。
- gather data slot OOB。
- replay 后 lane remap 恢复。

### 11.5 Paged attention 中的角色

ELENOR paged attention 的职责拆分：

```text
MFE   : page/segment metadata walk, stream preparation
BOA   : QK / AV dense path
EVU-MT: scale / mask / local softmax / tail
UCE   : launch order / event chain
```

EVU-MT 不负责 page walk，也不负责全局 attention partition。

## 12. Fault、CSR、debug 和 PMU

### 12.1 Fault record 最小字段

```c
typedef struct {
    uint16_t source;
    uint16_t fault_code;
    uint16_t cmd_id;
    uint16_t event_id;
    uint32_t fault_pc;
    uint32_t fault_lane;
    uint32_t fault_slot;
    uint32_t fault_addr;
    uint32_t aux0;
    uint32_t aux1;
} evu_mt_fault_record_v0_t;
```

要求：

- first-fault-wins；后续 fault 进入 secondary counter，不覆盖首个 syndrome。
- `fault_pc`、`fault_lane`、`fault_slot`、`fault_addr` 尽可能精确；若某字段无意义，必须定义无效编码。
- fault record 对外可见早于 fault event 可见。

### 12.2 Fault code 草案

```c
typedef enum {
    EVU_MT_FAULT_NONE              = 0,
    EVU_MT_FAULT_INVALID_LAUNCH    = 1,
    EVU_MT_FAULT_CODE_OOB          = 2,
    EVU_MT_FAULT_ILLEGAL_OPCODE    = 3,
    EVU_MT_FAULT_UNSUPPORTED_DTYPE = 4,
    EVU_MT_FAULT_INVALID_REG       = 5,
    EVU_MT_FAULT_SLOT_PERMISSION   = 6,
    EVU_MT_FAULT_ADDR_OOB          = 7,
    EVU_MT_FAULT_MASK_STACK_OVER   = 8,
    EVU_MT_FAULT_MASK_STACK_UNDER  = 9,
    EVU_MT_FAULT_REPLAY_TIMEOUT    = 10,
    EVU_MT_FAULT_INTERNAL          = 11
} evu_mt_fault_code_t;
```

### 12.3 CSR / status 建议

| Register                 | 说明                                      |
| ------------------------ | ----------------------------------------- |
| `EVU_MT_STATUS`          | idle / busy / fault / drain / replay_busy |
| `EVU_MT_CMD_ID`          | current command id                        |
| `EVU_MT_PC`              | current shared PC                         |
| `EVU_MT_ACTIVE_MASK`     | current active lane mask                  |
| `EVU_MT_LANE_VALID_MASK` | current tail-valid mask                   |
| `EVU_MT_FAULT_CODE`      | current or sticky fault code              |
| `EVU_MT_FAULT_PC`        | fault PC                                  |
| `EVU_MT_FAULT_LANE`      | first faulting lane                       |
| `EVU_MT_FAULT_SLOT`      | faulting slot id                          |
| `EVU_MT_FAULT_ADDR`      | fault byte offset                         |
| `EVU_MT_PMU_*`           | PMU counter window                        |

V1 建议提供只读 debug 视图，不提供运行时改写 architectural state 的 invasive debug 功能。

### 12.4 PMU required counters

```text
evu_mt_active_cycles
evu_mt_instruction_count
evu_mt_ifetch_stall_cycles
evu_mt_decode_stall_cycles
evu_mt_issue_stall_cycles
evu_mt_lsu_active_cycles
evu_mt_lsu_replay_cycles
evu_mt_replay_queue_full_cycles
evu_mt_sfu_active_cycles
evu_mt_reduce_active_cycles
evu_mt_shuffle_active_cycles
evu_mt_writeback_stall_cycles
evu_mt_commit_stall_cycles
evu_mt_masked_lane_count
evu_mt_branch_mask_cycles
evu_mt_fault_count_by_type
```

### 12.5 Primary stall attribution

primary stall 必须互斥：

```text
ACTIVE
IFETCH_STALL
DECODE_STALL
ISSUE_STALL
LSU_REPLAY
REPLAY_QUEUE_FULL
SFU_BUSY
REDUCE_BUSY
SHUFFLE_BUSY
WRITEBACK_STALL
COMMIT_STALL
FAULT_DRAIN
```

任何 cycle 最多归给一个 primary owner。secondary tag 仅用于 debug，不进入 utilization 汇总。

### 12.6 Reset / drain / clear-fault 行为

```text
reset:
  clear architectural state
  clear or invalidate PMU shadow
  return IDLE

fault:
  stop issue
  drain or kill outstanding request by fault_policy
  write fault record
  commit fault event

clear_fault:
  clear sticky fault state
  allow new launch
```

V1 不支持 fault 后继续在同一 kernel 内恢复执行。

## 13. Compiler、runtime 和系统 contract

### 13.1 Compiler 责任

compiler 必须提供：

```text
- EVU-MT kernel binary
- launch descriptor template
- slot binding table
- register allocation
- structured control-flow legality
- dtype legality
- local memory access legality
- scatter conflict-free proof if scatter enabled
```

不允许把无法 static/legalize 的 CFG 或 memory access 硬塞给 EVU-MT。

### 13.2 Lowering flow

```text
High-level IR
  -> tile-local kernel extraction
  -> EVU-MT kernel IR
  -> legality check
  -> instruction selection
  -> register allocation
  -> instruction scheduling
  -> binary encoding
```

### 13.3 Runtime / Tile UCE 责任

runtime / Tile UCE 负责：

```text
- program residency
- launch descriptor patch
- slot frame binding
- event chain / wait dependency
- stream token and DMA orchestration
- timeout policy
- fault domain reset policy
```

EVU-MT 只消费已准备好的 `launch + code slot + slot table + local data`。

### 13.4 与 ELENOR 其他模块的边界

| 模块         | EVU-MT 看到的内容                                  | 不应看到的内容                 |
| ------------ | -------------------------------------------------- | ------------------------------ |
| Tile UCE     | launch / wait / event                              | tile-level graph schedule 内幕 |
| BOA          | shared local slot、event chain                     | dense micro-loop 细节          |
| MFE          | 已落地到 local slot 的 stream / metadata           | page walk / reorder 内部状态   |
| USE          | 已准备好的 state slot / scalar param               | checkpoint lifecycle           |
| Stream Queue | error/EOS 已通过 UCE 映射为 launch policy 或 fault | queue credit/head/tail 所有权  |

## 14. 配置、PPA 和性能模型

### 14.1 Base configuration 建议

推荐 V1 balanced profile：

```text
LANES:              32
LANE_REGS:          16
SCALAR_REGS:        16
PRED_REGS:          8
MASK_STACK_DEPTH:   4
INST_WIDTH:         64-bit
DATA_WIDTH:         32-bit lane datapath
LSU:                unit / strided / masked / gather
SCATTER:            disabled or conflict-free only
ATOMIC:             disabled
REDUCE:             sum / max / sumsq
SHUFFLE:            pair / split-half / RoPE pair
SFU:                exp / gelu / rsqrt / rcp approx
MEMORY_DOMAIN:      local slot memory only
```

明确：这些是建议 profile，不是芯片最终冻结值。面积、频率、bank 数、slot 布局仍由 SRAM profile 和 PPA exploration 冻结。

### 14.2 关键 PPA 路径

| 路径                     | 风险         | 缓解                                 |
| ------------------------ | ------------ | ------------------------------------ |
| predicate -> byte enable | fanout 大    | mask 分段寄存、局部 byte-enable 生成 |
| addr gen -> bank check   | 组合路径长   | addr gen 和 bank check 分拍          |
| RF read -> ALU -> WB     | Fmax 风险    | 3-stage datapath                     |
| branch mask update -> PC | 控制复杂     | structured branch only               |
| reduction tree           | 宽度增长     | hierarchical multi-cycle reduce      |
| shuffle crossbar         | 面积/时序    | 限制 pair/split-half/small transpose |
| gather lane remap        | tag/CAM 压力 | replay tag 小窗口、单 active kernel  |

### 14.3 Clock gating 建议

必须支持：

```text
- inactive lane gating
- unused RF bank gating
- IFETCH idle gating
- LSU idle gating
- SFU idle gating
- reduction idle gating
- shuffle idle gating
```

### 14.4 Performance model

kernel latency：

```text
T_kernel = T_launch
         + T_ifetch
         + T_decode
         + T_issue
         + T_rf
         + T_execute
         + T_lsu
         + T_replay
         + T_sfu
         + T_reduce
         + T_shuffle
         + T_writeback
         + T_commit
```

吞吐上界：

```text
Perf = min(
    issue_width * lanes * lane_eff * divergence_eff,
    LSU_BW * lsu_eff,
    SFU_throughput,
    reduce_throughput,
    shuffle_throughput
)
```

定义：

```text
lane_eff       = active_lanes / total_lanes
divergence_eff = useful_path_cycles / executed_path_cycles
lsu_eff        = requested_bytes / (requested_bytes + replay_bytes)
issue_eff      = issued_instructions / total_cycles
```

对于 tail-heavy workload，`lane_eff` 是首要限制项；对于 local gather，`lsu_eff` 和 `lsu_replay_cycles` 是首要限制项。

## 15. RTL 实现建议

### 15.1 设计原则

- predicate/mask engine 必须是 first-class datapath，不要把 tail 当作 software 填充问题。
- descriptor validator 尽量前置，把结构性错误挡在 launch 阶段。
- replay queue 和 lane mapping 恢复逻辑分离；replay 只负责重发，不重写语义。
- masked store 的 byte-enable 必须直接来自 effective-active mask。
- multi-cycle 单元统一 valid/ready 契约，避免每个单元各自定义 backpressure 语义。
- first-fault-wins 必须贯穿 IFETCH、DECODE、LSU、SFU、WB 和 COMMIT。

### 15.2 建议的内部接口切分

推荐把下面几条接口单独模块化：

```text
- decode -> execute control bus
- mask engine -> lane-enable / byte-enable bus
- LSU req/resp + replay bus
- writeback bus
- fault syndrome bus
- PMU stall owner bus
```

这样可以把 CDC、lint、SVA 和 waveform debug 范围收窄到边界上。

### 15.3 建议的 SVA 优先级

优先写的不是高层 softmax golden，而是以下不变量：

```text
- inactive lane 无 side effect
- fault record before event
- same cmd_id no success-after-fault
- replay preserves lane mapping
- PMU primary stall one-hot
- mask stack never underflow/overflow silently
```

## 16. 验证计划、bring-up 和验收标准

### 16.1 SVA / formal 重点

必须覆盖：

```text
- inactive lane 不发 load/store
- inactive lane 不写 lane RF
- inactive lane 不写 memory
- masked store byte enable 正确
- tail lane 不访问 OOB
- active lane OOB 必须 fault
- illegal opcode 必须 fault
- unsupported dtype 必须 fault
- invalid register index 必须 fault
- mask stack overflow/underflow 必须 fault
- replay 后 lane mapping 不变
- all-inactive reduction 行为确定
- fault 后只 commit fault event
- PMU primary stall 互斥
```

### 16.2 单元验证矩阵

| 类别       | 关键 case                                             |
| ---------- | ----------------------------------------------------- |
| launch     | bad version、bad size、bad permission、entry_pc OOB   |
| control    | laneid、uniform branch、if_mask/else_mask/endif、exit |
| arithmetic | int/fp add/mul/max/min、compare、select、cvt          |
| predicate  | compare 生成 predicate、pred.and/or/not、tail mask    |
| LSU        | unit、stride、masked、gather、optional scatter reject |
| replay     | bank conflict、partial issue、queue full、timeout     |
| reduction  | sum/max/sumsq、all-inactive identity                  |
| SFU        | exp/rsqrt/gelu approx latency 和 tolerance            |
| fault      | code OOB、illegal opcode、slot OOB、replay timeout    |
| PMU        | active/stall/replay counter 与构造场景匹配            |

### 16.3 Bring-up 顺序

```text
1. reset / idle CSR
2. launch accept / reject
3. code fetch
4. illegal opcode fault
5. exit commit
6. laneid / scalar load
7. lane add / store
8. predicate compare
9. tail mask and masked store
10. unit-stride load/store
11. strided load/store
12. local gather
13. bank conflict replay
14. reduce.sum / reduce.max / reduce.sumsq
15. SFU exp / rsqrt / gelu
16. shuffle pair / split-half / RoPE pair
17. structured if_mask / else_mask / endif
18. full softmax local kernel
19. full RMSNorm kernel
20. PMU correlation
21. random fault injection
22. tile integration with UCE event chain
```

### 16.4 Golden model

必须提供：

```text
- EVU-MT ISA interpreter
- launch descriptor validator reference model
- local slot memory model
- replay randomizer
- C++ functional simulator
- Python numerical golden for softmax/norm
- fault injection testbench
```

### 16.5 V1 验收标准

```text
1. EVU-MT 能正确执行 laneid + load + compute + store kernel。
2. mask/tail corner case 全部通过。
3. inactive lane 不产生 memory side effect。
4. active lane OOB 能精确 fault。
5. illegal opcode / unsupported dtype / invalid register 能 fault。
6. gather bank conflict replay 后 lane mapping 正确。
7. reduce.sum / reduce.max / reduce.sumsq 与 golden 一致。
8. softmax local kernel 与 numerical golden 一致。
9. RMSNorm kernel 与 numerical golden 一致。
10. PMU active/stall/replay attribution 可用且 primary stall 互斥。
11. fault record 完整且先于 fault event 可见。
12. kernel done/fault commit 都只出现一次。
```

## 17. 风险、取舍和后续冻结项

### 17.1 主要风险

| 风险                                 | 影响                       | 缓解                                                                    |
| ------------------------------------ | -------------------------- | ----------------------------------------------------------------------- |
| EVU-MT 演化成 GPU SM                 | 面积/功耗/验证失控         | 禁止 warp scheduler、per-lane PC、多 context                            |
| ISA 过大                             | compiler/RTL 复杂          | V1 保持小 ISA，优先 lowering 到 predicate + local memory + small reduce |
| gather bank conflict 高              | replay cycles 高           | bank-aware layout、index locality hint                                  |
| SFU 精度不冻结                       | golden 难对齐              | 先冻结 approximation envelope                                           |
| structured branch 复杂               | 控制路径复杂               | if-conversion 优先，mask stack depth 限制                               |
| scatter 过早启用                     | duplicate index 一致性复杂 | V1 默认关闭或只支持 conflict-free                                       |
| fault 后 partial write 语义不清      | debug 困难                 | first-fault-wins + fault-policy 明确                                    |
| 把 global memory latency 推给 EVU-MT | 性能不可控                 | EVU-MT 只服务 local slot memory                                         |

### 17.2 后续必须冻结的规范项

```text
- opcode 编码
- dtype / convert / rounding policy
- SFU approximation envelope
- all-inactive reduction identity
- slot layout / bank hint 编码
- launch descriptor exact binary layout
- fault_policy 编码
- PMU counter 编号和 snapshot 协议
- scatter duplicate policy
- fault record 二进制 layout
```

## 18. 结论

EVU-MT 的最终边界应保持为：

```text
EVU-MT = tile-local shared-PC microthread vector engine
       + predicated lane execution
       + local slot LSU
       + local reduction / shuffle / SFU
       + precise replay / fault / PMU

EVU-MT ≠ GPU SM
EVU-MT ≠ dense matrix engine
EVU-MT ≠ metadata walk engine
EVU-MT ≠ tile-level scheduler
```

这个边界对硬件最重要：**把 EVU-MT 做成一个可综合、可验证、可被 compiler 稳定 targeting 的 tile-local kernel engine，而不是把所有 irregular 需求都塞进一个失控的“万能小 GPU”。**
