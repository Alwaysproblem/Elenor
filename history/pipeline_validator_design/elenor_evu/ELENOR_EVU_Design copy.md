# EVU 设计文档

## 1. 定位、目标和 First Silicon cutline

EVU（Enhanced Vector Unit）是 ELENOR 的 predicated vector engine，是 BOA 的补充而不是小 GPU。EVU 面向不适合映射到 BOA、或天然带 mask、index、branch、tail、small reduction 的 kernel：elementwise、activation、softmax、normalization、RoPE、layout transform、gather/scatter、dynamic shape tail 和边界处理。

EVU 的核心语义是：

```text
RVV/SVE-like predicated vector
+ indexed / strided LSU
+ shuffle / permute
+ mask engine
+ small reduction
```

EVU 不采用完整 GPU SIMT：不做 warp scheduler、不做 per-thread PC、不做 reconvergence stack、不保存大量 thread context。硬件执行 Tile UCE 发起的 EVU descriptor 或 vector kernel command，不直接消费高层 graph。

First Silicon V1 cutline：

| 优先级 | 能力                                                     | First Silicon V1 要求                                           | 后续能力                         |
| ------ | -------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------- |
| P0     | unit-stride、strided、masked load/store、tail            | 必须支持 elementwise、mask/tail、softmax/norm 基础路径          | 更复杂 addressing 由后续规格冻结 |
| P1     | indexed gather、bank conflict replay、shuffle/permute    | 支持 paged attention、embedding、layout transform 的基础 gather | 更大 reorder window              |
| P2     | scatter、duplicate index handling、limited atomic update | 可预留 descriptor 和 RTL hook，不阻塞 V1                        | MoE combine、GNN/recommender     |
| P3     | cross-tile irregular access、complex micro-thread assist | 不进入 First Silicon V1                                         | V2/V3 研究                       |

## 2. 职责、非职责和 ownership

EVU 负责：

- predicated vector arithmetic：add、mul、max、compare、approx exp、activation 子集。
- mask/tail：dynamic shape tail、padding、attention mask、sparse valid lane。
- vector reduction：sum、max、small prefix、softmax max/sum。
- vector memory：unit-stride、strided、indexed gather；scatter 在 V1 可限制或关闭。
- shuffle/permute：softmax row transform、RoPE、layout transform。
- bank conflict detection 和 replay。
- fault/PMU：invalid descriptor、slot fault、address fault、LSU replay、active/stall attribution。

EVU 不负责：

- 大规模 dense matmul 主路径；该路径属于 BOA。
- page/segment metadata walk；该路径属于 MFE。
- Tile Program PC、branch、engine orchestration；该路径属于 Tile UCE。
- state lifecycle、checkpoint/restore；该路径属于 USE。
- 全局 scatter atomic 一致性；V1 不默认支持跨 tile unordered atomic update。

ownership 表：

| 对象                       | owner                | EVU 行为                                                                |
| -------------------------- | -------------------- | ----------------------------------------------------------------------- |
| vector descriptor 静态字段 | compiler             | 消费 op_kind、dtype、elements、slot、mask policy                        |
| dynamic shape branch       | runtime / Tile UCE   | EVU 只执行 descriptor 给出的 vector_len 和 mask                         |
| MFE stream metadata        | MFE                  | EVU 消费 stream slot 或 index slot，不做 metadata walk                  |
| softmax row ownership      | Tile UCE / compiler  | EVU 对本 tile row/chunk 负责，跨 tile reduce 由 Collective/BOA 上层处理 |
| duplicate scatter index    | descriptor mode      | 未声明时必须 fault 或走 ordered fallback                                |
| PMU primary stall          | EVU PMU / global PMU | active、wait operand、LSU replay、SRAM bank、stream credit 唯一归因     |

## 3. 微架构和状态机

### 3.1 Pipeline

```text
Vector Decode
    |
VL / Predicate / Mask Setup
    |
Address Generation
    |
Vector LSU / Replay Queue
    |
Vector ALU / Compare / Reduce
    |
Shuffle / Permute
    |
Writeback / Event Commit
```

内部模块：

| 模块                    | 职责                                                   | 关键设计点                                               |
| ----------------------- | ------------------------------------------------------ | -------------------------------------------------------- |
| Launch Frontend         | 接收 Tile UCE launch、读取 descriptor                  | command id、event id、descriptor cache                   |
| Descriptor Validator    | 检查 dtype、slot、elements、mask policy、reserved bits | 输出 fault record                                        |
| Vector Sequencer        | 生成 vector loop、VL、tail、micro-op                   | 不暴露 full per-lane PC                                  |
| Predicate / Mask Engine | 合成 active lane mask                                  | 支持 tail mask、input mask、compare mask、attention mask |
| Address Generator       | unit/stride/indexed 地址生成                           | base+stride+index，slot boundary check                   |
| Vector LSU              | L1 vector buffer 访问、gather/scatter 子集             | bank conflict detection、replay                          |
| ALU / Approx Unit       | add/mul/max/compare/activation/exp approx              | exp 精度由后续规格冻结                                   |
| Reduction Unit          | row max、row sum、small prefix                         | softmax/norm 主路径                                      |
| Shuffle / Permute       | lane reorder、RoPE、layout transform                   | crossbar 宽度由 PPA exploration 冻结                     |
| Writeback Unit          | 写 dst slot、更新 event                                | masked store 不写 inactive lane                          |
| PMU/Fault Unit          | active、stall、replay、fault 计数                      | primary owner 唯一                                       |

### 3.2 Predicated vector 语义

EVU 的每条 vector micro-op 都在 `VL` 和 predicate mask 下执行：

```text
active_lane[i] = (i < VL) && tail_mask[i] && input_mask[i] && op_mask[i]
```

语义规则：

- inactive lane 不发起 memory side effect，不更新 destination element，不参与 reduce。
- masked load 的 inactive lane result 取值由 mask policy 决定：zero、undisturbed 或 poison-for-debug，First Silicon V1 推荐 zero/undisturbed 子集，由后续规格冻结。
- masked store 的 inactive lane 必须不写 SRAM。
- compare 产生 mask register，不直接 branch；branch 仍由 Tile UCE/runtime 处理。
- reduction 只规约 active lane；全 inactive lane 的 max/sum identity 必须由 descriptor 或 op_kind 定义。
- tail 处理不能依赖软件填充安全值；硬件 predicate 必须阻断越界 load/store。

### 3.3 状态机

```text
IDLE
  |
  v
DESC_FETCH
  |
  v
VALIDATE
  | invalid
  +---------> FAULT_COMMIT -> IDLE
  |
  v
SETUP_VL_MASK
  |
  v
ISSUE_LOOP
  |
  +--> LSU_REQ -> LSU_REPLAY? -> LSU_RESP
  |
  +--> ALU_REDUCE
  |
  +--> SHUFFLE
  |
  v
WRITEBACK
  | more elements
  +-------------> SETUP_VL_MASK
  |
  v
EVENT_COMMIT
  |
  v
IDLE
```

LSU replay 状态：

```text
LSU_REQ -> BANK_CHECK -> ISSUE_BANKS -> COLLECT_RESP
              | conflict
              v
           REPLAY_QUEUE -> ISSUE_BANKS
```

Replay 必须保持 program order 对可见写入的影响；gather load 可乱序返回但写回 vector lane 时恢复 lane order。scatter 若启用，duplicate index 行为必须由 descriptor mode 明确。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Descriptor

通用 vector kernel descriptor：

```c
typedef struct {
    uint64_t code_addr;
    uint64_t arg_addr;
    uint32_t vector_len;
    uint32_t elem_bits;
    uint32_t mask_policy;
    uint32_t scratch_bytes;
    uint32_t flags;
} elenor_vector_kernel_desc_t;
```

First Silicon V1 更推荐 slot-based descriptor，降低 code fetch 和可编程复杂度：

```c
typedef struct {
    uint32_t op_kind;
    uint16_t src0_slot;
    uint16_t src1_slot;
    uint16_t src2_slot;
    uint16_t dst_slot;
    uint16_t mask_slot;
    uint16_t index_slot;

    uint32_t elements;
    uint16_t dtype;
    uint16_t elem_bits;
    uint16_t vector_lanes;
    uint16_t mask_policy;

    uint32_t src0_stride;
    uint32_t src1_stride;
    uint32_t dst_stride;
    uint32_t index_scale;

    uint16_t activation_kind;
    uint16_t reduce_op;
    uint16_t shuffle_kind;
    uint16_t fault_policy;

    uint32_t flags;
    uint32_t reserved;
} evu_desc_v0_t;
```

字段说明：

| 字段                 | 语义                                                                | V1 约束                                |
| -------------------- | ------------------------------------------------------------------- | -------------------------------------- |
| `op_kind`            | elementwise、activation、softmax phase、norm phase、gather、scatter | 枚举由后续规格冻结                     |
| `src*_slot/dst_slot` | Tile Slot Frame 索引                                                | role/permission 必须匹配               |
| `mask_slot`          | 可选输入 mask                                                       | 无 mask 时只用 tail mask               |
| `index_slot`         | indexed gather/scatter index                                        | V1 gather 优先，scatter 可禁用         |
| `elements`           | logical element count                                               | 0 非法；tail 由 VL/mask 处理           |
| `dtype/elem_bits`    | element 类型和宽度                                                  | INT8/BF16/FP16/FP32 子集由后续规格冻结 |
| `vector_lanes`       | lane 数或 profile id                                                | 16/32/64 lanes 由配置决定              |
| `mask_policy`        | inactive lane 行为                                                  | zero/undisturbed 子集                  |
| `stride/index_scale` | 地址生成参数                                                        | boundary check 必须启用                |
| `activation_kind`    | relu、gelu approx、exp approx 等                                    | exp/gelu 误差由后续规格冻结            |
| `reduce_op`          | sum、max、prefix 子集                                               | softmax/norm V1 必需                   |
| `shuffle_kind`       | none、permute、RoPE、transpose tile                                 | crossbar 子集由 PPA exploration 冻结   |
| `fault_policy`       | invalid addr、duplicate index、overflow                             | 默认 fail-fast                         |

### 4.2 协议

Tile UCE launch：

```text
launch.evu desc_slot, event_id
wait event_id
```

EVU 与 SRAM：

- request 携带 slot id、byte offset、byte enable、lane mask、op id。
- response 携带 lane data、fault bit、replay tag。
- masked lane 不产生 request byte enable。
- bank conflict 可 replay，但必须保留 lane-to-element mapping。

EVU 与 MFE：

- MFE 可将 page/segment stream 写入 stream slot 或 metadata slot。
- EVU 读取 MFE 产物做 softmax、gather result combine、segment reduce。
- EOS/error token 由 Stream Queue 或 event path 传播，EVU 遇到 stream error 必须停止写 dst 并 fault commit。

### 4.3 状态寄存器和 PMU

| 寄存器                  | 说明                                                                                    |
| ----------------------- | --------------------------------------------------------------------------------------- |
| `EVU_STATUS`            | idle、busy、fault、drain、replay_busy                                                   |
| `EVU_CMD_ID`            | 当前 command id                                                                         |
| `EVU_FAULT_CODE`        | invalid descriptor、slot fault、address fault、duplicate index、timeout、internal fault |
| `EVU_FAULT_LANE`        | 首个 faulting lane，若不可用则为由后续规格冻结                                          |
| `EVU_PMU_ACTIVE`        | active cycles                                                                           |
| `EVU_PMU_LSU_REPLAY`    | bank conflict / dependency replay cycles                                                |
| `EVU_PMU_MASKED_LANES`  | inactive lane 计数，用于 tail 效率                                                      |
| `EVU_PMU_STALL_OPERAND` | operand/stream wait                                                                     |
| `EVU_PMU_STALL_WB`      | writeback stall                                                                         |

## 5. 数据流、控制流和时序路径

### 5.1 数据流

```text
Tile L1 Slot Frame
  ├── vector buffer
  ├── mask slot
  ├── index slot
  ├── MFE stream slot
  └── output/workspace slot
        |
        v
EVU LSU -> Predicate -> ALU/Reduce/Shuffle -> Writeback
```

Softmax 典型三阶段：

```text
score tile -> mask/scale -> row max reduce -> exp(score-max) -> row sum reduce -> normalize -> output
```

Norm 典型路径：

```text
input row -> sum/sumsq reduce -> scale parameter -> normalize -> activation optional -> output
```

Paged attention 中 EVU 位于 BOA QK 与 BOA AV 之间，执行 scale、mask、softmax、tail handling。

### 5.2 Banking 交互

- EVU vector buffer 访问模式包括 unit、strided、indexed，必须支持 bank conflict detection 和 replay。
- `evu_lsu_replay` 是 EVU primary stall owner；如果 replay 由 SRAM bank conflict 触发，global PMU 可记录 secondary tag，但 primary 不重复计数。
- EVU 与 BOA 并发时，EVU 不应访问 BOA accumulator hot bank；compiler memory planner 通过 slot bank hint 避免冲突。
- MFE stream buffer 使用 ping-pong，EVU 读取时通过 credit 或 event 确认 producer 完成。

### 5.3 关键时序路径

| 路径                                     | 风险                    | 建议                                                       |
| ---------------------------------------- | ----------------------- | ---------------------------------------------------------- |
| Mask generation -> LSU byte enable       | fanout 到所有 lane      | mask 分段寄存，byte enable 局部生成                        |
| Indexed address -> bank select -> replay | 组合复杂                | 地址生成和 bank check 分拍                                 |
| Shuffle / permute crossbar               | 面积和时序高            | V1 限制 shuffle_kind，crossbar 宽度由 PPA exploration 冻结 |
| Reduction tree                           | softmax/norm row 宽增长 | 分层 reduce，多拍规约                                      |
| exp approximation                        | 精度和延迟              | table/polynomial 方案由后续规格冻结                        |

### 5.4 工作负载映射示例

| 工作负载                | EVU 映射                                                              | 协同模块                                      | 关键检查                                           |
| ----------------------- | --------------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------- |
| Softmax                 | scale/mask、row max、exp approximation、row sum、normalize            | BOA 提供 QK score，BOA 消费 softmax 输出做 AV | all-masked row、tail row、误差阈值由后续规格冻结   |
| RMSNorm / LayerNorm     | sum/sumsq reduce、scale、normalize、optional activation               | BOA 负责前后 dense projection                 | reduction identity、rounding 和 dtype convert 明确 |
| RoPE                    | shuffle/permute + elementwise sin/cos 组合                            | compiler 提供 layout 和参数 slot              | lane permutation 与 bank layout 不冲突             |
| Paged Attention mask    | 对 MFE/BOA 产出的 score tile 应用 page mask、causal mask 和 tail mask | MFE 提供 page metadata，BOA 执行 QK/AV        | inactive lane 不参与 exp 和 store                  |
| Embedding / MoE combine | indexed gather result 的 elementwise combine 或 ordered scatter 子集  | MFE Segment Stream 提供 coalesced stream      | duplicate index policy 必须 descriptor 化          |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置

| 参数          | First Silicon V1 建议                | 冻结方式                              |
| ------------- | ------------------------------------ | ------------------------------------- |
| lanes         | 32 lanes 可作为 Balanced 配置        | Edge/High End 由 PPA exploration 冻结 |
| vector buffer | 约 256 KB / Tile 示例                | 由 SRAM profile 冻结                  |
| LSU           | unit/strided/masked + indexed gather | scatter/atomic 由后续规格冻结         |
| reduce        | sum/max + small prefix               | 更复杂 segment reduce 由后续规格冻结  |
| shuffle       | RoPE/layout 基础 permute             | crossbar 能力由 PPA exploration 冻结  |

### 6.2 性能模型

```text
EVU_perf = min(EVU_ALU_peak, EVU_LSU_bw * useful_ops_per_byte, reduce_throughput, shuffle_throughput)
EVU_lane_eff = active_lanes / total_lanes
EVU_lsu_eff = requested_bytes / (requested_bytes + replay_bytes)
```

Softmax latency：

```text
T_softmax = T_mask_scale + T_row_max + T_exp + T_row_sum + T_normalize + T_writeback
```

Tail-heavy dynamic shape 的有效吞吐由 `EVU_lane_eff` 决定；PMU 必须能区分正常 tail 效率下降和 LSU replay 导致的 stall。

### 6.3 PMU

必需 counter：

- `evu_active_cycles`。
- `evu_lsu_replay_cycles`。
- `evu_stall_operand`。
- `evu_stall_writeback`。
- `evu_masked_lane_count`。
- `evu_reduction_active_cycles`。
- `evu_shuffle_active_cycles`。
- `evu_stream_wait_cycles`。
- `evu_fault_count_by_type`。

PMU 指纹示例：

- softmax/norm：reduction active 高，LSU replay 低，masked lane 由 tail 决定。
- paged attention：EVU active 位于 BOA QK 与 BOA AV 之间；若 MFE 预取不足，EVU 前后 event wait 增加。
- embedding gather：LSU replay 和 bank conflict 上升，ALU active 低。

## 7. RTL/软件实现建议

### 7.1 RTL

- predicate/mask engine 必须是 first-class datapath，不要把 tail 当作软件填充问题。
- LSU replay queue 与 vector lane mapping 分离：replay 只重发冲突 bank，不改变 lane result 顺序。
- masked store 的 byte enable 必须由 active lane 直接控制，SVA 覆盖 inactive lane 不写。
- reduction tree 支持 identity value 注入，避免全 mask 行产生未定义值。
- exp/activation approximation 与 ALU 主路径解耦，允许多拍 latency。
- scatter/atomic hook 可以保留接口，但 First Silicon V1 未启用时 descriptor 必须 fault，而不是 silent no-op。
- clock gating 根据 lane mask、op_kind、LSU idle、shuffle idle 分域处理。

### 7.2 软件和 compiler

- compiler 将 elementwise、activation、softmax、norm、RoPE、tail path lowering 到 EVU descriptor。
- dynamic shape 通过 runtime branch 选择 descriptor 或调整 `elements`，EVU 只按 descriptor 执行。
- Paged Attention lowering：MFE page stream -> BOA QK -> EVU scale/mask/softmax -> BOA AV。
- MoE combine 若需要 scatter/duplicate update，V1 应通过 runtime 分桶、ordered output 或 group collective fallback，不能假定 EVU atomic。
- descriptor ABI 预留 `fault_policy`，使 unsupported scatter/atomic fail-fast。

## 8. 验证、bring-up 和验收标准

### 8.1 RTL/SVA 重点

- inactive lane 不发 load/store，不写 dst，不参与 reduce。
- `VL < lanes` 的 tail case 不越界访问 slot。
- unit/strided/indexed 地址边界检查覆盖首元素、末元素、跨 slot 越界。
- bank conflict replay 后 lane order 保持。
- all-masked reduction 返回 descriptor 定义的 identity 或 fault，不产生随机值。
- masked store byte enable 与 active mask 一致。
- duplicate scatter index 在未启用 mode 下 fault；启用 ordered mode 时结果可复现。
- PMU active、wait operand、LSU replay、writeback stall primary 互斥。

### 8.2 Bring-up 顺序

1. EVU descriptor validation 和 event/fault path。
2. unit-stride elementwise add/mul。
3. mask/tail corner：0 elements 非法、1 element、VL-1、VL、VL+1。
4. activation 和 compare mask。
5. row max/sum reduce。
6. softmax/norm random tensor golden。
7. strided load/store。
8. indexed gather + bank conflict replay。
9. 与 MFE/BOA 串联的 paged attention command trace。
10. scatter/duplicate index 只在启用 profile 下验证，否则验证 fail-fast。

### 8.3 验收标准

- softmax/norm random tensor test 通过，误差阈值由后续规格冻结。
- mask/tail corner case 全通过，硬件不依赖输入 padding。
- EVU PMU active/stall/replay counter 可用且唯一归因。
- Paged attention trace 中 EVU scale/mask/softmax 与 BOA QK/AV event chain 正确。
- invalid descriptor、slot fault、address fault、duplicate scatter fault 都能生成 fault record。

## 9. 风险、取舍和后续细化方向

| 风险                              | 影响                        | 缓解                                                   |
| --------------------------------- | --------------------------- | ------------------------------------------------------ |
| EVU 演化成 GPU SIMT               | 面积和验证失控，与 BOA 重叠 | 保持 predicated vector，不引入 per-thread PC           |
| scatter/atomic 过早实现           | 一致性和重复 index 复杂     | V1 gather 优先，scatter 需要 descriptor mode 明确      |
| shuffle crossbar 过宽             | 时序和面积恶化              | 限制 V1 shuffle 子集，RoPE/layout 优先                 |
| exp/gelu 精度未冻结               | golden 难对齐               | 规格冻结 approximation、rounding 和 tolerance          |
| bank conflict replay 影响可复现性 | gather/scatter 结果不稳定   | replay 保持 lane order，duplicate policy descriptor 化 |

后续需要冻结：EVU op_kind 编号、dtype/elem_bits 子集、mask policy、all-masked reduction identity、exp/activation approximation、scatter/atomic 一致性、PMU counter 编号和读取协议。

## 修改后的建议：

## 2. 职责、非职责和 Ownership

EVU-MT（Enhanced Vector Unit - Microthread Engine）是一个 **shared-PC microthread vector execution unit**。它通过 SIMT-like lane 编程模型执行 tile-local vector kernel，但不引入 GPU warp scheduler、per-thread PC 或 multi-warp residency。

EVU-MT 的核心职责是：

```text
predicated lane execution
+ local vector arithmetic
+ mask / tail handling
+ local reduction
+ unit / strided / indexed local memory access
+ limited shuffle / permute
+ bank conflict replay
+ precise fault / PMU attribution
```

---

### 2.1 EVU-MT 负责

#### 2.1.1 Predicated Microthread Arithmetic

EVU-MT 负责在 active lane mask 下执行基础 vector arithmetic：

```text
- integer add / sub / mul / max / min
- floating add / mul / max / min
- compare
- select
- dtype convert
- approximate exp / rsqrt / reciprocal
- activation subset: relu / gelu approx / silu approx optional
```

执行语义：

```text
每条指令在 effective_active mask 下执行。
inactive lane 不更新 lane register，不参与 memory side effect。
```

典型 kernel：

```c
tid = block_base + lane_id;

if (tid < N) {
    y[tid] = gelu(x[tid] * scale + bias);
}
```

---

#### 2.1.2 Mask / Tail / Predicate Handling

EVU-MT 负责所有 lane-level predicate 语义：

```text
- dynamic shape tail
- padding tail
- attention mask
- causal mask
- sparse valid lane
- compare-generated predicate
- structured branch active mask
```

effective mask 定义：

```text
effective_active[i] =
    lane_valid_mask[i]
  & active_mask[i]
  & instruction_predicate[i]
```

规则：

```text
- tail lane 不发 load/store
- tail lane 不参与 reduction
- tail lane 不更新 destination
- inactive lane 不触发 address fault
- masked store 的 inactive lane byte enable 必须为 0
```

EVU-MT 不依赖软件 padding 来保证越界安全，tail 必须由硬件 predicate 阻断。

---

#### 2.1.3 Local Vector Reduction

EVU-MT 负责 tile-local small reduction：

```text
- reduce.sum
- reduce.max
- reduce.sumsq
- optional small prefix
- softmax max / sum phase
- norm sum / sumsq phase
```

reduction 语义：

```text
- 只规约 active lane
- inactive lane 不参与 reduction
- all-inactive reduction 的行为必须由 numerical policy 定义
- BF16 / FP16 reduction 推荐使用 FP32 accumulate
```

典型用途：

```text
- softmax row max
- softmax row sum
- RMSNorm sumsq
- LayerNorm sum / sumsq
- local segment reduce
```

EVU-MT 只负责本单元内部可见 lane group 的局部 reduction。跨 tile、跨 group 或全局 reduction 不属于 EVU-MT 的职责。

---

#### 2.1.4 Local Vector Memory Access

EVU-MT 负责本地 slot memory 上的 vector memory operation：

```text
- unit-stride load/store
- strided load/store
- masked load/store
- indexed gather
- conflict-free scatter optional
```

memory address 必须是 local slot-relative address：

```text
slot_id + byte_offset
```

EVU-MT 不直接访问 virtual address、HBM physical address 或 page table。

gather 语义：

```text
idx[i]  = index_slot[lane_id[i]]
addr[i] = data_slot_base + base_offset + idx[i] * index_scale
```

限制：

```text
- gather 只面向 local memory domain
- active lane OOB 必须 fault
- inactive lane 不发 index load，不触发 OOB fault
- index dtype base profile 推荐 u32
- index_scale 推荐限制为 {1, 2, 4, 8, 16}
```

scatter 语义：

```text
- V1 可关闭 scatter
- 或仅支持 compiler/runtime 已证明 conflict-free 的 scatter
- duplicate index 默认 fault 或走 fallback
```

EVU-MT 不负责 unordered scatter atomic consistency。

---

#### 2.1.5 Shuffle / Permute

EVU-MT 负责有限 lane shuffle：

```text
- even/odd pair shuffle
- split-half pair shuffle
- RoPE pair transform support
- optional 8x8 / 16x16 small transpose
```

典型用途：

```text
- RoPE
- local layout pack / unpack
- softmax row local transform
- small tile transpose
```

EVU-MT 不建议在基础版本中实现 arbitrary full crossbar permute。复杂任意 permutation 会显著增加面积、时序和验证成本。

---

#### 2.1.6 Bank Conflict Detection and Replay

EVU-MT 负责本地 memory access 的 bank conflict detection 和 replay：

```text
- detect multi-lane bank conflict
- partial issue non-conflict lanes
- replay conflict lanes
- preserve logical lane mapping
- track replay cycles in PMU
```

规则：

```text
- replay 不改变 lane logical order
- gather response 可以乱序返回，但 writeback 必须恢复 lane mapping
- store replay 必须保持可见写入顺序
- replay queue full 时 backpressure pipeline
- replay timeout 必须 fault
```

EVU-MT 的 replay 只解决 local SRAM / local slot memory 的 bank conflict 或 structural hazard，不用于隐藏 HBM long latency。

---

#### 2.1.7 Instruction Execution and Shared-PC Control

EVU-MT 负责执行 microthread kernel ISA：

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

控制流约束：

```text
- 所有 lane 共享一个 PC
- 不支持 per-lane PC
- 不支持 GPU-style arbitrary divergence
- 小分支优先 if-conversion
- 支持 structured if_mask / else_mask / endif
- uniform branch 只允许 scalar condition
```

EVU-MT 内部可以有 branch unit，但不负责 Tile Program PC，也不负责 engine-level orchestration。

---

#### 2.1.8 Fault and PMU

EVU-MT 负责自身内部 fault detection、fault record 和 PMU attribution：

fault 类型包括：

```text
- invalid launch descriptor
- code slot fault
- illegal opcode
- unsupported dtype
- invalid register index
- slot permission fault
- active lane address OOB
- mask stack overflow / underflow
- replay timeout
- internal fault
```

fault record 至少包含：

```text
- cmd_id
- event_id
- fault_code
- fault_pc
- fault_lane
- fault_slot
- fault_addr
```

PMU 至少包含：

```text
- active cycles
- instruction count
- issue cycles
- ifetch stall
- decode stall
- LSU active cycles
- LSU replay cycles
- replay queue full cycles
- SFU active cycles
- reduction active cycles
- shuffle active cycles
- writeback stall cycles
- masked lane count
- fault count by type
```

PMU primary stall attribution 必须互斥，避免重复计数。

---

### 2.2 EVU-MT 不负责

#### 2.2.1 不负责 Dense Matmul / Conv 主路径

EVU-MT 不负责大规模 dense compute：

```text
- GEMM
- CONV
- large tensor contraction
- dense QK / AV matmul
```

这些属于专用 dense compute datapath，例如 BOA 或其他 contraction engine。

EVU-MT 可以负责 dense compute 前后的局部处理，例如：

```text
- bias / activation
- mask
- softmax local phase
- norm local phase
- RoPE
- layout local transform
```

但不应成为 dense matmul 的主执行单元。

---

#### 2.2.2 不负责 Page / Segment Metadata Walk

EVU-MT 不负责：

```text
- page table walk
- segment descriptor walk
- KV cache block table walk
- pointer chasing metadata traversal
- global address translation
```

EVU-MT 只消费已经准备好的 local slot：

```text
- data slot
- index slot
- mask slot
- parameter slot
- scratch slot
```

如果 workload 需要复杂 metadata walk，应由外部 memory flow / stream preparation 逻辑完成。EVU-MT 不直接把 metadata walk 内建到 lane program 中。

---

#### 2.2.3 不负责 Tile Program PC / Engine Orchestration

EVU-MT 的 PC 只表示 EVU-MT kernel 内部的 shared PC。

EVU-MT 不负责：

```text
- Tile Program PC
- engine-level command scheduling
- BOA / MFE / EVU 之间的整体调度
- producer-consumer event graph construction
- graph-level branch
- runtime-level dynamic shape branch
```

EVU-MT 只接收 launch，执行 kernel，然后 commit event 或 fault。

---

#### 2.2.4 不负责 State Lifecycle / Checkpoint / Restore

EVU-MT 不负责：

```text
- state lifecycle management
- checkpoint
- restore
- persistent model state ownership
- cross-kernel state migration
- cross-tile state consistency
```

EVU-MT 的内部 architectural state 只在当前 kernel 执行期间有效：

```text
- shared PC
- scalar registers
- lane registers
- predicate registers
- active mask
- mask stack
```

kernel 完成或 fault 后，该状态不保证对外可见，除非通过 debug/CSR 明确读取。

---

#### 2.2.5 不负责 Global Scatter Atomic Consistency

EVU-MT V1 不默认支持：

```text
- cross-tile unordered atomic add
- global scatter atomic max/min
- duplicate index unordered update
- global memory atomic consistency
- multi-producer write conflict resolution
```

如果启用 scatter，必须满足以下之一：

```text
- compiler/runtime 证明 conflict-free
- descriptor / launch policy 声明 ordered mode
- duplicate index 触发 fault
- fallback 到外部 ordered path
```

EVU-MT 不保证 unordered duplicate scatter 的确定性。

---

#### 2.2.6 不负责 Long-Latency Memory Hiding

EVU-MT 没有 warp scheduler，因此不负责通过多 warp 切换隐藏 long-latency memory miss。

EVU-MT 不适合直接执行：

```text
- random HBM gather
- linked-list pointer chasing
- cache-miss-heavy graph traversal
- unpredictable global load stream
```

EVU-MT 的性能前提是：

```text
外部逻辑提前准备数据到 local memory。
EVU-MT 只处理 local memory access、bank conflict replay 和 short-latency stall。
```

---

### 2.3 Ownership Table

| 对象 / 行为                   | Owner                         | EVU-MT 行为                                    |
| ----------------------------- | ----------------------------- | ---------------------------------------------- |
| microthread kernel binary     | compiler                      | EVU-MT 负责 fetch / decode / execute           |
| EVU-MT launch descriptor      | runtime / launch controller   | EVU-MT 负责 validate 和启动                    |
| shared PC                     | EVU-MT                        | 仅表示 EVU-MT kernel 内部 PC                   |
| lane register file            | EVU-MT                        | kernel 内部 per-lane temporary state           |
| scalar register file          | EVU-MT                        | kernel 内部 uniform state                      |
| predicate register file       | EVU-MT                        | lane mask state                                |
| active mask                   | EVU-MT                        | 控制当前有效 lane                              |
| tail mask                     | EVU-MT                        | 根据 logical_elements / block_base 生成        |
| structured mask stack         | EVU-MT                        | 支持 if_mask / else_mask / endif               |
| local arithmetic              | EVU-MT                        | 执行 add / mul / max / compare / activation    |
| local reduction               | EVU-MT                        | 执行 sum / max / sumsq / softmax max-sum phase |
| local memory load/store       | EVU-MT                        | 访问 local slot memory                         |
| indexed gather                | EVU-MT                        | 仅对 local slot memory 生效                    |
| scatter conflict policy       | compiler/runtime + EVU-MT     | 未声明 conflict-free 时 EVU-MT fault           |
| duplicate scatter index       | compiler/runtime              | EVU-MT 不默认解决 unordered duplicate          |
| bank conflict replay          | EVU-MT                        | 检测、replay、PMU 归因                         |
| memory metadata walk          | external memory flow logic    | EVU-MT 不做 walk                               |
| dense matmul / conv           | dense compute engine          | EVU-MT 不负责                                  |
| Tile-level command scheduling | external controller           | EVU-MT 不负责                                  |
| event dependency graph        | external controller / runtime | EVU-MT 只 commit event                         |
| state lifecycle               | external state engine         | EVU-MT 不负责                                  |
| fault detection               | EVU-MT                        | 检测本单元 fault                               |
| fault recovery policy         | external controller + EVU-MT  | EVU-MT 按 policy drain/kill                    |
| PMU primary attribution       | EVU-MT                        | 负责本单元 active/stall/replay 归因            |

---

### 2.4 EVU-MT 的输入输出边界

#### 输入

```text
- launch descriptor
- instruction memory response
- local data memory response
- CSR/debug request
```

#### 输出

```text
- instruction memory request
- local data memory request
- event commit
- fault record
- PMU counters
- CSR/debug response
```

EVU-MT 不假设输入数据来自哪里，也不假设输出数据被谁消费。它只保证：

```text
在给定 launch descriptor、kernel binary 和 local memory response 的条件下，
按照 EVU-MT ISA 和 predicated lane semantics 正确执行 kernel。
```

---

### 2.5 设计边界总结

EVU-MT 的职责边界可以浓缩为：

```text
EVU-MT 负责：
  tile-local shared-PC microthread vector execution

EVU-MT 不负责：
  system-level scheduling, metadata walk, dense compute, global atomic consistency
```

更具体地说：

```text
EVU-MT = SIMT-like lane programming model
       + predicated vector datapath
       + local indexed LSU
       + local reduction/shuffle/SFU
       + local replay/fault/PMU

EVU-MT ≠ GPU SM
EVU-MT ≠ dense matrix engine
EVU-MT ≠ memory flow engine
EVU-MT ≠ tile-level scheduler
```

# EVU-MT 设计文档

## 1. 定位、目标和 First Silicon cutline

EVU-MT（Enhanced Vector Unit - Microthread Engine）是 ELENOR Tile 内部的 **tile-local shared-PC microthread vector engine**。它面向 BOA 不适合处理、但又不值得引入完整 GPU SIMT 的 kernel：

- elementwise fusion
- activation
- softmax local phase
- normalization
- RoPE
- layout pack / unpack
- attention mask
- dynamic shape tail
- indexed gather
- small reduction
- sparse valid lane
- conflict-free scatter subset

EVU-MT 采用：

```text
Shared-PC Microthread Execution
+ SIMT-like lane_id programming model
+ predicated lane execution
+ local indexed / strided LSU
+ local reduction / shuffle / SFU
+ bank conflict replay
- warp scheduler
- per-thread PC
- multi-warp residency
- global memory programming model
```

EVU-MT 不是小 GPU SM。它只拿 SIMT 的 **lane programming model**，不拿 GPU 的 **warp scheduling machinery**。

核心语义：

```text
one EVU-MT kernel
    ↓
one shared PC
    ↓
N lanes execute same instruction
    ↓
each lane owns lane-local registers
    ↓
predicate mask controls active lanes
```

---

## 1.1 设计目标

EVU-MT 的设计目标是：

```text
1. 提供比 descriptor-only vector engine 更强的长尾算子表达能力。
2. 保留 predicated vector datapath 的能效优势。
3. 避免 GPU warp scheduler、per-thread PC、多 context 带来的面积和验证复杂度。
4. 只处理 tile-local memory access，不直接承担 HBM random latency hiding。
5. 支持 local gather、mask、tail、small reduction、RoPE、softmax/norm 等 AI workload 常见非 dense 路径。
6. 为 compiler 提供一个小型 EVU-MT ISA target，而不是为每类算子不断扩硬件 descriptor。
```

---

## 1.2 First Silicon V1 cutline

| 优先级 | 能力                            | First Silicon V1 要求                                | 后续能力                 |
| ------ | ------------------------------- | ---------------------------------------------------- | ------------------------ |
| P0     | shared-PC microthread execution | 必须支持 lane_id、shared PC、exit、basic issue       | 更复杂 structured branch |
| P0     | predicate / tail                | 必须支持 lane_valid、active_mask、predicate register | nested mask stack 扩展   |
| P0     | unit-stride load/store          | 必须支持 masked load/store                           | 更宽 LSU                 |
| P0     | basic arithmetic                | add、mul、max、min、compare、select                  | fma、bitwise             |
| P0     | local reduction                 | sum、max、sumsq                                      | prefix / segment reduce  |
| P0     | fault / PMU                     | illegal opcode、OOB、slot fault、PMU active/stall    | per-kernel snapshot      |
| P1     | strided load/store              | 支持 layout/norm/softmax 基础路径                    | 多维 stride              |
| P1     | indexed gather                  | 支持 local memory gather、bank replay                | reorder / coalesce hint  |
| P1     | SFU                             | exp.approx、rsqrt.approx、gelu.approx                | 更高精度 approximation   |
| P1     | shuffle                         | pair shuffle、split-half shuffle                     | small transpose          |
| P2     | structured mask branch          | if_mask / else_mask / endif                          | 更深 mask stack          |
| P2     | conflict-free scatter           | 可选；必须由 compiler/runtime 证明无 duplicate       | ordered scatter          |
| P3     | atomic scatter                  | 不进入 V1                                            | V2/V3 研究               |

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

---

# 2. 职责、非职责和 ownership

## 2.1 EVU-MT 负责

EVU-MT 负责以下单元内部能力：

### 2.1.1 Predicated microthread arithmetic

EVU-MT 负责在 lane predicate 下执行：

```text
- integer add / sub / mul / max / min
- floating add / mul / max / min
- compare
- select
- dtype convert
- approximate exp
- approximate rsqrt / reciprocal
- activation subset: relu / gelu.approx / silu.approx optional
```

每条 lane instruction 在 `effective_active` 下执行。

```text
effective_active[i] =
    lane_valid_mask[i]
  & active_mask[i]
  & instruction_predicate[i]
```

inactive lane 规则：

```text
- 不更新 lane register
- 不发起 memory request
- 不写 local memory
- 不参与 reduction
- 不触发 address fault
```

---

### 2.1.2 Mask / tail / predicate

EVU-MT 负责：

```text
- dynamic shape tail
- padding tail
- attention mask
- sparse valid lane
- compare-generated predicate
- input mask load
- structured mask stack
```

tail mask 由硬件生成：

```text
lane_valid_mask[i] = (block_base + i) < logical_elements
```

tail 不依赖软件 padding 保证安全。

---

### 2.1.3 Local vector reduction

EVU-MT 负责本 lane group 内部的 small reduction：

```text
- reduce.sum
- reduce.max
- reduce.sumsq
- small prefix optional
- softmax max / sum phase
- RMSNorm / LayerNorm sum / sumsq phase
```

规则：

```text
- 只规约 active lane
- inactive lane 不参与 reduction
- all-inactive reduction 行为由 numerical policy 定义
- BF16/FP16 reduction 推荐使用 FP32 accumulate
```

EVU-MT 不负责跨 tile、跨 group 或全局 reduction。

---

### 2.1.4 Local vector memory

EVU-MT 负责 local slot memory 上的：

```text
- unit-stride load/store
- strided load/store
- masked load/store
- indexed gather
- conflict-free scatter optional
```

memory address 是 slot-relative：

```text
slot_id + byte_offset
```

EVU-MT 不直接访问：

```text
- virtual address
- physical HBM address
- page table
- global pointer
```

indexed gather 规则：

```text
idx[i]  = index_slot[lane_id[i]]
addr[i] = data_slot_base + base_offset + idx[i] * index_scale
```

限制：

```text
- gather 只访问 local memory domain
- index dtype V1 推荐 u32
- index_scale ∈ {1, 2, 4, 8, 16}
- active lane OOB 必须 fault
- inactive lane 不加载 index，不触发 OOB
```

---

### 2.1.5 Shuffle / permute

EVU-MT 负责有限 lane shuffle：

```text
- even/odd pair shuffle
- split-half pair shuffle
- RoPE pair transform
- optional 8x8 / 16x16 small transpose
```

EVU-MT V1 不建议支持 arbitrary full crossbar permute。

---

### 2.1.6 Bank conflict detection and replay

EVU-MT 负责 local memory access 的 bank conflict 检测和 replay：

```text
- 检测 multi-lane bank conflict
- partial issue non-conflict lanes
- conflict lanes 进入 replay queue
- gather response 可乱序返回
- writeback 必须恢复 lane mapping
- PMU 记录 replay cycles
```

replay 只解决 local SRAM / local slot memory 的短延迟冲突，不用于隐藏 HBM long latency。

---

### 2.1.7 Fault / PMU

EVU-MT 负责本单元内部：

```text
- invalid launch descriptor
- code OOB
- illegal opcode
- unsupported dtype
- invalid register index
- slot permission fault
- active lane address OOB
- mask stack overflow / underflow
- replay timeout
- internal fault
```

PMU 负责：

```text
- active cycles
- instruction count
- issue cycles
- ifetch stall
- decode stall
- LSU active cycles
- LSU replay cycles
- replay queue full cycles
- SFU active cycles
- reduction active cycles
- shuffle active cycles
- writeback stall cycles
- masked lane count
- fault count by type
```

PMU primary stall attribution 必须互斥。

---

## 2.2 EVU-MT 不负责

EVU-MT 不负责：

```text
- 大规模 dense matmul / conv 主路径
- page / segment metadata walk
- KV cache block table walk
- HBM random pointer chasing
- Tile Program PC
- engine orchestration
- state lifecycle / checkpoint / restore
- global unordered scatter atomic consistency
- cross-tile atomic update
- graph-level dynamic branch
```

具体边界：

| 非职责               | 原因                                                |
| -------------------- | --------------------------------------------------- |
| dense matmul / conv  | 属于 dense compute engine                           |
| metadata walk        | 属于外部 memory flow / stream preparation           |
| Tile Program PC      | EVU-MT 只有 kernel 内部 shared PC                   |
| engine orchestration | EVU-MT 只接收 launch，commit event/fault            |
| state lifecycle      | EVU-MT 内部 state 只在当前 kernel 有效              |
| global atomic        | 无全局一致性模型，不支持 unordered duplicate update |
| HBM latency hiding   | 无 warp scheduler，依赖外部数据准备和 local memory  |

---

## 2.3 Ownership 表

| 对象 / 行为               | Owner                       | EVU-MT 行为                             |
| ------------------------- | --------------------------- | --------------------------------------- |
| microthread kernel binary | compiler                    | EVU-MT fetch / decode / execute         |
| launch descriptor         | runtime / launch controller | EVU-MT validate and start               |
| shared PC                 | EVU-MT                      | 只表示 EVU-MT kernel 内部 PC            |
| lane register file        | EVU-MT                      | per-lane temporary state                |
| scalar register file      | EVU-MT                      | uniform state                           |
| predicate register file   | EVU-MT                      | lane mask state                         |
| active mask               | EVU-MT                      | 当前有效 lane                           |
| tail mask                 | EVU-MT                      | 根据 block_base / logical_elements 生成 |
| structured mask stack     | EVU-MT                      | 支持 if_mask / else_mask / endif        |
| local arithmetic          | EVU-MT                      | 执行 ALU / SFU                          |
| local reduction           | EVU-MT                      | sum / max / sumsq / softmax phase       |
| local memory access       | EVU-MT                      | load/store/gather                       |
| bank conflict replay      | EVU-MT                      | 检测、replay、PMU attribution           |
| metadata walk             | external                    | EVU-MT 不做                             |
| dense compute             | external dense engine       | EVU-MT 不做                             |
| global scheduling         | external controller         | EVU-MT 不做                             |
| state lifecycle           | external state engine       | EVU-MT 不做                             |
| global scatter atomic     | external ordered path       | EVU-MT V1 不做                          |

---

# 3. EVU-MT 顶层接口

## 3.1 Top-level SystemVerilog 草案

```systemverilog
module evu_mt #(
    parameter int LANES              = 32,
    parameter int LANE_REGS          = 16,
    parameter int SCALAR_REGS        = 16,
    parameter int PRED_REGS          = 8,
    parameter int MASK_STACK_DEPTH   = 4,

    parameter int DATA_WIDTH         = 32,
    parameter int INST_WIDTH         = 64,
    parameter int ADDR_WIDTH         = 32,
    parameter int SLOT_ID_WIDTH      = 16,
    parameter int CMD_ID_WIDTH       = 16,
    parameter int EVENT_ID_WIDTH     = 16
) (
    input  logic clk,
    input  logic rst_n,

    // Launch interface
    input  logic                  launch_valid,
    output logic                  launch_ready,
    input  evu_mt_launch_req_t    launch_req,

    // Commit interface
    output logic                  commit_valid,
    input  logic                  commit_ready,
    output evu_mt_commit_t        commit,

    // Instruction memory interface
    output logic                  imem_req_valid,
    input  logic                  imem_req_ready,
    output evu_mt_imem_req_t      imem_req,
    input  logic                  imem_resp_valid,
    output logic                  imem_resp_ready,
    input  evu_mt_imem_resp_t     imem_resp,

    // Local data memory interface
    output logic                  dmem_req_valid,
    input  logic                  dmem_req_ready,
    output evu_mt_dmem_req_t      dmem_req,
    input  logic                  dmem_resp_valid,
    output logic                  dmem_resp_ready,
    input  evu_mt_dmem_resp_t     dmem_resp,

    // CSR / debug interface
    input  logic                  csr_req_valid,
    output logic                  csr_req_ready,
    input  evu_mt_csr_req_t       csr_req,
    output logic                  csr_resp_valid,
    input  logic                  csr_resp_ready,
    output evu_mt_csr_resp_t      csr_resp
);
```

---

## 3.2 输入输出边界

EVU-MT 输入：

```text
- launch request
- instruction memory response
- local data memory response
- CSR/debug request
```

EVU-MT 输出：

```text
- instruction memory request
- local data memory request
- event/fault commit
- PMU / CSR response
```

EVU-MT 不关心：

```text
- launch 来自哪个上级调度器
- local memory 数据由谁准备
- output 被谁消费
- graph 如何 partition
- tile 如何调度
```

---

# 4. Launch Descriptor

EVU-MT 只接受 kernel launch descriptor。
descriptor 不描述具体算子，只描述如何启动一个 EVU-MT microthread kernel。

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

字段说明：

| 字段               | 说明                                |
| ------------------ | ----------------------------------- |
| `version`          | descriptor version                  |
| `cmd_id`           | 当前 kernel command id              |
| `event_id`         | commit event id                     |
| `code_slot`        | kernel binary 所在 local slot       |
| `arg_slot`         | kernel 参数区                       |
| `scratch_slot`     | scratch memory                      |
| `slot_table_slot`  | kernel 可见 slot binding table      |
| `num_lanes`        | 本次启动使用 lane 数                |
| `max_lane_regs`    | kernel 声明的 lane register 数      |
| `max_scalar_regs`  | kernel 声明的 scalar register 数    |
| `max_pred_regs`    | kernel 声明的 predicate register 数 |
| `entry_pc`         | kernel 入口 PC                      |
| `code_size_bytes`  | code 区大小                         |
| `logical_elements` | 逻辑元素数，用于 tail               |
| `block_base`       | 当前 lane group 的 logical base     |
| `fault_policy`     | fault 后 drain / kill 策略          |

V1 建议：

```text
- num_lanes <= LANES
- max_lane_regs <= LANE_REGS
- max_scalar_regs <= SCALAR_REGS
- max_pred_regs <= PRED_REGS
- entry_pc 必须 instruction aligned
- code_size_bytes 必须覆盖 entry_pc
```

---

# 5. 内部模块和微架构

## 5.1 模块划分

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

---

## 5.2 Pipeline

推荐 pipeline：

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

### S0: Launch / Init

职责：

```text
- 接收 launch request
- 检查 descriptor 基本合法性
- 初始化 cmd_id / event_id
- 初始化 PC
- 初始化 lane_valid_mask
- 清空 active_mask / pred_reg / mask_stack
- 进入 IFETCH
```

### S1: IFETCH

职责：

```text
- 根据 PC 请求 instruction word
- 检查 PC alignment
- 检查 code range
```

fault：

```text
- code OOB
- unaligned PC
- imem response error
```

### S2: DECODE

职责：

```text
- opcode decode
- src/dst register index decode
- dtype decode
- immediate decode
- instruction legality check
```

fault：

```text
- illegal opcode
- unsupported dtype
- invalid register index
```

### S3: Predicate + RF Read

职责：

```text
- 计算 effective_active
- 读取 scalar RF
- 读取 lane RF
- 读取 predicate RF
```

### S4: Execute / Addr Gen

职责：

```text
- ALU operation
- compare
- branch mask update
- memory address generation
```

### S5: LSU / SFU / Reduce / Shuffle

职责：

```text
- local memory request
- gather issue
- bank conflict replay
- SFU multi-cycle operation
- reduction tree
- shuffle network
```

### S6: Writeback

职责：

```text
- lane RF writeback
- scalar RF writeback
- predicate RF writeback
- memory response writeback
```

inactive lane 不写回。

### S7: Commit / Fault

职责：

```text
- exit event commit
- fault event commit
- PMU snapshot
- state clear
```

---

# 6. Architectural State

## 6.1 Shared architectural state

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

---

## 6.2 Lane Register File

基础配置：

```text
LANES     = 32
LANE_REGS = 16
REG_WIDTH = 32-bit
```

逻辑结构：

```text
lane_rf[LANES][LANE_REGS]
```

语义：

```text
- v0 在每个 lane 中是不同值
- r0-r15 是所有 lane 共享的 scalar register
- p0-p7 是 predicate mask，每个 bit 对应一个 lane
```

物理实现建议：

```text
- lane RF 按 lane group banking
- inactive lane clock gating
- RF read / execute / writeback 分拍
- 不支持 multi-kernel context save/restore
```

---

## 6.3 Predicate State

Predicate state 包括：

```text
- lane_valid_mask
- active_mask
- instruction predicate
- predicate registers p0-p7
- structured mask stack
```

effective active：

```text
effective_active = lane_valid_mask & active_mask & instruction_predicate
```

reset / launch 后：

```text
lane_valid_mask[i] = (block_base + i) < logical_elements
active_mask        = lane_valid_mask
pred_reg[*]        = 0
mask_stack_ptr     = 0
```

---

# 7. EVU-MT ISA

## 7.1 Register Model

```text
r0-r15    scalar registers
v0-v15    lane registers
p0-p7     predicate registers
pc        shared PC
amask     active mask
```

---

## 7.2 Opcode 分类

| 类别        | 指令                                                             |
| ----------- | ---------------------------------------------------------------- |
| Control     | `exit`, `br.u`, `if_mask`, `else_mask`, `endif`                  |
| Lane        | `laneid`, `setvl`                                                |
| Predicate   | `cmp`, `pred.and`, `pred.or`, `pred.not`                         |
| Memory      | `load`, `store`, `gather`, `load.scalar`, `load.mask`            |
| Integer ALU | `add`, `sub`, `mul`, `max`, `min`, `select`                      |
| FP ALU      | `fadd`, `fmul`, `fmax`, `fmin`, optional `fma`                   |
| SFU         | `exp.approx`, `gelu.approx`, `rsqrt.approx`, `rcp.approx`        |
| Convert     | `cvt`, `round`, `sat`                                            |
| Reduction   | `reduce.sum`, `reduce.max`, `reduce.sumsq`                       |
| Shuffle     | `shuffle.pair`, `shuffle.split_half`, optional `transpose.small` |

---

## 7.3 Opcode enum 草案

```c
typedef enum {
    EVU_INST_EXIT          = 0x00,
    EVU_INST_LANEID        = 0x01,
    EVU_INST_SETVL         = 0x02,

    EVU_INST_CMP           = 0x10,
    EVU_INST_PRED_AND      = 0x11,
    EVU_INST_PRED_OR       = 0x12,
    EVU_INST_PRED_NOT      = 0x13,

    EVU_INST_LOAD          = 0x20,
    EVU_INST_STORE         = 0x21,
    EVU_INST_GATHER        = 0x22,
    EVU_INST_LOAD_SCALAR   = 0x23,
    EVU_INST_LOAD_MASK     = 0x24,

    EVU_INST_ADD           = 0x30,
    EVU_INST_SUB           = 0x31,
    EVU_INST_MUL           = 0x32,
    EVU_INST_MAX           = 0x33,
    EVU_INST_MIN           = 0x34,
    EVU_INST_SELECT        = 0x35,

    EVU_INST_FADD          = 0x40,
    EVU_INST_FMUL          = 0x41,
    EVU_INST_FMAX          = 0x42,
    EVU_INST_FMIN          = 0x43,
    EVU_INST_FMA           = 0x44,

    EVU_INST_EXP_APPROX    = 0x50,
    EVU_INST_GELU_APPROX   = 0x51,
    EVU_INST_RSQRT_APPROX  = 0x52,
    EVU_INST_RCP_APPROX    = 0x53,

    EVU_INST_REDUCE_SUM    = 0x60,
    EVU_INST_REDUCE_MAX    = 0x61,
    EVU_INST_REDUCE_SUMSQ  = 0x62,

    EVU_INST_SHUFFLE_PAIR  = 0x70,
    EVU_INST_SHUFFLE_SPLIT = 0x71,
    EVU_INST_TRANSPOSE_SM  = 0x72,

    EVU_INST_IF_MASK       = 0x80,
    EVU_INST_ELSE_MASK     = 0x81,
    EVU_INST_ENDIF         = 0x82,
    EVU_INST_BR_UNIFORM    = 0x83
} evu_mt_opcode_t;
```

---

## 7.4 Instruction encoding

基础 64-bit 指令格式：

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

编码约束：

```text
- instruction address aligned
- illegal opcode 必须 fault
- unsupported dtype 必须 fault
- invalid register index 必须 fault
- branch target 必须 instruction aligned
```

---

# 8. Control Flow

## 8.1 Shared-PC

EVU-MT 不支持 per-lane PC：

```text
所有 lane 共享同一个 pc。
```

禁止：

```text
lane0 pc = A
lane1 pc = B
lane2 pc = C
```

---

## 8.2 If-conversion

小分支推荐由 compiler 转成 select/predicate：

```c
if (x > 0)
    y = a;
else
    y = b;
```

lower 为：

```asm
cmp.gt.f32      p0, v_x, r_zero
select.f32      v_y, v_a, v_b, p0
```

---

## 8.3 Structured mask stack

支持结构化分支：

```asm
cmp.gt.f32      p0, v0, r_zero

if_mask         p0
  fmul.f32      v1, v0, r_a, p0
else_mask
  fmul.f32      v1, v0, r_b, p0
endif
```

硬件行为：

```text
if_mask:
  push(active_mask)
  active_mask = active_mask & p0

else_mask:
  active_mask = stack_top & ~p0

endif:
  active_mask = pop()
```

限制：

```text
- mask stack depth = 4
- overflow fault
- underflow fault
- 不支持 arbitrary divergent branch
- uniform branch 只允许 scalar condition
```

---

# 9. Vector LSU

## 9.1 Memory domain

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

---

## 9.2 Access modes

| 模式               | V1 支持     | 说明                |
| ------------------ | ----------- | ------------------- |
| unit-stride load   | required    | 连续 load           |
| unit-stride store  | required    | 连续 store          |
| strided load/store | required    | layout / row stride |
| masked load/store  | required    | tail / mask         |
| indexed gather     | required    | local indexed load  |
| scatter            | optional    | conflict-free only  |
| atomic             | unsupported | V1 禁止             |

---

## 9.3 Address generation

unit load/store：

```text
addr[i] = slot_base + base_offset + lane_id[i] * elem_size
```

strided：

```text
addr[i] = slot_base + base_offset + lane_id[i] * stride
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
- gather index dtype V1 = u32
```

---

## 9.4 LSU request / response

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
  logic [7:0]  op_id;
  logic [7:0]  replay_tag;
  logic [31:0] lane_valid;
  logic        fault;
  logic [5:0]  fault_lane;
  logic [15:0] fault_code;
  logic [511:0] data;
} evu_mt_dmem_resp_t;
```

---

# 10. Replay Queue

## 10.1 Replay 触发条件

```text
- SRAM bank conflict
- LSU structural hazard
- store port conflict
- dependency replay
```

---

## 10.2 Replay 规则

```text
- replay 不改变 logical lane order
- gather response 可乱序返回
- writeback 必须恢复 lane mapping
- store replay 必须保持可见顺序
- replay queue full 会 backpressure pipeline
- replay timeout 触发 fault
```

---

## 10.3 Replay entry

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

---

# 11. Datapath

## 11.1 ALU

基础 ALU：

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

FP ALU：

```text
- fadd
- fmul
- fmax
- fmin
- optional fma
```

---

## 11.2 SFU

SFU 支持：

```text
- exp.approx
- gelu.approx
- rsqrt.approx
- rcp.approx
```

要求：

```text
- multi-cycle pipeline
- valid/ready backpressure
- approximation error policy frozen
- inactive lane 不进入 SFU
```

---

## 11.3 Reduction unit

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
- FP16/BF16 input 推荐 FP32 accumulate
- reduction tree 分层多拍
```

---

## 11.4 Shuffle unit

基础支持：

```text
- pair even/odd shuffle
- split-half pair shuffle
```

可选支持：

```text
- 8x8 transpose
- 16x16 transpose
```

V1 不建议支持 arbitrary permute crossbar。

---

# 12. Numerical Policy

必须冻结：

```text
- dtype_src
- dtype_acc
- dtype_dst
- rounding mode
- saturation mode
- NaN policy
- Inf policy
- denorm policy
- reduction identity
- SFU approximation tolerance
```

推荐基础策略：

| Path           | Input          | Accumulate   | Output         |
| -------------- | -------------- | ------------ | -------------- |
| elementwise    | BF16/FP16/FP32 | same or FP32 | BF16/FP16/FP32 |
| reduction      | BF16/FP16      | FP32         | FP32/BF16/FP16 |
| softmax kernel | BF16/FP16      | FP32         | BF16/FP16      |
| norm kernel    | BF16/FP16      | FP32         | BF16/FP16      |
| INT8           | INT8           | INT16/INT32  | INT8           |

identity：

```text
sum identity = 0
max identity = dtype_min
softmax max identity = -inf
softmax sum identity = 0
all-masked reduction = policy-defined zero/fault
```

---

# 13. Kernel mapping examples

## 13.1 Fused activation

C-like semantics：

```c
tid = block_base + lane_id;

if (tid < N) {
    y[tid] = gelu(x[tid] * scale + bias);
}
```

pseudo ISA：

```asm
laneid.u32      v0
add.u32         v0, v0, r_block_base
cmp.lt.u32      p0, v0, r_N

load.bf16       v1, [slot_x + v0 * 2], p0
fmul.f32        v2, v1, r_scale, p0
fadd.f32        v3, v2, r_bias, p0
gelu.approx     v4, v3, p0
store.bf16      [slot_y + v0 * 2], v4, p0

exit
```

---

## 13.2 Local softmax

C-like semantics：

```c
x = load(score[lane])
x = apply_mask_and_scale(x)
m = reduce_max(x)
e = exp(x - m)
s = reduce_sum(e)
y = e / s
store(out[lane], y)
```

pseudo ISA：

```asm
laneid.u32        v0
cmp.lt.u32        p0, v0, r_inner

load.bf16         v1, [slot_score + v0 * 2], p0
load.mask         p1, [slot_mask + v0], p0
pred.and          p2, p0, p1

fmul.f32          v2, v1, r_scale, p2
select.masked     v2, v2, r_neg_inf, p2

reduce.max.f32    r_m, v2, p2
fsub.f32          v3, v2, r_m, p2
exp.approx.f32    v4, v3, p2
reduce.sum.f32    r_s, v4, p2
rcp.approx.f32    r_inv, r_s
fmul.f32          v5, v4, r_inv, p2

store.bf16        [slot_out + v0 * 2], v5, p2
exit
```

---

## 13.3 RMSNorm

C-like semantics：

```c
x = load(input[lane])
ss = reduce_sum(x * x)
rstd = rsqrt(ss / N + eps)
y = x * rstd * gamma
store(out[lane], y)
```

pseudo ISA：

```asm
laneid.u32        v0
cmp.lt.u32        p0, v0, r_N

load.bf16         v1, [slot_x + v0 * 2], p0
fmul.f32          v2, v1, v1, p0
reduce.sum.f32    r_ss, v2, p0

fmul.f32.scalar   r_mean, r_ss, r_invN
fadd.f32.scalar   r_var, r_mean, r_eps
rsqrt.approx.f32  r_rstd, r_var

load.bf16         v3, [slot_gamma + v0 * 2], p0
fmul.f32          v4, v1, r_rstd, p0
fmul.f32          v5, v4, v3, p0

store.bf16        [slot_out + v0 * 2], v5, p0
exit
```

---

## 13.4 Local gather + activation

C-like semantics：

```c
tid = block_base + lane_id;

if (tid < N) {
    idx = index[tid];
    x = src[idx];
    y = relu(x);
    dst[tid] = y;
}
```

pseudo ISA：

```asm
laneid.u32      v0
add.u32         v0, v0, r_block_base
cmp.lt.u32      p0, v0, r_N

load.u32        v1, [slot_index + v0 * 4], p0
gather.bf16     v2, [slot_src + v1 * 2], p0
max.bf16        v3, v2, r_zero, p0
store.bf16      [slot_dst + v0 * 2], v3, p0

exit
```

---

# 14. State Machine

## 14.1 Main FSM

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
  |
  +--> EXEC_LSU -> LSU_WAIT -> LSU_REPLAY? -> WRITEBACK
  |
  +--> EXEC_SFU -> SFU_WAIT -> WRITEBACK
  |
  +--> EXEC_REDUCE -> REDUCE_WAIT -> WRITEBACK
  |
  +--> EXEC_SHUFFLE -> WRITEBACK
  |
  v
PC_UPDATE
  | exit
  +---------> EVENT_COMMIT -> IDLE
  |
  +---------> IFETCH
```

---

## 14.2 LSU replay FSM

```text
LSU_REQ
  |
  v
ADDR_GEN
  |
  v
BANK_CHECK
  | no conflict
  +------------> ISSUE_BANKS -> WAIT_RESP -> WRITEBACK
  |
  | conflict
  v
PARTIAL_ISSUE
  |
  v
REPLAY_ENQUEUE
  |
  v
REPLAY_ISSUE
  |
  v
WAIT_RESP
  |
  v
WRITEBACK
```

---

## 14.3 Fault FSM

```text
FAULT_DETECT
  |
  v
STOP_ISSUE
  |
  v
DRAIN_OR_KILL_OUTSTANDING
  |
  v
WRITE_FAULT_RECORD
  |
  v
COMMIT_FAULT_EVENT
  |
  v
IDLE
```

---

# 15. Registers and CSR

## 15.1 Status registers

| Register                  | Description                               |
| ------------------------- | ----------------------------------------- |
| `EVU_MT_STATUS`           | idle / busy / fault / drain / replay_busy |
| `EVU_MT_CMD_ID`           | current command id                        |
| `EVU_MT_PC`               | current shared PC                         |
| `EVU_MT_ACTIVE_MASK`      | current active lane mask                  |
| `EVU_MT_LANE_VALID_MASK`  | current tail-valid lane mask              |
| `EVU_MT_FAULT_CODE`       | current fault code                        |
| `EVU_MT_FAULT_PC`         | fault PC                                  |
| `EVU_MT_FAULT_LANE`       | first faulting lane                       |
| `EVU_MT_FAULT_SLOT`       | faulting slot id                          |
| `EVU_MT_FAULT_ADDR`       | faulting byte offset                      |
| `EVU_MT_PMU_ACTIVE`       | active cycles                             |
| `EVU_MT_PMU_INST`         | instruction count                         |
| `EVU_MT_PMU_LSU_REPLAY`   | LSU replay cycles                         |
| `EVU_MT_PMU_MASKED_LANES` | inactive lane count                       |
| `EVU_MT_PMU_STALL_*`      | stall counters                            |

---

## 15.2 Fault code

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

---

# 16. PMU and Performance Model

## 16.1 Required PMU counters

```text
evu_mt_active_cycles
evu_mt_instruction_count
evu_mt_ifetch_cycles
evu_mt_decode_cycles
evu_mt_issue_cycles
evu_mt_lsu_active_cycles
evu_mt_lsu_replay_cycles
evu_mt_replay_queue_full_cycles
evu_mt_sfu_active_cycles
evu_mt_reduce_active_cycles
evu_mt_shuffle_active_cycles
evu_mt_writeback_stall_cycles
evu_mt_masked_lane_count
evu_mt_branch_mask_cycles
evu_mt_fault_count
```

---

## 16.2 Primary stall attribution

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

---

## 16.3 Performance model

EVU-MT kernel latency：

```text
T_kernel =
    T_launch
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
Perf =
  min(
    issue_width * lanes * lane_eff * divergence_eff,
    LSU_BW * lsu_eff,
    SFU_throughput,
    reduce_throughput,
    shuffle_throughput
  )
```

定义：

```text
lane_eff =
    active_lanes / total_lanes

divergence_eff =
    useful_path_cycles / executed_path_cycles

lsu_eff =
    requested_bytes / (requested_bytes + replay_bytes)

issue_eff =
    issued_instructions / total_cycles
```

---

# 17. RTL 实现建议

## 17.1 关键路径

| 路径                       | 风险       | 缓解                                 |
| -------------------------- | ---------- | ------------------------------------ |
| predicate -> byte enable   | fanout 大  | mask 分段寄存，局部 byte-enable      |
| address gen -> bank check  | 组合路径长 | addr gen 和 bank check 分拍          |
| RF read -> ALU -> RF write | Fmax 风险  | 3-stage datapath                     |
| branch mask update -> PC   | 控制复杂   | structured branch only               |
| reduction tree             | 宽度增长   | hierarchical multi-cycle reduce      |
| shuffle crossbar           | 面积/时序  | 限制 pair/split-half/small transpose |

---

## 17.2 Clock gating

必须支持：

```text
- inactive lane gating
- unused RF bank gating
- SFU idle gating
- reduction idle gating
- shuffle idle gating
- LSU idle gating
- IFETCH idle gating
```

---

## 17.3 Reset / drain / clear fault

基础策略：

```text
reset:
  clear all architectural state
  clear PMU or mark PMU invalid
  state = IDLE

fault:
  stop issue
  drain or kill outstanding request by fault_policy
  commit fault event
  state = FAULT or IDLE depending on control policy

clear_fault:
  clear fault record
  allow new launch
```

---

# 18. Software / Compiler Contract

EVU-MT 需要 compiler 提供：

```text
- EVU-MT kernel binary
- launch descriptor
- slot binding
- register allocation
- structured control flow legality
- dtype legality
- local memory access legality
- scatter conflict-free proof if scatter enabled
```

Compiler lowering flow：

```text
High-level IR
  ↓
Tile-local kernel extraction
  ↓
EVU-MT Kernel IR
  ↓
EVU-MT legalization
  ↓
instruction selection
  ↓
register allocation
  ↓
instruction scheduling
  ↓
binary encoding
```

EVU-MT Kernel IR 示例：

```mlir
evu_mt.kernel @gather_gelu(
    %src   : !evu.slot,
    %idx   : !evu.slot,
    %dst   : !evu.slot,
    %scale : f32,
    %bias  : f32,
    %n     : i32) {

  %lane = evu_mt.lane_id : i32
  %p0 = evu_mt.cmpi ult, %lane, %n : i32

  %i = evu_mt.load %idx[%lane] pred %p0 : i32
  %x = evu_mt.gather %src[%i] pred %p0 : bf16
  %y0 = evu_mt.mulf %x, %scale pred %p0 : bf16
  %y1 = evu_mt.addf %y0, %bias pred %p0 : bf16
  %y2 = evu_mt.gelu_approx %y1 pred %p0 : bf16

  evu_mt.store %dst[%lane], %y2 pred %p0 : bf16
  evu_mt.return
}
```

---

# 19. Verification Plan

## 19.1 SVA 重点

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
- fault 后 commit fault event
- PMU primary stall 互斥
```

示例：

```systemverilog
property inactive_lane_no_mem_req;
  @(posedge clk) disable iff (!rst_n)
    (lane_active[i] == 1'b0) |-> !lane_mem_req_valid[i];
endproperty

assert property (inactive_lane_no_mem_req);
```

```systemverilog
property inactive_lane_no_writeback;
  @(posedge clk) disable iff (!rst_n)
    wb_valid |-> ((wb_lane_mask & ~effective_active_mask) == '0);
endproperty

assert property (inactive_lane_no_writeback);
```

```systemverilog
property masked_store_byte_enable;
  @(posedge clk) disable iff (!rst_n)
    dmem_req_valid && dmem_req.is_store |->
      ((dmem_req.byte_enable & ~active_lane_byte_mask) == '0);
endproperty

assert property (masked_store_byte_enable);
```

---

## 19.2 Bring-up 顺序

```text
1. reset / idle status
2. launch accept / reject
3. code fetch
4. illegal opcode fault
5. exit instruction commit
6. laneid instruction
7. scalar load
8. lane add/store
9. predicate compare
10. tail mask
11. masked store
12. unit-stride load/store
13. strided load/store
14. gather local
15. bank conflict replay
16. reduce.sum / reduce.max
17. SFU exp / rsqrt / gelu
18. shuffle pair
19. structured if_mask / else_mask / endif
20. full softmax kernel
21. full RMSNorm kernel
22. PMU correlation
23. random fault injection
```

---

## 19.3 Golden model

必须提供：

```text
- EVU-MT ISA interpreter
- launch descriptor validator
- local slot memory model
- replay randomizer
- C++ functional simulator
- Python numerical golden for softmax/norm
- fault injection testbench
```

---

# 20. Base Configuration

推荐 V1 balanced configuration：

```text
LANES:              32
LANE_REGS:          16
SCALAR_REGS:        16
PRED_REGS:          8
MASK_STACK_DEPTH:   4
CMD_QUEUE_DEPTH:    1
INST_WIDTH:         64-bit
DATA_WIDTH:         32-bit
LSU:                unit / strided / masked / gather
SCATTER:            disabled or conflict-free only
ATOMIC:             disabled
REDUCE:             sum / max / sumsq
SHUFFLE:            pair / split-half
SFU:                exp / gelu / rsqrt / rcp approx
MEMORY_DOMAIN:      local slot memory only
```

---

# 21. 验收标准

V1 acceptance criteria：

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
11. fault event 能正确 commit。
12. kernel 完成 event 能正确 commit。
```

---

# 22. 风险、取舍和后续方向

| 风险                            | 影响                          | 缓解                                         |
| ------------------------------- | ----------------------------- | -------------------------------------------- |
| EVU-MT 演化成 GPU SM            | 面积/功耗/验证失控            | 禁止 warp scheduler、per-lane PC、多 context |
| ISA 过大                        | compiler/RTL 复杂             | V1 只保留小 ISA                              |
| gather bank conflict 高         | replay cycles 高              | bank-aware layout、compiler hint             |
| SFU 精度不冻结                  | golden 难对齐                 | 提前冻结 approximation policy                |
| structured branch 复杂          | 控制路径复杂                  | if-conversion 优先，mask stack depth 限制    |
| scatter 过早启用                | 一致性和 duplicate index 复杂 | V1 关闭或只支持 conflict-free                |
| fault 后 partial write 语义不清 | debug 困难                    | fault event + poisoned output policy         |
| memory latency 被误放进 EVU     | 性能不可控                    | EVU-MT 只支持 local slot memory              |

---

# 23. 总结

EVU-MT 是一个：

```text
tile-local shared-PC microthread vector execution unit
```

它的设计边界是：

```text
EVU-MT 负责：
  predicated lane execution
  local arithmetic
  local memory load/store/gather
  local reduction
  limited shuffle
  bank replay
  fault/PMU

EVU-MT 不负责：
  warp scheduling
  dense matmul
  metadata walk
  global memory programming
  global unordered atomic
  Tile-level orchestration
```

一句话：

> EVU-MT 拿 SIMT 的 lane 编程便利性，但不拿 GPU 的 warp scheduler 和 global-memory execution model。
