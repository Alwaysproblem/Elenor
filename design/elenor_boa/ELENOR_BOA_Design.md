# BOA 设计文档

## 1. 定位、目标和 First Silicon cutline

BOA（Block Outer-product Accelerator）是 ELENOR 的 dense compute 主路径，面向规则、密集、block 化的矩阵型计算。硬件执行 Tile UCE 发起的 command、descriptor 和 Tile Program，不直接执行高层 graph。BOA 内部只处理局部 contraction、局部 accumulate、局部 reduce 和受限 epilogue；全局 tensor mapping、page/segment 数据流、runtime scheduling 和状态生命周期分别由 compiler/runtime、MFE、Tile Group Sequencer、Tile UCE 和 USE 承担。

目标工作负载：

- GEMM、batched GEMM、Conv lowering 后的 GEMM / implicit GEMM。
- Dense attention 的 QK 和 AV。
- MoE expert MLP 的 dense 主计算。
- split-K partial sum、block-level reduce、受限 bias / activation / requant epilogue。

非目标工作负载：

- fine-grained gather、scatter、dynamic sparse graph traversal。
- per-element branch、复杂 permutation、跨 tile irregular access。
- page table walk、segment walk、global memory policy。
- 高层 graph 解释或每个物理 PE 的软件可见调度。

Architecture V1 可以保留完整 dataflow search、更多 dtype、复杂 epilogue fusion、sparse matmul 和更多跨 tile reduce 形态；First Silicon V1 必须收敛到可验证的闭环：

| 能力     | First Silicon V1                                                         | 后续能力                                      |
| -------- | ------------------------------------------------------------------------ | --------------------------------------------- |
| dtype    | INT8、BF16，INT32/BF16 accumulate                                        | INT4、FP16、FP32 accumulate 由后续规格冻结    |
| shape    | GEMM、attention QK/AV、expert MLP 的固定 tile family                     | 更广泛 dynamic shape 和 conv implicit mapping |
| reduce   | OPA 内 accumulate、BOA 内 reduce tree、基础 split-K                      | 跨 group reduce policy、memory merge 优化     |
| epilogue | bias、relu、requant、round/saturate 子集                                 | gelu、residual add、复杂 fused epilogue       |
| 启动     | Tile UCE launch descriptor + event completion                            | compiler-guided dataflow autotune             |
| PMU      | active、operand stall、accumulator stall、writeback stall、SRAM conflict | 完整 stall taxonomy 和采样 trace              |

## 2. 职责、非职责和 ownership

BOA ownership 边界必须清晰，避免 dense datapath 被不规则访问和控制语义污染。

| 对象                                   | owner                                | BOA 行为                                                    |
| -------------------------------------- | ------------------------------------ | ----------------------------------------------------------- |
| BOA descriptor 静态字段                | compiler                             | 消费 shape、dtype、layout、tile、dataflow、reduce、epilogue |
| context/base IOVA patch                | runtime / firmware                   | 通过 descriptor cache 看到已 patch 后字段                   |
| tile_id / group_id / slot offset patch | Tile UCE auto-patch                  | 使用 slot id 和 layout id 访问 L1                           |
| page list / segment offset             | MFE                                  | 只消费 MFE 写入的 stream/metadata slot，不做 walk           |
| Tile Program PC 和 launch/wait         | Tile UCE                             | BOA 只接收 launch、返回 event/fault                         |
| state lifecycle                        | USE                                  | BOA 不管理 state cache 或 checkpoint                        |
| accumulator slot                       | Tile Frame / BOA descriptor          | BOA 独占 accumulate 权限，EVU/DMA 不能无声明覆盖            |
| partial result merge                   | BOA / Collective / Memory merge 之一 | descriptor 必须声明 owner，不允许隐式共享                   |

BOA 必须提供：

- descriptor validation 结果：unsupported dtype、tile shape illegal、slot permission violation、alignment fault、reserved bits nonzero。
- event completion：done、fault、timeout、cancelled。
- PMU local counter：active、stall reason、bank conflict、reduce tree saturation、writeback bandwidth。
- deterministic replay contract：相同 descriptor、slot frame 和输入数据应产生相同输出；不同 reduction order 带来的数值差异必须由 dtype/rounding mode 明确。
- reduce tree 的 bypass 的旁路，简单支持非 split-K 场景的。
- 需要暴露 reduce tree 的输入以便于直接可以使用 reduce tree 做局部规约的场景
- 计算基地址的能力，因为需要 pingpong buffer 来隐藏 operand fetch latency，BOA 需要能够在不阻塞 OPA hot path 的情况下切换 tile buffer。

BOA 不拥有：

- HBM 全局地址策略。
- MFE page/segment stream 的一致性边界。
- EVU predicated vector mask 的解释。
- UCE 的 program branch、event wait 和 descriptor patch policy。

## 3. 微架构和状态机

### 3.1 Pipeline

```text
Descriptor Fetch
    |
Descriptor Validate / Tile Decode
    |
Operand Fetch / Double Buffer Fill
    |
OPA Array Compute
    |
Local Accumulate
    |
Reduce Network
    |
Post-Op / Output Convert
    |
Writeback / Event Commit
```

推荐内部子模块：

| 子模块               | 职责                                                            | 关键接口                             |
| -------------------- | --------------------------------------------------------------- | ------------------------------------ |
| Launch Frontend      | 接收 Tile UCE launch、读取 descriptor、建立 command context     | cmd_id、desc_slot、event_id          |
| Descriptor Validator | 检查 ABI version、size、dtype、tile、slot 权限、reserved bits   | fault record                         |
| Tile Decoder         | 生成 m/n/k loop、split-K、dataflow、transpose 和 layout control | micro-loop issue                     |
| Operand Fetcher      | 从 L1 operand slot 读取 A/B tile，填充 ping-pong buffer         | SRAM read port、bank conflict signal |
| OPA Array            | 多个 OPA 并行执行 outer product / local reduce                  | fragment valid/ready                 |
| Accumulator File     | 保存 partial sum，支持 read-modify-write 和 clear/reuse         | accumulator SRAM / register file     |
| Reduce Network       | OPA 间、K tile 间和 split-K 局部规约                            | reduce_op、rounding mode             |
| Epilogue Unit        | bias、activation、requant、saturate、convert                    | param slot、out dtype                |
| Writeback Unit       | 写 C slot 或 output stream，提交 event                          | SRAM write port、event unit          |
| PMU/Fault Unit       | 记录 cycle、stall、fault、overflow、ECC 可选事件                | PMU bus、fault record                |

### 3.2 状态机

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
INIT_LOOP
  |
  v
PREFETCH_A_B
  | operands ready
  v
COMPUTE_K_TILE <----+
  |                  |
  v                  |
ACCUMULATE           |
  | more k tile -----+
  v
LOCAL_REDUCE
  | split-K pending -> WAIT_PARTIAL / COLLECTIVE_HANDOFF
  v
EPILOGUE
  |
  v
WRITEBACK
  |
  v
EVENT_COMMIT
  |
  v
IDLE
```

状态语义：

- `DESC_FETCH` 只读取 descriptor region，不访问 operand slot。
- `VALIDATE` 必须在任何输出写入前完成，避免半执行 descriptor 污染 accumulator。
- `PREFETCH_A_B` 与上一 tile 的 `COMPUTE_K_TILE` 可重叠；ping-pong buffer 的 valid/ready 必须可由 SVA 覆盖。
- `COMPUTE_K_TILE` 不应因 epilogue 参数读取阻塞 OPA hot path；epilogue 参数提前进入小寄存器或独立 buffer。
- `WAIT_PARTIAL / COLLECTIVE_HANDOFF` 只在 descriptor 声明 split-K 或跨 tile reduce 时进入。
- `FAULT_COMMIT` 必须记录 command id、descriptor id、tile id、fault code 和第一个 faulting field。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Launch 协议

Tile UCE 通过 Tile Program 发起 BOA command：

```text
launch.boa desc_slot, event_id
wait event_id
```

硬件可见握手：

```text
uce_boa_launch_valid
uce_boa_launch_ready
uce_boa_desc_slot
uce_boa_cmd_id
uce_boa_event_id
boa_uce_done_valid
boa_uce_fault_valid
boa_uce_fault_code
```

`launch_ready=0` 的 primary stall owner 是 BOA busy 或 descriptor frontend full；如果 BOA 等待 operand 或 stream credit，PMU 必须归因到 `boa_operand_stall` 或 `stream_credit_empty_or_full`，不能重复计入 UCE stall。

### 4.2 Slot-based BOA descriptor v0

```c
typedef struct {
    uint16_t op_kind;
    uint16_t flags;

    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t batch;

    uint16_t tile_m;
    uint16_t tile_n;
    uint16_t tile_k;
    uint16_t split_k;

    uint16_t dtype_a;
    uint16_t dtype_b;
    uint16_t dtype_acc;
    uint16_t dtype_out;

    uint16_t a_slot;
    uint16_t b_slot;
    uint16_t c_slot;
    uint16_t bias_slot;

    uint16_t a_layout;
    uint16_t b_layout;
    uint16_t c_layout;
    uint16_t dataflow;

    uint16_t transpose_flags;
    uint16_t rounding_mode;
    uint16_t saturation_mode;
    uint16_t epilogue_kind;

    uint16_t quant_param_slot;
    uint16_t epilogue_param_slot;
    uint32_t reserved;
} boa_desc_v0_t;
```

字段约束：

| 字段               | 语义                                                         | V1 约束                                     |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------- |
| `op_kind`          | GEMM、QK、AV、expert MLP、reduce-like                        | 枚举值由后续规格冻结                        |
| `flags`            | accumulate clear、append partial、event policy、fault policy | reserved bit 必须为 0                       |
| `m/n/k/batch`      | logical shape                                                | 0 非法；tail 由 mask 或 tile boundary 处理  |
| `tile_m/n/k`       | BOA micro tile                                               | 必须匹配支持的 OPA array shape              |
| `split_k`          | K 维切分数                                                   | 1 为普通路径；大于 1 必须声明 partial owner |
| `dtype_*`          | 输入、权重、累加、输出类型                                   | First Silicon V1 为 INT8/BF16 子集          |
| `*_slot`           | Tile Slot Frame 索引                                         | role 和权限必须匹配 read/write/accumulate   |
| `*_layout`         | bank-aware layout id                                         | layout 编码由后续规格冻结                   |
| `dataflow`         | output-stationary、weight-stationary 等                      | V1 推荐 output-stationary                   |
| `transpose_flags`  | A/B transpose 或 packed layout                               | 必须与 stride/layout 一致                   |
| `rounding_mode`    | requant / convert rounding                                   | 默认值由后续规格冻结                        |
| `saturation_mode`  | overflow handling                                            | INT8 输出必须声明                           |
| `epilogue_kind`    | none、bias、relu、requant 等                                 | gelu/residual 可预留                        |
| `quant_param_slot` | scale/zero point 参数                                        | per-tensor / per-channel 子集由后续规格冻结 |
| `reserved`         | ABI 扩展                                                     | validation 要求为 0                         |

### 4.3 寄存器与状态视图

BOA 不需要软件可见的逐 PE 寄存器。建议只暴露 debug/status CSR 或 memory-mapped status region：

| 寄存器                  | 访问 | 说明                                  |
| ----------------------- | ---- | ------------------------------------- |
| `BOA_STATUS`            | RO   | idle、busy、fault、drain、clock_gated |
| `BOA_CMD_ID`            | RO   | 当前 command id                       |
| `BOA_FAULT_CODE`        | RO   | 最近 fault code                       |
| `BOA_FAULT_FIELD`       | RO   | descriptor field 或 slot index        |
| `BOA_PMU_ACTIVE`        | RO   | active cycles                         |
| `BOA_PMU_STALL_OPERAND` | RO   | operand wait cycles                   |
| `BOA_PMU_STALL_ACC`     | RO   | accumulator conflict cycles           |
| `BOA_PMU_STALL_WB`      | RO   | writeback stall cycles                |
| `BOA_PMU_BANK_CONFLICT` | RO   | SRAM bank conflict cycles             |

软件不应通过 CSR 编排 BOA micro-loop；CSR 只用于 bring-up、fault inspection 和 PMU readout。

## 5. 数据流、控制流和时序路径

### 5.1 数据流

```text
L2 / MFE stream / Tile DMA
        |
        v
Tile L1 Slot Frame
  ├── A slot: bank-aware packed tile
  ├── B slot: bank-aware packed tile
  ├── C slot: accumulator/output
  ├── bias / quant param slot
  └── descriptor/event region
        |
        v
BOA operand fetcher -> OPA array -> accumulator -> reduce -> epilogue -> C slot
```

BOA operand fetch 应按 bank-aware layout 发起 sequential burst。A/B hot path 不应与 Program / Descriptor / Event region 固定落在同一组 bank。compiler memory planner 可以通过 `bank_policy` 和 `layout` hint 控制 slot placement；硬件必须能检测冲突并通过 PMU 归因。

### 5.2 Banking 交互

- A/B operand buffer 使用 ping-pong double buffer，prefetch 侧和 compute 侧独立 valid bit。
- accumulator slot 是 read-modify-write 模式，端口需求高于普通 output slot；First Silicon V1 应限制同时运行的 BOA + EVU 写热点。
- 如果 MFE 正在向 BOA 输入 slot 填充 page stream，MFE 是 writer，BOA 是 reader；slot frame 必须表达 producer/consumer 和 stream credit。
- DMA 不允许覆盖 `ELENOR_SLOT_ACCUMULATOR`，除非 command 明确执行 clear/init。

### 5.3 关键时序路径

| 路径                                    | 风险                             | 约束                                              |
| --------------------------------------- | -------------------------------- | ------------------------------------------------- |
| OPA multiply-accumulate                 | 乘加树和 accumulator feedback    | pipeline reg 切分，tile_k 由 PPA exploration 冻结 |
| Reduce network                          | fan-in 随 OPA 数增长             | 分层 reduce，避免单周期全宽 fan-in                |
| SRAM read -> operand align -> OPA input | bank mux 和 layout swizzle       | operand buffer 预取，避免直接跨大 mux 进 OPA      |
| Accumulator RMW                         | SRAM macro read/write turnaround | 局部 register accumulator 或分 bank accumulator   |
| Descriptor decode                       | 多字段组合控制 fanout            | decode 后寄存，micro-loop 使用压缩控制字          |
| Event/fault commit                      | 控制面跨域                       | async FIFO 或同步边界由后续规格冻结               |

### 5.4 工作负载映射示例

| 工作负载           | BOA 映射                                                       | 协同模块                                                        | 关键检查                                                   |
| ------------------ | -------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- |
| Dense GEMM         | A/B tile 进入 OPA array，K 维循环 accumulate，C slot writeback | Tile DMA 负责 L2->L1，Tile UCE 管 launch/wait                   | BOA active 高，operand stall 低，golden matmul 对齐        |
| Dense Attention QK | Q tile 与 K tile 做 BOA GEMM，输出 score tile 到 workspace     | EVU 后续执行 scale/mask/softmax                                 | QK descriptor 的 layout 与 EVU score layout 一致           |
| Attention AV       | softmax 后概率 tile 与 V tile 做 BOA GEMM                      | MFE 可提供 paged V stream，EVU 提供 softmax 输出                | `T_prefetch <= T_qk` case 下 AV operand stall 不应异常升高 |
| MoE Expert MLP     | 每个 expert batch 映射为 BOA GEMM 或 batched GEMM              | MFE Segment Stream 做 token grouping，EVU/Collective 做 combine | BOA utilization 可由 expert imbalance model 解释           |
| Conv lowering      | im2col 或 implicit tile 后进入 OPA MUL                         | MFE 可选做 layout stream，EVU 处理尾部                          | tile_m/n/k 与 packed layout 匹配                           |

## 6. 配置、PPA、性能模型和 PMU

### 6.1 配置参数

| 参数               | First Silicon V1 建议              | 冻结方式                     |
| ------------------ | ---------------------------------- | ---------------------------- |
| OPA 数量 / Tile    | 4 个 OPA                           | 由 PPA exploration 冻结      |
| OPA shape          | 16x16 或 32x16 outer-product tile  | 由 PPA exploration 冻结      |
| L1 banks           | 至少 16 banks                      | 由 SRAM profile 冻结         |
| BOA operand buffer | A/B double buffer                  | 容量由 SRAM profile 冻结     |
| accumulator 容量   | 支持一个或多个 output tile partial | 由 SRAM profile 冻结         |
| dataflow           | output-stationary first            | 其他 dataflow 由后续规格冻结 |

### 6.2 性能模型

```text
BOA_perf = min(BOA_peak, SRAM_bw * AI_sram, HBM_bw * AI_hbm)
BW_sram_required = BW_boa_A_read + BW_boa_B_read + BW_boa_acc_read_write + BW_boa_writeback
BW_eff = BW_peak * (1 - conflict_rate)
```

Paged attention 的 BOA 利用率与 MFE overlap 相关：

```text
T_prefetch <= T_qk
```

满足该条件时，KV page stream 的大部分 latency 可被 QK 计算隐藏；否则 BOA PMU 应显示 `boa_operand_stall` 上升，并与 MFE `prefetch miss` 或 `stream stall` 同步出现。

MoE expert MLP 的 BOA 有效利用率受 token imbalance 影响：

```text
imbalance = max(tokens_per_expert) / avg(tokens_per_expert)
U_boa = 1 / imbalance
```

compiler/runtime 应通过 token sorting、expert batching、capacity padding 和 group-level placement 改善 BOA batch fullness。

### 6.3 PMU

BOA local PMU 必须支持唯一归因：

- `boa_active_cycles`：OPA array 有效计算。
- `boa_stall_operand`：A/B tile、MFE stream 或 DMA 填充不足。
- `boa_stall_accumulator`：accumulator bank conflict 或 RMW busy。
- `boa_stall_writeback`：C slot、stream credit 或 SRAM write port 不可用。
- `boa_reduce_active`：reduce tree 有效周期。
- `boa_epilogue_active`：epilogue / convert 有效周期。
- `boa_sram_bank_conflict`：由 BOA 请求触发的 bank conflict。
- `boa_fault_count_by_type`：invalid descriptor、slot fault、dtype fault、timeout、internal fault。

每个 stall cycle 只能有一个 primary owner；secondary debug tag 可记录但不进入 utilization 统计。

## 7. RTL/软件实现建议

### 7.1 RTL

- 将 descriptor decode 与 OPA datapath 解耦：frontend 输出压缩 micro-loop 控制字，datapath 不直接扇出完整 descriptor。
- OPA array、operand buffer、accumulator、reduce network 分别有清晰 ready/valid 边界。
- operand fetcher 对 bank conflict 可 replay；replay 不改变 accumulator commit 顺序。
- accumulator clear、partial append、writeback commit 必须有明确 epoch，避免 timeout/retry 后重复累加。
- epilogue 参数在 compute 前预取，禁止在 OPA hot loop 中随机读取 parameter slot。
- clock gating 按 idle、waiting operand、waiting writeback 和 debug freeze 分级；operand gating 避免 masked/tail lane 无效切换。

### 7.2 软件和 compiler

- MLIR lowering 应生成 slot-based descriptor，不把绝对物理地址绑定进算子语义。
- compiler memory planner 应同时输出 slot role、alignment、layout、bank hint 和 descriptor patch metadata。
- Tile kernel library 应提供固定模板：GEMM、QK、AV、expert MLP、split-K reduce。
- runtime 负责 context id、base IOVA、descriptor residency；Tile UCE 负责 tile/group offset patch；MFE 负责 page/segment 相关动态地址。
- 对 unsupported shape，compiler 选择 EVU tail path、smaller BOA tile 或 runtime multi-versioning，不能让 BOA 执行未定义 tile。

## 8. 验证、bring-up 和验收标准

### 8.1 RTL/SVA 重点

- launch handshake：`launch_valid && launch_ready` 后必须进入 `DESC_FETCH`，直到 done/fault 前不得接受同一 context 的覆盖启动。
- descriptor validation：非法 dtype、slot permission、reserved bit、tile shape 必须在输出写入前 fault。
- ping-pong buffer：compute buffer 被消费时 prefetch 不能覆盖；valid/ready 不丢 token。
- accumulator epoch：clear、accumulate、writeback 顺序不可交换；fault path 不提交 partial output。
- SRAM permission：只有 descriptor 声明的 C/accumulator slot 可被 BOA 写入。
- reduce determinism：相同 reduction mode 和 rounding mode 下 RTL 与 Python golden 对齐。
- event liveness：非 fault descriptor 在 operand/event 依赖满足后最终产生 done。
- PMU attribution：active 与各类 primary stall 计数互斥。

### 8.2 Bring-up 顺序

1. command queue + event + DMA copy 闭环。
2. BOA descriptor validation smoke test。
3. 单 OPA INT8/BF16 micro tile golden compare。
4. 多 OPA GEMM，通过 Tile UCE command queue 触发。
5. L1 double buffer overlap，观察 operand stall 下降。
6. split-K local reduce，验证 partial owner 和 event chain。
7. QK / AV attention trace，与 EVU softmax 串联。
8. MFE page stream feeding BOA，验证 paged attention PMU 指纹。

### 8.3 验收标准

- BOA GEMM 必须通过 command queue 触发，而不是 testbench 直接拉 datapath。
- Python golden 与 RTL 输出在 dtype tolerance 内一致。
- invalid descriptor、timeout、slot permission fault 都能生成 fault record。
- BOA active/stall counter 与人工构造的 operand starvation、accumulator conflict、writeback backpressure 场景一致。
- SRAM bank conflict 率可被 layout 改变显著影响，证明 bank-aware layout 生效。

## 9. 风险、取舍和后续细化方向

| 风险                        | 影响                              | 缓解                                                    |
| --------------------------- | --------------------------------- | ------------------------------------------------------- |
| BOA epilogue 过宽           | 面积和时序恶化，和 EVU 重叠       | V1 只做 bias/relu/requant 子集，复杂 elementwise 放 EVU |
| dataflow search 过早复杂化  | descriptor ABI 和验证面扩大       | V1 固定 output-stationary，保留 dataflow 字段           |
| accumulator 带宽不足        | OPA 利用率下降                    | accumulator banking、tile_k 限制、PMU 归因              |
| MFE/BOA slot ownership 不清 | page stream 与 operand fetch 冲突 | Slot Frame 声明 producer/consumer 和权限                |
| split-K 数值不可复现        | golden mismatch                   | descriptor 明确 reduction owner、order 和 rounding      |
| 高 fanout 控制              | 时序难闭合                        | decode 寄存、局部控制字、分层 reduce                    |

后续需要冻结：

- BOA descriptor binary ABI 的 version、size、alignment、endianness 和 validation 规则。
- 支持的 tile family、OPA shape、dtype 子集和 PPA target。
- SRAM profile：bank 数、端口、容量、延迟、bank conflict replay 成本。
- canonical 工作负载 trace：GEMM、dense attention、paged attention、MoE expert MLP。
- PMU counter 编号、宽度、overflow 行为和读取一致性。
