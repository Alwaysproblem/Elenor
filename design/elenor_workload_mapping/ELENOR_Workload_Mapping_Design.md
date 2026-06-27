# ELENOR Workload Mapping 设计文档

## 1. 定位、目标和 First Silicon cutline

Workload Mapping 文档定义 Dense Transformer、Paged Attention、MoE、SSM/Mamba/RWKV、Embedding/GNN 和多模型并发如何映射到 ELENOR 的 BOA、EVU、MFE、USE、Tile UCE、Tile Group Sequencer、Device Runtime、Stream Queue 和 Slot Frame。它面向 compiler pass、runtime package、kernel library、PMU model 和 verification trace。

核心原则：

- Dense compute 归 BOA。
- Irregular compute 归 EVU。
- 数据相关动态内存访问归 MFE。
- Stateful compute 归 USE。
- Program control 归 Runtime / Tile Group Sequencer / Tile UCE。
- 硬件执行 command、descriptor、program，不执行高层 graph。

First Silicon V1 切线：

| Workload        | First Silicon V1                                                            | 后续扩展                              |
| --------------- | --------------------------------------------------------------------------- | ------------------------------------- |
| GEMM/Conv       | BOA INT8/BF16 GEMM、split-K 基础 reduce                                     | epilogue fusion、advanced tiling      |
| Dense Attention | QK/AV BOA、softmax/norm EVU、event chain                                    | 更复杂 fusion                         |
| Paged Attention | MFE Page Stream、KV prefetch/reorder、BOA QK/AV、EVU softmax                | Sparse Block Stream                   |
| MoE             | router top-k、MFE Segment Stream、expert batching、BOA expert GEMM、combine | advanced load balance、atomic combine |
| SSM             | USE scan/recurrence、checkpoint/restore、BOA projection、EVU local op       | advanced recurrence transform         |
| Multi-model     | group partition、queue priority、SRAM quota metadata                        | preemption、QoS scheduler             |

ABI v0 结构体和 workload descriptor 都是样例，不是最终冻结定义；未冻结 field、canonical case 参数和 performance target 由后续规格冻结。

## 2. 职责、非职责和 ownership

### 2.1 职责

Workload mapping 负责：

1. 为每类 workload 指定 engine ownership、dataflow、control flow、descriptor set 和 PMU fingerprint。
2. 指导 compiler engine partition、kernel library selection、memory planning 和 command template generation。
3. 指导 runtime load/patch/submit：哪些 metadata 在 package build 固定，哪些 launch 时 patch。
4. 定义 golden tests 和 canonical traces。
5. 暴露风险：SRAM/NoC contention、stream backpressure、routing imbalance、state cache miss、fault handling。

### 2.2 非职责

- 不定义最终 RTL microarchitecture。
- 不冻结所有 tile size/page size/head_dim/prefetch depth；这些由后续规格冻结或由 SRAM/PPA profile 冻结。
- 不把 MFE 扩展成任意 graph traversal engine。
- 不让 Vector/EVU 承担 dense matrix 主路径。
- 不让 USE 取代 Tile UCE 控制 tile program。

### 2.3 Ownership matrix

| Workload component            | Owner                                  | 说明                                   |
| ----------------------------- | -------------------------------------- | -------------------------------------- |
| QKV/MLP/expert GEMM           | BOA                                    | Dense compute 主路径                   |
| softmax/norm/activation/tail  | EVU                                    | Predicated vector 与 mask/tail         |
| KV page walk/prefetch/reorder | MFE                                    | 数据相关动态地址和 stream fill         |
| token grouping/segment stream | MFE + EVU + optional USE               | MFE 管数据流，USE 可辅助 state/counter |
| recurrence/state update       | USE                                    | state slot、checkpoint/restore         |
| dynamic branch                | Runtime / Tile UCE                     | command path 和 tile-local branch      |
| role overlap                  | Tile Group Sequencer + Stream Queue    | credit/backpressure/EOS/error          |
| descriptor patch              | Compiler/Runtime/Tile UCE/MFE/USE 分层 | owner 不得重叠                         |

## 3. 微架构和状态机

### 3.1 通用 pipeline pattern

```text
TileGroupTask:
  init stream queues
  prefetch block metadata/data
  dispatch role 0
  dispatch role 1
  overlap DMA/MFE with BOA/EVU/USE
  propagate EOS/error
  signal group task done

Tile Program:
  pop stream token
  branch on EOS/error
  load or stream operands
  launch engine descriptors
  wait events
  push output token
  release input credit
  signal tile done
```

### 3.2 Stream Queue 状态

```text
ProducerAcquireCredit
  -> FillPayload
  -> PushToken
  -> ConsumerPop
  -> ConsumePayload
  -> ReleaseCredit
```

EOS/error token 必须沿 workload pipeline 传播。Paged Attention、MoE 和 GNN 都依赖 stream queue 的 credit/backpressure 来避免 role 间无限 buffering。

### 3.3 Workload-specific 状态机

Paged Attention：

```text
DecodeToken
  -> BuildOrReadKVMetadata
  -> MFEPageWalk
  -> KVPrefetchReorder
  -> BOAQK
  -> EVUScaleMaskSoftmax
  -> BOAAV
  -> StoreOutput
  -> UpdateEvents
```

MoE：

```text
RouterLogits
  -> TopK
  -> TokenGroup
  -> ExpertBatch
  -> ExpertGEMM
  -> Combine
  -> OutputScatterOrReduce
```

SSM：

```text
Projection
  -> LocalVectorOp
  -> StateRead
  -> ScanOrRecurrence
  -> StateWriteOrCheckpoint
  -> OutputProjection
```

## 4. 接口、descriptor、寄存器和协议

### 4.1 Object model

```text
WorkloadPlan
  workload_type
  shape_class
  engine_pipeline[]
  kernel_bindings[]
  descriptor_templates[]
  stream_queues[]
  slot_frames[]
  command_templates[]
  pmu_expectations[]
  golden_trace
```

### 4.2 Binary layout/versioning examples

Workload plan manifest 示例：

```c
typedef struct {
    uint32_t workload_plan_version;
    uint32_t workload_type;
    uint32_t shape_class_id;
    uint32_t flags;
    uint32_t command_template_offset;
    uint32_t descriptor_table_offset;
    uint32_t stream_table_offset;
    uint32_t slot_frame_offset;
    uint32_t pmu_manifest_offset;
    uint32_t golden_trace_id;
} elenor_workload_plan_v0_example_t;
```

Paged Attention descriptor linkage 示例：

```c
typedef struct {
    uint32_t plan_version;
    uint32_t page_stream_desc_id;
    uint32_t qk_boa_desc_id;
    uint32_t softmax_evu_desc_id;
    uint32_t av_boa_desc_id;
    uint32_t store_desc_id;
    uint32_t input_event;
    uint32_t output_event;
    uint32_t fault_record_slot;
} elenor_paged_attention_plan_v0_example_t;
```

MoE descriptor linkage 示例：

```c
typedef struct {
    uint32_t plan_version;
    uint32_t router_desc_id;
    uint32_t topk_desc_id;
    uint32_t segment_stream_desc_id;
    uint32_t expert_gemm_desc_base;
    uint32_t combine_desc_id;
    uint32_t expert_count;
    uint32_t capacity_policy;
} elenor_moe_plan_v0_example_t;
```

这些 layout 只表达对象引用，不冻结最终 ABI。exact enum、alignment、field width 和 extension policy 由后续规格冻结。

### 4.3 APIs

Compiler/runtime API 示例：

```c
int elenor_compile_workload_plan(const elenor_graph_t *graph, const elenor_target_profile_t *profile, elenor_package_t **pkg);
int elenor_select_shape_class(const elenor_loaded_package_t *pkg, const elenor_launch_params_t *params, uint32_t *shape_class_id);
int elenor_launch_workload(elenor_loaded_package_t *pkg, uint32_t entry_id, const elenor_launch_params_t *params, elenor_event_handle_t *done);
int elenor_read_workload_pmu(elenor_runtime_t *rt, uint32_t workload_instance_id, elenor_pmu_snapshot_t *out);
```

Workload mapping 通过 executable package 提供 entry，不要求 runtime 理解高层 graph。

### 4.4 Pass pipeline/dialect strategy

```text
-stablehlo-to-linalg
-elenor-workload-detect
-elenor-shape-specialize
-elenor-engine-partition
-elenor-kernel-library-select
-elenor-stream-plan
-elenor-slot-frame-plan
-elenor-workload-command-lowering
-elenor-descriptor-abi-lowering
-elenor-package-layout
-elenor-workload-golden-trace-emit
```

Dialect 使用：

- `elenor.boa`：GEMM、QK、AV、expert MLP、projection。
- `elenor.evu`：softmax、norm、activation、mask/tail、top-k 子步骤、combine。
- `elenor.mfe`：Page Stream、Segment Stream、embedding gather、layout/reorder。
- `elenor.use`：scan、recurrence、state checkpoint、routing counter assist。
- `elenor.runtime`：shape branch、launch_group_task、event、barrier、reset/fault path。

Clean MLIR-like workload 示例：

```mlir
elenor.workload @moe_ffn(%x, %router_w, %experts, %out) {
  %logits = elenor.boa.matmul %x, %router_w
  %topk = elenor.evu.topk %logits { k = 2 }
  %groups = elenor.mfe.segment_group %x by %topk
  %expert_out = elenor.boa.expert_batch %groups, %experts
  %combined = elenor.evu.combine %expert_out by %topk
  elenor.mfe.store %combined, %out
}
```

## 5. 数据流、控制流和时序路径

### 5.1 Dense Transformer mapping

```text
Q/K/V projection -> BOA
QK score         -> BOA
scale/mask       -> EVU
softmax          -> EVU
AV               -> BOA
output projection-> BOA
```

Command sequence：

```text
DMA weights -> BOA QKV -> BOA QK -> EVU scale/mask/softmax -> BOA AV -> BOA output -> event signal
```

关键：EVU softmax 与 BOA AV 之间需要 event dependency；large attention 需要 split-K/collective reduce。

### 5.2 Paged Attention mapping

```text
Runtime/Compiler KV metadata
  -> MFE page walk / KV prefetch / reorder
  -> BOA QK
  -> EVU scale / mask / softmax
  -> BOA AV
  -> MFE/DMA store output
```

Tile program pseudocode：

```text
launch.mfe desc_gather_k_pages -> e0
launch.mfe desc_gather_v_pages -> e1
waitall e0 | e1
launch.boa desc_qk_matmul -> e2
wait e2
launch.evu desc_scale_mask -> e3
wait e3
launch.evu desc_softmax -> e4
wait e4
launch.boa desc_av_matmul -> e5
wait e5
launch.mfe desc_store_output -> e6
wait e6
ret
```

关键性能条件：

```text
T_prefetch <= T_qk
```

若不满足，BOA operand stall 会上升，PMU 应显示 MFE prefetch miss/stream backpressure 与 BOA stall 相关。

### 5.3 MoE mapping

```text
Router logits       -> BOA / EVU
Top-k               -> EVU / USE assist
Token grouping      -> MFE Segment Stream
Expert weight fetch -> MFE / DMA
Expert GEMM         -> BOA
Combine             -> EVU / Collective
```

有效利用率近似：

```text
imbalance = max(tokens_per_expert) / avg(tokens_per_expert)
U_boa = 1 / imbalance
```

缓解策略：token sorting、expert batching、capacity padding、group-level expert placement、Segment Stream。duplicate index、capacity overflow、combine precision 必须由 descriptor mode 明确。

### 5.4 SSM / Mamba / RWKV mapping

```text
Projection        -> BOA
Depthwise/local   -> EVU
State update      -> USE
Scan/recurrence   -> USE
Output projection -> BOA
```

并行 scan：

```text
sequence -> chunks -> local scan -> chunk summary scan -> fixup -> state checkpoint
```

USE 管 state slot、state cache、checkpoint/restore；Tile UCE 发起 launch/wait/branch。

### 5.5 Embedding / GNN mapping

```text
indices / offsets -> MFE Segment Stream
embedding gather  -> MFE + EVU LSU
segment reduce    -> EVU / USE
feature transform -> BOA
```

主要瓶颈是随机访存与 coalescing。MFE 应将 index stream 合并成 burst stream，EVU 处理 tail、duplicate 和局部 reduce。

### 5.6 Multi-model mapping

- group-level partition：按 Tile Group 切分模型。
- command queue priority：latency-sensitive 模型优先。
- SRAM quota：避免一个模型占满 shared SRAM。
- PMU feedback：根据 stall、queue occupancy、NoC congestion 调整分配。
- fault isolation：context_id、queue_id、reset domain 分离。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 Canonical case 参数

| Workload        | 参数                                                               | 状态           |
| --------------- | ------------------------------------------------------------------ | -------------- |
| Dense attention | batch、heads、seq_len、head_dim、dtype                             | 由后续规格冻结 |
| Paged Attention | page size、head_dim、prefetch depth、stream depth、L1/L2 footprint | 由后续规格冻结 |
| MoE             | expert_count、top_k、capacity_factor、tokens_per_batch             | 由后续规格冻结 |
| SSM             | state_dim、chunk_size、checkpoint interval                         | 由后续规格冻结 |
| Embedding/GNN   | index distribution、segment length、duplicate policy               | 由后续规格冻结 |
| Multi-model     | context count、priority policy、SRAM quota                         | 由后续规格冻结 |

SRAM/NoC dependent 值由 SRAM profile 冻结或由 PPA exploration 冻结。

### 6.2 Performance models

```text
Perf = min(BOA_compute, EVU_compute, MFE_stream, USE_state, Memory_bw)

T_paged_attention =
  T_page_walk + T_prefetch + T_qk + T_softmax + T_av + T_writeback - T_overlap

T_moe =
  T_router + T_topk + T_grouping + max(T_expert_gemm_i) + T_combine

T_ssm =
  T_projection + T_local_op + T_scan_or_recurrence + T_state_write + T_output
```

### 6.3 PMU fingerprints

| Workload        | 期望 PMU                                                                  |
| --------------- | ------------------------------------------------------------------------- |
| GEMM            | BOA active 高，operand stall 与 SRAM conflict 可解释                      |
| Dense attention | BOA QK/AV 与 EVU softmax 交替，collective stall 可见                      |
| Paged Attention | MFE prefetch hit/miss、stream queue occupancy、BOA operand stall          |
| MoE             | routing imbalance、MFE segment stall、BOA utilization 波动、combine stall |
| SSM             | USE active、state cache hit/miss、event wait                              |
| Embedding/GNN   | MFE/EVU LSU replay、memory bandwidth、segment reduce stall                |
| Multi-model     | queue occupancy、QoS latency、NoC VC congestion、fault isolation          |

PMU primary owner 必须唯一，避免同一 stall cycle 多头计数。

## 7. RTL/软件实现建议

### 7.1 Compiler

- 先用 workload pattern table 映射 engine，不做不可解释的全局自动 search。
- 每类 workload 输出 canonical trace 和 descriptor golden。
- Memory planner 避免 BOA operand、EVU vector、MFE stream、USE state 同时打满同一 SRAM bank。
- Dynamic shape 采用有限 shape class + tail/ragged descriptor。

### 7.2 Runtime

- load package 后基于 shape class 选择 command path。
- patch context/base IOVA、page table base、KV cache base、state buffer base。
- Warm launch 复用 program residency，只更新 descriptor/event。
- Fault 后停止复用相关 descriptor cache，直到 reset/drain 完成。

### 7.3 Firmware/RTL

- Tile Group Sequencer 负责 pipeline role 和 stream queue 初始化。
- Tile UCE 负责 tile program control、L2->L1 DMA、engine launch/wait、descriptor patch。
- MFE Page/Segment Stream 要定义 EOS/error token、timeout、invalid page/segment 行为。
- USE checkpoint/restore 在 fault/reset path 下必须 deterministic。

### 7.4 Kernel library strategy

First Silicon V1 推荐 kernel：

```text
matmul_boa_v1
qkv_projection_boa_v1
attention_qk_boa_v1
softmax_evu_v1
attention_av_boa_v1
page_attention_qk_v1
page_attention_av_v1
moe_router_topk_v1
moe_expert_boa_v1
segment_group_mfe_v1
ssm_scan_use_v1
embedding_segment_v1
```

Compiler 选择 kernel，runtime 校验 ABI，firmware 管 residency，Tile UCE 执行 program control。

## 8. 验证、bring-up 和验收标准

### 8.1 Golden tests

| Workload        | Golden                             | Fault cases                        | PMU check                    |
| --------------- | ---------------------------------- | ---------------------------------- | ---------------------------- |
| GEMM            | Python matmul / dtype tolerance    | bad descriptor、DMA timeout        | BOA active/stall             |
| Dense attention | QK/softmax/AV reference            | mask/tail/event order              | BOA/EVU overlap              |
| Paged Attention | page reorder + attention reference | invalid page、EOS/error、timeout   | MFE prefetch vs BOA stall    |
| MoE             | 8/16 expert routing benchmark      | duplicate index、capacity overflow | imbalance vs BOA utilization |
| SSM             | Mamba/RWKV recurrence reference    | checkpoint fault/reset             | USE state cache              |
| Embedding/GNN   | segment reduce reference           | bad offset、duplicate policy       | MFE/EVU stall                |
| Multi-model     | context isolation trace            | one context fault                  | queue/QoS/fault isolation    |

### 8.2 Verification plan

1. Python golden：每个 workload 的 numerics、shape edge、fault scenario。
2. Compiler FileCheck：workload detect、engine partition、descriptor lowering。
3. Descriptor golden：binary descriptor、stream queue descriptor、slot frame。
4. Runtime trace：load/patch/submit/wait/fault。
5. RTL unit：BOA、EVU、MFE、USE standalone。
6. Tile integration：command queue + SRAM + Tile UCE + engine smoke。
7. Group integration：stream queue、collective、DMA overlap。
8. System integration：driver + firmware + runtime end-to-end。
9. Performance validation：PMU fingerprint 与 performance model 对齐。

### 8.3 Bring-up order

1. command/event/barrier。
2. DMA 1D/2D。
3. BOA GEMM。
4. PMU basic。
5. EVU elementwise/mask/tail。
6. EVU softmax/norm。
7. MFE Page Stream。
8. Paged Attention end-to-end。
9. EVU gather/scatter。
10. MFE Segment Stream。
11. USE scan/recurrence。
12. MoE dispatch。
13. multi-model scheduling。

### 8.4 验收标准

- 每个 workload 都能从 compiler package 通过 runtime command queue 启动，不绕过控制面。
- Golden numerics、event order、fault behavior 和 PMU fingerprint 都有可复现 trace。
- Paged Attention 必须证明 page reorder、timeout、invalid page 行为确定。
- MoE 必须证明 duplicate/capacity/combine policy 与 descriptor mode 一致。
- SSM 必须证明 checkpoint/restore 在 reset path 下行为确定。

## 9. 风险、取舍和后续细化方向

| 风险                    | 影响                      | 缓解                                                 |
| ----------------------- | ------------------------- | ---------------------------------------------------- |
| Workload 覆盖过宽       | 每类只完成部分路径        | 按 bring-up 顺序冻结 canonical traces                |
| MFE 过度设计            | 验证面积失控              | V1 Page + Segment，Sparse Block 后续                 |
| USE 范围不清            | 演化成通用控制器          | 绑定 scan/recurrence/checkpoint/event assist         |
| MoE imbalance           | BOA 利用率低              | token sorting、expert batching、PMU feedback         |
| Paged Attention latency | BOA 等 KV stream          | prefetch overlap，`T_prefetch <= T_qk` 作为目标 case |
| SRAM/NoC contention     | 多 engine 互相阻塞        | bank-aware layout、NoC VC、PMU primary owner         |
| Dynamic shape 爆炸      | package/verification 变大 | shape class + mask/tail + MFE ragged descriptor      |
| Multi-model fault 扩散  | 影响其他 context          | context isolation、reset domain、fault record        |

后续应冻结：canonical workload 参数、descriptor mode、page/segment semantics、duplicate policy、state checkpoint ABI、kernel library manifest、PMU counter mapping 和 performance target。未冻结项统一写为由后续规格冻结。
