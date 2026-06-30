# ELENOR Compiler Stack 设计文档

## 1. 定位、目标和 First Silicon cutline

ELENOR Compiler Stack 把 PyTorch/JAX/ONNX/StableHLO/Torch-MLIR 等高层表示 lowering 为 ELENOR executable package、descriptor template、command template、Tile-SPMD kernel binding 和 runtime metadata。Compiler 不直接生成硬件内部每个 engine 的所有控制细节；First Silicon V1 依赖 tile kernel library，compiler 负责 pattern-based selection、descriptor template generation、command buffer packing 和 golden binary descriptor。

核心边界：

```text
High-level graph
  -> StableHLO / Torch-MLIR
  -> Tensor/Linalg
  -> Shape specialization
  -> Engine partition
  -> BOA/EVU/MFE/USE/Runtime dialects
  -> Descriptor template
  -> Command template
  -> Executable package
```

First Silicon V1 切线：

| 能力       | First Silicon V1                                                        | 后续扩展                             |
| ---------- | ----------------------------------------------------------------------- | ------------------------------------ |
| Frontend   | StableHLO/Torch-MLIR 子集导入                                           | eager/dynamic graph 更完整覆盖       |
| Partition  | pattern-based engine partition                                          | cost-model 自动 partition            |
| Kernel     | tile kernel library selection                                           | compiler 生成更多 tile program       |
| Descriptor | BOA GEMM、DMA、EVU elementwise/softmax、MFE Page Stream、USE state 样例 | Segment/Sparse/Persistent 扩展       |
| Runtime    | command template、event dependency、package manifest                    | PMU feedback scheduler               |
| Tests      | FileCheck、golden descriptor、canonical traces                          | fuzzing、profile-guided optimization |

ABI v0 结构体和 descriptor layout 只作为样例，不是最终冻结定义；field、alignment、endianness、versioning 和兼容策略由后续规格冻结。

## 2. 职责、非职责和 ownership

### 2.1 职责

Compiler 负责：

1. 从高层 graph 中识别 ELENOR 支持的 workload pattern：dense GEMM/Conv/Attention、Paged Attention、MoE、SSM、Embedding/GNN、多模型并发。
2. Shape specialization 与 dynamic shape multi-versioning。
3. Engine partition：BOA、EVU、MFE、USE、Runtime/Tile UCE 的职责映射。
4. 选择 tile kernel library 中的 kernel template。
5. 生成 descriptor template、slot frame template、共享 `TensorView` 语义的 view binding、stream queue descriptor 和 event dependency。
6. 生成 executable package：section table、program reference、descriptor table、relocation table、command template、PMU manifest。
7. 生成 golden tests：FileCheck IR、golden binary descriptor、canonical workload trace。

### 2.2 非职责

Compiler 不负责：

- 不让硬件直接解释高层 graph。
- First Silicon V1 不从零生成所有 tile microcode。
- 不在 compiler 中实现 driver memory policy 或 IOMMU。
- 不把 data-dependent dynamic memory access 放入 UCE 控制程序；MFE 拥有 page/segment stream 的数据相关访问。
- 不把 USE 扩展成通用控制器；USE 管 state/scan/recurrence。

### 2.3 Ownership

| 对象                | Compiler owner             | Runtime/Firmware owner        | Hardware consumer    |
| ------------------- | -------------------------- | ----------------------------- | -------------------- |
| Shape class         | shape-specialize pass      | launch-time branch            | Runtime / Tile UCE   |
| Engine partition    | engine-partition pass      | 不重分区                      | BOA/EVU/MFE/USE      |
| Tile kernel binding | kernel-library-select pass | validate residency            | Tile UCE             |
| Descriptor template | engine dialect lowering    | context/tile/data/state patch | Engines              |
| Slot frame template | memory-plan pass           | bind physical/local slots     | Tile UCE / DMA       |
| Stream queue        | pipeline-schedule pass     | init/reset/drain              | Tile Group Sequencer |
| Command template    | runtime lowering pass      | instantiate/submit            | Device Runtime       |
| PMU manifest        | performance pass           | read/compare                  | PMU tools            |

## 3. 微架构和状态机

Compiler pipeline 状态机：

```text
ImportedGraph
  -> CanonicalTensorIR
  -> ShapeSpecializedIR
  -> PartitionedEngineIR
  -> KernelBoundIR
  -> DescriptorTemplateIR
  -> RuntimeCommandIR
  -> MemoryPlannedIR
  -> PackagedBinary
  -> VerifiedArtifact
```

每个状态的 exit 条件：

- `ShapeSpecializedIR`：所有 First Silicon V1 kernel 的 rank/dtype/layout 已知；动态 shape 被分类为有限版本或交给 mask/ragged descriptor。
- `PartitionedEngineIR`：每个 op/group 有唯一 primary engine owner，跨 engine 边界通过 descriptor/event/stream 表达。
- `KernelBoundIR`：每个 kernel_call 绑定 kernel_id、kernel ABI、descriptor ABI、slot frame ABI。
- `DescriptorTemplateIR`：静态字段已填，runtime/tile/MFE/USE patch 字段显式标记。
- `RuntimeCommandIR`：event dependency、barrier、timeout、fault slot 完整。
- `MemoryPlannedIR`：TensorView/view binding 已解析到 slot frame backing store；live writable alias 不重叠，phase-disjoint alias 有显式 barrier/release。
- `PackagedBinary`：section、relocation、CRC、version manifest 完整。

## 4. 接口、descriptor、寄存器和协议

### 4.1 Object model

```text
CompilerModule
  TargetProfile
  ShapeClasses[]
  EnginePartitions[]
  KernelBindings[]
  DescriptorTemplates[]
  SlotFrameTemplates[]
  StreamQueues[]
  RuntimeCommands[]
  Relocations[]
  GoldenArtifacts[]

KernelBinding
  kernel_id
  kernel_abi_version
  descriptor_abi_version
  supported_dtype_layout
  required_slots
  pmu_fingerprint
```

### 4.2 Compiler APIs

Compiler 对外 API 以 target profile 和 executable package 为边界，不暴露内部 pass 的临时 IR 给 runtime。

```c
int elenor_compiler_create(const elenor_compiler_options_t *opts, elenor_compiler_t **out);
int elenor_compile_module(elenor_compiler_t *compiler, const elenor_input_module_t *input, elenor_compiled_module_t **out);
int elenor_emit_package(elenor_compiled_module_t *module, const elenor_target_profile_t *profile, elenor_package_blob_t *out);
int elenor_emit_descriptor_golden(elenor_compiled_module_t *module, elenor_golden_blob_t *out);
int elenor_emit_workload_trace(elenor_compiled_module_t *module, elenor_trace_blob_t *out);
```

### 4.3 Binary layout/versioning 示例

Compiler 输出的 manifest 必须让 runtime 在不读取 compiler IR 的情况下完成 load/patch/submit。以下 layout 是 ABI v0 样例，不是最终冻结定义。

```c
typedef struct {
    uint32_t compiler_manifest_version;
    uint32_t target_profile_id;
    uint32_t descriptor_abi_version;
    uint32_t command_abi_version;
    uint32_t kernel_binding_count;
    uint32_t shape_class_count;
    uint32_t command_template_count;
    uint32_t relocation_count;
} elenor_compiler_manifest_v0_example_t;

typedef struct {
    uint32_t kernel_id;
    uint32_t kernel_abi_version;
    uint32_t descriptor_abi_version;
    uint32_t slot_frame_abi_version;
    uint32_t required_feature_bits;
    uint32_t pmu_fingerprint_id;
} elenor_kernel_binding_v0_example_t;
```

Versioning 规则：compiler manifest、package container、command ABI、descriptor ABI 和 kernel ABI 分开编号；runtime 必须同时校验 target profile 与 ABI tuple。未冻结的 enum、alignment 和 feature bit 由后续规格冻结。

### 4.4 Dialect strategy

| Dialect          | 语义                                                                   | 输出                    |
| ---------------- | ---------------------------------------------------------------------- | ----------------------- |
| `elenor.boa`     | dense GEMM/Conv/QK/AV/expert MLP                                       | BOA descriptor template |
| `elenor.evu`     | elementwise、activation、softmax、norm、mask/tail、gather/scatter 子集 | EVU descriptor template |
| `elenor.mfe`     | Page Stream、Segment Stream、layout/reorder/data stream                | MFE stream descriptor   |
| `elenor.use`     | state、scan、recurrence、checkpoint/restore                            | USE state descriptor    |
| `elenor.runtime` | command、event、barrier、branch_shape、launch_group_task               | command template        |
| `elenor.package` | section、relocation、manifest、kernel binding                          | executable package      |

Dialect 不应互相吞并：BOA dialect 不表达 event wait；Runtime dialect 不表达 matmul tile layout；MFE dialect 不表达 high-level graph traversal。

### 4.5 Clean MLIR-like pseudocode

```mlir
func.func @decode_step(%q: tensor<?x?xbf16>, %kv: !elenor.kv_pages, %out: tensor<?x?xbf16>) {
  %shape = elenor.runtime.shape_class %q
  elenor.runtime.branch %shape {
    case #elenor.shape<short> {
      elenor.runtime.launch @dense_attention_short(%q, %kv, %out)
    }
    case #elenor.shape<paged> {
      elenor.runtime.launch @paged_attention_task(%q, %kv, %out)
    }
  }
  return
}

elenor.group_task @paged_attention_task(%q, %kv, %out) {
  %pages = elenor.mfe.page_stream %kv
    page_table = @kv_page_table
    prefetch_depth = #elenor.param<由后续规格冻结>

  %score = elenor.boa.matmul %q, %pages.k
    tile = #elenor.tile<由后续规格冻结>

  %prob = elenor.evu.softmax %score
    mask = #elenor.mask<causal>

  %value = elenor.boa.matmul %prob, %pages.v
  elenor.mfe.store %value, %out
  elenor.runtime.signal #elenor.event<group_task_done>
}
```

### 4.6 Descriptor template 示例

```c
typedef struct {
    uint32_t descriptor_abi_version;
    uint32_t descriptor_bytes;
    uint32_t engine_type;
    uint32_t op_type;
    uint32_t static_flags;
    uint32_t patch_mask;
    uint64_t tensor_a_template;
    uint64_t tensor_b_template;
    uint64_t tensor_c_template;
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
    uint32_t event_in;
    uint32_t event_out;
} elenor_desc_template_v0_example_t;
```

`patch_mask` 明确哪些字段由 runtime、Tile UCE、MFE 或 USE patch。未声明字段不得被 runtime 修改。

### 4.7 Pass pipeline

推荐 pass pipeline：

```text
-stablehlo-to-linalg
-canonicalize
-cse
-linalg-tile-and-fuse
-elenor-shape-specialize
-elenor-engine-partition
-elenor-kernel-library-select
-elenor-tensor-view-alias-plan
-elenor-slot-frame-plan
-elenor-boa-desc-template
-elenor-evu-irregular-lowering
-elenor-mfe-stream-detect
-elenor-use-state-detect
-elenor-bufferize
-elenor-memory-plan
-elenor-runtime-command-buffer-lowering
-elenor-descriptor-abi-lowering
-elenor-package-layout
-elenor-golden-artifact-emit
```

## 5. 数据流、控制流和时序路径

### 5.1 Compile-time flow

```text
Input model
  -> frontend import
  -> graph canonicalization
  -> shape class creation
  -> engine partition
  -> tile/kernel selection
  -> descriptor template generation
  -> memory and slot frame planning
  -> stream/event scheduling
  -> command template packing
  -> package emission
  -> golden trace emission
```

### 5.2 Runtime load/patch/submit flow emitted by compiler

Compiler 生成 runtime metadata，使 runtime 执行：

```text
load package
allocate buffers and events
upload program/descriptor/weight
patch context-level relocation
select command path by shape class
submit command sequence
wait completion event
read PMU/fault if needed
```

Compiler 必须显式输出：binding table、relocation table、shape class table、entry table、event dependency、PMU manifest。Runtime 不应通过反向解析 high-level IR 做这些决定。

### 5.3 Engine partition rules

| Pattern                              | Target                   | 边界                                      |
| ------------------------------------ | ------------------------ | ----------------------------------------- |
| matmul/conv/dense attention QK/AV    | BOA                      | 大规模 dense compute                      |
| softmax/norm/activation/tail         | EVU                      | irregular scalar/vector path              |
| gather/scatter/layout/page/KV stream | MFE + EVU                | data-related dynamic memory access 归 MFE |
| MoE token dispatch                   | MFE + EVU + optional USE | grouping/segment stream/combine           |
| scan/recurrence/state update         | USE                      | state lifecycle 归 USE                    |
| dynamic branch                       | Runtime / Tile UCE       | branch over command/program path          |

### 5.4 Tile-SPMD 与 kernel library

Compiler 不为每个 tile 生成独立 program，而生成 group/grid launch 和 descriptor offset。Tile program 通过 tile_id、group_id、slot frame 和 descriptor template 决定数据切片。

Kernel library manifest 示例：

```text
matmul_boa_v1:
  descriptor_abi = 由后续规格冻结
  slots = A,B,C,workspace
  dtype = int8,bf16
  pmu = boa_active,boa_operand_stall,sram_bank_conflict

page_attention_qk_v1:
  descriptor_abi = 由后续规格冻结
  slots = Q,K_pages,score,metadata
  engines = MFE,BOA,EVU
  pmu = mfe_prefetch,boa_operand_stall,stream_backpressure
```

## 6. 配置、PPA、性能模型和 PMU

Compiler target profile 输入：

| 参数                   | 用途                 | 未冻结值                |
| ---------------------- | -------------------- | ----------------------- |
| tile_count/group_count | grid mapping         | 由后续规格冻结          |
| tile_l1_bytes          | slot frame capacity  | 由 SRAM profile 冻结    |
| group_sram_bytes       | region staging       | 由 SRAM profile 冻结    |
| SRAM banks/ports       | bank-aware layout    | 由 SRAM profile 冻结    |
| BOA peak               | tiling/cost model    | 由 PPA exploration 冻结 |
| MFE bandwidth          | Page Stream overlap  | 由 PPA exploration 冻结 |
| EVU lanes              | softmax/norm mapping | 由 PPA exploration 冻结 |
| USE state cache        | recurrence chunking  | 由 SRAM profile 冻结    |

Compiler static performance estimates：

```text
Perf = min(BOA_compute, EVU_compute, MFE_stream, USE_state, Memory_bw)
T_paged_attention = T_page_walk + T_prefetch + T_qk + T_softmax + T_av + T_writeback - T_overlap
```

Compiler 输出 PMU expectation：

- GEMM：BOA active 高，operand stall 与 SRAM conflict 可解释。
- Paged Attention：MFE prefetch 与 BOA QK overlap，`T_prefetch <= T_qk` case 中 BOA stall 降低。
- MoE：routing imbalance 与 BOA utilization 对齐。
- SSM：USE active/state cache hit 与 recurrence chunk 对齐。

## 7. RTL/软件实现建议

### 7.1 Compiler implementation

- 每个 lowering pass 有单一职责和 FileCheck 测试。
- Descriptor ABI lowering 必须输出 byte-level golden。
- Shape specialization 先覆盖固定 shape 和有限 dynamic shape class，再扩展 search。
- Engine partition 初期使用 pattern table，不引入难解释的全局自动 scheduler。
- Memory planner 必须避免 BOA、EVU、MFE、USE 的 SRAM 峰值访问叠加。

### 7.2 Package emitter

- 生成 deterministic executable package。
- 将 descriptor template 与 relocation table 分离。
- 生成 command template，不让 runtime 重建 dependency graph。
- 记录 kernel library binding 和 ABI tuple。

### 7.3 Runtime/compiler contract

Compiler 产物必须告诉 runtime：

- entry_id 到 command span 的映射。
- binding_id 到 descriptor relocation 的映射。
- shape class 到 command path 的映射。
- event id allocation strategy。
- fault record slot policy。
- PMU counter list。

### 7.4 Golden and debug artifacts

- IR dump：每个 pass 后可稳定打印。
- Descriptor golden：二进制 bytes + human-readable decode。
- Command trace：command_id、event wait/signal、descriptor_id。
- Workload trace：dense attention、paged attention、MoE、SSM canonical cases。
- PMU manifest：expected counter owner 和 qualitative fingerprint。

### 7.5 Fault handling strategy

Compiler 必须在编译期拒绝 unsupported op、unsupported dtype/layout、slot capacity overflow、kernel ABI mismatch 和 descriptor ABI mismatch。对 runtime 才能确定的错误，compiler 必须在 package 中保留 fault attribution metadata：command_id、descriptor_id、kernel_id、shape_class_id、fault_record_slot 和 human-readable decode。编译器不得通过生成 fallback 空 kernel 掩盖不支持路径；无法映射的 workload 必须产生明确 diagnostic。

## 8. 验证、bring-up 和验收标准

### 8.1 Golden tests

| 测试                     | 覆盖                                       | 验收                                       |
| ------------------------ | ------------------------------------------ | ------------------------------------------ |
| FileCheck per pass       | dialect 边界、op lowering                  | IR pattern 稳定                            |
| descriptor binary golden | BOA/EVU/MFE/USE/DMA descriptor             | bytes 与 golden 一致                       |
| package golden           | section/relocation/command template        | deterministic build                        |
| memory plan              | slot capacity / bank hint / alias lifetime | 不超过 SRAM profile，writable alias 不重叠 |
| kernel binding           | kernel ABI/descriptor ABI mismatch         | compile 或 load 拒绝                       |
| workload canonical       | GEMM/attention/paged/MoE/SSM               | Python golden 对齐                         |
| fault compile tests      | unsupported op/layout/dtype                | 明确 diagnostic                            |

### 8.2 Bring-up alignment

- Phase 0：冻结 ownership、ABI v0 样例、golden trace 格式。
- Phase 1：`boa.gemm` descriptor lowering + command queue GEMM。
- Phase 2：EVU softmax/norm/tail lowering + random tensor tests。
- Phase 3：MFE Page Stream + paged attention trace。
- Phase 4：MFE Segment + MoE routing imbalance benchmark。
- Phase 5：USE scan/recurrence + checkpoint/restore golden。
- Phase 6：multi-model context/priority/quota compiler metadata。

### 8.3 Verification plan

1. Frontend import tests：StableHLO/Torch-MLIR 子集。
2. Pass regression：每个 pass FileCheck。
3. ABI regression：descriptor、command、package bytes。
4. Python golden：operator-level 与 workload-level compare。
5. Runtime integration：load package、patch descriptor、submit command、read event。
6. RTL integration：compiler 生成的 command 不能绕过 firmware queue。
7. Performance validation：PMU fingerprint 与 compiler static model 对齐。

### 8.4 验收标准

- First Silicon V1 的 matmul、softmax/norm、paged attention、MoE dispatch、SSM recurrence 都有 canonical compiler trace。
- 所有 descriptor template 都有 owner-tagged patch fields。
- Runtime command buffer lowering 生成可由 ABI 文档消费的 command sequence。
- Unsupported workload 有明确 diagnostic，不生成伪可执行 package。
- MLIR-like 示例不包含 harness artifacts。

## 9. 风险、取舍和后续细化方向

| 风险                     | 影响                          | 缓解                                                 |
| ------------------------ | ----------------------------- | ---------------------------------------------------- |
| Compiler complexity 过高 | 每个 engine 都不稳定          | pattern-based first deliverable，tile kernel library |
| Dialect 边界混乱         | Runtime/engine 语义耦合       | BOA/EVU/MFE/USE/Runtime/Package 分层                 |
| Descriptor ABI 频繁变化  | runtime/RTL 联调困难          | golden binary descriptor，version/size/reserved      |
| Memory planner 不准      | SRAM/NoC contention           | SRAM profile + PMU feedback + bank-aware layout      |
| Dynamic shape 爆炸       | package 太大                  | finite shape class + tail/ragged descriptor          |
| Kernel library 版本漂移  | load 后失败或 silent mismatch | kernel ABI 与 descriptor ABI 同时校验                |
| 自动 partition 不可解释  | 性能调优困难                  | 初期 pattern table，后续引入 cost model              |

后续应冻结：ELENOR dialect op/type、descriptor exact layout、slot frame ABI、shape class policy、kernel library manifest、package section schema、canonical trace schema 和 PMU counter mapping。未冻结项统一写为由后续规格冻结。
