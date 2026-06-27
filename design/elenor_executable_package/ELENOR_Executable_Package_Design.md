# ELENOR Executable Package 设计文档

## 1. 定位、目标和 First Silicon cutline

Executable Package 是 compiler/runtime 与硬件执行域之间的离线交付物。它不承载高层 graph 解释逻辑，而把已经 partition、lowering、kernel 选择、descriptor template 生成和 command packing 后的内容打包成可校验、可加载、可 patch、可提交的二进制对象集合。

核心目标：

- 固化 graph schedule、group task、tile program、descriptor table、relocation table、weight section 和 runtime metadata 的组织方式。
- 支持 cold path 时 runtime 注册 program/descriptor/weight section，支持 warm path 时只 patch descriptor/context/shape metadata。
- 明确 Architecture V1 的长期 package 模型，同时给 First Silicon V1 一个可实现的最小切线。
- 保证硬件只执行 command、descriptor、TileGroupTask 和 Tile Program；program residency 由 hardware/firmware 隐式保证，而不是 package 中的显式 load 指令。

First Silicon V1 切线：

| 能力              | First Silicon V1 必须支持                                         | V1.x / V2 可扩展                                 |
| ----------------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| Package container | 单文件或目录式 package，固定 header、section table、CRC           | 签名、压缩、多芯片 shard                         |
| Program           | Tile Program section 存储，program_id 稳定                        | 多版本 program cache policy                      |
| Descriptor        | BOA、DMA、EVU、MFE Page Stream、基础 event descriptor template    | Segment Stream full mode、Sparse Block 预留 flag |
| Relocation        | context base IOVA、descriptor base、weight base、slot/frame patch | 更复杂的 residency、priority、QoS patch          |
| Runtime metadata  | command template、event table 初值、PMU manifest                  | profile-guided re-layout                         |
| Golden            | package manifest、section CRC、descriptor golden binary           | ABI 兼容性矩阵                                   |

所有 ABI v0 结构体只作为样例，不是最终冻结定义；field、alignment、endianness、versioning、validation、兼容策略由后续规格冻结。

## 2. 职责、非职责和 ownership

### 2.1 职责

Executable Package 负责：

1. 描述执行对象之间的静态关系：Graph Schedule -> TileGroupTask -> TileRoleBinding -> Tile Program -> Descriptor Template。
2. 保存 runtime 可直接上传到 device memory 的 program text、descriptor bytes、weight blob 和 const blob。
3. 提供 version、feature bit、target profile、section size、alignment、CRC 和 relocation 信息。
4. 提供 compiler 生成的 command template，使 runtime 在 launch 时无需重新理解 graph。
5. 提供 kernel library binding：每个 kernel_call 绑定具体 tile kernel 名称、版本、descriptor ABI 和 slot frame ABI。
6. 给 driver/firmware 做静态校验入口：禁止不匹配 ABI、越界 section、非法 relocation 和不支持的 feature。

### 2.2 非职责

Executable Package 不负责：

- 不在 device 侧解释 PyTorch/JAX/ONNX graph。
- 不在 hardware 中执行任意 tensor algebra。
- 不把每个 tile 的 program 都复制成独立二进制；推荐 Tile-SPMD program template。
- 不把绝对物理地址永久绑定到算子语义；package 只保存 relocation slot 和 descriptor template。
- 不负责 OS 级任务调度、IOMMU policy 或进程隔离；这些属于 driver/runtime。

### 2.3 Ownership

| 对象                | 生成方                                  | 加载/校验方           | 运行时 patch 方                        | 消费方               |
| ------------------- | --------------------------------------- | --------------------- | -------------------------------------- | -------------------- |
| Package header      | Compiler packer                         | User runtime / driver | 不 patch                               | Runtime              |
| Graph schedule      | Compiler scheduler                      | Firmware runtime      | context/queue/event base               | Device Runtime       |
| TileGroupTask       | Compiler scheduler + firmware           | Firmware              | group task metadata、role bindings     | Tile Group Sequencer |
| Tile Program        | Tile kernel library + compiler selector | Firmware / Tile UCE   | 不在 running 状态 patch                | Tile UCE             |
| Descriptor template | Compiler lowering                       | Runtime / firmware    | Runtime、Tile UCE、MFE、USE 分层 patch | BOA/EVU/MFE/USE/DMA  |
| Relocation table    | Compiler packer                         | Runtime               | Runtime                                | Runtime / firmware   |
| Weight blob         | Compiler/exporter                       | Runtime / driver      | base IOVA                              | DMA/BOA              |
| PMU manifest        | Compiler + runtime                      | Runtime               | counter routing                        | PMU tools            |

## 3. 微架构和状态机

Package 自身是软件对象，但必须映射到硬件装载状态机。推荐状态：

```text
Created
  -> Verified
  -> LoadedToHostRuntime
  -> BoundToContext
  -> UploadedToDevice
  -> ResidentReady
  -> Submitted
  -> Running
  -> Completed
  -> ReusableWarm
  -> Evicted
```

关键状态语义：

- `Verified`：完成 magic、version、section table、CRC、alignment、feature bit 和 ABI profile 检查。
- `BoundToContext`：绑定 context_id、IOMMU domain、queue_id、event base、fault record region。
- `UploadedToDevice`：program repository、descriptor table、weights、const、workspace manifest 已获得 device IOVA。
- `ResidentReady`：hot Tile Program 已可被 Tile UCE fetch 或可按需从 HBM 装入。
- `ReusableWarm`：program text 未变，只允许 descriptor/context/shape metadata patch。
- `Evicted`：program cache slot 或 descriptor cache 被回收，下一次 launch 需要走 cold 或 semi-cold path。

Residency 分层：

| 层级        | 存放对象                                     | 管理者              | Miss 行为                   |
| ----------- | -------------------------------------------- | ------------------- | --------------------------- |
| Host memory | package 原始文件、manifest                   | User runtime        | 重新 map 或重新读取 package |
| Device HBM  | program repository、descriptor table、weight | Runtime / driver    | DMA 上传                    |
| Tile SRAM   | hot Tile Program、hot descriptor cache       | Tile UCE / firmware | HBM -> Tile SRAM load       |

## 4. 接口、descriptor、寄存器和协议

### 4.1 Object model

Package object model：

```text
ElenorPackage
  Header
  TargetProfile
  SectionTable
  GraphSchedule
    GroupTaskTable
    RoleBindingTable
    DependencyTable
    MemoryLifetimeTable
    LaunchMetadata
  ProgramRepository
    TilePrograms[]
  DescriptorRepository
    TensorDescTable
    DmaDescTable
    BoaDescTable
    EvuDescTable
    MfeStreamDescTable
    UseStateDescTable
    EventInitTable
  CommandTemplates[]
  SlotFrameTemplates[]
  RelocationTable[]
  WeightSections[]
  ConstSections[]
  PmuManifest
  DebugManifest
```

### 4.2 Binary layout/versioning 示例

```c
#define ELENOR_PKG_MAGIC 0x45504B47u /* 'EPKG' */
#define ELENOR_PKG_ABI_V0_EXAMPLE 1u

typedef struct {
    uint32_t magic;
    uint16_t package_abi_version;
    uint16_t header_bytes;
    uint32_t package_flags;
    uint32_t target_profile_id;

    uint32_t section_count;
    uint32_t section_table_offset;

    uint32_t command_abi_version;
    uint32_t descriptor_abi_version;
    uint32_t slot_frame_abi_version;
    uint32_t stream_queue_abi_version;

    uint64_t package_bytes;
    uint64_t build_id_lo;
    uint64_t build_id_hi;
    uint32_t header_crc32;
    uint32_t package_crc32;
} elenor_pkg_header_v0_t;

typedef struct {
    uint32_t section_type;
    uint32_t section_flags;
    uint64_t file_offset;
    uint64_t file_bytes;
    uint64_t required_alignment;
    uint64_t device_alignment;
    uint32_t crc32;
    uint32_t reserved_or_zero;
} elenor_pkg_section_v0_t;
```

Versioning 规则：

- `package_abi_version` 控制 container 与 section table 兼容性。
- `command_abi_version`、`descriptor_abi_version`、`slot_frame_abi_version` 分开检查，避免 package 版本升级牵连所有 ABI。
- minor-compatible 变更只能追加字段或新增 section，不得改变已存在字段语义。
- runtime 必须拒绝 major 不兼容 package，并在 fault record 中记录 expected/actual version。
- 未冻结的 size、alignment、endianness 和 CRC policy 由后续规格冻结。

Section type 示例：

```c
typedef enum {
    ELENOR_SEC_GRAPH_SCHEDULE = 1,
    ELENOR_SEC_GROUP_TASK_TABLE = 2,
    ELENOR_SEC_TILE_PROGRAM = 3,
    ELENOR_SEC_DESCRIPTOR_TABLE = 4,
    ELENOR_SEC_COMMAND_TEMPLATE = 5,
    ELENOR_SEC_RELOCATION_TABLE = 6,
    ELENOR_SEC_WEIGHT = 7,
    ELENOR_SEC_CONST = 8,
    ELENOR_SEC_SLOT_FRAME_TEMPLATE = 9,
    ELENOR_SEC_PMU_MANIFEST = 10,
    ELENOR_SEC_DEBUG_MANIFEST = 11,
} elenor_pkg_section_type_v0_t;
```

### 4.3 API 示例

```c
typedef struct elenor_package elenor_package_t;
typedef struct elenor_loaded_package elenor_loaded_package_t;

int elenor_pkg_open(const void *bytes, uint64_t size, elenor_package_t **pkg);
int elenor_pkg_validate(const elenor_package_t *pkg, const elenor_device_caps_t *caps);
int elenor_pkg_bind_context(elenor_package_t *pkg, uint32_t context_id, uint32_t queue_id);
int elenor_pkg_upload(elenor_runtime_t *rt, elenor_package_t *pkg, elenor_loaded_package_t **loaded);
int elenor_pkg_patch_binding(elenor_loaded_package_t *loaded, const elenor_binding_table_t *bindings);
int elenor_pkg_make_launch(const elenor_loaded_package_t *loaded, uint32_t entry_id, elenor_launch_t *launch);
int elenor_pkg_release(elenor_loaded_package_t *loaded);
```

API 不得重新做 graph lowering，只能验证 package、绑定资源、patch relocation、生成 command submit payload。

## 5. 数据流、控制流和时序路径

### 5.1 Cold load/patch/submit flow

```text
Host runtime:
  open package and validate ABI tuple
  upload program / descriptor / weight sections
  register program_id -> section_id / iova / hash / abi metadata
  bind context, queue, event base, fault record base
  patch context-level relocation
  flush or invalidate descriptor cache if needed
  build launch command from command template
  ring doorbell

Device Runtime:
  consume command
  validate command and descriptor version
  parse graph schedule entry
  dispatch group task

Tile Group Sequencer:
  lookup Tile Program residency contract per role binding
  prefetch descriptor window and init stream queue/event resources
  dispatch prepared tile tasks per role

Tile UCE:
  accept prepared tile task with local program handle
  check local handle / frame generation / descriptor window
  run Tile-SPMD program
  signal tile done
```

### 5.2 Warm submit flow

```text
Host runtime:
  reuse program_id / section metadata
  patch descriptor fields changed by shape/context/buffer binding
  update event sequence
  ring doorbell

Device Runtime:
  validate command sequence number
  issue group task

Tile Group Sequencer / Tile UCE:
  hit resident Tile Program
  bind patched descriptor
  run without program reload
```

Warm path 的性能目标是把 launch 开销限制在 descriptor patch、event reset 和 doorbell，不重复搬运 program text，也不要求 compiler/runtime 发出显式 `program.load`。

### 5.3 时序关键路径

| 路径                                 | 风险                | 控制策略                                        |
| ------------------------------------ | ------------------- | ----------------------------------------------- |
| package validation -> submit         | 软件开销过高        | open/validate 与 launch 分离，validation 可缓存 |
| descriptor patch -> descriptor cache | stale descriptor    | patch 后显式 flush/invalidate policy            |
| program residency miss               | launch latency 抖动 | Program Table 与 hot kernel pinning             |
| event sequence reuse                 | 虚假 completion     | event_id + sequence 双字段                      |
| relocation 越界                      | address fault       | runtime 与 firmware 双层 bounds check           |

## 6. 配置、PPA、性能模型和 PMU

Package 需要携带 target profile，不硬编码某个配置的 SRAM 容量：

| 字段               | 用途                                       | 未冻结值                |
| ------------------ | ------------------------------------------ | ----------------------- |
| tile_count         | tile grid 上限                             | 由后续规格冻结          |
| group_count        | group grid 上限                            | 由后续规格冻结          |
| tile_l1_bytes      | slot frame capacity 校验                   | 由 SRAM profile 冻结    |
| group_sram_bytes   | region residency 校验                      | 由 SRAM profile 冻结    |
| noc_vc_count       | command/event/data/collective traffic 校验 | 由 PPA exploration 冻结 |
| program_sram_bytes | hot tile kernel 数量估计                   | 由 SRAM profile 冻结    |

Package PMU manifest：

- 列出每个 group task 的 expected PMU fingerprint，例如 BOA active、MFE prefetch miss、stream credit stall。
- 标记 command_id、task_id、program_id、descriptor_id 到 counter group 的映射。
- 支持 golden trace 对齐：每条 canonical workload trace 绑定 expected event order 和 counter owner。

性能模型应在 package build 阶段输出静态估计：

```text
T_launch = T_validate_cached + T_patch + T_doorbell + T_residency_miss
T_group_task = max(T_group_dma, T_tile_compute_pipeline, T_stream_backpressure)
T_warm = T_patch + T_event_reset + T_submit
```

PPA 相关 section 只作为 manifest，不驱动硬件时序。硬件最终 SRAM/NoC/clock/power 由 PPA exploration 冻结。

## 7. RTL/软件实现建议

### 7.1 Compiler packer

- 以 deterministic build 为目标，同一输入和 target profile 生成字节一致的 package。
- section table 按类型排序，所有 offset/alignment 显式记录。
- descriptor template 与 relocation 分离，避免 runtime 扫描任意 bytes。
- 生成 golden binary descriptor 文件，用于 FileCheck 与 byte-level regression。

### 7.2 Runtime loader

- validation cache 以 build_id + target_profile + ABI tuple 为 key。
- package upload 使用大粒度 DMA，descriptor patch 使用小粒度映射或 staging buffer。
- Program Registry 管理 `program_id -> section_id / HBM IOVA / version / hash`；group/tile local slot 由 Program Residency Manager 动态分配。
- warm launch 不修改 program text；需要新 program 时先完成 quiesce 或切换 epoch。

### 7.3 Firmware / RTL hook

- Device Runtime 只接收 command template 实例，不读取 package 文件格式。
- Tile Group Sequencer 通过 `program_id/template_id/program_iova` 触发隐式 residency lookup/fetch/verify/install。
- Tile Dispatcher 向 Tile UCE 发送 prepared tile task，只携带 local program handle，不携带 global program_iova。
- Tile UCE 执行 descriptor template patch，但只 patch 允许字段：tile_id、group_id、slot offset、local event id。
- MFE 拥有数据相关动态内存访问，例如 page list、segment offset 和 stream fill。
- USE 拥有 state slot 生命周期和 checkpoint/restore 语义。

### 7.4 Pass pipeline/dialect 策略

Executable Package 来自 compiler 后端 pass：

```text
stablehlo-to-linalg
canonicalize
shape-specialize
elenor-engine-partition
elenor-kernel-library-select
elenor-descriptor-template
elenor-memory-plan
elenor-command-template
elenor-package-layout
elenor-package-verify
```

Dialect 边界：

- BOA/EVU/MFE/USE dialect 产生 engine descriptor template。
- Runtime dialect 产生 command template、event dependency、group task launch。

Clean MLIR-like 示例：

```mlir
elenor.package @llm_block target(#elenor.profile<balanced>) {
  elenor.entry @decode_step(%q, %kv_pages, %out) {
    %task = elenor.group_task @paged_attention_task
    elenor.command_template launch_group_task %task
      wait_events = []
      signal_event = #elenor.event<graph_done>

  elenor.kernel_ref @page_attention_qk_v1
    descriptor = @qk_desc
    frame = @attention_frame
}
```

## 8. 验证、bring-up 和验收标准

### 8.1 Golden tests

| 测试                  | 输入                                  | 验收                                                                     |
| --------------------- | ------------------------------------- | ------------------------------------------------------------------------ |
| package header golden | 固定 package manifest                 | header bytes 与 golden 一致                                              |
| section CRC           | 随机 section reorder/损坏             | loader 拒绝损坏 package                                                  |
| relocation golden     | context/base/slot 多组 binding        | patch 后 descriptor bytes 与 golden 一致                                 |
| descriptor ABI golden | BOA/DMA/MFE/EVU descriptor template   | byte-level compare                                                       |
| cold launch trace     | program cache empty                   | event 顺序、隐式 residency miss/fetch/verify、descriptor validation 正确 |
| warm launch trace     | program resident                      | 不发生 program reload，只 patch descriptor 并复用 resident handle        |
| fault injection       | bad version、bad CRC、越界 relocation | fault record 含 package_id、section、offset、reason                      |

### 8.2 Verification plan

1. Compiler FileCheck：package IR 到 section manifest 的结构检查。
2. Binary golden：header、section table、descriptor table、relocation table 字节级回归。
3. Runtime unit：open/validate/bind/upload/patch/make_launch。
4. Firmware sim：command consume、program residency miss/hit、event sequence。
5. System trace：GEMM、dense attention、paged attention、MoE 三条 canonical package 端到端。
6. PMU 对齐：package manifest 中 expected fingerprint 与 runtime counter snapshot 对齐。

### 8.3 验收标准

- ABI 不匹配、CRC 错误、alignment 错误、section 越界必须 deterministic fail。
- Cold launch 与 warm launch 产生可区分 trace。
- Warm launch 不修改 program text，不触发不必要 program reload。
- descriptor patch owner 清晰：runtime patch context/base，Tile UCE patch tile/group/slot，MFE patch data-dependent stream，USE patch state lifecycle。
- First Silicon V1 至少支持 command -> DMA -> BOA -> event -> PMU 的 package 路径。

## 9. 风险、取舍和后续细化方向

| 风险                            | 影响                                      | 缓解                                                           |
| ------------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| Package ABI 过早复杂            | loader、firmware、compiler 同时不稳定     | container 最小化，descriptor ABI 分版本                        |
| Program residency 策略不清      | launch latency 抖动、cache stale          | Program Table、event quiesce、warm path 规则                   |
| Relocation 任意化               | patch 成本高且难验证                      | relocation entry 显式类型化                                    |
| Section 与 engine 语义耦合      | package 升级牵连 RTL                      | package 只承载 bytes、manifest、relocation                     |
| Descriptor cache coherence 漏洞 | warm launch 使用旧参数                    | patch 后严格 invalidate/flush                                  |
| Kernel library 版本漂移         | compiler 选择的 kernel 与 firmware 不匹配 | kernel_id、kernel_abi_version、descriptor_abi_version 同时校验 |

后续应冻结：package header exact layout、section alignment、CRC/签名策略、relocation type、Program Table entry、descriptor cache coherence、target profile 编码和 golden trace 格式。未冻结字段统一标记为由后续规格冻结。
