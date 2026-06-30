# ELENOR Tile Slot Frame 设计文档

## 1. 定位、目标和 First Silicon cutline

Tile Slot Frame 是 ELENOR Tile L1 SRAM 的二进制 binding contract。它把 Tile Program 和 descriptor 中的逻辑 slot 引用绑定到可变的 L1 base、size、layout、权限、bank placement 和生命周期，避免把固定物理地址永久绑定到算子语义。

核心原则：

```text
固定 slot ABI + 可变 Tile Frame
```

硬件执行 command、descriptor 和 Tile Program；slot frame 只描述 L1 memory binding。它不解释高层 graph，不替代 compiler memory planner，也不提供全局 cache coherency。

First Silicon cutline：

| 能力           | First Silicon V1                                           | V1.x / V2 保留                               |
| -------------- | ---------------------------------------------------------- | -------------------------------------------- |
| slot 数量      | 16 个固定 slot                                             | 可扩展 slot table、dynamic allocation        |
| 权限           | read、write、accumulate、persistent、bank_pinned           | fine-grained engine mask、capability token   |
| 生命周期       | per-command、per-tile-program、per-role、resident          | preemption save/restore                      |
| patch          | tile_id/group_id/slot offset auto-patch                    | complex affine patch、late binding optimizer |
| coherency      | descriptor cache invalidate、program text running 禁 patch | hardware descriptor coherence                |
| bank placement | compiler/runtime hint + hardware check                     | automatic bank remap                         |

## 2. 职责、非职责和 ownership

### 2.1 ownership matrix

| 对象 / 动作                        | owner                           | 说明                                                         |
| ---------------------------------- | ------------------------------- | ------------------------------------------------------------ |
| slot role、layout、alignment       | Compiler                        | package build 时由 kernel library ABI 和 memory planner 决定 |
| context base、IOVA、residency      | Runtime / firmware              | load package、bind context、warm launch patch                |
| frame bind                         | Tile UCE                        | launch Tile Program 前绑定 frame，检查版本和权限             |
| tile_id/group_id/slot offset patch | Tile UCE auto-patch             | per tile launch 时生成 effective address                     |
| page list / segment offset patch   | MFE                             | 数据相关动态地址由 MFE 管理                                  |
| state slot / checkpoint pointer    | USE / Tile UCE                  | USE 拥有 state 生命周期，UCE 发起控制                        |
| descriptor cache invalidate        | Runtime / firmware + Tile UCE   | warm patch 后确保 UCE 看到新 descriptor                      |
| slot violation fault               | Tile UCE / DMA / engine wrapper | 记录 command id、program id、tile id、slot id                |

### 2.2 非职责

- Slot Frame 不分配 HBM，不做 IOMMU translation。
- Slot Frame 不决定高层 tensor lifetime；compiler/runtime 必须显式生成 frame。
- Slot Frame 不允许 running program text 被 patch。
- Slot Frame 不自动解决 bank conflict；它只提供 bank policy 和 PMU attribution 所需 metadata。

## 3. 微架构和状态机

### 3.1 L1 SRAM layout

```text
Tile L1 SRAM
├── Program Region       # tile.text
├── Descriptor Region    # tile.desc
├── Const Region         # tile.const
├── Event/Status Region  # tile.event
├── Slot 0: A / input
├── Slot 1: B / input
├── Slot 2: C / accumulator
├── Slot 3: workspace
├── Slot 4: metadata / page list / shape info
├── Slot 5: state / checkpoint
└── Slot 6..15: kernel-specific binding
```

推荐 2 MB / Tile 的起始 profile：Program/Descriptor/Event 128 KB，BOA operand 512 KB，BOA accumulator 384 KB，EVU vector 256 KB，MFE stream 384 KB，USE state 128 KB，DMA staging/shared 256 KB。具体分区由 SRAM profile 冻结。

### 3.2 frame bind 状态机

```text
FRAME_IDLE
  -> FETCH_FRAME_DESC
  -> VALIDATE_ABI
  -> VALIDATE_SLOT_TABLE
  -> CHECK_OVERLAP_ALIGNMENT
  -> CHECK_BANK_POLICY
  -> INSTALL_SHADOW
  -> FRAME_ACTIVE

错误边：
VALIDATE_ABI / VALIDATE_SLOT_TABLE / CHECK_OVERLAP_ALIGNMENT / CHECK_BANK_POLICY
  -> INVALID_DESCRIPTOR_FAULT
  -> FRAME_FAULTED
```

`INSTALL_SHADOW` 后，Tile UCE、Tile DMA、MFE tile port、BOA、EVU、USE wrapper 都只访问 shadow copy，避免运行中 descriptor memory 被软件修改影响正在执行的 tile。

### 3.3 descriptor patch 状态机

```text
PATCH_IDLE
  -> SELECT_TEMPLATE
  -> READ_FRAME_SHADOW
  -> COMPUTE_EFFECTIVE_ADDR
  -> CHECK_PERMISSION
  -> WRITE_DESC_SHADOW
  -> PATCH_COMMIT
  -> PATCH_DONE

错误边：
READ_FRAME_SHADOW / COMPUTE_EFFECTIVE_ADDR / CHECK_PERMISSION
  -> PATCH_FAULT_RECORD
  -> PATCH_ABORT
```

规则：

- patch 失败不得 launch engine。
- patch commit 后 descriptor read 必须看到新值。
- warm launch 可 patch descriptor data，但必须先 invalidate/flush descriptor cache。
- program text 在 running 状态下不可 patch；违反时产生 invalid descriptor fault。

### 3.4 slot 生命周期状态机

```text
UNBOUND -> BOUND -> IN_USE -> PRODUCED -> CONSUMED -> REUSABLE -> UNBOUND
                         \-> RESIDENT_VALID -> RESIDENT_DIRTY
FAULTED -> RESET_CLEAN
```

生命周期语义：

- per-command：单个 engine task 的 input/output/workspace；event 完成后可释放。
- per-tile-program：Tile Program 内多个 engine 复用；Tile Program 结束后释放。
- per-role：role 间缓存或 partial result；Tile Group Sequencer / Stream Queue 管理可见性。
- resident：program、const、hot descriptor、USE state cache；只有 firmware/runtime 或 checkpoint path 可替换。

## 4. 接口、descriptor、寄存器和协议

### 4.1 binary ABI v0

```c
#define ELENOR_TILE_SLOT_COUNT 16

typedef enum {
    ELENOR_SLOT_INPUT        = 1u << 0,
    ELENOR_SLOT_OUTPUT       = 1u << 1,
    ELENOR_SLOT_ACCUMULATOR  = 1u << 2,
    ELENOR_SLOT_WORKSPACE    = 1u << 3,
    ELENOR_SLOT_METADATA     = 1u << 4,
    ELENOR_SLOT_CONST        = 1u << 5,
    ELENOR_SLOT_STATE        = 1u << 6,
    ELENOR_SLOT_PROGRAM      = 1u << 7,
    ELENOR_SLOT_EVENT_STATUS = 1u << 8,
} elenor_slot_role_t;

typedef enum {
    ELENOR_SLOT_READ         = 1u << 0,
    ELENOR_SLOT_WRITE        = 1u << 1,
    ELENOR_SLOT_ACCUMULATE   = 1u << 2,
    ELENOR_SLOT_PERSISTENT   = 1u << 3,
    ELENOR_SLOT_BANK_PINNED  = 1u << 4,
    ELENOR_SLOT_EXECUTE      = 1u << 5,
    ELENOR_SLOT_NO_DMA_WRITE = 1u << 6,
} elenor_slot_flags_t;

typedef enum {
    ELENOR_SLOT_LIFE_PER_COMMAND      = 0,
    ELENOR_SLOT_LIFE_PER_TILE_PROGRAM = 1,
    ELENOR_SLOT_LIFE_PER_ROLE        = 2,
    ELENOR_SLOT_LIFE_RESIDENT         = 3,
} elenor_slot_lifetime_t;

typedef struct {
    uint32_t base;
    uint32_t size;
    uint16_t layout;
    uint16_t role;
    uint16_t alignment;
    uint16_t bank_policy;
    uint16_t lifetime;
    uint16_t owner;
    uint32_t flags;
} elenor_tile_slot_v0_t;

typedef struct {
    uint16_t abi_version;
    uint16_t slot_count;
    uint32_t frame_id;
    uint32_t generation;
    uint32_t l1_bytes;
    uint32_t flags;
    elenor_tile_slot_v0_t slots[ELENOR_TILE_SLOT_COUNT];
} elenor_tile_frame_v0_t;
```

### 4.2 address template 和 patch record

```c
typedef enum {
    ELENOR_ADDR_DIRECT           = 0,
    ELENOR_ADDR_BASE_TILE_LINEAR = 1,
    ELENOR_ADDR_BASE_TILE_2D     = 2,
    ELENOR_ADDR_BASE_GROUP_TILE  = 3,
    ELENOR_ADDR_PAGE_LIST        = 4,
    ELENOR_ADDR_SLOT_RELATIVE    = 5,
} elenor_addr_mode_t;

typedef struct {
    uint64_t base_addr;
    uint32_t tile_stride_x;
    uint32_t tile_stride_y;
    uint32_t group_stride;
    uint32_t element_stride;
    uint16_t slot_id;
    uint16_t layout;
    uint16_t addr_mode;
    uint16_t required_flags;
    uint32_t byte_offset;
} elenor_tensor_addr_template_v0_t;

typedef struct {
    uint32_t patch_id;
    uint32_t desc_id;
    uint16_t field_offset;
    uint16_t field_width;
    uint16_t owner;
    uint16_t addr_mode;
    uint32_t slot_id;
} elenor_desc_patch_record_v0_t;
```

Effective address：

```text
effective_addr = template.base_addr
               + frame.slots[slot_id].base
               + template.byte_offset
               + tile_id  * tile_stride
               + group_id * group_stride
```

`ADDR_PAGE_LIST` 和 segment offset 由 MFE 管理；UCE 不把 MFE 数据相关动态访问抢过来。

### 4.3 寄存器草案

| 寄存器                          | 说明                                                           |
| ------------------------------- | -------------------------------------------------------------- |
| `FRAME_CTRL`                    | bind、unbind、invalidate_desc、reset_shadow                    |
| `FRAME_STATUS`                  | idle、binding、active、faulted、generation                     |
| `FRAME_ID`                      | 当前 frame id                                                  |
| `FRAME_FAULT`                   | invalid slot、permission、overlap、alignment、patch fault code |
| `SLOT_BASE[i]` / `SLOT_SIZE[i]` | shadow slot range                                              |
| `SLOT_ATTR[i]`                  | role、flags、lifetime、bank_policy                             |
| `PATCH_STATUS`                  | patch active、last patch id、stall cycles                      |
| `PATCH_FAULT_PTR`               | fault record index                                             |

## 5. 数据流、控制流和时序路径

### 5.1 cold launch

1. Host Runtime 上传 package、descriptor、frame table、program。
2. Device Runtime 校验 ABI version 和 context。
3. Tile Group Sequencer dispatch prepared tile task。
4. Tile UCE 加载 Tile Program，fetch frame descriptor。
5. Tile UCE validate slot table：范围、重叠、对齐、权限、bank policy。
6. UCE install shadow，执行 descriptor template auto-patch。
7. UCE launch DMA/BOA/EVU/MFE/USE；engine wrapper 只使用 patched descriptor 和 frame shadow。
8. completion event 或 fault record 返回。

### 5.2 warm launch

Warm launch 不 reload program，只 patch descriptor、context、shape metadata：

```text
Runtime patch descriptor -> descriptor cache invalidate/flush
Device Runtime issue group task
Tile UCE bind existing program + new frame generation
Tile UCE patch descriptor shadow
Tile Program run
```

若 frame generation 与 descriptor template generation 不匹配，必须 fault；不得使用旧 slot binding 继续运行。

### 5.3 ordering / coherency 规则

- Slot Frame shadow 是 Tile Program 执行期间的唯一权威视图。
- frame descriptor memory 可被 runtime 修改，但修改对 active frame 不生效，直到下一次 bind。
- descriptor patch commit 是 launch engine 的前置条件。
- running program text 不允许 patch；resident const 若需替换，必须先 drain tile program。
- DMA 写 output slot 后，consumer engine 需要 event/fence；Stream Queue token 只排序 token，不替代 L1 memory fence。
- accumulator slot 只能被 BOA/EVU accumulate path 或明确 storeback path 修改；普通 DMA 覆盖需要显式 flag，否则 fault。
- USE state slot 的 checkpoint/restore 由 USE 生命周期控制，UCE 发起；DMA 不能绕过 checkpoint path 写 state。

### 5.4 bank placement

`bank_policy` 描述编译器/runtime 对 bank 的约束：

| policy            | 语义                                                            |
| ----------------- | --------------------------------------------------------------- |
| `DEFAULT`         | 硬件按 base address 映射 bank                                   |
| `PINNED_MASK`     | slot 只使用指定 bank mask，mask 编码由 SRAM profile 冻结        |
| `INTERLEAVE`      | 连续 cache line / SRAM row 跨 bank 交织                         |
| `NO_HOT_CONFLICT` | 不与 program/descriptor/event 或 accumulator hot path 共享 bank |

硬件必须检查无法满足的 pinned policy 并 fault 或降级为明确可观测状态；不得静默忽略影响 correctness 的权限/重叠规则。性能 hint 可降级，但 PMU 必须记录。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 capacity assumptions

| 配置     | L1 / Tile | slot count |   banks | Program/Desc/Event | 备注                    |
| -------- | --------: | ---------: | ------: | -----------------: | ----------------------- |
| Edge     |      1 MB |         16 | 至少 16 |       64 到 128 KB | 小模型 / 小 batch       |
| Balanced |      2 MB |         16 | 至少 16 |             128 KB | 推荐 First Silicon 起点 |
| High End |      4 MB |   16 或 32 |      32 |      128 到 256 KB | bank policy 更关键      |

slot table shadow 面积约为 `slot_count * sizeof(slot_entry)` 加校验逻辑；相对 1 到 4 MB L1 很小。bank conflict、alignment checker 和 patch datapath 的时序成本需要由 PPA exploration 冻结。

### 6.2 bandwidth model

Slot Frame 不产生带宽，但决定 L1 访问冲突：

```text
BW_eff(slot_i) = BW_bank_peak(bank_policy_i) * (1 - conflict_rate_i)
```

Compiler/runtime 应避免以下峰值重叠：BOA A/B operand burst、BOA accumulator RMW、MFE stream write、Tile DMA load/store、EVU gather replay、UCE descriptor fetch。

### 6.3 PMU / error hooks

必需 counter：

- `slot_bind_count`
- `slot_bind_fault_count`
- `slot_permission_fault_count`
- `slot_overlap_fault_count`
- `slot_alignment_fault_count`
- `slot_bank_policy_violation_count`
- `desc_patch_count`
- `desc_patch_fault_count`
- `desc_patch_stall_cycles`
- `descriptor_cache_invalidate_count`
- `l1_bank_conflict_by_slot[slot]`

Fault record 必须包含：command id、program id、frame id、generation、tile id、slot id、patch id、fault code、offending address、required permission、actual flags。

## 7. RTL/软件实现建议

- Slot table validate 使用独立 combinational checker + registered result，不要把所有 slot overlap 比较压在 launch critical path；可多周期 bind。
- Shadow table 使用双 buffer：active shadow 与 next shadow，bind commit 原子切换 generation。
- Descriptor patch unit 支持 small ALU：add、shift、multiply-by-stride；复杂公式由 compiler 预展开。
- Engine wrapper 接收 slot-resolved address，不让 BOA/EVU/MFE 自行解释 slot ABI。
- Tile DMA 的地址生成先做 slot permission，再 issue L1 SRAM request。
- Software package 中 frame table 与 descriptor table 都带 ABI version；runtime patch context base 后递增 generation。
- Compiler kernel library 使用固定 slot role 约定，但不固定绝对地址。
- Firmware debug dump 输出 frame shadow，便于 fault triage。

## 8. 验证、bring-up 和验收标准

### 8.1 SVA / formal checks

- Slot range：`base + size <= l1_bytes`，加法不得溢出。
- Slot overlap：除明确允许 alias 的只读 const case 外，两个 writable slot 不得重叠。
- TensorView / MFE view descriptor 可以在一个 backing slot 上创建多个逻辑 view，但这不放宽 writable alias 规则；V1 只允许只读 alias，或由显式 release/barrier 分隔的 phase-disjoint handoff alias。
- Permission：write request 必须命中 WRITE；accumulate request 必须命中 ACCUMULATE；execute request 必须命中 EXECUTE。
- Generation：engine launch 使用的 descriptor generation 必须等于 active frame generation。
- Patch atomicity：patch commit 前 engine launch 不得看到部分写入 descriptor。
- Running text immutable：active program text slot 不允许 write/patch。
- Bank policy：BANK_PINNED slot 的 request bank 必须在 mask 内。
- Reset clean：tile reset 后 active frame invalid 或 generation 递增，旧 descriptor handle 不可继续使用。

### 8.2 测试矩阵

| 测试                       | 目的                  | 验收                                      |
| -------------------------- | --------------------- | ----------------------------------------- |
| frame ABI version mismatch | descriptor validation | invalid descriptor fault                  |
| slot overlap random        | range checker         | writable overlap 全部被拦截               |
| accumulator protection     | 权限                  | DMA 普通写 accumulator fault              |
| warm launch patch          | descriptor coherency  | invalidate 后读取新 descriptor            |
| generation mismatch        | stale descriptor      | launch 被拒绝并记录 fault                 |
| bank pinned stress         | bank policy           | PMU 能按 slot 记录 conflict               |
| reset during active frame  | reset semantics       | 旧 handle 无效，credit/event 不污染新 run |

Bring-up：先 frame bind checker formal，再 Tile DMA slot access，再 BOA GEMM slot A/B/C binding，再 Stream Queue payload slot，再 MFE metadata/page-list slot。

### 8.3 跨模块 contract checklist

- Binary struct / protocol：tile frame、slot entry、address template、patch record、frame 寄存器均有 v0 草案。
- State machine：frame bind、descriptor patch、slot lifecycle、reset generation invalidation 必须有 transition coverage。
- Capacity / bandwidth / area：slot count、L1 profile、bank 数量、patch datapath 和 checker 面积分开记录；未冻结容量由 SRAM profile 冻结，checker 时序由 PPA exploration 冻结。
- NoC VC behavior：Slot Frame 本身不发 bulk packet；slot violation、patch fault、descriptor fault 必须通过 VC0/event fault path 可见；slot 指向的 DMA/stream payload 按 Memory / NoC VC contract 使用 VC1/VC2。
- Credit / EOS / error / reset：Slot Frame 不拥有 stream credit/EOS，但 payload slot 必须支持 EOS/error token 引用；tile reset 后旧 frame generation、token handle 和 descriptor patch 失效。
- Patch ownership：Compiler 冻结静态 shape/layout，Runtime/firmware patch context/residency，Tile UCE patch tile/group/slot offset，MFE patch page/segment，USE 管理 state slot。
- Ordering / coherency：active shadow 是唯一权威视图；warm patch 必须 invalidate descriptor cache；engine launch 只能使用 patch commit 后的 descriptor。
- SVA / formal：range、overlap、permission、generation、patch atomicity、running text immutable、bank pinned check 必须覆盖。
- PMU / error hooks：slot fault、patch fault、descriptor invalidate 和 per-slot bank conflict 必须带 frame id、slot id、patch id。

## 9. 风险、取舍和后续细化方向

| 风险                            | 影响                                   | 缓解                                                       |
| ------------------------------- | -------------------------------------- | ---------------------------------------------------------- |
| 固定地址语义回流                | dynamic shape / paged attention 难扩展 | 强制 slot ABI + frame binding                              |
| patch ownership 混乱            | UCE/MFE/Runtime 覆盖彼此字段           | patch record owner 和 fault code                           |
| descriptor cache coherence 错误 | warm launch 使用旧值                   | invalidate/flush protocol + generation check               |
| bank hint 被误认为 correctness  | 静默性能退化                           | correctness 属性必须 fault，性能 hint 只记录 PMU           |
| slot alias 规则过宽             | 数据破坏                               | First Silicon 禁止 writable alias                          |
| patch datapath 过复杂           | UCE timing 风险                        | 限制为 stride/add 模式，复杂计算由 compiler/runtime 预处理 |

后续需要冻结：slot alias policy、layout 编码、bank_policy 编码、engine owner mask、alignment 最小值、frame table residency、descriptor cache line size、patch field width 和 canonical kernel slot ABI。
