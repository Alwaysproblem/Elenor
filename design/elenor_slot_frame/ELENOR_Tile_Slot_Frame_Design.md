# ELENOR Tile Slot Frame 设计文档

## 1. 定位、目标和 First Silicon cutline

Tile Slot Frame 描述 Tile Program 对 Task-owned L1 Arena 的静态绑定视图。软件编译产物中的 Frame Slot 是 buffer 的绑定索引；它不是独立的分配器、跨 Task 池或 SRAM 配额。编译器给每个逻辑 buffer 预先计算 Slot/offset，运行时仅按该布局绑定/失效 view，并在 Task 安全退休时归还整个 L1 Arena。

核心原则：

```text
固定 slot ABI + 可变 Tile Frame
```

当前源合同以 immutable CompiledProgram 中的 ArenaLayout、Task owner、allocation/Profile generation 和逐条 view 生命周期为准；本文件的 C descriptor、寄存器与 Tile Frame binary ABI 均是尚未冻结的硬件 v0 草案，不表示当前芯片或硬件 ABI 已实现。

First Silicon cutline：

| 能力       | 当前 validator 软件合同                                                                    | 后续硬件规格草案                                              |
| ---------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| Frame Slot | Tile Program ArenaLayout 中的静态 buffer 绑定索引；slot 容量来自 target 配置               | 固定物理 slot table、字段编码、表项数量由硬件 ABI/PPA 冻结    |
| owner      | 每个逻辑 Task (`TaskIdentity`) 拥有一个 L1 Arena；root `Context` 单独拥有 L2 Arena         | window/role/resident owner-tag 组织仍待定义                   |
| 生命周期   | `tile.alloc` 绑定预计算 layout；`tile.free` 失效 view；安全 Task 退休归还 Arena 与 R lease | preemption、save/restore、硬件动态分配不属于当前模型          |
| reuse      | 编译器证明 L1 view lifetime 不重叠时静态复用同一 Slot/offset；运行时不搜索空闲区           | 多 buffer/window 的物理实现由后续规格决定                     |
| Profile    | 全 Tile 的 L1 统一使用该层活动 Profile；L1 与 Group L2 Profile 独立                        | bank mask、物理地址映射和可编程重分区由 SRAM profile 冻结     |
| binary ABI | 软件产物为独立的 schema 2 / compiler ABI v2 `CompiledProgram`                              | 本文 `*_v0_t` C layout 不是该产物 codec，也不是已冻结硬件 ABI |

## 2. 职责、非职责和 ownership

### 2.1 ownership matrix

| 对象 / 动作                                          | owner                              | 当前合同                                                                                                   |
| ---------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 静态 buffer lifetime、ArenaLayout、Frame Slot/offset | Compiler；独立 verifier 复算并验证 | 编译器从 Tile Program 的 `tile.alloc` / `tile.free` 得出可复用 lifetime；不可变布局随 CompiledProgram 封存 |
| L1 Arena reservation                                 | TaskIdentity / Tile ArenaPool      | 每个已准入 Task 拥有完整、带 padding 的 L1 Arena；Reservation 随 Profile/allocation generation 标记        |
| Frame prepare/bind 与已编译 Slot view                | Tile UCE / SlotFrame               | Task admission 准备 Arena Frame；Tile Program 执行 `tile.alloc` 时绑定已编译 Slot，不能改选 Slot/offset    |
| 单 view release                                      | Tile UCE / ArenaPool               | `tile.free` 只失效 view；等待该 Task 安全退休后才归还整个 Arena 和 R lease                                 |
| Group L2 buffers/backings/claims                     | root Context / Group ArenaPool     | 属于独立 L2 所有权；Frame Slot 不分配 L2，也不因 L1 view free 回收 L2                                      |
| L1/L2 Profile 事务                                   | Group 的唯一 ProfileController     | 分层同层统一配置；L1-only 切换不重置活跃 L2 Arena                                                          |
| 可选 binding/generation 校验                         | SlotFrame / runtime                | 检查 owner、Arena、allocation/Profile generation；不把软件 artifact 字段解释为硬件 binary ABI              |

### 2.2 非职责

- Slot Frame 不分配 HBM，不做 IOMMU translation。
- Slot Frame 不决定高层 tensor lifetime；compiler/runtime 必须显式生成 frame。
- Slot Frame 不允许 running program text 被 patch。
- Slot Frame 不自动解决 bank conflict；它只提供 bank policy 和 PMU attribution 所需 metadata。

## 3. 微架构和状态机

### 3.1 L1 Arena 与 Frame Slot

```text
每 Tile 的 L1 ArenaPool
  └── TaskIdentity-owned Arena（完整 per-bank reserve + stripe padding）
        └── ArenaLayout: buffer_id -> 逻辑字节、Frame Slot、静态 offset、lifetime
              └── SlotFrame: 当前已绑定的有效 view 与 owner/generation
```

Compiler 按活动 L1 Profile 的 bank geometry 产生确定的 striped ArenaLayout。Arena reservation 按每个 bank 校验；Frame Slot 是 view 绑定索引，不等同于 Arena、UCE execution context 或 ABI 隔离域 `context_id`。硬件 SRAM 容量、内部 region 划分和 frame table 实现仍未冻结。

逻辑字节数、padding 和 owner reservation 是不同量：view 只暴露声明的 logical bytes；每个物理段按 stripe/bank geometry 对齐，Arena 预留编译布局所需的全部容量。相邻 Task 不能因某一个 view `free` 而取得该 Arena 的余量。

### 3.2 physical frame-bind state proposal

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

软件绑定还要求 `SlotFrame.prepare` 对照已提交 Task Arena 与封存 layout，随后 `bind` 激活 Frame generation。`ALLOC_L1` 只能绑定该 layout 预先选定的 Slot/offset；generation、owner 或 Slot 不匹配时拒绝。上述是 validator 的对象边界，不定义物理 shadow table 的总线或原子更新电路。

### 3.3 physical descriptor-patch state proposal

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

### 3.4 Task L1 view 生命周期与 Arena 退休

```text
编译期 layout：buffer lifetime -> 固定 Frame Slot / Arena offset
Task admission：完整 L1 Arena reservation -> Frame prepare -> R lease commit
Tile Program：ALLOC_L1 -> LIVE VIEW -> FREE_L1 / invalidated
             （若 lifetime 已静态分隔，后继 ALLOC_L1 可重绑同一 Slot/offset）
安全 Task retirement：剩余 view invalidated -> pins/inflight 收敛
                      -> 整个 L1 Arena 与 R lease 归还
```

`tile.free` 只撤销局部访问权；它不返回物理 extent、不释放 Task Arena、不归还 R lease。只有编译器从指令 lifetime 证明不重叠的 L1 buffers，才可预先安排相同的 Slot/offset；之后的 `ALLOC_L1` 是按冻结布局重新绑定，不是运行时动态放置或空闲表搜索。`tile.free` 前，相关异步访问和 opaque engine events 必须已等待；死 scratch 无须 Store，Task 结束后仍需保留的数据须先完成 `tile.store.async`。

L2 生命周期不适用此复用规则。root Context 拥有独立 L2 Arena；当前 L2 布局永久 no-rebind，每个本地 buffer 保留自己的完整 stripe-round padded span，即使 `nest.release` 后也不在同一 Arena 内复用。L1 Frame 不管理 readonly backing、shared claim 或 L2 pin。

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

本文 C struct 是独立的硬件 Frame/slot ABI v0 草案，字段、宽度、地址编码均未冻结。当前编译产物 schema 2 / compiler ABI v2 保存不可变 ArenaLayout；Loader 独立验证后只绑定调用点提供的实际 memory bindings，不把此 C struct 当作 JSON codec、硬件 binary ABI 或运行时布局搜索请求。

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

当前执行合同只使用编译期确定的连续 striped layout 与 Slot/offset 绑定；`BASE_TILE_2D`、多维 affine patch 和 page-list 等条目保留为硬件 descriptor v0 设计空间，不表示模型实现了任意 shape 重解释或 DMA stride。Tile UCE 不接管 MFE 的 page/segment 动态地址。

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

### 5.1 当前软件执行与 Frame bind

```text
source xDSL ModuleOp
  -> compile_program：计算 Task L1 ArenaLayout、Frame Slot/offset 与 lifetime
  -> 封存 schema 2 / compiler ABI v2 CompiledProgram
  -> independent verifier + load_program：只读校验并绑定实际调用点
  -> Simulator.run：root 完整 L2 准入 -> Grid Route -> 每 Tile Task admission
  -> Task L1 Arena reservation / Frame prepare
  -> UCE Frame bind -> 按 Tile Program 执行 ALLOC_L1 / FREE_L1
  -> 所有 view、pin、inflight 与 Task lease 收敛后整块 Task Arena 退休
```

编译器会检查允许的 L1 Profile 与 `requested_contexts_per_tile=R` 包络；运行时按冻结 layout 做实际 Task 准入。一次 Tile 阻塞不撤销其他 Tile 已提交的 Task。Frame Slot 的存在不证明特定物理 bank/NoC 布局已经实现。

### 5.2 bind generation 与硬件 warm-patch 草案

运行时 Frame 绑定到已提交 Task Arena，并受其 owner、allocation generation 和 L1 Profile generation 约束。程序内 `ALLOC_L1` 只能 materialize layout 中指定的 view；`FREE_L1` 只失效该 view。Frame/Tile Program 退休后，新的 Task 取得新的 owner-scoped Arena 生命周期。

旧式 package warm-launch 对 frame table/descriptor 的 patch、cache invalidate/flush 与物理地址自动 patch 仍是硬件 v0 草案，不是当前 `load_program` 主路径：Loader 不调用 compiler、不补依赖、不重选 Profile 或改变 layout。实际 source module 不因执行而改写；新运行使用新编译/加载的 immutable executable 和显式 binding。

### 5.3 ordering / coherency 规则

- 当前 SlotFrame 是已绑定 Task Arena layout 的访问视图；owner、Arena ID、allocation generation、Profile generation 和 Frame state 必须匹配后才能读写。
- `tile.free` 只失效指定 L1 view；pending async access、opaque compute event 和引用收敛前不能 free。死 scratch 不要求 Store；要跨 Task 持续使用的值必须在 Task 退休前完成对应 Store。
- 同一 Task 内仅允许编译 lifetime 已证明不重叠的 view 静态复用 Slot/offset；不同 Task 之间通过整个 Arena 的安全退休自动回收，不存在跨 Task view free 即返还配额。
- L2 `nest.release` 与 L1 `tile.free` 是不同 owner/view 合同；前者不改变 L2 no-rebind span，也不由 Frame 管理。
- Profile 配置与数据可见性是分离合同。普通 event/fence 约束 producer/consumer；Profile 切换由显式完整配置事务执行，普通 await 或 Frame bind 不是维护事务。
- 硬件 program text 正在执行时不得 patch 的规则属于未来 binary/descriptor 实现要求，不改变当前只读 executable 主路径。

### 5.4 bank geometry 与物理 placement 草案

`bank_policy`、mask 编码、硬件地址映射与 Conflict 旁路仍是待冻结物理 ABI。当前软件用目标 Registry 中每层统一的 bank geometry、SPM/cache bytes、system reservation 与 alignment 构造静态 striped layout，并在编译/独立 verifier 中逐 bank 验证 Arena 预算；它不模拟任意自动 bank remap。

| policy            | 硬件 v0 设计意图（尚未冻结）                          |
| ----------------- | ----------------------------------------------------- |
| `DEFAULT`         | 按物理地址/目标 geometry 映射 bank                    |
| `PINNED_MASK`     | 限定 bank mask；mask 编码与可行性由 SRAM profile 冻结 |
| `INTERLEAVE`      | 连续数据跨 bank 交织                                  |
| `NO_HOT_CONFLICT` | 尽量避开热点，但作为性能 hint 时不得改变正确性        |

Layout 中的 bank 几何是软件容量合同，不等于实物仲裁器、bank conflict 率或地址映射已通过 RTL 验证。

## 6. 配置、PPA、性能模型和 PMU

### 6.1 capacity assumptions

| 对象            | 当前模型合同                                                                              | 物理规格状态                                               |
| --------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| logical bytes   | 每个 L1 view 仅暴露源声明字节数                                                           | Frame entry 字段宽度未冻结                                 |
| padding / Arena | per-bank striped layout；整 stripe-round reserve 纳入 Task Arena                          | stripe、alignment 与 bank mapping 由目标 SRAM profile 冻结 |
| Frame Slot      | buffer 的静态绑定索引；Slot 数校验使用 target `frame_slot_capacity`                       | 实际 table 深度/面积未冻结                                 |
| L1 Profile      | 同一 Tile Group 中所有 Tile 统一；mode 明确 `spm_bytes_per_bank` / `cache_bytes_per_bank` | 硅片容量、频率、组织仍未冻结                               |
| owner/reclaim   | 一个 Task 拥有一个 Arena；全部 Task views/pins/inflight 收敛后退休                        | 实物 owner-tag 与回收电路待规格冻结                        |

每 Bank 准入容量应从活动 Profile 的 SPM 与系统预留计算；仅比较 aggregate free bytes 不能证明跨 bank 合法。Frame Slot 数、UCE execution context 数、Group execution slot、program-resident slot 和 ABI `context_id` 不得统称为同一种 slot/context。

### 6.2 bandwidth model

Frame Slot 本身不提供带宽。软件模型按有限 per-bank Arena/Transfer stages 计容量等待和 bank wait；公式

```text
BW_eff ~= BW_peak * (1 - conflict_rate)
```

只作为物理性能估算形式，不是当前芯片实测值、compiler 的运行时 occupancy 预测或固定加速保证。当前模拟配置的 L1/L2 Profile 分层独立，但每层所有成员使用统一活动 mode；Profile 切换带完整 frontier、issue gate、maintenance 与成员 ACK。

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

当前 validator PMU / snapshot 观测的是绑定与资源合同的模型数据，例如 `frame_bind`、`l1_frame_fault`、memory wait、Arena/view/pin/inflight/lease 与各 Profile generation；其名称和统计范围不是上表硬件寄存器编号。按组件/域归因；并发 stall 类别不可直接相加为一个全局 stall cycle。硬件 counter id、快照与 fault-record 格式仍为 v0 草案。

## 7. RTL/软件实现建议

以下是物理 Slot Frame RTL/software integration 建议，不是当前 simulator 中已经存在的硬件实现；软件可执行语义以 §3.1/§3.4 的 Arena/Frame 生命周期为准。

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

以下 Slot/permission/descriptor 的 SVA 项是未来硬件 v0 验收建议；当前软件验收范围见 §8.2 的 Arena/layout/owner 检查，不能据此声称 RTL 已通过。

- Slot range：`base + size <= l1_bytes`，加法不得溢出。
- 同时存活的 L1 view 不能重叠；仅当编译器证明 lifetime 不相交时，才能为其预先安排相同 Slot/offset。当前合同没有通用别名例外。
- Lifetime reuse：仅对编译器证明前一 L1 view 在后续 `ALLOC_L1` 前已 `FREE_L1` 的布局复用同一 Slot/offset；运行时绑定不得搜索新位置。
- Generation：engine launch 使用的 descriptor generation 必须等于 active frame generation。
- Patch atomicity：patch commit 前 engine launch 不得看到部分写入 descriptor。
- Running text immutable：active program text slot 不允许 write/patch。
- Bank policy：BANK_PINNED slot 的 request bank 必须在 mask 内。
- Task isolation / generation：新 Task 不能使用旧 Task 的 Frame owner、allocation generation 或 Profile generation；Task retirement 前必须等待 view、pin、inflight 和 R lease 收敛。

### 8.2 测试矩阵

| 测试                         | 目的                             | 验收                                                                     |
| ---------------------------- | -------------------------------- | ------------------------------------------------------------------------ |
| Layout lifetime reuse        | 编译期静态 L1 复用               | 不重叠 lifetime 可复用预先计算的 Slot/offset；重叠 lifetime 不可重叠放置 |
| Frame binding mismatch       | owner / layout / generation 校验 | 错误 Task Arena 或 generation 拒绝绑定                                   |
| `tile.free` with live access | view invalidation safety         | 未完成访问/event 时不得失效该 view                                       |
| Task retirement              | whole-Arena ownership            | 所有 view/pin/inflight 收敛后归还 Arena 与 R lease                       |
| L1/L2 Profile split          | 配置隔离                         | L1-only reconfigure 不重置 L2 Arena/Profile                              |
| Hardware v0 warm patch       | descriptor coherency proposal    | 仅作为未来 RTL 验收；不作为当前 loader 路径证明                          |
| Reset during active Task     | reset generation isolation       | 未确认 drain/isolation 前不复用旧 Task Arena                             |

当前软件模型验证适用于编译、独立 verifier、load 与 Simulator 行为；上表中硬件 patch、slot table RTL、bank-mask 与物理 timing/PPA 仍需独立 RTL/SVA 验收。

Bring-up：先 frame bind checker formal，再 Tile DMA slot access，再 BOA GEMM slot A/B/C binding，再 Stream Queue payload slot，再 MFE metadata/page-list slot。

### 8.3 跨模块 contract checklist

- Software artifact：schema 2 / compiler ABI v2 的 ArenaLayout 与真实 bindings 经独立只读 verifier 校验；它不是 `tile_frame_v0_t`。
- Lifetime / reclaim：L1 lifetime 复用仅为编译期静态 Slot/offset 复用；`tile.free` 失效 view，完整 Task Arena 与 R lease 在安全退休/取消隔离后归还。
- L2 boundary：root Context 拥有 L2 Arena；L2 no-rebind、readonly backing/claim 与 pin 由 Group L2 owner 管理，不归 SlotFrame。
- Capacity / Profile：logical bytes、padding、Frame Slot、L1 Arena 和 per-level Profile 独立报告；物理容量与 slot count 不因软件配置冻结。
- Binary / registers：本文 C struct、register、address-template 均为独立硬件 v0 草案，字段宽度和物理接口由后续规格冻结。
- Verification：当前编译/loader/runtime 检查不替代 frame-table RTL、SVA、CDC/RDC 或器件 bank-conflict 测量。
- PMU / error hooks：模型 snapshot 与硬件 PMU 名称分开；fault 必须携带可定位的 owner、generation 与 view 信息，精确 hardware encoding 待定。

## 9. 风险、取舍和后续细化方向

| 风险                                | 影响                                   | 缓解                                                       |
| ----------------------------------- | -------------------------------------- | ---------------------------------------------------------- |
| 固定地址语义回流                    | dynamic shape / paged attention 难扩展 | 强制 slot ABI + frame binding                              |
| patch ownership 混乱                | UCE/MFE/Runtime 覆盖彼此字段           | patch record owner 和 fault code                           |
| descriptor cache coherence 错误     | warm launch 使用旧值                   | invalidate/flush protocol + generation check               |
| bank hint 被误认为 correctness      | 静默性能退化                           | correctness 属性必须 fault，性能 hint 只记录 PMU           |
| slot alias 规则过宽                 | 数据破坏                               | First Silicon 禁止 writable alias                          |
| sliding window 误引入动态 allocator | scope 膨胀、slot ABI 和验证面失控      | 仅允许预配置 multi-buffer slot + owner/hazard 检查         |
| patch datapath 过复杂               | UCE timing 风险                        | 限制为 stride/add 模式，复杂计算由 compiler/runtime 预处理 |

后续需要冻结：slot alias policy、layout 编码、bank_policy 编码、engine owner mask、alignment 最小值、frame table residency、descriptor cache line size、patch field width、canonical kernel slot ABI、per-window owner tag 和 multi-buffer profile。
