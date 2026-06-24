# ELENOR 架构文档评审报告

> **评审日期**: 2026-06-23
> **评审范围**: `design/` 目录下全部 27 份设计文档 + README + 项目结构
> **评审方法**: 主架构文档全文通读 + 4 组并行子模块深度评审 + 跨文档一致性分析
> **评审版本**: v0.1-review (架构评审稿)

---

## 目录

- [1. 总体评价](#1-总体评价)
- [2. 架构亮点](#2-架构亮点)
- [3. 系统性阻塞问题 (P0)](#3-系统性阻塞问题-p0)
- [4. 跨文档一致性问题](#4-跨文档一致性问题)
- [5. 分模块详细评审](#5-分模块详细评审)
  - [5.1 计算引擎 (BOA / EVU / MFE / USE)](#51-计算引擎-boa--evu--mfe--use)
  - [5.2 芯片组织与数据流 (Tile Group / Compute Tile / NoC / DMA / Collective)](#52-芯片组织与数据流-tile-group--compute-tile--noc--dma--collective)
  - [5.3 控制平面与基础设施 (Runtime ABI / Host IF / Scheduler / Fault-Reset / PMU / Region Sequencer / Slot Frame / Stream Queue / Tile UCE / Physical)](#53-控制平面与基础设施-runtime-abi--host-if--scheduler--fault-reset--pmu--region-sequencer--slot-frame--stream-queue--tile-uce--physical)
  - [5.4 软件栈与系统支撑 (Compiler / Driver-Firmware / Workload Mapping / Verification / Package / OPA)](#54-软件栈与系统支撑-compiler--driver-firmware--workload-mapping--verification--package--opa)
- [6. 工程规范问题](#6-工程规范问题)
- [7. 冻结优先级排序建议](#7-冻结优先级排序建议)
- [8. 总结](#8-总结)

---

## 1. 总体评价

ELENOR 是一套面向未来 5-10 年 AI 工作负载的加速器架构设计，覆盖 Dense Transformer、Paged Attention、MoE、SSM/Mamba/RWKV、Dynamic Shape 和多模型并发等场景。文档体系包含 27 份设计文档，约 15,000 行，从芯片顶层到引擎微架构、从编译器到验证 bring-up，形成了完整的架构描述。

**核心判断：架构方向正确，文档质量在架构评审稿中属于上乘，但跨文档契约尚未统一，这是进入 RTL/编译器/runtime 并行拆解阶段的最大障碍。**

### 评分总览

| 维度         | 评分 | 说明                                                                |
| ------------ | :--: | ------------------------------------------------------------------- |
| 架构方向     |  A   | Compute ≠ Control ≠ Data Movement 的拆分精准，四引擎分工合理        |
| 文档完整性   |  A-  | 覆盖全面，结构统一，每份文档有职责/非职责/ownership/状态机/PMU/风险 |
| V1 范围控制  |  B+  | Architecture V1 / First Silicon V1 / V2 Reserved 三层切分清晰       |
| 跨文档一致性 |  C+  | 多处 ABI 定义冲突，命名规范不统一，共享契约未冻结                   |
| 可验证性     |  B   | 验证计划结构化，但依赖未冻结的 ABI schema                           |
| 可实现性     |  B-  | 大量"由后续规格冻结"参数，需尽快收敛                                |

---

## 2. 架构亮点

### 2.1 四引擎职责拆分精准

`Compute != Control != Data Movement` 的原则贯穿全文，四引擎分工明确：

- **BOA**: dense compute (GEMM, attention QK/AV, expert MLP) — 不处理 fine-grained gather/scatter
- **EVU**: irregular compute (softmax, norm, RoPE, gather/scatter, tail) — 不做大规模矩阵乘主路径
- **MFE**: memory flow (page walk, prefetch, reorder, stream fill) — 不做任意图遍历
- **USE**: state/control (scan, recurrence, checkpoint/restore) — 不演化成通用 CPU

这种拆分避免了任一硬件模块承担过多职责，是本架构最大的设计优势。

### 2.2 分层控制流模型清晰

```text
Graph Schedule PC → Region PC → Tile PC → Engine tasks → Micro-ops
```

五层控制流（Graph / Region / Kernel / State / Compute）各有明确的控制器和职责边界，从 Device Runtime 到 BOA/EVU Sequencer 形成了完整的控制链路。

### 2.3 Tile-SPMD 编程范式

同一份 Tile Program template + descriptor auto-patch 的设计避免了为每个 tile 生成独立程序，降低了 program cache 压力和编译器复杂度。CUDA 对应关系表清晰实用。

### 2.4 Slot Frame 抽象

"固定 slot ABI + 可变 Tile Frame" 的设计比固定物理地址绑定算子语义更灵活，支持 dynamic shape、paged attention 和 workspace 切换。shadow installation 机制保护了运行中的 tile program。

### 2.5 V1 范围切分

Architecture V1 / First Silicon V1 / V1.x-V2 Reserved 的三层切分是正确的方法论。First Silicon V1 的 cutline 表格明确区分了"必须实现"和"可预留"，避免了范围蔓延。

### 2.6 验证计划务实

bring-up 顺序（command/event/DMA → BOA → EVU → MFE → USE → multi-model）正确地先打通控制面和数据搬运，再扩展复杂 engine。Phase Exit Criteria 可量化。

### 2.7 EVU-MT 的 SIMT-lite 设计

EVU 从最初的 "predicated vector" 演进到 "shared-PC microthread engine"，复用 SIMT 的 lane 编程模型但不引入 GPU 的 warp scheduling machinery，是正确的面积/性能/验证复杂度平衡。

### 2.8 PMU 唯一归因规则

stall attribution hierarchy (9 级互斥 primary owner) 的设计避免了同一 cycle 被多个模块重复计数，这在早期架构文档中并不多见。

---

## 3. 系统性阻塞问题 (P0)

以下问题如不解决，将直接阻塞 First Silicon V1 的正确性或可验证性。

### P0-1: 共享 ABI 尚未冻结——所有文档仍使用"样例"结构体

**问题**: 所有 ABI 结构体（command、event、fault record、descriptor）仍标注为"v0 样例"或"示例级接口"，没有机器可读的冻结定义。各文档对同一概念给出了不同的定义：

- `elenor_command_v0_t` 在架构文档和 Runtime ABI 文档中枚举值和命名不同（见 [§4.1](#41-command-type-枚举冲突)）
- `elenor_fault_record_v0_t` 在 Runtime ABI 和 Fault/Reset 文档中字段完全不同（见 [§4.3](#43-fault-record-结构体冲突)）
- `elenor_event_status` 在不同文档中使用了 enum 和 bitmask 两种编码（见 [§4.2](#42-event-status-编码冲突)）

**影响**: 编译器、runtime、firmware、RTL 和验证团队无法基于同一份契约并行开发。验证可以通过过时的 schema 而真实系统失败。

**建议**: 产出 5 份机器可读的 ABI schema 文件，所有文档 normative reference：

1. Descriptor ABI schema (common header + engine-specific payloads + relocation targets)
2. Command/Event ABI schema (command header/opcodes + event record + memory-ordering rules)
3. Executable-package schema (section taxonomy + kernel-binding table + debug manifest)
4. Workload semantic schema (shape classes, paged-attention page semantics, MoE modes, USE checkpoint ABI)
5. Verification schema bundle (same IDs/builds/ABIs carried into traces, goldens, PMU checks)

### P0-2: Tile Program 驻留路径未冻结

**问题**: Compute Tile 的执行依赖 `program_local_slot` 和 `program_epoch`，但没有任何文档冻结 tile program 如何安装到 tile-local instruction storage：

- Tile Group 的 Program Residency Manager 只定义了 region program fetch
- Compute Tile 的状态机假设 `PREPARED_TASK_CHECK` 会验证 local program handle，但未定义安装路径
- Global DMA 明确说 tile-program load to tile SRAM "may be" Group DMA 或 Tile DMA，留待后续冻结

**影响**: Cold launch 可能向 program image 缺失或已失效的 tile 派发任务。warm launch 后 reset 无法正确 invalidate local handle。

**建议**: 冻结一条 V1 路径：

```text
Global/Group DMA prefetch tile program to group L2
  → Tile DMA installs into tile program SRAM/I-cache
  → tile returns {program_local_slot, program_epoch} ready ack
  → Tile Dispatcher may issue prepared task
```

Residency manager 必须拥有 epoch 并在 tile/group reset 时 invalidate。

### P0-3: 分页注意力跨引擎 ABI 未冻结

**问题**: Paged Attention 是 V1 标题级 workload，依赖 MFE → BOA → EVU → BOA 的完整流水线，但跨引擎的 K/V 数据布局契约仍未定义：

- MFE page descriptor 的 `head_dim` 和 `layout_transform` 只说"需匹配 BOA/EVU 期望布局"
- canonical case 参数（page size, head_dim, prefetch depth, stream queue depth）全部待冻结
- BOA 没有定义 input stream contract（valid/credit/EOS/error）
- MFE stream token/credit ABI 未冻结

**影响**: 四个引擎可以各自局部正确，但完整 paged attention pipeline 仍然失败。这是 V1 最重要的端到端闭环项。

**建议**: 立即冻结 canonical paged-attention inter-engine ABI：

- K/V 元素顺序、per-head packing、lane/tile 粒度
- stream token 单元语义、credit 计算、EOS/error 原子性
- MFE → BOA 的 producer/consumer credit 语义
- 一个 frozen "must work" 参数向量

### P0-4: 事件等待缺少 sequence/generation 字段

**问题**: 事件模型要求 wait 方匹配 `event_id + sequence`，但 command header 只携带 `wait_event_base` 和 `wait_event_count`，没有 expected sequence：

- Runtime ABI §3.2 要求 sequence 匹配，但 §4.2 的 command layout 无 sequence 字段
- Scheduler 内部 record 同样只有 `wait_event_base/count`
- Stream Queue token 缺少 generation 字段，但 reset 要求 generation-safe 行为

**影响**: event table wrap/realloc 后，stale completion 可以被误认为是新 completion。reset 后旧 event ID 可以 alias 新 work。

**建议**: 将 `wait_event_base/count` 替换为 `(event_id, expected_sequence)` pair 数组或指向 wait-list descriptor 的指针。Stream token 增加 generation 字段。

### P0-5: Fault Record 有两个不兼容的 v0 定义

**问题**: 两个文档定义了同名但字段完全不同的 `elenor_fault_record_v0_t`：

| 来源             | 字段特点                                                                                                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runtime ABI §4.4 | queue/event 导向：fault_id, context_id, queue_id, command_index, command_type, event_id, event_sequence, fault_code, descriptor_iova, descriptor_offset, producer_id, domain, detail0, detail1                                        |
| Fault/Reset §4.1 | 全面诊断导向：abi_version, code, source, severity, fault_record_index, context_id, command_id, event_id, program_id, desc_id, group_id, tile_id, queue_id, slot_id, patch_id, engine_id, offending_addr, aux0, aux1, pmu_snapshot_ptr |

**影响**: Driver/firmware 无法安全解码两种 fault record。Stream Queue error token 传播、host interrupt 路径都会受影响。

**建议**: 以 Fault/Reset 文档的定义为唯一权威，Runtime ABI 及所有其他文档 normative reference 它。统一 `fault_record_slot` / `fault_record_index` 命名和 ring 分配规则。

### P0-6: BOA split-K 语义未闭合

**问题**: BOA descriptor 要求 `split_k > 1` 时声明 partial owner，FSM 有 `WAIT_PARTIAL / COLLECTIVE_HANDOFF` 状态，但从未定义：

- partial group identity
- expected fan-in count
- reduction order
- accumulator epoch/tagging
- "所有 partial 已消费"的精确 event 语义

**影响**: duplicate append 或 premature writeback 会静默损坏输出。这对 split-K GEMM 和 attention 是 First Silicon 正确性阻塞项。

**建议**: 扩展 descriptor 添加 `partial_group_id`, `partial_index`, `partial_count`, `reduce_owner`, `reduction_order`, `acc_epoch`。要求 completion 仅在所有 expected partials retire 后发生。

### P0-7: EVU V1 cutline 与验证验收计划矛盾

**问题**: EVU 的 V1 cutline 将 SFU 标为 P1、indexed gather 标为 P1、structured mask control 标为 P2，但 bring-up 要求 local gather、SFU exp/rsqrt/gelu、structured `if_mask/else_mask/endif`、完整 softmax 和 RMSNorm，且 V1 acceptance 要求 softmax 和 RMSNorm golden 对齐。

**影响**: 这是一个 First Silicon 计划 bug——cutline 和验证闭环目标不一致。

**建议**: 要么将 gather/SFU/structured mask 移入 V1 must-have，要么将 softmax/RMSNorm/structured-control kernels 从 V1 acceptance 中移除。

### P0-8: USE checkpoint/event-assist 语义被推迟但 V1 依赖它们

**问题**: Checkpoint 和 event-assist 都在 First Silicon V1 cutline 中，验收计划依赖 fault/reset 恢复正确性，但文档明确说"Checkpoint policy enum 和 rollback semantics: 由后续规格冻结"和"event assist 确切语义: 由后续规格冻结"。

**影响**: 这是直接的 RTL 阻塞。Mamba/RWKV 类 workload 的恢复和正确性依赖精确的 commit/rollback 语义。

**建议**: 立即冻结 checkpoint 策略（output-slot 写入与 state 写入是否原子、`on_fault` 含义、version 递增规则、dirty line 清除时机）和 event-assist 语义（legal modes、visibility/order、timeout 行为、与 Local Event Unit 的交互）。

---

## 4. 跨文档一致性问题

### 4.1 Command Type 枚举冲突

**架构文档** (§18.4, line 2110) 使用从 0 开始的隐式枚举：

```c
ELENOR_CMD_BOA_GEMM,        // = 0
ELENOR_CMD_VECTOR_KERNEL,   // = 1
...
```

**Runtime ABI 文档** (§4.2, line 195) 使用从 1 开始的显式枚举，且命名不同：

```c
ELENOR_CMD_LAUNCH_REGION = 1,
ELENOR_CMD_DMA = 2,
ELENOR_CMD_BOA_GEMM = 3,
ELENOR_CMD_EVU_KERNEL = 4,  // 非 VECTOR_KERNEL
...
```

**差异清单**:

- 枚举起始值不同 (0 vs 1)
- `ELENOR_CMD_VECTOR_KERNEL` vs `ELENOR_CMD_EVU_KERNEL` 命名不同
- 架构文档有 `ELENOR_CMD_MFE_SPARSE_STREAM`，Runtime ABI 没有
- Runtime ABI 有 `ELENOR_CMD_LAUNCH_REGION = 1` 作为首项，架构文档将其排在后面
- Scheduler 的 V1 cutline 只处理 `LAUNCH_REGION`, `DMA`, `BARRIER`, `EVENT_WAIT`, `EVENT_SIGNAL`，不消费直接 engine commands

**建议**: 冻结一份 V1 command matrix：哪些枚举值在 Runtime ABI 中合法、哪些是 pass-through vs lowered into `LAUNCH_REGION`、不支持的值产生什么确定性错误。

### 4.2 Event Status 编码冲突

三处定义使用了两种不同的编码模型：

| 来源             | 编码模型 | 值                                                     |
| ---------------- | -------- | ------------------------------------------------------ |
| 架构文档 §18.6   | enum     | PENDING=0, DONE=1, ERROR=2, TIMEOUT=3, RESET=4         |
| Runtime ABI §4.3 | enum     | PENDING=0, DONE=1, ERROR=2, TIMEOUT=3, RESET=4         |
| Fault/Reset §4.3 | bitmask  | COMPLETE=1<<0, FAULT=1<<1, TIMEOUT=1<<2, CANCELED=1<<3 |

此外，Region Sequencer 的 `signal.event` opcode 接受 generic `status`，example 使用 `status=ready`，这不是上述任何模型中的合法值。

**建议**: 冻结一个 canonical event object：推荐 `status enum + optional flags + producer_id + sequence + context/namespace`。如需 stage marker，使用独立的 non-event 信号通道。

### 4.3 Fault Record 结构体冲突

详见 [P0-5](#p0-5-fault-record-有两个不兼容的-v0-定义)。

### 4.4 Descriptor 命名规范不统一

各文档的 descriptor 命名遵循三种不同模式：

| 模式                | 示例                                     | 使用文档                  |
| ------------------- | ---------------------------------------- | ------------------------- |
| `elenor_*_desc_t`   | `elenor_boa_desc_t`, `elenor_dma_desc_t` | 架构文档, MFE, Global DMA |
| `*_desc_v0_t`       | `boa_desc_v0_t`, `mfe_page_desc_v0_t`    | BOA, MFE slot-based       |
| `*_desc_t` (无前缀) | `evu_desc_t`, `mfe_load_desc_t`          | 架构文档 EVU/MFE          |

此外，descriptor 是否包含 version/size header 也不一致：

- EVU 有 `version`, `size_bytes`
- USE 有 `abi_version`, `desc_size`
- BOA 没有 version/size 字段
- MFE 混用 raw-address 和 slot-based 两种形式

**建议**: 统一为 `elenor_<engine>_desc_v0_t` 命名 + 共同 header（`abi_version`, `desc_size`, `flags/reserved`）+ engine-specific payload。

### 4.5 PMU Counter 命名碎片化

PMU 架构文档定义了稳定的 `PMU_*` ID 和分层 ID 编码，但各 block 文档使用 block-local 命名：

| Block            | 命名模式                  | 示例                     |
| ---------------- | ------------------------- | ------------------------ |
| Host Interface   | `hi_*`                    | `hi_doorbell_count`      |
| Scheduler        | `sched_*`                 | `sched_queue_occupancy`  |
| Region Sequencer | `rs_*`                    | `rs_issue_count`         |
| Stream Queue     | `sq_*`                    | `sq_credit_full`         |
| Tile UCE         | `uce_*`                   | `uce_fetch_stall`        |
| Slot Frame       | `slot_*` / `desc_patch_*` | `slot_check_fail`        |
| BOA              | `boa_*` / `BOA_PMU_*`     | `BOA_PMU_STALL_OPERAND`  |
| EVU              | `EVU_MT_PMU_*`            | `EVU_MT_PMU_STALL_*`     |
| MFE              | `MFE_PMU_*`               | `MFE_PMU_STALL`          |
| Global DMA       | `dma_*`                   | `dma_wait_memory_cycles` |

**建议**: 发布一份架构级 counter ID registry，要求每个 block 文档将 local counter 映射到 registry。包含 primary-owner 和 optional secondary-tag 语义。

### 4.6 Stall Attribution 层级不统一

架构文档定义了 9 级互斥 stall owner：

```text
1. engine_active
2. engine_wait_event
3. engine_wait_operand
4. stream_credit_empty_or_full
5. sram_bank_conflict
6. noc_backpressure
7. dma_wait_memory
8. uce_program_or_descriptor_stall
9. unknown_or_unclassified
```

但各 engine 的 PMU 使用不同的 stall 分类粒度和命名：

- BOA: `stall_operand`, `stall_acc`, `stall_wb`, `bank_conflict` (未映射到统一层级)
- EVU: `IFETCH_STALL`, `DECODE_STALL`, `ISSUE_STALL`, `LSU_REPLAY`, `REPLAY_QUEUE_FULL`, `SFU_BUSY`, `WRITEBACK_STALL`, `COMMIT_STALL` (粒度更细但未映射)
- MFE: `MFE_PMU_STALL` (单一 counter，无分类)
- USE: 部分使用通用类别 (`engine_wait_operand`, `dma_wait_memory`)，部分使用 local 命名

**建议**: 保留 engine-local 细粒度 counter 用于 debug，但要求每个 engine 同时输出映射到统一 9 级层级的 primary stall owner。

### 4.7 地址引用模型不统一

不同模块使用不同的地址引用方式：

| 模块               | 地址模型                                                           |
| ------------------ | ------------------------------------------------------------------ |
| Compute Tile       | slot-relative L1 (`slot_id + byte_offset`)                         |
| Global DMA         | raw 64-bit address + `address_domain` flag                         |
| Memory/NoC DMA     | overloaded `src_iova_or_l2` / `dst_iova_or_l2` + destination flags |
| Collective         | 32-bit L2 offsets                                                  |
| Tile Group         | `l2_window_base/bytes` (flat window)                               |
| Stream Queue token | 32-bit `payload_addr` (地址空间含义未定义)                         |

**建议**: 统一地址域编码：添加 `src_domain` 和 `dst_domain` enum (HBM/host, group L2, tile L1, device-local)，不通过 flag 组合重载地址含义。Stream token `payload_addr` 明确为 L1 slot-relative 或 L2 offset。

### 4.8 Context ID 位宽不统一

| 文档                                              | context_id 位宽 |
| ------------------------------------------------- | --------------- |
| Memory/NoC                                        | `uint16_t`      |
| Tile Group, Compute Tile, Collective, Runtime ABI | `uint32_t`      |

**建议**: 统一为 `uint32_t context_id`。

### 4.9 Event ID 位宽不统一

| 文档                                | event_id 位宽           |
| ----------------------------------- | ----------------------- |
| Tile Group, Collective              | `uint16_t signal_event` |
| Memory/NoC, Global DMA, Runtime ABI | `uint32_t event_id`     |

**建议**: 统一为 `uint32_t event_id`。

### 4.10 Stream Queue 语义未完全冻结

- EOS 是否消耗 credit 仍未冻结（§3.2 line 85 建议消耗但未确认）
- `ELENOR_STREAM_Q_BROADCAST` 在 v0 enum 中存在但 First Silicon 排除 broadcast
- multi-consumer 行为（broadcast token vs refcount token vs per-consumer queue）未定义
- Region Sequencer 使用 `wait.credit`，Tile UCE 使用 `stream.acquire/release`，两种抽象层级不同

**建议**: 冻结 V1 credit 模型：EOS 消耗并随后释放一个 credit（与其他 token 一致）。从 v0 enum 移除 BROADCAST 或用 capability bit 门控。统一 Region Sequencer 和 Tile UCE 的 stream 操作抽象。

---

## 5. 分模块详细评审

### 5.1 计算引擎 (BOA / EVU / MFE / USE)

#### 5.1.1 BOA — Block Outer-product Accelerator

**文档**: `elenor_boa/ELENOR_BOA_Design.md` (428 行)

**亮点**:

1. 边界纪律强：明确拒绝 page walk、state lifecycle、graph control 和 irregular vector 语义
2. 执行 pipeline 简洁且硬件可实现：fetch → validate → prefetch → compute → reduce → epilogue → writeback，`VALIDATE` 门控在 writeback 前
3. PMU 分离 operand/accumulator/writeback/reduce/epilogue/bank-conflict，不是单一 busy counter
4. SVA 列表命中真实故障模式：launch 原子性、pre-write descriptor fault、accumulator epoch、reduction 确定性

**P0 问题**:

1. **Descriptor 缺少 version/size header**: validator 被要求检查 ABI version 和 size，但 `boa_desc_v0_t` 不包含这些字段。无法安全拒绝 stale descriptor。建议添加共同 descriptor header。
2. **Split-K 语义未闭合**: 详见 [P0-6](#p0-6-boa-split-k-语义未闭合)。
3. **Timeout/cancel 和 MFE-fed operand 语义不精确**: completion 可以是 done/fault/timeout/cancelled，但未定义 timeout source、cancel handshake、timeout/cancel 后哪些 write 仍允许 visible。BOA 也没有显式 stream/EOS/error contract。

**P1 问题**:

1. Numeric/epilogue 行为太软：`rounding_mode`, `saturation_mode`, quant 参数格式全部待冻结，但验收依赖 dtype-tolerant golden 对齐
2. V1 dataflow 应在 ABI 中显式：文档推荐 output-stationary 但未声明非 OS 值必须 hard-fault
3. Event/fault commit ordering 跨域边界被推迟

**P2 建议**:

1. 将 `boa_stall_operand` 拆分为 `dma_underfill`, `mfe_credit_wait`, `bank_conflict_replay`
2. 移除 V1 的 output stream writeback 或完整定义它
3. 添加 accumulator occupancy/high-watermark 可观测性

#### 5.1.2 EVU — Enhanced Vector Unit / Microthread Engine

**文档**: `elenor_evu/ELENOR_EVU_Design.md` (1680 行)

**亮点**:

1. 范围纪律严格：反复拒绝"小 GPU SM"陷阱，保持 shared-PC、tile-local、slot-memory engine
2. Predicate/tail 语义是一等公民：`effective_active`、inactive-lane 副作用抑制、硬件生成 tail mask
3. Bank conflict replay 和 fault attribution 被当作架构问题而非实现细节
4. Structured control flow 模型合理：V1 限制为 uniform branch + structured mask control
5. 验证计划检查真实不变量：inactive lane 不触碰 memory、replay 保持 lane mapping、fault record 先于 fault event

**P0 问题**:

1. **LSU request/response 与 lane 配置不匹配**: base profile 是 `LANES=32`, `DATA_WIDTH=32`（1024 bit vector payload），但 response packet 只有 `data[511:0]`，request 只有 `byte_enable[63:0]`。Gather addressing 是 per-lane 的，但 `evu_mt_dmem_req_t` 只有一个 `byte_offset`，不是 per-lane offset 数组。接口无法表示 ISA 声称支持的语义。
2. **V1 cutline 与验收计划矛盾**: 详见 [P0-7](#p0-7-evu-v1-cutline-与验证验收计划矛盾)。
3. **Fault-policy 对 outstanding stores 和 replayed LSU 操作的可见性边界未定义**: fault FSM 允许 drain 或 kill，replay 规则要求 store order 保留，但未定义 fault 后 already-issued stores、masked stores、optional scatter 的精确可见性边界。

**P1 问题**:

1. Numeric 行为不够冻结：denorm/NaN/inf policy、saturation/wrap、approximation envelope、all-inactive reduction identity 全部推迟
2. 多个 binary contract 未冻结：opcode encoding、launch descriptor binary layout、`fault_policy` encoding、PMU snapshot protocol
3. Commit payload 的 `fault_record_slot` 和 `pmu_snapshot_slot` 缺少分配/生命周期管理说明

**P2 建议**:

1. V1 移除 optional transpose（pair/split-half/RoPE 已足够）
2. 添加 per-bank replay 可观测性
3. 区分 tail-inactive 和 predicate-inactive 的 PMU/debug counter

#### 5.1.3 MFE — Memory Flow Engine

**文档**: `elenor_mfe/ELENOR_MFE_Design.md` (607 行)

**亮点**:

1. Ownership 选择正确：page walk、block decode、address generation、prefetch、reorder、stream fill 统归 MFE
2. 内部 pipeline 分解合理：metadata decode → walkers → request tracker → reorder buffer → coalescer → stream buffer → commit
3. 一致性边界显式声明：logical token order、duplicate-index policy、scatter 限制、local-reduce scope
4. 性能建模绑定实际 workload：`T_prefetch <= T_qk` overlap 条件和 MoE/embedding/GNN 映射示例

**P0 问题**:

1. **Fault 后 page/segment error 的可见性不够精确**: FSM 说 fault 后 MFE 可能 drain 已完成 response，但未定义 BOA/EVU 在 page fault mid-stream 时合法能消费什么。建议定义精确可见性边界。
2. **Canonical paged-attention K/V layout contract 缺失**: 详见 [P0-3](#p0-3-分页注意力跨引擎-abi-未冻结)。
3. **Stream token/credit ABI 未冻结**: MFE 是 main producer 但 token ABI 仍 open。建议冻结 token size、credit unit、metadata validity、EOS/error exclusivity、ordering guarantees。

**P1 问题**:

1. Descriptor ABI 不一致：raw-address 和 slot-based 两种形式并存，无 version/size header
2. Segment 边缘语义需闭合：duplicate policy、output order policy、all-empty segment identity
3. Timeout 语义太开放：per-request 还是 per-command 未定义

**P2 建议**:

1. 将 DMA/window-generator scope 从核心 page/segment engine 在 spec 中分离
2. 在 fault 时暴露 token-indexed debug metadata（logical token index + page id + stream kind）
3. PMU 分离 page-walk stall 和 data-fetch stall

#### 5.1.4 USE — Unified State Engine

**文档**: `elenor_use/ELENOR_USE_Design.md` (720 行)

**亮点**:

1. 控制/状态/数据搬运拆分出色：UCE 管 program control，USE 管 state lifecycle，MFE 管 dynamic memory flow
2. 状态机分解合理：DESC_VALIDATE → STATE_ACQUIRE → INPUT_READY → EXECUTE → COMMIT → CHECKPOINT_OPTIONAL → EVENT_SIGNAL
3. Checkpoint/restore 作为架构特性而非事后补充
4. 保持 USE 有界范围：抵抗变成通用 CPU 或 memory processor
5. 验证计划强：recurrence golden、checkpoint after fault/reset、event assist、token metadata、dirty-eviction illegality

**P0 问题**:

1. **Checkpoint policy/rollback 语义被推迟但 V1 依赖**: 详见 [P0-8](#p0-8-use-checkpointevent-assist-语义被推迟但-v1-依赖它们)。
2. **Event-assist 语义被推迟但 V1 依赖**: 同上。
3. **State locking/coherence protocol 未充分指定**: `STATE_ACQUIRE` 说 USE 获得 "state lock/tag"，但未定义 lock 粒度、谁可以 block、lock 是否存活于 checkpoint/reset、并发 metadata/state 访问如何仲裁。

**P1 问题**:

1. Token-metadata 边缘语义仍开放：overflow、saturation、duplicate-update ordering
2. StateView/slot binding 需要更强的 bounds 语义：缺少 `bytes_per_state` 和 bounds check
3. Fault/event ordering 不如 EVU 显式：未明确声明 "fault record visible before fault event visible"

**P2 建议**:

1. 将 descriptor namespace 拆分为 state-compute vs metadata/event-assist 类
2. 添加 lock contention 和 rollback frequency PMU counter
3. 暴露 state-version transition tracing in debug

---

### 5.2 芯片组织与数据流 (Tile Group / Compute Tile / NoC / DMA / Collective)

#### 5.2.1 Tile Group

**文档**: `elenor_tile_group/ELENOR_Tile_Group_Design.md` (481 行)

**P0 问题**:

1. **Region resource descriptor 无法执行 L2 ownership 模型**: descriptor 只暴露 `l2_window_base/bytes` + `stream_buffer_base/bytes`，不足以描述或验证 program/cache、stream、prefetch、partial-result、event/status、collective scratch/output 等独立区域。建议替换为 L2 allocation table。
2. **Tile program residency 路径未冻结**: 详见 [P0-2](#p0-2-tile-program-驻留路径未冻结)。
3. **Event record ABI 缺少 epoch/sequence 字段**: 控制规则和 SVA 要求 monotonically increasing event sequence 和 epoch check，但 launch/resource descriptor 只有 `wait_event_base/count` 和 `signal_event`。

**P1 问题**:

1. Stream Queue descriptor table 被引用但从未规范
2. "Explicit fence" 要求在 RUN state 中 descriptor patching，但 fence scope/ordering 未定义
3. L2 仲裁和饥饿边界未足够紧凑
4. MFE 使用 group-level L2 stream/prefetch space 被承认但未资源描述

#### 5.2.2 Compute Tile

**文档**: `elenor_compute_tile/ELENOR_Compute_Tile_Design.md` (747 行)

**P0 问题**:

1. **Prepared-task 执行依赖未冻结的 local program residency contract**: 详见 [P0-2](#p0-2-tile-program-驻留路径未冻结)。
2. **Descriptor patch coherence 被标记为风险但未架构化指定**: warm launch 可观察 partially patched descriptor across UCE, DMA, BOA, EVU, MFE, USE fetch points。建议冻结 descriptor-cache protocol。
3. **Tile-local event 语义对 reset-safe execution 不完整**: task descriptor 只有 `event_base`，reset 后旧 event ID 可 alias 新 work。

**P1 问题**:

1. Tile DMA 缺少 canonical ABI
2. Multi-consumer/refcount/broadcast stream descriptor 缺失
3. L1 仲裁分类了但未冻结 starvation bound 和 protected-path guarantee
4. `state_preserve_reset` 未定义

#### 5.2.3 Memory / NoC

**文档**: `elenor_memory_noc/ELENOR_Memory_NoC_Design.md` (374 行)

**P0 问题**:

1. **共享 DMA descriptor ABI 与 Global DMA 文档冲突**: 本文挡定义了 `elenor_dma_desc_v0_t` with `kind`, `context_id`, `queue_id`, `desc_id`, `event_id`, `timeout_cycles`，Global DMA 定义了不同的 minimal/ext descriptor。建议冻结一份 canonical split。
2. **Address-domain 编码不够精确**: `src_iova_or_l2` / `dst_iova_or_l2` 重载了 external IOVA、group L2 和 potential local endpoint，但只有 destination-style flags。建议添加 `src_domain` 和 `dst_domain` enum。
3. **NoC VC plan 不完整**: DMA request 和 write-ack traffic 未分配到 frozen virtual network。deadlock analysis、buffer sizing、starvation proof 依赖完整 channel-dependency graph。

#### 5.2.4 Global DMA

**文档**: `elenor_global_dma/ELENOR_Global_DMA_Design.md` (466 行)

**P0 问题**:

1. **定义了竞争性 DMA descriptor ABI**: 与 Memory/NoC spec 冲突。建议采用一份 canonical `dma_launch + dma_desc_v0` contract。
2. **Tile program load 路径未冻结**: 详见 [P0-2](#p0-2-tile-program-驻留路径未冻结)。
3. **Completion/event ordering 仍模糊**: `event_update` 可以是 VC0 或 sideband。建议 V1 强制 event update 走 ordered VC0 path。

**P1 问题**:

1. Zero-length/zero-row 语义未解决
2. DMA request/write-ack path mapping 未与 NoC contract 对齐
3. Same-cycle completion/fault priority 推迟

#### 5.2.5 Collective Engine

**文档**: `elenor_collective/ELENOR_Collective_Engine_Design.md` (470 行)

**P0 问题**:

1. **Collective operation 身份不完整**: pipeline 语义要求 `collective_id + block_id + sequence_id` 正确性，但 descriptor 只有 `collective_id`。建议添加 `block_id`, `sequence_id`, `epoch`。
2. **Numeric 语义不够冻结**: rounding、saturation、overflow、NaN/Inf、accumulator mode、deterministic reduction order 全部推迟。
3. **Broadcast/output mode 对 First Silicon 太开放**: descriptor 允许 output to L2、stream 或 broadcast fanout，但无具体 per-consumer credit accounting 或 recipient encoding。建议 V1 缩窄到 `L2 output buffer + Tile DMA consumers`。

---

### 5.3 控制平面与基础设施 (Runtime ABI / Host IF / Scheduler / Fault-Reset / PMU / Region Sequencer / Slot Frame / Stream Queue / Tile UCE / Physical)

#### 5.3.1 Runtime ABI

**文档**: `elenor_runtime_abi/ELENOR_Runtime_Command_Event_ABI_Design.md` (481 行)

**P0 问题**: 详见 [P0-4](#p0-4-事件等待缺少-sequencegeneration-字段)、[P0-5](#p0-5-fault-record-有两个不兼容的-v0-定义)。

**P1 问题**:

1. V1 command subset 未冻结：enum 暴露了直接 engine commands 但 Scheduler 不消费
2. Timeout clock-domain 语义推迟
3. `fault_record_slot` ownership 不够紧凑

#### 5.3.2 Host Interface

**文档**: `elenor_host_interface/ELENOR_Host_Interface_Design.md` (463 行)

**P0 问题**:

1. **Queue/context generation 缺失**: queue config 和 doorbell payload 没有 queue generation / context epoch。replayed doorbell 可在 reset 后 target 重建的 queue。建议添加 `queue_generation`。
2. **DOORBELL 语义在 register boundary 模糊**: CSR table 说 "queue_id, tail or increment"，internal entry 使用 absolute `tail_snapshot`。两种模型有不同 replay/idempotency 行为。建议冻结 `DOORBELL(queue_id, absolute_tail, doorbell_seq)`。
3. **Completion authority 在 event-table memory 和 optional host mirror 之间分裂**: 建议每种 protocol mode 定义唯一 authoritative completion object。

#### 5.3.3 Global Scheduler

**文档**: `elenor_global_scheduler/ELENOR_Global_Scheduler_Design.md` (498 行)

**P0 问题**:

1. **`RESET_DOMAIN` 是 V1 command 但无 Scheduler contract**: cutline 包含 `RESET_DOMAIN`，但 decode/descriptor/flow 只详述 `LAUNCH_REGION`, `DMA`, `EVENT_WAIT`, `EVENT_SIGNAL`。建议添加 `RESET_DOMAIN` command format、arbitration behavior、affected-resource model。
2. **Multi-group launch 结构性不一致**: region launch descriptor 只有 singular `group_id` + `tile_mask`，但 §5.1 说 "Region task 可跨多个 group"。建议 V1 禁止 multi-group region 或添加 `group_mask[]` + completion policy。
3. **Scheduler-visible wait 不携带 event generation/sequence**: 详见 [P0-4](#p0-4-事件等待缺少-sequencegeneration-字段)。

#### 5.3.4 Fault / Reset

**文档**: `elenor_fault_reset/ELENOR_Fault_Reset_Design.md` (376 行)

**P0 问题**:

1. **Event-fault 编码与其他所有 event model 冲突**: 详见 [§4.2](#42-event-status-编码冲突)。
2. **Fault record v0 与 Runtime ABI 不 wire-compatible**: 详见 [P0-5](#p0-5-fault-record-有两个不兼容的-v0-定义)。
3. **`ELENOR_RESET_F_CLEAR_PMU` 不安全**: reset flags 允许 PMU clear 作为 reset 的一部分，但同一文档要求 PMU snapshot before clear。建议重新定义为 post-snapshot, post-publication action。

#### 5.3.5 PMU

**文档**: `elenor_pmu/ELENOR_PMU_Design.md` (375 行)

**P0 问题**:

1. **PMU ABI 在子系统文档中不统一**: 详见 [§4.5](#45-pmu-counter-命名碎片化)。
2. **Snapshot/freeze 语义仍开放**: `FROZEN` 状态下 counter 是否停止更新或继续在 shadow bank 中运行未冻结。Host Interface、Fault/Reset 和 runtime tooling 需要确定性 snapshot visibility。建议冻结 V1 模型：double-buffered atomic snapshot with generation 或 freeze-then-read with explicit bounded latency。

#### 5.3.6 Region Sequencer

**文档**: `elenor_region_sequencer/ELENOR_Region_Sequencer_Design.md` (463 行)

**P0 问题**:

1. **`elenor_region_stage_desc_t` 无法编码文档其他部分展示的 stage behavior**: opcode list 和 example 需要 stream init/bindings、DMA descriptor IDs、collective refs、EOS behavior、stage-local wait/signal 语义，但 stage descriptor 只有 `stage_id`, `tile_mask`, template/program IDs, flags, in/out stream masks, event base/signal。
2. **Region Program event signaling 使用未定义的 status vocabulary**: `signal.event event_id, status` 使用 `status=ready`，但 architecture-wide event model 只定义 `PENDING/DONE/ERROR/TIMEOUT/RESET`。
3. **Queue/stream 资源操作上必须但缺失于 launch ABI**: First-Silicon opcodes 包括 `init.stream`, `wait.stream`, `wait.credit`，但 region launch descriptor 没有 queue-descriptor table pointer/count。

#### 5.3.7 Slot Frame

**文档**: `elenor_slot_frame/ELENOR_Tile_Slot_Frame_Design.md` (399 行)

**P0 问题**:

1. **Address-template formula 对 2D/linear tile addressing 未充分指定**: template 有 `tile_stride_x` 和 `tile_stride_y`，但 effective-address formula 只用 singular `tile_stride`，未定义 canonical tile-index mapping。不同实现可能从同一 descriptor 计算出不同地址。
2. **Generation-safe patching 在 prose 中要求但 binary patch contract 中缺失**: `elenor_tile_frame_v0_t` 有 `generation`，warm launch 要求 frame-generation matching，但 `elenor_desc_patch_record_v0_t` 没有 generation/epoch 字段。
3. **`base` 字段的 address-space 含义未冻结**: slot entry 用 32-bit `base` 和 `size`，但 address generation 将它们与 64-bit `base_addr` 混用。未说明 slot base 是 L1-local byte offset、bank-local physical address 还是其他。

#### 5.3.8 Stream Queue

**文档**: `elenor_stream_queue/ELENOR_Stream_Queue_Design.md` (396 行)

**P0 问题**:

1. **Token ABI 没有 generation**: 详见 [§4.10](#410-stream-queue-语义未完全冻结)。
2. **EOS credit consumption 未冻结**: producer state machine 说是否消耗 credit 仍 open。
3. **Base ABI 暴露 `BROADCAST` queue kind 但 First Silicon 排除它**。

#### 5.3.9 Tile UCE

**文档**: `elenor_tile_uce/ELENOR_Tile_UCE_Design.md` (682 行)

**P0 问题**:

1. **Tile ISA 内部不一致**: opcode catalog 定义了 Control, Engine Launch, Sync, Stream, Descriptor, Profiling/Error 组，但 example 使用 `br.err` 和 `signal.tile.done`，两者均未定义。catalog 也没有 signal instruction。
2. **Patch ABI 无法编码文档中展示的 stream example**: stream pipeline example patches `token.payload(r_in)` 和 `token.payload(r_out)`，但 patch descriptor 只有 generic `ELENOR_PATCH_STREAM_TOKEN` + 一个 `source_ref`。未指定是 token payload address、bytes、metadata 还是 fault index。
3. **Engine launch request 缺少 generation/tag 信息**: prepared-task validation 检查 program/frame/descriptor freshness，warm patch flow 要求 version/invalidate discipline，但 `elenor_engine_launch_req_v0_t` 只有 `desc_slot`, `event_id`, `context_id`, `timeout_cycles`。

#### 5.3.10 Physical / Timing / Power

**文档**: `elenor_physical_timing_power/ELENOR_Physical_Timing_Power_Design.md` (339 行)

**P0 问题**:

1. **Timeout correctness 仍物理上未定义**: 多个控制面文档使用 `timeout_cycles` 作为架构正确性，但本文挡仍将 "event timeout 与 clock 关系" 留待后续。event broadcast latency、quiesce、reset、PMU timestamp alignment 都依赖它。
2. **Low-power restore 语义对架构可见控制面状态未冻结**: `WAKE_RESTORE` 后 descriptor cache、program SRAM、state cache、event/status region 的有效性明确留 open，但同一文档将 sleep/retention 包含在 V1 physical state machine 中。

---

### 5.4 软件栈与系统支撑 (Compiler / Driver-Firmware / Workload Mapping / Verification / Package / OPA)

#### 5.4.1 Compiler Stack

**文档**: `elenor_compiler/ELENOR_Compiler_Stack_Design.md` (447 行)

**P0 问题**:

1. **Shape-class selection 不是正式 ABI**: compiler 要求 dynamic shape 变成有限 version 或 ragged descriptor，但缺少 predicate language、precedence rule、overlap rule、default/fail behavior。runtime 可能选择错误的 command path。
2. **Descriptor ABI 太 generic 无法驱动 BOA/EVU/MFE/USE RTL**: 只有一个 generic template example，缺少 BOA-specific fields (`reduce_op`, `dataflow`, `rounding_mode`, `saturation_mode`)。
3. **Command-template lowering 没有对应的 binary command contract**: compiler 声明 `RuntimeCommandIR` 必须有完整 event dependency、barrier、timeout、fault-slot，但未定义 binary command header、opcode set、wait semantics。

**P1 问题**:

1. Kernel-library manifest 是 prose 不是 loadable contract
2. Memory/slot lifetime 是隐含的但不是 compiler object model 中的一等公民
3. PMU expectation 只有定性描述
4. Fault attribution metadata 未标准化

#### 5.4.2 Driver / Firmware / Runtime

**文档**: `elenor_driver_firmware/ELENOR_Driver_Firmware_Runtime_Design.md` (466 行)

**P0 问题**:

1. **没有具体的 binary command-entry format 或 opcode contract**: runtime 构建 command，firmware 消费，但只有 context creation、submit、doorbell struct，没有 command header、opcode enum、payload union。
2. **Event-table 语义和 memory ordering 不够指定**: 未定义 event record layout、write atomicity、sequence-width/wrap rules、producer identity、fence/order guarantees。
3. **Reset/drain 未定义 in-flight work 的 quiesce criteria**: 未定义 outstanding DMA、MFE page walk、BOA/EVU operation、USE checkpoint write 何为 "safe" drain vs must kill。

**P1 问题**:

1. Firmware 对 package metadata 的可见性不清晰
2. Program residency 被引用但未规范
3. PMU snapshot 缺少 atomicity 语义
4. Security/boot chain 推迟太晚

#### 5.4.3 Workload Mapping

**文档**: `elenor_workload_mapping/ELENOR_Workload_Mapping_Design.md` (525 行)

**P0 问题**:

1. **Paged-attention page 语义仍推迟**: 详见 [P0-3](#p0-3-分页注意力跨引擎-abi-未冻结)。
2. **MoE 语义 mode 需要正确定义但未指定**: duplicate index policy、capacity overflow policy、combine precision 均要求 "由 descriptor mode 明确" 但推迟冻结。
3. **USE checkpoint/restore 语义不够具体**: 详见 [P0-8](#p0-8-use-checkpointevent-assist-语义被推迟但-v1-依赖它们)。

**P1 问题**:

1. `WorkloadPlan` 重复 package/container 概念但无 normalization rule
2. Paged-attention pseudocode 未展示性能模型依赖的 overlap
3. Multi-model scheduling 太 policy-light
4. Embedding/GNN 覆盖太浅

#### 5.4.4 Verification / Bring-up

**文档**: `elenor_verification_bringup/ELENOR_Verification_Bringup_Design.md` (298 行)

**P0 问题**:

1. **Plan 依赖共享 ABI/schema lock 但未定义 source of truth**: 未说明 schema 是从 compiler/package source 生成、手工维护 header 还是 verification-local table。
2. **Fault taxonomy 和 checksum policy 推迟但 Phase 1 依赖它们**。
3. **Transaction record 缺少 package/build identity**: 包含 command ID、descriptor ID、event ID 等，但不包含 package build ID、ABI tuple、kernel ID、shape-class ID。

#### 5.4.5 Executable Package

**文档**: `elenor_executable_package/ELENOR_Executable_Package_Design.md` (418 行)

**P0 问题**:

1. **Section taxonomy 相对于 object model 和其他文档不完整**: object model 包含 `EventInitTable`, `CommandTemplates[]`, `SlotFrameTemplates[]`, `RelocationTable[]`, `PmuManifest`, `DebugManifest`，但 section enum 没有 kernel binding、stream-queue descriptor、target-profile metadata、event-init table 的显式 section。
2. **Relocation model 不够深**: 没有 relocation-entry struct、relocation types、field widths、endianness、target-section references、bounds semantics、patch-owner rules。
3. **Mutable launch metadata 未与 immutable program/schedule bytes 干净分离**: warm launch 要么 in-place patch program/schedule bytes，要么在 program text 和 descriptor 之间不一致地复制 state。

#### 5.4.6 OPA

**文档**: `elenor_opa/ELENOR_OPA_Design.md` (351 行)

**P0 问题**:

1. **BOA descriptor contract 需要驱动 OPA 但在 stack 中未冻结**: OPA behavior 依赖 BOA descriptor fields (`op_kind`, `reduce_op`, `dataflow`, `rounding_mode`, `saturation_mode`)，但 compiler 的 descriptor example 只有 generic fields。
2. **Arithmetic 语义在 correctness-critical case 中定义不足**: 未定义 accumulation order、BF16/INT conversion points、NaN/Inf behavior for max/min、reduction associativity。
3. **OPA reduce mode vs EVU ownership 的合法使用边界模糊**: 架构文档将 softmax/norm/tail 分配给 EVU，但 OPA example 使用 `OPA_PASS + OPA_REDUCE_ADD/MAX` 做 reduce-sum tail 和 softmax-max assist。

---

## 6. 工程规范问题

### 6.1 重复文件

`design/elenor_evu/ELENOR_EVU_Design copy.md` (2934 行) 是 EVU 设计文档的旧版本（标题为 "EVU 设计文档"），当前版本 `ELENOR_EVU_Design.md` (1680 行) 已演进为 "EVU-MT 设计文档"。旧文件应删除以避免混淆。

### 6.2 Git 状态

`README.md` 和 `.omp/` 目录处于 untracked 状态，建议加入版本控制。

### 6.3 SKILL.md 未填写

`.omp/skills/ai-chips/SKILL.md` 的 `description` 字段仍为模板默认值 ("Describe what this skill does and when to use it...")，建议填写实际描述。

### 6.4 README 文档导航不完整

README 的文档导航只列出了 16 份文档，但 `design/` 目录下有 27 份设计文档。以下文档未在 README 中列出：

- elenor_global_scheduler
- elenor_host_interface
- elenor_fault_reset
- elenor_pmu
- elenor_region_sequencer
- elenor_slot_frame
- elenor_stream_queue
- elenor_tile_uce
- elenor_executable_package
- elenor_opa
- elenor_physical_timing_power

### 6.5 图片引用

部分文档引用了 `.drawio` 文件，但 README 只展示了 `.png` / `.svg` 导出。建议确认所有 `.drawio` 源文件都有对应的导出图片，并在文档中统一引用导出格式。

---

## 7. 冻结优先级排序建议

### 第一优先级：ABI 冻结 (阻塞所有并行开发)

| 序号 | 冻结项                                                                               | 阻塞原因                                        | 涉及文档                                 |
| :--: | ------------------------------------------------------------------------------------ | ----------------------------------------------- | ---------------------------------------- |
|  1   | Command enum + command header binary layout                                          | firmware 无法 decode compiler 输出              | Runtime ABI, 架构文档, Scheduler         |
|  2   | Event record layout (status enum + producer + sequence + context)                    | wait/wakeup/fault 传播语义发散                  | Runtime ABI, Scheduler, Fault/Reset, UCE |
|  3   | Fault record binary layout (统一为一份)                                              | driver/firmware 无法安全解码                    | Runtime ABI, Fault/Reset, 所有 engine    |
|  4   | Descriptor common header (abi_version + desc_size + flags) + engine-specific payload | compiler/RTL/verification 对同一 bytes 解释不同 | 所有 engine, Compiler, Package           |
|  5   | PMU counter ID registry                                                              | firmware/driver tooling 碎片化                  | PMU, 所有 block                          |

### 第二优先级：正确性契约冻结 (阻塞 First Silicon V1)

| 序号 | 冻结项                                                                     | 阻塞原因                                         |
| :--: | -------------------------------------------------------------------------- | ------------------------------------------------ |
|  6   | Tile program residency/install path + epoch                                | cold launch 可能向无 program 的 tile 派发        |
|  7   | Paged attention inter-engine ABI (K/V layout + stream credit + EOS/error)  | V1 标题 workload 无法端到端闭环                  |
|  8   | Event wait sequence/generation field                                       | stale completion 误用                            |
|  9   | BOA split-K partial identity + reduction order + accumulator epoch         | split-K GEMM 和 attention 静默损坏               |
|  10  | USE checkpoint/restore policy + event-assist semantics                     | Mamba/RWKV 恢复正确性                            |
|  11  | Stream Queue credit/EOS/generation model                                   | credit leak 和 deadlock                          |
|  12  | Reset domain generation model (queue/event/token/descriptor/frame/program) | reset 后 stale handle 复用                       |
|  13  | Timeout clock domain 和 conversion rules                                   | event broadcast/quiesce/reset/PMU timestamp 依赖 |

### 第三优先级：范围澄清 (避免 V1 范围争议)

| 序号 | 冻结项                                                                         | 阻塞原因                                        |
| :--: | ------------------------------------------------------------------------------ | ----------------------------------------------- |
|  14  | EVU V1 cutline 与 acceptance 对齐                                              | gather/SFU/structured mask 是否 V1 must-have    |
|  15  | MoE descriptor mode (duplicate/overflow/combine)                               | 两个实现都能过 smoke test 但行为不同            |
|  16  | V1 command subset bitmap (哪些 enum 值合法)                                    | 软件可生成 Scheduler 不消费的 command           |
|  17  | Numeric profile per mode (BF16/INT8 conversion, rounding, saturation, NaN/Inf) | golden 与 RTL drift                             |
|  18  | First Silicon V1 SRAM/NoC profile                                              | 所有文档的 "由后续规格冻结" 参数需要一个 target |

### 第四优先级：工程清理

| 序号 | 行动                             | 说明            |
| :--: | -------------------------------- | --------------- |
|  19  | 删除 `ELENOR_EVU_Design copy.md` | 旧版本残留      |
|  20  | 补全 README 文档导航             | 11 份文档未列出 |
|  21  | 填写 SKILL.md description        | 模板默认值      |
|  22  | Git track README.md 和 .omp/     | 当前 untracked  |

---

## 8. 总结

### 架构判断

ELENOR 的架构方向是正确的：

1. **四引擎分工** (BOA/EVU/MFE/USE) 精准地拆开了 dense compute、irregular compute、memory flow 和 state/control，避免了任一模块承担过多职责。
2. **分层控制流** (Graph → Region → Tile → Engine) 和 **Tile-SPMD** 编程范式是成熟且可实现的选择。
3. **V1 范围切分** (Architecture V1 / First Silicon V1 / V2 Reserved) 是正确的方法论。
4. **验证计划** 务实，bring-up 顺序正确。

### 核心风险

架构的核心风险不在概念层面，而在**契约定义层面**：

- 27 份文档描述了同一个系统，但共享 ABI 仍是"样例"而非冻结定义
- 同一概念在不同文档中有不同的定义（command enum、event status、fault record、descriptor naming、PMU counter、address model、ID 位宽）
- 多个 V1-critical 语义被推迟（paged attention ABI、split-K、checkpoint/restore、event sequence、tile program residency）

### 建议的下一步

**不是更多 feature work，而是一次 contract-freeze pass**：

1. 产出 5 份机器可读的 ABI schema 文件（[§3 P0-1](#p0-1-共享-abi-尚未冻结所有文档仍使用样例结构体)）
2. 按 [§7](#7-冻结优先级排序建议) 的优先级顺序冻结 13 个正确性契约
3. 统一所有文档的命名规范、ID 位宽、枚举值
4. 确保所有文档 normative reference 冻结后的 ABI schema

这些 contract 补齐后，ELENOR 文档才能从架构评审稿进入 RTL、compiler、runtime、driver、firmware 可以并行拆解的规格阶段。

---

> **本评审基于 2026-06-23 的文档状态。所有行号和章节引用均基于该日期的文件版本。**
